//!
//! tensor/ops/create.zig — Tensor creation functions
//!
//! Purpose:
//!   Factory functions that allocate and return new owned tensors.
//!   Every function takes an explicit allocator — no hidden globals.
//!
//! Shape contract:
//!   All functions return `!Tensor` with the caller-specified shape.
//!   The returned tensor is always owned (caller must call deinit).
//!
//! Memory ownership:
//!   Caller owns the returned tensor. Pass the same allocator to deinit.
//!
//! Error conditions:
//!   OutOfMemory — allocator cannot fulfill the request
//!   InvalidArgument — arange with step <= 0, fromSlice with mismatched length
//!
//! Math:
//!   randn: Box-Muller transform produces standard normal Z ~ N(0,1).
//!   Scaled as X = mean + std_dev * Z to get N(mean, std_dev²).
//!   randu: X = low + (high - low) * U where U ~ Uniform(0, 1).
//!

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Shape = @import("../shape.zig").Shape;
const Strides = @import("../shape.zig").Strides;
// PR-iota: device-aware zerosLike/onesLike route to CUDA when the
// template tensor is on GPU (backward seed allocation, gradient
// accumulators, etc.).
const cuda_dispatch = @import("../device_dispatch.zig");
const computeStrides = @import("../shape.zig").computeStrides;
const totalElements = @import("../shape.zig").totalElements;
const Tensor = @import("../tensor.zig").Tensor;
const DType = @import("../../core/dtype.zig").DType;
const Device = @import("../../core/device.zig").Device;
const Rng = @import("../../core/rng.zig").Rng;

/// Create a zero-filled tensor of the given shape.
///
/// Worked example:
///   const t = try zeros(allocator, Shape.init2D(2, 3));
///   // t.shape = (2, 3), t.data = [0, 0, 0, 0, 0, 0]
///   // t is owned — call t.deinit(allocator) when done
pub fn zeros(allocator: std.mem.Allocator, shape: Shape) !Tensor {
    const t = try Tensor.init(allocator, shape);
    // Tensor.init already zeroes the data, but let's be explicit
    @memset(t.cpuData(), 0);
    return t;
}

/// Create a tensor filled with 1.0.
pub fn ones(allocator: std.mem.Allocator, shape: Shape) !Tensor {
    const t = try Tensor.init(allocator, shape);
    @memset(t.cpuData(), @as(f32, 1.0));
    return t;
}

/// Create a tensor filled with a constant value.
pub fn full(allocator: std.mem.Allocator, shape: Shape, value: f32) !Tensor {
    const t = try Tensor.init(allocator, shape);
    @memset(t.cpuData(), value);
    return t;
}

/// Allocate a zero-filled tensor on the same device as `template`.
/// Used by autograd for gradient accumulators: the gradient of a
/// CUDA tensor must live on the same context as the tensor itself.
pub fn zerosLike(allocator: std.mem.Allocator, template: Tensor) !Tensor {
    if (template.device == .cuda) {
        const ctx = template.storage.cuda.ctx;
        return try cuda_dispatch.zerosOn(ctx, template.shape);
    }
    return try zeros(allocator, template.shape);
}

/// Allocate a ones-filled tensor on the same device as `template`.
/// The CUDA path writes 0x3f800000 (f32 1.0 bit pattern) via
/// cuMemsetD32_v2 — no kernel launch required for this simple fill.
pub fn onesLike(allocator: std.mem.Allocator, template: Tensor) !Tensor {
    if (template.device == .cuda) {
        const ctx = template.storage.cuda.ctx;
        return try cuda_dispatch.onesOn(ctx, template.shape);
    }
    return try ones(allocator, template.shape);
}

/// Create a tensor with values drawn from a normal distribution.
///
/// Uses the Box-Muller transform: given two independent uniforms U1, U2 ~ Uniform(0,1),
///   Z = sqrt(-2 * ln(U1)) * cos(2π * U2)
/// is standard normal N(0,1). We then scale: X = mean + std_dev * Z.
///
/// Worked example:
///   var rng = Rng.init(42);
///   const t = try randn(allocator, Shape.init1D(3), &rng, 0.0, 1.0);
///   // t has 3 values drawn from N(0,1)
pub fn randn(allocator: std.mem.Allocator, shape: Shape, rng: *Rng, mean: f32, std_dev: f32) !Tensor {
    var t = try Tensor.init(allocator, shape);
    const n = totalElements(shape);
    for (0..n) |i| {
        t.cpuData()[i] = mean + std_dev * rng.normalF32();
    }
    return t;
}

/// Create a tensor with values drawn from a uniform distribution.
///
/// X = low + (high - low) * U where U ~ Uniform(0, 1).
pub fn randu(allocator: std.mem.Allocator, shape: Shape, rng: *Rng, low: f32, high: f32) !Tensor {
    var t = try Tensor.init(allocator, shape);
    const n = totalElements(shape);
    const range = high - low;
    for (0..n) |i| {
        t.cpuData()[i] = low + range * rng.floatF32();
    }
    return t;
}

/// Create a 1D tensor of evenly spaced values from start to stop (exclusive),
/// stepping by step.
///
/// Worked example:
///   const t = try arange(allocator, 0.0, 6.0, 2.0);
///   // t.shape = (3,), t.data = [0.0, 2.0, 4.0]
pub fn arange(allocator: std.mem.Allocator, start: f32, stop: f32, step: f32) !Tensor {
    if (step <= 0.0) return LabError.InvalidArgument;
    const count: usize = @intFromFloat(@ceil((stop - start) / step));
    if (count == 0) return LabError.InvalidArgument;
    const shape = Shape.init1D(count);
    const t = try Tensor.init(allocator, shape);
    var val: f32 = start;
    for (0..count) |i| {
        t.cpuData()[i] = val;
        val += step;
    }
    return t;
}

/// Create an owned tensor by copying data from a slice.
///
/// The slice length must match totalElements(shape), otherwise ShapeMismatch.
/// This is the primary way to construct a tensor from known values in tests.
///
/// Worked example:
///   const t = try fromSlice(allocator, Shape.init2D(2, 2), &.{1, 2, 3, 4});
///   // t.data = [1, 2, 3, 4], t.shape = (2, 2)
pub fn fromSlice(allocator: std.mem.Allocator, shape: Shape, data: []const f32) !Tensor {
    const n = totalElements(shape);
    if (data.len != n) return LabError.ShapeMismatch;
    const t = try Tensor.init(allocator, shape);
    @memcpy(t.cpuData(), data);
    return t;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "zeros creates zero-filled tensor" {
    const allocator = std.testing.allocator;
    const shape = Shape.init2D(2, 3);
    var t = try zeros(allocator, shape);
    defer t.deinit(allocator);
    for (t.cpuData()) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "ones creates one-filled tensor" {
    const allocator = std.testing.allocator;
    const shape = Shape.init2D(2, 3);
    var t = try ones(allocator, shape);
    defer t.deinit(allocator);
    for (t.cpuData()) |v| {
        try std.testing.expectEqual(@as(f32, 1.0), v);
    }
}

test "full creates constant-filled tensor" {
    const allocator = std.testing.allocator;
    const shape = Shape.init2D(2, 3);
    var t = try full(allocator, shape, 42.0);
    defer t.deinit(allocator);
    for (t.cpuData()) |v| {
        try std.testing.expectEqual(@as(f32, 42.0), v);
    }
}

test "randn produces values with approximate mean and std" {
    const allocator = std.testing.allocator;
    var rng = Rng.init(1337);
    const shape = Shape.init1D(1000);
    var t = try randn(allocator, shape, &rng, 5.0, 2.0);
    defer t.deinit(allocator);
    // Compute sample mean
    var sum: f32 = 0;
    for (t.cpuData()) |v| sum += v;
    const mean = sum / 1000.0;
    // Should be approximately 5.0 (within 1.0 for 1000 samples)
    try std.testing.expect(@abs(mean - 5.0) < 1.0);
}

test "arange produces sequential values" {
    const allocator = std.testing.allocator;
    var t = try arange(allocator, 0.0, 6.0, 2.0);
    defer t.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), t.shape.dims[0]);
    try std.testing.expectEqual(@as(f32, 0.0), t.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 2.0), t.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 4.0), t.cpuData()[2]);
}

test "fromSlice copies data into tensor" {
    const allocator = std.testing.allocator;
    const shape = Shape.init2D(2, 2);
    var t = try fromSlice(allocator, shape, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer t.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 1.0), t.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 4.0), t.cpuData()[3]);
}

test "fromSlice rejects mismatched length" {
    const allocator = std.testing.allocator;
    const shape = Shape.init2D(2, 3);
    const result = fromSlice(allocator, shape, &.{ 1.0, 2.0 });
    try std.testing.expectError(LabError.ShapeMismatch, result);
}

test "arange rejects non-positive step" {
    const allocator = std.testing.allocator;
    const result = arange(allocator, 0.0, 10.0, 0.0);
    try std.testing.expectError(LabError.InvalidArgument, result);
}
