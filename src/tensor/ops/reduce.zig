//!
//! tensor/ops/reduce.zig — Reduction operations with axis
//!
//! Purpose:
//!   Reduce a tensor along a specified axis by summing, averaging, or taking
//!   the maximum. The axis dimension collapses to size 1 in the output.
//!
//! What "reducing along an axis" means:
//!   If you have a (2,3) tensor and reduce along axis 0, you combine the
//!   two rows into one. The output is (1,3) — each output element is the
//!   sum/mean/max of the two elements stacked above it.
//!   If you reduce along axis 1, you combine the three columns into one.
//!   The output is (2,1) — each output element is the sum/mean/max across
//!   the three elements in that row.
//!
//! Shape contract:
//!   sum(t: (2,3), axis=0) -> (1,3)
//!   sum(t: (2,3), axis=1) -> (2,1)
//!   sumAll(t: (2,3))      -> (1,)  (scalar in a rank-0-ish wrapper)
//!
//! Memory ownership:
//!   All functions return a new owned tensor. Caller must deinit.
//!
//! Error conditions:
//!   InvalidArgument — axis >= tensor's ndim
//!   OutOfMemory — allocation failure
//!
//! TODO: support keep_dims=false (remove the axis dim entirely, not just set to 1)
//!

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Shape = @import("../shape.zig").Shape;
const computeStrides = @import("../shape.zig").computeStrides;
const totalElements = @import("../shape.zig").totalElements;
const Tensor = @import("../tensor.zig").Tensor;
const DType = @import("../../core/dtype.zig").DType;
const Device = @import("../../core/device.zig").Device;

/// Sum all elements along the given axis.
///
/// Worked example:
///   t = [[1, 2, 3],
///        [4, 5, 6]]   shape (2,3)
///   sum(t, axis=0) = [[5, 7, 9]]   shape (1,3)
///   sum(t, axis=1) = [[6],
///                     [15]]         shape (2,1)
pub fn sum(allocator: std.mem.Allocator, tensor: Tensor, axis: u2) !Tensor {
    if (@as(usize, axis) >= tensor.shape.ndim()) return LabError.InvalidArgument;

    // Output shape is the same as input, but the reduced axis has dim=1
    var out_dims = tensor.shape.dims;
    out_dims[axis] = 1;
    const out_shape = Shape{ .dims = out_dims, .rank = tensor.shape.rank };
    var out = try Tensor.init(allocator, out_shape);

    const axis_size = tensor.shape.dims[axis];
    const out_n = totalElements(out_shape);
    const ndim = tensor.shape.ndim();

    // For each position in the output, iterate over the reduction axis
    // in the input and accumulate.
    for (0..out_n) |out_i| {
        // Decompose flat output index into multi-dim coordinates
        var coords = tensor.shape.dims;
        var remaining = out_i;
        for (0..ndim) |d| {
            // Process from rightmost dim
            const rev_d = ndim - 1 - d;
            coords[rev_d] = remaining % out_dims[rev_d];
            remaining /= out_dims[rev_d];
        }
        // Now accumulate along the reduction axis
        var acc: f32 = 0.0;
        for (0..axis_size) |k| {
            coords[axis] = k;
            // Compute flat index from coords
            var in_i: usize = 0;
            for (0..ndim) |d| {
                in_i += coords[d] * tensor.strides.values[d];
            }
            acc += tensor.data[in_i];
        }
        out.data[out_i] = acc;
    }
    return out;
}

/// Mean along the given axis = sum / axis_size.
pub fn mean(allocator: std.mem.Allocator, tensor: Tensor, axis: u2) !Tensor {
    if (@as(usize, axis) >= tensor.shape.ndim()) return LabError.InvalidArgument;
    const out = try sum(allocator, tensor, axis);
    const axis_size = @as(f32, @floatFromInt(tensor.shape.dims[axis]));
    for (out.data) |*v| {
        v.* /= axis_size;
    }
    return out;
}

/// Max along the given axis.
pub fn max(allocator: std.mem.Allocator, tensor: Tensor, axis: u2) !Tensor {
    if (@as(usize, axis) >= tensor.shape.ndim()) return LabError.InvalidArgument;

    var out_dims = tensor.shape.dims;
    out_dims[axis] = 1;
    const out_shape = Shape{ .dims = out_dims, .rank = tensor.shape.rank };
    var out = try Tensor.init(allocator, out_shape);

    const axis_size = tensor.shape.dims[axis];
    const out_n = totalElements(out_shape);
    const ndim = tensor.shape.ndim();

    for (0..out_n) |out_i| {
        var coords = tensor.shape.dims;
        var remaining = out_i;
        for (0..ndim) |d| {
            const rev_d = ndim - 1 - d;
            coords[rev_d] = remaining % out_dims[rev_d];
            remaining /= out_dims[rev_d];
        }
        var max_val: f32 = -std.math.inf(f32);
        for (0..axis_size) |k| {
            coords[axis] = k;
            var in_i: usize = 0;
            for (0..ndim) |d| {
                in_i += coords[d] * tensor.strides.values[d];
            }
            if (tensor.data[in_i] > max_val) max_val = tensor.data[in_i];
        }
        out.data[out_i] = max_val;
    }
    return out;
}

/// Sum all elements into a scalar (rank-0-ish: shape {1}).
pub fn sumAll(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    const out_shape = Shape.init1D(1);
    var out = try Tensor.init(allocator, out_shape);
    var acc: f32 = 0.0;
    for (tensor.data) |v| acc += v;
    out.data[0] = acc;
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "sum along axis 0 of (2,3)" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    // [[1, 2, 3],
    //  [4, 5, 6]]
    t.data[0] = 1;
    t.data[1] = 2;
    t.data[2] = 3;
    t.data[3] = 4;
    t.data[4] = 5;
    t.data[5] = 6;

    var out = try sum(allocator, t, 0);
    defer out.deinit(allocator);
    // axis 0: sum each column -> [1+4, 2+5, 3+6] = [5, 7, 9]
    try std.testing.expectEqual(@as(f32, 5.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 7.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), out.data[2]);
}

test "sum along axis 1 of (2,3)" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    t.data[0] = 1;
    t.data[1] = 2;
    t.data[2] = 3;
    t.data[3] = 4;
    t.data[4] = 5;
    t.data[5] = 6;

    var out = try sum(allocator, t, 1);
    defer out.deinit(allocator);
    // axis 1: sum each row -> [1+2+3, 4+5+6] = [6, 15]
    try std.testing.expectEqual(@as(f32, 6.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 15.0), out.data[1]);
}

test "mean along axis 0" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    t.data[0] = 2;
    t.data[1] = 4;
    t.data[2] = 6;
    t.data[3] = 8;
    t.data[4] = 10;
    t.data[5] = 12;

    var out = try mean(allocator, t, 0);
    defer out.deinit(allocator);
    // mean of each column: [(2+8)/2, (4+10)/2, (6+12)/2] = [5, 7, 9]
    try std.testing.expectEqual(@as(f32, 5.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 7.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), out.data[2]);
}

test "max along axis 0" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    t.data[0] = 1;
    t.data[1] = 5;
    t.data[2] = 3;
    t.data[3] = 4;
    t.data[4] = 2;
    t.data[5] = 6;

    var out = try max(allocator, t, 0);
    defer out.deinit(allocator);
    // max of each column: [max(1,4), max(5,2), max(3,6)] = [4, 5, 6]
    try std.testing.expectEqual(@as(f32, 4.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, 6.0), out.data[2]);
}

test "sumAll reduces to scalar" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    t.data[0] = 1;
    t.data[1] = 2;
    t.data[2] = 3;
    t.data[3] = 4;
    t.data[4] = 5;
    t.data[5] = 6;

    var out = try sumAll(allocator, t);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), out.shape.dims[0]);
    try std.testing.expectEqual(@as(f32, 21.0), out.data[0]);
}

test "reduce rejects invalid axis" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    try std.testing.expectError(LabError.InvalidArgument, sum(allocator, t, 3));
}
