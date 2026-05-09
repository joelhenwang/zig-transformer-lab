//!
//! tensor/ops/elementwise.zig — Elementwise binary/unary ops with broadcasting
//!
//! Purpose:
//!   Elementwise operations between tensors, supporting NumPy-style
//!   broadcasting for ranks up to 3. Broadcasting lets us combine tensors
//!   of different but compatible shapes without explicitly replicating data.
//!
//! Broadcasting rules (NumPy, simplified for rank <= 3):
//!   1. Align shapes from the RIGHT (trailing dimensions first).
//!   2. Two dims are compatible if they are equal, or one of them is 1.
//!   3. The output shape takes the MAX of each pair.
//!   4. A dim of size 1 is "stretched" — the same value is reused for
//!      every position along that dimension.
//!
//! Shape contract:
//!   add(a: (1,3), b: (2,3)) -> (2,3)
//!   add(a: (2,1), b: (2,3)) -> (2,3)
//!   add(a: (1,1), b: (2,3)) -> (2,3)
//!
//! Implementation strategy:
//!   1. Call broadcastShapes(a.shape, b.shape) to get output shape.
//!   2. Allocate output tensor.
//!   3. Iterate flat index i from 0..totalElements(output).
//!   4. Compute multi-dim indices from flat i (using output strides).
//!   5. Map to input indices: if input dim == 1, use 0; else use output index.
//!   6. Read a_val, b_val, compute, write to output.
//!
//! Memory ownership:
//!   All binary ops return a new owned tensor. Caller must deinit.
//!   addInPlace modifies a in-place and does NOT allocate.
//!
//! Error conditions:
//!   ShapeMismatch — shapes are not broadcast-compatible
//!   OutOfMemory — allocation failure
//!
//! TODO: rank > 3 broadcasting, SIMD vectorization
//!

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Shape = @import("../shape.zig").Shape;
const Strides = @import("../shape.zig").Strides;
const computeStrides = @import("../shape.zig").computeStrides;
const totalElements = @import("../shape.zig").totalElements;
const broadcastShapes = @import("../shape.zig").broadcastShapes;
const Tensor = @import("../tensor.zig").Tensor;
const DType = @import("../../core/dtype.zig").DType;
const Device = @import("../../core/device.zig").Device;
const equals = @import("../shape.zig").equals;

/// Map a flat output index to a multi-dimensional input index,
/// accounting for broadcasting. For each dimension: if the input
/// has dim==1 (broadcast), we use index 0; otherwise we use the
/// output's index for that dimension.
fn broadcastIndex(out_idx: usize, out_shape: Shape, in_shape: Shape, in_strides: Strides) usize {
    var result: usize = 0;
    // Walk dimensions from right to left (broadcasting aligns trailing dims)
    var i: usize = out_shape.ndim();
    var remaining = out_idx;
    while (i > 0) {
        i -= 1;
        const dim_size = out_shape.dims[i];
        const coord = remaining % dim_size;
        remaining = remaining / dim_size;
        // If the input has size 1 at this dim (broadcast), coord should be 0
        const in_coord: usize = if (i < in_shape.ndim() and in_shape.dims[i] == 1) 0 else coord;
        if (i < in_shape.ndim()) {
            result += in_coord * in_strides.values[i];
        }
    }
    return result;
}

/// Broadcast elementwise addition: out[i,j,...] = a[i,j,...] + b[i,j,...]
pub fn add(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.data[i] = a.data[a_idx] + b.data[b_idx];
    }
    return out;
}

/// Broadcast elementwise subtraction: out = a - b
pub fn sub(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.data[i] = a.data[a_idx] - b.data[b_idx];
    }
    return out;
}

/// Broadcast elementwise multiplication (Hadamard product, NOT matmul).
pub fn mul(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.data[i] = a.data[a_idx] * b.data[b_idx];
    }
    return out;
}

/// Broadcast elementwise division: out = a / b
/// Division by zero produces Inf or NaN — let IEEE 754 happen, document it.
pub fn div(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.data[i] = a.data[a_idx] / b.data[b_idx];
    }
    return out;
}

/// Add a scalar to every element: out[i] = a[i] + scalar
pub fn addScalar(allocator: std.mem.Allocator, a: Tensor, scalar: f32) !Tensor {
    const out_shape = a.shape;
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        out.data[i] = a.data[i] + scalar;
    }
    return out;
}

/// Multiply every element by a scalar: out[i] = a[i] * scalar
pub fn mulScalar(allocator: std.mem.Allocator, a: Tensor, scalar: f32) !Tensor {
    const out_shape = a.shape;
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        out.data[i] = a.data[i] * scalar;
    }
    return out;
}

/// In-place addition: a += b. No broadcasting — shapes must match exactly.
pub fn addInPlace(a: *Tensor, b: Tensor) !void {
    if (!equals(a.shape, b.shape)) return LabError.ShapeMismatch;
    const n = totalElements(a.shape);
    for (0..n) |i| {
        a.data[i] += b.data[i];
    }
}

/// Elementwise negation: out[i] = -a[i]
pub fn neg(allocator: std.mem.Allocator, a: Tensor) !Tensor {
    const out_shape = a.shape;
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        out.data[i] = -a.data[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "add (2,3) + (2,3) = (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;
    a.data[3] = 4;
    a.data[4] = 5;
    a.data[5] = 6;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.data[0] = 10;
    b.data[1] = 20;
    b.data[2] = 30;
    b.data[3] = 40;
    b.data[4] = 50;
    b.data[5] = 60;

    var out = try add(allocator, a, b);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 11.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 66.0), out.data[5]);
}

test "add broadcast (1,3) + (2,3) = (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(1, 3));
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.data[0] = 10;
    b.data[1] = 20;
    b.data[2] = 30;
    b.data[3] = 40;
    b.data[4] = 50;
    b.data[5] = 60;

    var out = try add(allocator, a, b);
    defer out.deinit(allocator);
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    try std.testing.expectEqual(@as(f32, 11.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 22.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, 33.0), out.data[2]);
    // Row 1: [1+40, 2+50, 3+60] = [41, 52, 63]  (a is broadcast)
    try std.testing.expectEqual(@as(f32, 41.0), out.data[3]);
    try std.testing.expectEqual(@as(f32, 52.0), out.data[4]);
    try std.testing.expectEqual(@as(f32, 63.0), out.data[5]);
}

test "sub (2,3) - (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer a.deinit(allocator);
    a.data[0] = 10;
    a.data[1] = 20;
    a.data[2] = 30;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.data[0] = 1;
    b.data[1] = 2;
    b.data[2] = 3;

    var out = try sub(allocator, a, b);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 9.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 27.0), out.data[2]);
}

test "mul elementwise" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 2;
    a.data[1] = 3;
    a.data[2] = 4;

    var b = try Tensor.init(allocator, Shape.init1D(3));
    defer b.deinit(allocator);
    b.data[0] = 5;
    b.data[1] = 6;
    b.data[2] = 7;

    var out = try mul(allocator, a, b);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 10.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 28.0), out.data[2]);
}

test "div produces Inf for division by zero" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(2));
    defer a.deinit(allocator);
    a.data[0] = 1.0;
    a.data[1] = 0.0;

    var b = try Tensor.init(allocator, Shape.init1D(2));
    defer b.deinit(allocator);
    b.data[0] = 0.0;
    b.data[1] = 0.0;

    var out = try div(allocator, a, b);
    defer out.deinit(allocator);
    // 1.0/0.0 = +Inf, 0.0/0.0 = NaN
    try std.testing.expect(std.math.isPositiveInf(out.data[0]));
    try std.testing.expect(std.math.isNan(out.data[1]));
}

test "addScalar" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;

    var out = try addScalar(allocator, a, 10.0);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 11.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 13.0), out.data[2]);
}

test "mulScalar" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;

    var out = try mulScalar(allocator, a, 3.0);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 3.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 9.0), out.data[2]);
}

test "neg" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = -2;
    a.data[2] = 0;

    var out = try neg(allocator, a);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, -1.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), out.data[2]);
}

test "addInPlace" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;

    var b = try Tensor.init(allocator, Shape.init1D(3));
    defer b.deinit(allocator);
    b.data[0] = 10;
    b.data[1] = 20;
    b.data[2] = 30;

    try addInPlace(&a, b);
    try std.testing.expectEqual(@as(f32, 11.0), a.data[0]);
    try std.testing.expectEqual(@as(f32, 33.0), a.data[2]);
}

test "addInPlace rejects shape mismatch" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer a.deinit(allocator);
    var b = try Tensor.init(allocator, Shape.init2D(3, 2));
    defer b.deinit(allocator);
    try std.testing.expectError(LabError.ShapeMismatch, addInPlace(&a, b));
}

test "add broadcast (2,1) + (2,3) = (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 1));
    defer a.deinit(allocator);
    a.data[0] = 100;
    a.data[1] = 200;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.data[0] = 1;
    b.data[1] = 2;
    b.data[2] = 3;
    b.data[3] = 4;
    b.data[4] = 5;
    b.data[5] = 6;

    var out = try add(allocator, a, b);
    defer out.deinit(allocator);
    // Row 0: 100 + [1,2,3] = [101, 102, 103]
    try std.testing.expectEqual(@as(f32, 101.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 103.0), out.data[2]);
    // Row 1: 200 + [4,5,6] = [204, 205, 206]
    try std.testing.expectEqual(@as(f32, 204.0), out.data[3]);
    try std.testing.expectEqual(@as(f32, 206.0), out.data[5]);
}
