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
//! Layout policy (PR-β):
//!   - Binary broadcast ops (add/sub/mul/div): strided inputs OK, output
//!     is always fresh contiguous.
//!   - Scalar ops (addScalar/mulScalar/neg): strided inputs OK, output
//!     is always fresh contiguous.
//!   - addInPlace: both sides MUST be contiguous. Non-contiguous input
//!     returns `LabError.InvalidLayout` rather than silently corrupting
//!     data through a flat-buffer loop.
//!
//! Error conditions:
//!   ShapeMismatch — shapes are not broadcast-compatible
//!   InvalidLayout — addInPlace called on a non-contiguous tensor
//!   OutOfMemory — allocation failure
//!
//! TODO: future: rank-5+ broadcasting (today the CUDA fast path is
//!       stride-aware through rank 4); SIMD vectorization of the
//!       CPU elementwise loops.
//!

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Shape = @import("../shape.zig").Shape;
const Strides = @import("../shape.zig").Strides;
const computeStrides = @import("../shape.zig").computeStrides;
const totalElements = @import("../shape.zig").totalElements;
const broadcastShapes = @import("../shape.zig").broadcastShapes;
const logicalOffsetFromLinear = @import("../shape.zig").logicalOffsetFromLinear;
const Tensor = @import("../tensor.zig").Tensor;
const DType = @import("../../core/dtype.zig").DType;
const Device = @import("../../core/device.zig").Device;
const equals = @import("../shape.zig").equals;
const Tape = @import("../../autograd/tape.zig").Tape;
const Node = @import("../../autograd/node.zig").Node;
const OpKind = @import("../../autograd/node.zig").OpKind;
const SavedData = @import("../../autograd/node.zig").SavedData;
const NodeId = @import("../tensor.zig").NodeId;
// PR-η.2: elementwise dispatch routes CUDA inputs to GPU kernels via
// src/backend/cuda/dispatch.zig. The CPU ops fall back to this file's
// loop-based implementations when inputs live on CPU.
const cuda_dispatch = @import("../device_dispatch.zig");
const shape_isContiguous = @import("../shape.zig").isContiguous;

/// Pick the right CUDA elementwise entry point based on input
/// shape / layout. Same-shape + contiguous inputs take the fast
/// flat-index path; anything else (broadcast, transposed, sliced)
/// goes through the stride-aware rank-4 kernel.
///
/// `op` is one of "add" / "sub" / "mul" / "div" and selects which
/// fast-path / broadcast-path pair to use.
fn cudaBinary(op: enum { add, sub, mul, div }, a: Tensor, b: Tensor) !Tensor {
    const fast = equals(a.shape, b.shape) and
        shape_isContiguous(a.shape, a.strides) and
        shape_isContiguous(b.shape, b.strides);
    if (fast) {
        return switch (op) {
            .add => cuda_dispatch.add(a, b),
            .sub => cuda_dispatch.sub(a, b),
            .mul => cuda_dispatch.mul(a, b),
            .div => cuda_dispatch.div(a, b),
        };
    }
    return switch (op) {
        .add => cuda_dispatch.addBroadcast(a, b),
        .sub => cuda_dispatch.subBroadcast(a, b),
        .mul => cuda_dispatch.mulBroadcast(a, b),
        .div => cuda_dispatch.divBroadcast(a, b),
    };
}

/// Map a flat output index to a multi-dimensional input index,
/// accounting for broadcasting. For each dimension: if the input
/// has dim==1 (broadcast), we use index 0; otherwise we use the
/// output's index for that dimension.
fn broadcastIndex(out_idx: usize, out_shape: Shape, in_shape: Shape, in_strides: Strides) usize {
    var result: usize = 0;
    const out_ndim = out_shape.ndim();
    const in_ndim = in_shape.ndim();
    // Extra leading dims in the output that don't exist in the input.
    // These are broadcasted (input index 0 for all of them).
    // Example: output (B,T,D), input (1,D) → extra_dims = 1
    const extra_dims: usize = if (out_ndim > in_ndim) out_ndim - in_ndim else 0;
    // Walk dimensions from right to left (broadcasting right-aligns shapes)
    var i: usize = out_ndim;
    var remaining = out_idx;
    while (i > 0) {
        i -= 1;
        const dim_size = out_shape.dims[i];
        const coord = remaining % dim_size;
        remaining = remaining / dim_size;
        if (i >= extra_dims) {
            // This output dim maps to an input dim (right-aligned).
            // Output dim i → input dim (i - extra_dims).
            const in_dim = i - extra_dims;
            // If the input has size 1 at this dim (broadcast), coord should be 0
            const in_coord: usize = if (in_shape.dims[in_dim] == 1) 0 else coord;
            result += in_coord * in_strides.values[in_dim];
        }
        // else: extra leading dim in output not present in input → broadcast → add 0
    }
    return result;
}

/// Broadcast elementwise addition: out[i,j,...] = a[i,j,...] + b[i,j,...]
///
/// Device dispatch (PR-η.2):
///   - CPU inputs  -> CPU loop below.
///   - CUDA inputs -> forward the call to backend.cuda.dispatch.add
///     and record the result on the tape. The CUDA dispatch path
///     requires same-shape contiguous inputs for now; broadcast
///     inputs on CUDA are PR-θ territory.
pub fn add(allocator: std.mem.Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    if (a.device == .cuda or b.device == .cuda) {
        var out = try cudaBinary(.add, a, b);
        // Tape recording uses the same path as CPU — cloneTensorData
        // now handles CUDA snapshots via DtoD copy.
        try recordBinaryOp(tape, &out, &a, &b, .add);
        return out;
    }
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.cpuData()[i] = a.cpuData()[a_idx] + b.cpuData()[b_idx];
    }
    try recordBinaryOp(tape, &out, &a, &b, .add);
    return out;
}

/// Broadcast elementwise subtraction: out = a - b
pub fn sub(allocator: std.mem.Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    if (a.device == .cuda or b.device == .cuda) {
        var out = try cudaBinary(.sub, a, b);
        try recordBinaryOp(tape, &out, &a, &b, .sub);
        return out;
    }
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.cpuData()[i] = a.cpuData()[a_idx] - b.cpuData()[b_idx];
    }
    try recordBinaryOp(tape, &out, &a, &b, .sub);
    return out;
}

/// Broadcast elementwise multiplication (Hadamard product, NOT matmul).
pub fn mul(allocator: std.mem.Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    if (a.device == .cuda or b.device == .cuda) {
        var out = try cudaBinary(.mul, a, b);
        try recordBinaryOp(tape, &out, &a, &b, .mul);
        return out;
    }
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.cpuData()[i] = a.cpuData()[a_idx] * b.cpuData()[b_idx];
    }
    try recordBinaryOp(tape, &out, &a, &b, .mul);
    return out;
}

/// Broadcast elementwise division: out = a / b
/// Division by zero produces Inf or NaN — let IEEE 754 happen, document it.
pub fn div(allocator: std.mem.Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    if (a.device == .cuda or b.device == .cuda) {
        var out = try cudaBinary(.div, a, b);
        try recordBinaryOp(tape, &out, &a, &b, .div);
        return out;
    }
    const out_shape = try broadcastShapes(a.shape, b.shape);
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_idx = broadcastIndex(i, out_shape, a.shape, a.strides);
        const b_idx = broadcastIndex(i, out_shape, b.shape, b.strides);
        out.cpuData()[i] = a.cpuData()[a_idx] / b.cpuData()[b_idx];
    }
    try recordBinaryOp(tape, &out, &a, &b, .div);
    return out;
}

/// Add a scalar to every element: out[i] = a[i] + scalar.
///
/// The input `a` may be non-contiguous (e.g. a transposed view) — we
/// translate each logical index through `a.strides` before reading.
/// The output is a fresh contiguous tensor in row-major order.
pub fn addScalar(allocator: std.mem.Allocator, a: Tensor, scalar: f32, tape: ?*Tape) !Tensor {
    if (a.device == .cuda) {
        var out = try cuda_dispatch.addScalar(a, scalar);
        if (tape) |t| {
            if (a.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .add_scalar,
                    .parents = .{ a.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_scalar = .{ .shape = a.shape, .scalar = scalar } },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const out_shape = a.shape;
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_off = logicalOffsetFromLinear(a.shape, a.strides, i);
        out.cpuData()[i] = a.cpuData()[a_off] + scalar;
    }
    if (tape) |t| {
        if (a.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .add_scalar,
                .parents = .{ a.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_scalar = .{ .shape = a.shape, .scalar = scalar } },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }
    return out;
}

/// Multiply every element by a scalar: out[i] = a[i] * scalar.
/// Strided input `a` is handled via logical-to-physical offset translation.
pub fn mulScalar(allocator: std.mem.Allocator, a: Tensor, scalar: f32, tape: ?*Tape) !Tensor {
    if (a.device == .cuda) {
        var out = try cuda_dispatch.mulScalar(a, scalar);
        if (tape) |t| {
            if (a.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .mul_scalar,
                    .parents = .{ a.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_scalar = .{ .shape = a.shape, .scalar = scalar } },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const out_shape = a.shape;
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_off = logicalOffsetFromLinear(a.shape, a.strides, i);
        out.cpuData()[i] = a.cpuData()[a_off] * scalar;
    }
    if (tape) |t| {
        if (a.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .mul_scalar,
                .parents = .{ a.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_scalar = .{ .shape = a.shape, .scalar = scalar } },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }
    return out;
}

/// In-place addition: a += b. Shapes must match exactly (no broadcasting).
///
/// LAYOUT POLICY: both `a` and `b` must be contiguous. An in-place op that
/// walks a non-contiguous lhs via `data[i]` would write into the wrong
/// physical slots of the parent buffer; combined with aliasing between
/// `a` and `b` (e.g. a view and its parent), it is a subtle data-
/// corruption trap. We reject upfront with `error.InvalidLayout` instead.
pub fn addInPlace(a: *Tensor, b: Tensor) !void {
    if (!equals(a.shape, b.shape)) return LabError.ShapeMismatch;
    if (!a.isContiguous() or !b.isContiguous()) return LabError.InvalidLayout;
    const n = totalElements(a.shape);
    for (0..n) |i| {
        a.cpuData()[i] += b.cpuData()[i];
    }
}

/// Elementwise negation: out[i] = -a[i]. Strided input `a` is handled
/// via logical-to-physical offset translation.
pub fn neg(allocator: std.mem.Allocator, a: Tensor, tape: ?*Tape) !Tensor {
    if (a.device == .cuda) {
        var out = try cuda_dispatch.neg(a);
        if (tape) |t| {
            if (a.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .neg,
                    .parents = .{ a.tape_node, null },
                    .n_parents = 1,
                    .saved = .nothing,
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const out_shape = a.shape;
    var out = try Tensor.init(allocator, out_shape);
    const n = totalElements(out_shape);
    for (0..n) |i| {
        const a_off = logicalOffsetFromLinear(a.shape, a.strides, i);
        out.cpuData()[i] = -a.cpuData()[a_off];
    }
    if (tape) |t| {
        if (a.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .neg,
                .parents = .{ a.tape_node, null },
                .n_parents = 1,
                .saved = .nothing,
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Autograd recording helpers
// ---------------------------------------------------------------------------

/// Record a binary elementwise op on the tape if either input requires grad.
///
/// Stores snapshots of both input tensors by value in the Node's
/// SavedData. The `data` slices in the snapshots point to the
/// original heap buffers, which must outlive the tape. This avoids
/// dangling pointers that would occur if we stored `@constCast(a)`
/// where `a` is a pointer to a by-value parameter (stack-local copy).
fn recordBinaryOp(tape: ?*Tape, out: *Tensor, a: *const Tensor, b: *const Tensor, op: OpKind) !void {
    if (tape) |t| {
        if (a.requires_grad or b.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = op,
                .parents = .{ a.tape_node, b.tape_node },
                .n_parents = 2,
                .saved = .{ .tensor_pair = .{ .a = a.*, .b = b.* } },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "add (2,3) + (2,3) = (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = 2;
    a.cpuData()[2] = 3;
    a.cpuData()[3] = 4;
    a.cpuData()[4] = 5;
    a.cpuData()[5] = 6;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 10;
    b.cpuData()[1] = 20;
    b.cpuData()[2] = 30;
    b.cpuData()[3] = 40;
    b.cpuData()[4] = 50;
    b.cpuData()[5] = 60;

    var out = try add(allocator, a, b, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 11.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 66.0), out.cpuData()[5]);
}

test "add broadcast (1,3) + (2,3) = (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(1, 3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = 2;
    a.cpuData()[2] = 3;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 10;
    b.cpuData()[1] = 20;
    b.cpuData()[2] = 30;
    b.cpuData()[3] = 40;
    b.cpuData()[4] = 50;
    b.cpuData()[5] = 60;

    var out = try add(allocator, a, b, null);
    defer out.deinit(allocator);
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    try std.testing.expectEqual(@as(f32, 11.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 22.0), out.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 33.0), out.cpuData()[2]);
    // Row 1: [1+40, 2+50, 3+60] = [41, 52, 63]  (a is broadcast)
    try std.testing.expectEqual(@as(f32, 41.0), out.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 52.0), out.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 63.0), out.cpuData()[5]);
}

test "sub (2,3) - (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 10;
    a.cpuData()[1] = 20;
    a.cpuData()[2] = 30;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 1;
    b.cpuData()[1] = 2;
    b.cpuData()[2] = 3;

    var out = try sub(allocator, a, b, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 9.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 27.0), out.cpuData()[2]);
}

test "mul elementwise" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 2;
    a.cpuData()[1] = 3;
    a.cpuData()[2] = 4;

    var b = try Tensor.init(allocator, Shape.init1D(3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 5;
    b.cpuData()[1] = 6;
    b.cpuData()[2] = 7;

    var out = try mul(allocator, a, b, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 10.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 28.0), out.cpuData()[2]);
}

test "div produces Inf for division by zero" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(2));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1.0;
    a.cpuData()[1] = 0.0;

    var b = try Tensor.init(allocator, Shape.init1D(2));
    defer b.deinit(allocator);
    b.cpuData()[0] = 0.0;
    b.cpuData()[1] = 0.0;

    var out = try div(allocator, a, b, null);
    defer out.deinit(allocator);
    // 1.0/0.0 = +Inf, 0.0/0.0 = NaN
    try std.testing.expect(std.math.isPositiveInf(out.cpuData()[0]));
    try std.testing.expect(std.math.isNan(out.cpuData()[1]));
}

test "addScalar" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = 2;
    a.cpuData()[2] = 3;

    var out = try addScalar(allocator, a, 10.0, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 11.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 13.0), out.cpuData()[2]);
}

test "mulScalar" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = 2;
    a.cpuData()[2] = 3;

    var out = try mulScalar(allocator, a, 3.0, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 3.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 9.0), out.cpuData()[2]);
}

test "neg" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = -2;
    a.cpuData()[2] = 0;

    var out = try neg(allocator, a, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(f32, -1.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 2.0), out.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 0.0), out.cpuData()[2]);
}

test "addInPlace" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = 2;
    a.cpuData()[2] = 3;

    var b = try Tensor.init(allocator, Shape.init1D(3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 10;
    b.cpuData()[1] = 20;
    b.cpuData()[2] = 30;

    try addInPlace(&a, b);
    try std.testing.expectEqual(@as(f32, 11.0), a.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 33.0), a.cpuData()[2]);
}

test "add broadcast 3D + 2D: (2,3,8) + (1,8) = (2,3,8)" {
    const allocator = std.testing.allocator;
    // gamma-like broadcasting: (1,8) over (2,3,8)
    var a = try Tensor.init(allocator, Shape.init3D(2, 3, 8));
    defer a.deinit(allocator);
    for (0..48) |i| a.cpuData()[i] = @floatFromInt(i);

    var b = try Tensor.init(allocator, Shape.init2D(1, 8));
    defer b.deinit(allocator);
    for (0..8) |i| b.cpuData()[i] = @floatFromInt(i * 10);

    var out = try add(allocator, a, b, null);
    defer out.deinit(allocator);

    // Position (0, 0, 5): a[0,0,5]=5, b[0,5]=50 → out = 55
    try std.testing.expectApproxEqAbs(@as(f32, 55.0), out.cpuData()[0 * 24 + 0 * 8 + 5], 1e-4);
    // Position (1, 2, 5): a[1,2,5]=1*24+2*8+5=45, b[0,5]=50 → out = 95
    try std.testing.expectApproxEqAbs(@as(f32, 95.0), out.cpuData()[1 * 24 + 2 * 8 + 5], 1e-4);
    // Position (0, 1, 3): a[0,1,3]=11, b[0,3]=30 → out = 41
    try std.testing.expectApproxEqAbs(@as(f32, 41.0), out.cpuData()[0 * 24 + 1 * 8 + 3], 1e-4);
}

test "add broadcast 2D + 1D: (2,3) + (3,) = (2,3)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1;
    a.cpuData()[1] = 2;
    a.cpuData()[2] = 3;
    a.cpuData()[3] = 4;
    a.cpuData()[4] = 5;
    a.cpuData()[5] = 6;

    var b = try Tensor.init(allocator, Shape.init1D(3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 10;
    b.cpuData()[1] = 20;
    b.cpuData()[2] = 30;

    var out = try add(allocator, a, b, null);
    defer out.deinit(allocator);
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    try std.testing.expectEqual(@as(f32, 11.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 22.0), out.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 33.0), out.cpuData()[2]);
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    try std.testing.expectEqual(@as(f32, 14.0), out.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 25.0), out.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 36.0), out.cpuData()[5]);
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
    a.cpuData()[0] = 100;
    a.cpuData()[1] = 200;

    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);
    b.cpuData()[0] = 1;
    b.cpuData()[1] = 2;
    b.cpuData()[2] = 3;
    b.cpuData()[3] = 4;
    b.cpuData()[4] = 5;
    b.cpuData()[5] = 6;

    var out = try add(allocator, a, b, null);
    defer out.deinit(allocator);
    // Row 0: 100 + [1,2,3] = [101, 102, 103]
    try std.testing.expectEqual(@as(f32, 101.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 103.0), out.cpuData()[2]);
    // Row 1: 200 + [4,5,6] = [204, 205, 206]
    try std.testing.expectEqual(@as(f32, 204.0), out.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 206.0), out.cpuData()[5]);
}

// -- Strided-input correctness (PR-β) ---------------------------------------
//
// After PR-β, the scalar/unary ops must honour the input's strides when
// reading. These tests pin down the behaviour so a future flat-loop
// regression fails loudly instead of silently producing wrong answers.

test "addScalar on transposed view reads logical elements" {
    const allocator = std.testing.allocator;
    // Underlying (2,3) buffer in row-major order:
    //   [1, 2, 3,
    //    4, 5, 6]
    var base = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer base.deinit(allocator);
    base.cpuData()[0] = 1;
    base.cpuData()[1] = 2;
    base.cpuData()[2] = 3;
    base.cpuData()[3] = 4;
    base.cpuData()[4] = 5;
    base.cpuData()[5] = 6;

    // Transposed view, shape (3,2), strides (1,3). Logical row-major
    // iteration of the view visits the original values in order
    //   [1,4,  2,5,  3,6].
    const view = try base.transpose2d();

    var out = try addScalar(allocator, view, 10.0, null);
    defer out.deinit(allocator);

    try std.testing.expectEqual(@as(f32, 11.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 14.0), out.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 12.0), out.cpuData()[2]);
    try std.testing.expectEqual(@as(f32, 15.0), out.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 13.0), out.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 16.0), out.cpuData()[5]);
}

test "mulScalar on transposed view reads logical elements" {
    const allocator = std.testing.allocator;
    var base = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer base.deinit(allocator);
    base.cpuData()[0] = 1;
    base.cpuData()[1] = 2;
    base.cpuData()[2] = 3;
    base.cpuData()[3] = 4;
    base.cpuData()[4] = 5;
    base.cpuData()[5] = 6;
    const view = try base.transpose2d();

    var out = try mulScalar(allocator, view, 2.0, null);
    defer out.deinit(allocator);

    const expected = [_]f32{ 2, 8, 4, 10, 6, 12 };
    for (expected, 0..) |want, i| {
        try std.testing.expectEqual(want, out.cpuData()[i]);
    }
}

test "neg on transposed view reads logical elements" {
    const allocator = std.testing.allocator;
    var base = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer base.deinit(allocator);
    for (base.cpuData(), 0..) |*v, i| v.* = @floatFromInt(i + 1);
    const view = try base.transpose2d();

    var out = try neg(allocator, view, null);
    defer out.deinit(allocator);

    const expected = [_]f32{ -1, -4, -2, -5, -3, -6 };
    for (expected, 0..) |want, i| {
        try std.testing.expectEqual(want, out.cpuData()[i]);
    }
}

test "addInPlace rejects non-contiguous lhs with InvalidLayout" {
    const allocator = std.testing.allocator;
    var base = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer base.deinit(allocator);
    var view = try base.transpose2d(); // non-contiguous (3,2)

    var b = try Tensor.init(allocator, Shape.init2D(3, 2));
    defer b.deinit(allocator);

    try std.testing.expectError(LabError.InvalidLayout, addInPlace(&view, b));
}

test "addInPlace rejects non-contiguous rhs with InvalidLayout" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init2D(3, 2));
    defer a.deinit(allocator);

    var base = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer base.deinit(allocator);
    const view = try base.transpose2d(); // non-contiguous (3,2)

    try std.testing.expectError(LabError.InvalidLayout, addInPlace(&a, view));
}
