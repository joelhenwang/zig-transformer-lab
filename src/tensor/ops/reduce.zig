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
const shape_equals = @import("../shape.zig").equals;
const isContiguous = @import("../shape.zig").isContiguous;
const Tensor = @import("../tensor.zig").Tensor;
const DType = @import("../../core/dtype.zig").DType;
const Device = @import("../../core/device.zig").Device;
const Tape = @import("../../autograd/tape.zig").Tape;
const Node = @import("../../autograd/node.zig").Node;
const OpKind = @import("../../autograd/node.zig").OpKind;
const ops_shape = @import("shape_ops.zig");

/// Sum all elements along the given axis.
///
/// Worked example:
///   t = [[1, 2, 3],
///        [4, 5, 6]]   shape (2,3)
///   sum(t, axis=0) = [[5, 7, 9]]   shape (1,3)
///   sum(t, axis=1) = [[6],
///                     [15]]         shape (2,1)
pub fn sum(allocator: std.mem.Allocator, tensor: Tensor, axis: u2, tape: ?*Tape) !Tensor {
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

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .sum,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .reduce_info = .{ .shape = tensor.shape, .axis = axis } },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Mean along the given axis = sum / axis_size.
pub fn mean(allocator: std.mem.Allocator, tensor: Tensor, axis: u2, tape: ?*Tape) !Tensor {
    if (@as(usize, axis) >= tensor.shape.ndim()) return LabError.InvalidArgument;
    var out = try sum(allocator, tensor, axis, null);
    const axis_size = @as(f32, @floatFromInt(tensor.shape.dims[axis]));
    for (out.data) |*v| {
        v.* /= axis_size;
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .mean,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .reduce_info = .{ .shape = tensor.shape, .axis = axis } },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
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
///
/// If a tape is provided and the input requires_grad, records a .sum
/// node so backward can broadcast the gradient back to the input shape.
pub fn sumAll(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) !Tensor {
    const out_shape = Shape.init1D(1);
    var out = try Tensor.init(allocator, out_shape);
    var acc: f32 = 0.0;
    for (tensor.data) |v| acc += v;
    out.data[0] = acc;

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .sum,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .reduce_info = .{ .shape = tensor.shape, .axis = 0 } },
            });
            out.tape_node = node_id;
            out.requires_grad = true;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// sumToShape — broadcast-backward helper (used by autograd)
// ---------------------------------------------------------------------------

/// Reduce a gradient tensor back to a smaller (original) shape by summing
/// along axes where the original shape had dim=1.
///
/// This is THE key operation for broadcasting backward. When a forward
/// op broadcasted shape (1,3) → (2,3), the backward receives a gradient
/// of shape (2,3) but needs to return a gradient of shape (1,3). We sum
/// along the broadcasted axis (axis 0) to collapse the gradient back.
///
/// Algorithm:
///   1. If shapes are equal → return a view (no computation needed).
///   2. Right-align the shapes and compare dimensions.
///   3. For each axis where grad has a larger dim than target:
///      a. If target dim is 1 → sum along this axis (keep dims).
///      b. If target has fewer dims → sum along this axis (it was
///         broadcasted from a lower-rank shape).
///   4. After summing all broadcasted axes, reshape to target.
///
/// Worked example:
///   sumToShape(grad: (2,3), target: (1,3))
///     → sum along axis 0 → (1,3)   [axis 0 had dim 2 in grad, 1 in target]
///
///   sumToShape(grad: (2,3), target: (3,))
///     → sum along axis 0 → (1,3), then reshape/squeeze to (3,)
///
///   sumToShape(grad: (2,3), target: (2,3))
///     → return view of grad (no-op)
///
/// Memory ownership:
///   Returns a new owned tensor. Caller must deinit.
///   Caller must deinit the result.
pub fn sumToShape(allocator: std.mem.Allocator, grad: Tensor, target: Shape) LabError!Tensor {
    // Same shape → return an owned copy (not a view) because callers
    // may deinit the source tensor, which would invalidate a view.
    // CRITICAL: must check contiguity before @memcpy. If grad is a
    // non-contiguous view (e.g., from transpose2d), @memcpy copies
    // raw buffer bytes in memory order, ignoring strides, producing
    // silently wrong data. reshapeTracked handles non-contiguous
    // inputs correctly via stride-aware element-by-element copy.
    if (shape_equals(grad.shape, target)) {
        if (isContiguous(grad.shape, grad.strides)) {
            const out = try Tensor.init(allocator, target);
            @memcpy(out.data, grad.data[0..out.data.len]);
            return out;
        } else {
            return ops_shape.reshapeTracked(allocator, grad, target, null);
        }
    }

    // Collect axes that need summing.
    const grad_ndim = grad.shape.ndim();
    const target_ndim = target.ndim();
    const extra_dims = if (grad_ndim > target_ndim) grad_ndim - target_ndim else 0;

    var axes_to_sum: [4]u2 = .{ 0, 0, 0, 0 };
    var n_axes: usize = 0;

    // Extra leading dims: e.g., grad (2,3) target (3,) → sum axis 0
    for (0..extra_dims) |i| {
        axes_to_sum[n_axes] = @intCast(i);
        n_axes += 1;
    }

    // Right-aligned comparison: axes where target dim is 1 but grad dim > 1
    for (0..target_ndim) |ti| {
        const grad_axis = extra_dims + ti;
        const grad_dim = grad.shape.dims[grad_axis];
        const target_dim = target.dims[ti];
        if (target_dim == 1 and grad_dim > 1) {
            axes_to_sum[n_axes] = @intCast(grad_axis);
            n_axes += 1;
        }
    }

    if (n_axes == 0) {
        // No axes to sum, but shapes differ — reshape.
        // Use reshapeTracked which handles non-contiguous views correctly.
        return ops_shape.reshapeTracked(allocator, grad, target, null);
    }

    // Sum axes from rightmost to leftmost to avoid index shifting.
    var current_owned = false;
    var current = grad;
    var i: usize = n_axes;
    while (i > 0) : (i -= 1) {
        const axis: u2 = axes_to_sum[i - 1];
        const summed = try sum(allocator, current, axis, null);
        if (current_owned) current.deinit(allocator);
        current = summed;
        current_owned = true;
    }

    // After summing, we may still need to reshape (e.g., (1,3) → (3,))
    if (!shape_equals(current.shape, target)) {
        if (totalElements(current.shape) == totalElements(target)) {
            // reshapeTracked handles non-contiguous tensors correctly.
            const result = try ops_shape.reshapeTracked(allocator, current, target, null);
            if (current_owned) current.deinit(allocator);
            return result;
        }
    }

    return current;
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

    var out = try sum(allocator, t, 0, null);
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

    var out = try sum(allocator, t, 1, null);
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

    var out = try mean(allocator, t, 0, null);
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

    var out = try sumAll(allocator, t, null);
    defer out.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), out.shape.dims[0]);
    try std.testing.expectEqual(@as(f32, 21.0), out.data[0]);
}

test "reduce rejects invalid axis" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    try std.testing.expectError(LabError.InvalidArgument, sum(allocator, t, 3, null));
}

test "sumToShape — same shape returns owned copy" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    t.data[0] = 1;
    t.data[1] = 2;

    var result = try sumToShape(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);
    try std.testing.expect(result.owned);
    try std.testing.expect(result.data.ptr != t.data.ptr);
    try std.testing.expectEqual(@as(f32, 1.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), result.data[1]);
}

test "sumToShape — sum broadcasted axis 0" {
    const allocator = std.testing.allocator;
    // grad shape (2,3), target shape (1,3)
    // axis 0 was broadcasted from 1 to 2, so sum along axis 0
    var grad = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer grad.deinit(allocator);
    // [[1, 2, 3],
    //  [4, 5, 6]]
    grad.data[0] = 1;
    grad.data[1] = 2;
    grad.data[2] = 3;
    grad.data[3] = 4;
    grad.data[4] = 5;
    grad.data[5] = 6;

    var result = try sumToShape(allocator, grad, Shape.init2D(1, 3));
    defer result.deinit(allocator);
    // sum axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
    try std.testing.expectEqual(@as(f32, 5.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 7.0), result.data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), result.data[2]);
    try std.testing.expectEqual(@as(usize, 1), result.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), result.shape.dims[1]);
}

test "sumToShape — sum broadcasted axis 1" {
    const allocator = std.testing.allocator;
    // grad shape (2,3), target shape (2,1)
    var grad = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer grad.deinit(allocator);
    grad.data[0] = 1;
    grad.data[1] = 2;
    grad.data[2] = 3;
    grad.data[3] = 4;
    grad.data[4] = 5;
    grad.data[5] = 6;

    var result = try sumToShape(allocator, grad, Shape.init2D(2, 1));
    defer result.deinit(allocator);
    // sum axis 1: [1+2+3, 4+5+6] = [6, 15]
    try std.testing.expectEqual(@as(f32, 6.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 15.0), result.data[1]);
}

test "sumToShape — reduce rank (2,3) → (3,)" {
    const allocator = std.testing.allocator;
    var grad = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer grad.deinit(allocator);
    grad.data[0] = 1;
    grad.data[1] = 2;
    grad.data[2] = 3;
    grad.data[3] = 4;
    grad.data[4] = 5;
    grad.data[5] = 6;

    var result = try sumToShape(allocator, grad, Shape.init1D(3));
    defer result.deinit(allocator);
    // sum the leading axis (axis 0), then reshape to (3,)
    try std.testing.expectEqual(@as(f32, 5.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 7.0), result.data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), result.data[2]);
}

test "sumToShape — transposed (non-contiguous) view" {
    const allocator = std.testing.allocator;
    // Build a 2x3 tensor, transpose it, then sumToShape back to (3,2).
    // This tests the fix for the @memcpy-on-non-contiguous bug:
    // before the fix, @memcpy copied raw bytes ignoring strides,
    // silently producing the original (non-transposed) data.
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    // t = [[1, 2, 3],
    //      [4, 5, 6]]
    t.data[0] = 1;
    t.data[1] = 2;
    t.data[2] = 3;
    t.data[3] = 4;
    t.data[4] = 5;
    t.data[5] = 6;

    // transpose2d returns a non-contiguous VIEW with strides [1, 3]
    // and shape (3, 2). Logically: [[1, 4], [2, 5], [3, 6]]
    const t_view = try t.transpose2d();
    try std.testing.expect(!t_view.isContiguous());

    // sumToShape with same shape must produce an owned contiguous copy
    // that respects the logical (transposed) element ordering.
    var result = try sumToShape(allocator, t_view, Shape.init2D(3, 2));
    defer result.deinit(allocator);

    // result[0,0] = t_view[0,0] = t[0,0] = 1
    // result[0,1] = t_view[0,1] = t[1,0] = 4
    // result[1,0] = t_view[1,0] = t[0,1] = 2
    // result[1,1] = t_view[1,1] = t[1,1] = 5
    // result[2,0] = t_view[2,0] = t[0,2] = 3
    // result[2,1] = t_view[2,1] = t[1,2] = 6
    try std.testing.expectEqual(@as(f32, 1.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), result.data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), result.data[2]);
    try std.testing.expectEqual(@as(f32, 5.0), result.data[3]);
    try std.testing.expectEqual(@as(f32, 3.0), result.data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), result.data[5]);
}
