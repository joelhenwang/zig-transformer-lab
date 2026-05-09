//!
//! zig-transformer-lab — Numerically stable softmax and log_softmax
//!
//! Purpose:
//!   Provides softmax and log_softmax along the last axis.  Softmax is
//!   THE most numerically tricky operation in a transformer: it converts
//!   raw logits into probabilities, and its naive implementation
//!   overflows for large values.
//!
//!   Key insight: softmax(x) = softmax(x - max(x)).  Subtracting a
//!   constant from every element doesn't change the result because
//!     exp(x - c) / sum(exp(x - c)) = exp(x) / sum(exp(x))
//!   The max acts as c, keeping all exponents <= 0 and thus bounded.
//!
//! Shape contract:
//!   softmax:     in:(..., C) -> out:(..., C)  [last axis only]
//!   logSoftmax:  in:(..., C) -> out:(..., C)  [last axis only]
//!   The output has the same shape as the input.  Each group of C
//!   elements along the last axis is independently normalized to
//!   sum to 1 (softmax) or to have log-sum = 0 (log_softmax).
//!
//! Math:
//!   softmax(x)[j] = exp(x[j] - max(x)) / sum_k exp(x[k] - max(x))
//!
//!   log_softmax(x)[j] = x[j] - max(x) - log(sum_k exp(x[k] - max(x)))
//!
//!   The log_softmax formula avoids the division-then-log pattern,
//!   which can lose precision when softmax values are very small.
//!   Computing log directly from the sum is more stable.
//!
//! Memory ownership:
//!   Both functions return new owned tensors.  The caller must call
//!   deinit(allocator) on the returned tensor when done.
//!
//! Errors:
//!   OutOfMemory — allocator could not fulfill the output buffer
//!
//! TODO:
//!   - Stage 7: CUDA kernel using warp-level reductions
//!   - Support for axis parameter (softmax along arbitrary axis)
//!
//! Credits:
//!   The max-subtraction trick is standard in every deep learning
//!   framework.  Described in Goodfellow et al. (2016), Section 4.1.
//!   No code copied.

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Tensor = @import("../tensor.zig").Tensor;
const Shape = @import("../shape.zig").Shape;
const totalElements = @import("../shape.zig").totalElements;

/// Compute the strided offset for a flat element index.
///
/// This helper decodes a flat index into multi-dimensional indices
/// using the tensor's shape, then computes the data offset using
/// strides.  It works for any rank and any stride layout, including
/// non-contiguous tensors (e.g., after transpose).
///
/// This is an O(ndim) computation per element, but ndim <= 4 in our
/// system, so the overhead is negligible compared to @exp.
fn stridedOffset(tensor: Tensor, flat: usize) usize {
    var offset: usize = 0;
    var remaining: usize = flat;
    const ndim = tensor.shape.ndim();
    var axis: usize = 0;
    while (axis + 1 < ndim) : (axis += 1) {
        // Product of all dims after this axis = number of elements
        // in each "block" at this axis level.  Used to decode the
        // flat index into a per-axis index via integer division.
        var block: usize = 1;
        var a2: usize = axis + 1;
        while (a2 < ndim) : (a2 += 1) {
            block *= tensor.shape.dims[a2];
        }
        const idx = remaining / block;
        remaining %= block;
        offset += idx * tensor.strides.values[axis];
    }
    // Last axis: remaining is the index along the last dimension
    offset += remaining * tensor.strides.values[axis];
    return offset;
}

/// Numerically stable softmax along the last axis.
///
/// For each group of C elements along the last axis (where C is the
/// size of the last dimension), independently computes:
///   out[j] = exp(x[j] - max(x)) / sum_k exp(x[k] - max(x))
///
/// The max-subtraction ensures all exponents are <= 0, preventing
/// overflow.  Underflow to 0 for very negative values is harmless
/// (0 / sum is just a very small probability).
///
/// Shape: in:(..., C) -> out:(..., C)
///
/// Worked example:
///   // in = [[1, 2, 3]]  shape (1, 3)
///   // max = 3
///   // exp([-2, -1, 0]) = [0.1353, 0.3679, 1.0]
///   // sum = 1.5032
///   // out = [[0.0900, 0.2447, 0.6652]]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn softmax(allocator: std.mem.Allocator, tensor: Tensor) LabError!Tensor {
    const ndim = tensor.shape.ndim();
    const C = tensor.shape.dims[ndim - 1];
    // Number of independent softmax groups (one per "row" of the
    // last axis).  Each group contains C elements.
    const num_groups = totalElements(tensor.shape) / C;

    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    // Iterate over each group.  For rank-2 (B, C), each group is a
    // row.  For rank-3 (D, B, C), each group is a 1D slice indexed
    // by (d, b, :).  The general decoding uses the prefix axes.
    for (0..num_groups) |g| {
        // Decode group index g into prefix indices for axes 0..ndim-2,
        // then compute the base offset for this group in the input
        // tensor's data buffer using strides.
        var base_offset: usize = 0;
        var prefix_idx: usize = g;
        for (0..ndim - 1) |axis| {
            // Number of groups contained in one "step" along this
            // axis.  E.g., for shape (2, 3, C), axis 0 has 3 groups
            // per step (B=3), and axis 1 has 1 group per step.
            var groups_per_step: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim - 1) : (a2 += 1) {
                groups_per_step *= tensor.shape.dims[a2];
            }
            const idx = prefix_idx / groups_per_step;
            prefix_idx %= groups_per_step;
            base_offset += idx * tensor.strides.values[axis];
        }

        // --- Step 1: Find the maximum value in this group ---
        // This is the key to numerical stability.  Without max
        // subtraction, exp(100) = Inf and the sum becomes Inf,
        // giving NaN after division.
        const stride_last = tensor.strides.values[ndim - 1];
        var max_val: f32 = -std.math.inf(f32);
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last];
            max_val = @max(max_val, val);
        }

        // --- Step 2: Compute exp(x - max) and accumulate the sum ---
        var sum_exp: f32 = 0;
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last] - max_val;
            const ev = @exp(val);
            out.data[g * C + c] = ev;
            sum_exp += ev;
        }

        // --- Step 3: Normalize by the sum ---
        // After this, each group sums to 1.0 (within float precision).
        for (0..C) |c| {
            out.data[g * C + c] /= sum_exp;
        }
    }

    return out;
}

/// Numerically stable log-softmax along the last axis.
///
/// For each group of C elements along the last axis, computes:
///   log_softmax[j] = x[j] - max(x) - log(sum_k exp(x[k] - max(x)))
///
/// This is more stable than log(softmax(x)) because it avoids the
/// division-then-log pattern.  When softmax values are very small
/// (close to 0), log(softmax) requires computing log of a tiny
/// number, which amplifies rounding errors.  The direct formula
/// avoids this by computing log of the sum directly.
///
/// Shape: in:(..., C) -> out:(..., C)
///
/// Worked example:
///   // in = [[1, 2, 3]]  shape (1, 3)
///   // max = 3, sum_exp = 1.5032
///   // log_softmax = [1-3-ln(1.5032), 2-3-ln(1.5032), 3-3-ln(1.5032)]
///   //             = [-2.4076, -1.4076, -0.4076]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn logSoftmax(allocator: std.mem.Allocator, tensor: Tensor) LabError!Tensor {
    const ndim = tensor.shape.ndim();
    const C = tensor.shape.dims[ndim - 1];
    const num_groups = totalElements(tensor.shape) / C;

    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    for (0..num_groups) |g| {
        // Decode group index into prefix indices (same as softmax)
        var base_offset: usize = 0;
        var prefix_idx: usize = g;
        for (0..ndim - 1) |axis| {
            var groups_per_step: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim - 1) : (a2 += 1) {
                groups_per_step *= tensor.shape.dims[a2];
            }
            const idx = prefix_idx / groups_per_step;
            prefix_idx %= groups_per_step;
            base_offset += idx * tensor.strides.values[axis];
        }

        // --- Step 1: Find max ---
        const stride_last = tensor.strides.values[ndim - 1];
        var max_val: f32 = -std.math.inf(f32);
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last];
            max_val = @max(max_val, val);
        }

        // --- Step 2: Compute sum(exp(x - max)) ---
        // We don't store the exp values — we only need the sum
        // for log_softmax.
        var sum_exp: f32 = 0;
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last] - max_val;
            sum_exp += @exp(val);
        }

        // --- Step 3: Compute log_softmax directly ---
        // log_softmax = x - max - log(sum_exp)
        // This avoids the unstable log(softmax) path.
        const log_sum_exp = @log(sum_exp);
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last];
            out.data[g * C + c] = val - max_val - log_sum_exp;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "softmax [[1,2,3]] = [[0.0900, 0.2447, 0.6652]]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(1, 3));
    defer t.deinit(alloc);
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;

    var out = try softmax(alloc, t);
    defer out.deinit(alloc);

    // exp(1-3)=0.1353, exp(2-3)=0.3679, exp(3-3)=1.0
    // sum=1.5032, normalized: [0.0900, 0.2447, 0.6652]
    try std.testing.expectApproxEqAbs(@as(f32, 0.0900), out.data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2447), out.data[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6652), out.data[2], 1e-3);
}

test "softmax [[0,0]] = [[0.5, 0.5]]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(1, 2));
    defer t.deinit(alloc);
    t.data[0] = 0.0;
    t.data[1] = 0.0;

    var out = try softmax(alloc, t);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out.data[1], 1e-4);
}

test "softmax [[100,101]] does not overflow" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(1, 2));
    defer t.deinit(alloc);
    t.data[0] = 100.0;
    t.data[1] = 101.0;

    var out = try softmax(alloc, t);
    defer out.deinit(alloc);

    // max=101, exp(-1)=0.3679, exp(0)=1.0
    // sum=1.3679, [0.2689, 0.7311]
    try std.testing.expect(std.math.isFinite(out.data[0]));
    try std.testing.expect(std.math.isFinite(out.data[1]));
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), out.data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), out.data[1], 1e-3);
}

test "softmax rows sum to 1.0" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(3, 5));
    defer t.deinit(alloc);
    // Fill with various values
    for (0..15) |i| t.data[i] = @as(f32, @floatFromInt(i));

    var out = try softmax(alloc, t);
    defer out.deinit(alloc);

    // Each row should sum to 1.0
    for (0..3) |row| {
        var sum: f32 = 0;
        for (0..5) |col| {
            sum += out.data[row * 5 + col];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-4);
    }
}

test "logSoftmax [[1,2,3]]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(1, 3));
    defer t.deinit(alloc);
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;

    var out = try logSoftmax(alloc, t);
    defer out.deinit(alloc);

    // log_softmax = [1-3-ln(1.5032), 2-3-ln(1.5032), 3-3-ln(1.5032)]
    // ln(1.5032) ~ 0.4076
    // = [-2.4076, -1.4076, -0.4076]
    try std.testing.expectApproxEqAbs(@as(f32, -2.4076), out.data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -1.4076), out.data[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.4076), out.data[2], 1e-3);
}
