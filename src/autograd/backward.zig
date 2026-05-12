//!
//! zig-transformer-lab — Backward pass implementations for each OpKind
//!
//! Purpose:
//!   Implements the gradient computation (vector-Jacobian product) for
//!   every operation recorded on the autograd tape. Each function takes
//!   the forward-pass node (containing saved data) and the upstream
//!   gradient, and returns the gradient contributions for each parent.
//!
//!   This file is the mathematical heart of the autograd engine. Every
//!   gradient formula here corresponds to a specific derivative rule
//!   from calculus, translated into tensor operations.
//!
//! Shape contract:
//!   backward(node, grad_output) → [2]?*Tensor
//!   - grad_output has the same shape as the node's output tensor.
//!   - Each returned gradient has the same shape as the corresponding
//!     parent input tensor.
//!   - If a parent doesn't need a gradient, its entry is null.
//!
//! Math — see individual backward cases for derivations.
//!
//! Memory ownership:
//!   backward() allocates new gradient tensors via the allocator.
//!   The caller (tape.backward()) takes ownership and either:
//!     - Accumulates them into existing grad_map entries (+=)
//!     - Stores them directly for leaf tensors
//!     - Frees them when retain_graph=false and the node is intermediate
//!
//! Errors:
//!   OutOfMemory — gradient tensor allocation failure
//!
//! Credits:
//!   Gradient formulas are standard calculus. The matmul backward
//!   follows the standard linear algebra identity. The fused
//!   cross-entropy backward follows PyTorch's implementation.
//!   No code copied.
//;

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const totalElements = @import("../tensor/shape.zig").totalElements;
const shape_equals = @import("../tensor/shape.zig").equals;
const isContiguous = @import("../tensor/shape.zig").isContiguous;
const Node = @import("node.zig").Node;
const OpKind = @import("node.zig").OpKind;
const SavedData = @import("node.zig").SavedData;


// Shared helpers (extracted to break circular imports in Phase 4).
const grad_helpers = @import("grad_helpers.zig");
const heapAlloc = grad_helpers.heapAlloc;

const max_parent_grads = 2;
pub const BackwardResult = grad_helpers.BackwardResult;

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

/// Compute parent gradients for the given node.
///
/// Switches on node.op to call the appropriate backward implementation.
/// The exhaustive switch gives compile-time safety: adding a new OpKind
/// without a backward case is a compilation error.
pub fn backward(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
) LabError!BackwardResult {
    var result: BackwardResult = .{ null, null };

    switch (node.op) {
        // ----- Elementwise binary -----
        .add => try bw_elementwise.backwardAdd(allocator, node, grad_output, &result),
        .sub => try bw_elementwise.backwardSub(allocator, node, grad_output, &result),
        .mul => try bw_elementwise.backwardMul(allocator, node, grad_output, &result),
        .div => try bw_elementwise.backwardDiv(allocator, node, grad_output, &result),

        // ----- Elementwise scalar -----
        .add_scalar => try bw_elementwise.backwardAddScalar(allocator, node, grad_output, &result),
        .mul_scalar => try bw_elementwise.backwardMulScalar(allocator, node, grad_output, &result),

        // ----- Linear algebra -----
        .matmul => try bw_matmul.backwardMatmul(allocator, node, grad_output, &result),
        .matmul_batch => try bw_matmul.backwardMatmulBatch(allocator, node, grad_output, &result),

        // ----- Shape transforms -----
        .transpose2d => try bw_shape.backwardTranspose2d(allocator, node, grad_output, &result),
        .reshape => try bw_shape.backwardReshape(allocator, node, grad_output, &result),

        // ----- Reductions -----
        .sum => try bw_reduce.backwardSum(allocator, node, grad_output, &result),
        .mean => try bw_reduce.backwardMean(allocator, node, grad_output, &result),

        // ----- Unary activations -----
        .exp => try bw_unary.backwardExp(allocator, node, grad_output, &result),
        .log => try bw_unary.backwardLog(allocator, node, grad_output, &result),
        .neg => try bw_unary.backwardNeg(allocator, node, grad_output, &result),
        .relu => try bw_unary.backwardRelu(allocator, node, grad_output, &result),
        .gelu => try bw_unary.backwardGelu(allocator, node, grad_output, &result),

        // ----- Normalization -----
        .softmax => try bw_softmax.backwardSoftmax(allocator, node, grad_output, &result),
        .log_softmax => try bw_softmax.backwardLogSoftmax(allocator, node, grad_output, &result),

        // ----- Loss -----
        .cross_entropy => try bw_loss.backwardCrossEntropy(allocator, node, grad_output, &result),

        // ----- Unary math -----
        .sqrt => try bw_unary.backwardSqrt(allocator, node, grad_output, &result),

        // ----- Embedding (Stage 4) -----
        .embedding => try bw_embedding.backwardEmbedding(allocator, node, grad_output, &result),

        // ----- 3D inner transpose -----
        .transpose_inner2d => try bw_shape.backwardTransposeInner2d(allocator, node, grad_output, &result),

        // ----- 4D axis-1/2 transpose -----
        .transpose_axes12_4d => try bw_shape.backwardTransposeAxes12_4d(allocator, node, grad_output, &result),
    }

    return result;
}

// ---------------------------------------------------------------------------
// Individual backward implementations — dispatched to colocated ops files
// ---------------------------------------------------------------------------
//
// Each backward function lives alongside its forward op for locality.
// The switch in backward() above is a one-liner dispatch table.
// See grad_helpers.zig for shared utilities (heapAlloc, broadcastTo,
// accumulateGrad).

// Backward functions are accessed via these module imports:
const bw_elementwise = @import("../tensor/ops/elementwise.zig");
const bw_unary = @import("../tensor/ops/unary.zig");
const bw_matmul = @import("../tensor/ops/matmul.zig");
const bw_reduce = @import("../tensor/ops/reduce.zig");
const bw_softmax = @import("../tensor/ops/softmax.zig");
const bw_loss = @import("../tensor/ops/loss.zig");
const bw_shape = @import("../tensor/ops/shape_ops.zig");
const bw_embedding = @import("../nn/embedding.zig");

// ---------------------------------------------------------------------------
// Shared helpers (re-exported from grad_helpers.zig)
// ---------------------------------------------------------------------------

pub const broadcastTo = grad_helpers.broadcastTo;
pub const accumulateGrad = grad_helpers.accumulateGrad;

// ---------------------------------------------------------------------------
// GELU derivative
// ---------------------------------------------------------------------------

/// Derivative of the exact GELU function.
///
/// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
///
/// GELU'(x) = 0.5 * (1 + erf(x/sqrt(2)))
///          + x * exp(-x²/2) / sqrt(2*pi)
///
/// The first term is the "cumulative" part (CDF of standard normal),
/// and the second is the "density" part (PDF of standard normal times x).
fn geluDerivative(x: f32) f32 {
    const x_f64: f64 = @floatCast(x);
    const sqrt2: f64 = 1.4142135623730951;
    const sqrt2pi: f64 = 2.5066282746310002;
    const z = x_f64 / sqrt2;
    const erf_val = erfApprox(z);
    const cumulative = 0.5 * (1.0 + erf_val);
    const density = x_f64 * @exp(-0.5 * x_f64 * x_f64) / sqrt2pi;
    const result_f64 = cumulative + density;
    return @floatCast(result_f64);
}

/// Approximate error function erf(x) — Abramowitz & Stegun formula 7.1.26.
/// Duplicated from unary.zig to avoid circular dependency once ops take tape.
fn erfApprox(x: f64) f64 {
    const sign: f64 = if (x >= 0) 1.0 else -1.0;
    const a = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * a);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-a * a);
    return sign * y;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "backward neg — dL/da = -dL/dc" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 1.0;
    a.cpuData()[1] = 2.0;
    a.cpuData()[2] = 3.0;

    var grad_out = try Tensor.init(allocator, Shape.init1D(3));
    defer grad_out.deinit(allocator);
    grad_out.cpuData()[0] = 1.0;
    grad_out.cpuData()[1] = 1.0;
    grad_out.cpuData()[2] = 1.0;

    const node = Node{
        .id = 0,
        .op = .neg,
        .parents = .{ null, null },
        .n_parents = 1,
        .saved = .nothing,
    };

    const result = try backward(allocator, node, &grad_out);
    const da = result[0] orelse unreachable;
    defer {
        da.deinit(allocator);
        allocator.destroy(da);
    }

    try std.testing.expectEqual(@as(f32, -1.0), da.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, -1.0), da.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, -1.0), da.cpuData()[2]);
}

test "backward relu — dL/da = dL/dc * (a > 0)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(4));
    defer a.deinit(allocator);
    a.cpuData()[0] = 2.0; // positive
    a.cpuData()[1] = -1.0; // negative
    a.cpuData()[2] = 0.0; // zero
    a.cpuData()[3] = 5.0; // positive

    var grad_out = try Tensor.init(allocator, Shape.init1D(4));
    defer grad_out.deinit(allocator);
    grad_out.fill(1.0);

    const node = Node{
        .id = 0,
        .op = .relu,
        .parents = .{ null, null },
        .n_parents = 1,
        .saved = .{ .tensor_ref = a },
    };

    const result = try backward(allocator, node, &grad_out);
    const da = result[0] orelse unreachable;
    defer {
        da.deinit(allocator);
        allocator.destroy(da);
    }

    try std.testing.expectEqual(@as(f32, 1.0), da.cpuData()[0]); // a > 0 → grad passes
    try std.testing.expectEqual(@as(f32, 0.0), da.cpuData()[1]); // a < 0 → grad blocked
    try std.testing.expectEqual(@as(f32, 0.0), da.cpuData()[2]); // a == 0 → grad blocked
    try std.testing.expectEqual(@as(f32, 1.0), da.cpuData()[3]); // a > 0 → grad passes
}

test "backward add — dL/da = sumToShape(dL/dc, a.shape)" {
    const allocator = std.testing.allocator;
    // a: (1,3), b: (2,3) → c: (2,3)  [a was broadcast]
    var a = try Tensor.init(allocator, Shape.init2D(1, 3));
    defer a.deinit(allocator);
    var b = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer b.deinit(allocator);

    var grad_out = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer grad_out.deinit(allocator);
    grad_out.cpuData()[0] = 1.0;
    grad_out.cpuData()[1] = 2.0;
    grad_out.cpuData()[2] = 3.0;
    grad_out.cpuData()[3] = 4.0;
    grad_out.cpuData()[4] = 5.0;
    grad_out.cpuData()[5] = 6.0;

    const node = Node{
        .id = 0,
        .op = .add,
        .parents = .{ null, null },
        .n_parents = 2,
        .saved = .{ .tensor_pair = .{ .a = a, .b = b } },
    };

    const result = try backward(allocator, node, &grad_out);
    const da = result[0] orelse unreachable;
    defer {
        da.deinit(allocator);
        allocator.destroy(da);
    }
    const db = result[1] orelse unreachable;
    defer {
        db.deinit(allocator);
        allocator.destroy(db);
    }

    // dL/da = sumToShape(grad, (1,3)) → sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
    try std.testing.expectEqual(@as(f32, 5.0), da.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 7.0), da.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 9.0), da.cpuData()[2]);

    // dL/db = sumToShape(grad, (2,3)) → same shape, no reduction: [1, 2, 3, 4, 5, 6]
    try std.testing.expectEqual(@as(f32, 1.0), db.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 6.0), db.cpuData()[5]);
}

test "backward matmul — dL/dA = dL/dC @ Bᵀ, dL/dB = Aᵀ @ dL/dC" {
    const allocator = std.testing.allocator;
    // A: (2,3), B: (3,4), C: (2,4)
    var A = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer A.deinit(allocator);
    var B = try Tensor.init(allocator, Shape.init2D(3, 4));
    defer B.deinit(allocator);

    // Fill with small values
    A.cpuData()[0] = 1.0;
    A.cpuData()[1] = 2.0;
    A.cpuData()[2] = 3.0;
    A.cpuData()[3] = 4.0;
    A.cpuData()[4] = 5.0;
    A.cpuData()[5] = 6.0;
    B.cpuData()[0] = 0.1;
    B.cpuData()[1] = 0.2;
    B.cpuData()[2] = 0.3;
    B.cpuData()[3] = 0.4;
    B.cpuData()[4] = 0.5;
    B.cpuData()[5] = 0.6;
    B.cpuData()[6] = 0.7;
    B.cpuData()[7] = 0.8;
    B.cpuData()[8] = 0.9;
    B.cpuData()[9] = 1.0;
    B.cpuData()[10] = 1.1;
    B.cpuData()[11] = 1.2;

    var grad_out = try Tensor.init(allocator, Shape.init2D(2, 4));
    defer grad_out.deinit(allocator);
    grad_out.fill(1.0);

    const node = Node{
        .id = 0,
        .op = .matmul,
        .parents = .{ null, null },
        .n_parents = 2,
        .saved = .{ .tensor_pair = .{ .a = A, .b = B } },
    };

    const result = try backward(allocator, node, &grad_out);
    const da = result[0] orelse unreachable;
    defer {
        da.deinit(allocator);
        allocator.destroy(da);
    }
    const db = result[1] orelse unreachable;
    defer {
        db.deinit(allocator);
        allocator.destroy(db);
    }

    // dL/dA shape should be (2,3)
    try std.testing.expectEqual(@as(usize, 2), da.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), da.shape.dims[1]);

    // dL/dB shape should be (3,4)
    try std.testing.expectEqual(@as(usize, 3), db.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), db.shape.dims[1]);
}

test "backward exp — dL/da = dL/dc * exp(a)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.cpuData()[0] = 0.0; // exp(0) = 1
    a.cpuData()[1] = 1.0; // exp(1) = e
    a.cpuData()[2] = 2.0; // exp(2) = e²

    var grad_out = try Tensor.init(allocator, Shape.init1D(3));
    defer grad_out.deinit(allocator);
    grad_out.fill(1.0);

    const node = Node{
        .id = 0,
        .op = .exp,
        .parents = .{ null, null },
        .n_parents = 1,
        .saved = .{ .tensor_ref = a },
    };

    const result = try backward(allocator, node, &grad_out);
    const da = result[0] orelse unreachable;
    defer {
        da.deinit(allocator);
        allocator.destroy(da);
    }

    try std.testing.expect(std.math.approxEqAbs(f32, da.cpuData()[0], 1.0, 1e-5));
    try std.testing.expect(std.math.approxEqAbs(f32, da.cpuData()[1], std.math.e, 1e-4));
    try std.testing.expect(std.math.approxEqAbs(f32, da.cpuData()[2], std.math.e * std.math.e, 1e-3));
}

test "accumulateGrad — dst += src" {
    const allocator = std.testing.allocator;
    var dst = try Tensor.init(allocator, Shape.init1D(3));
    defer dst.deinit(allocator);
    dst.cpuData()[0] = 1.0;
    dst.cpuData()[1] = 2.0;
    dst.cpuData()[2] = 3.0;

    var src = try Tensor.init(allocator, Shape.init1D(3));
    defer src.deinit(allocator);
    src.cpuData()[0] = 10.0;
    src.cpuData()[1] = 20.0;
    src.cpuData()[2] = 30.0;

    accumulateGrad(&dst, &src);

    try std.testing.expectEqual(@as(f32, 11.0), dst.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 22.0), dst.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 33.0), dst.cpuData()[2]);
}

test "broadcastTo — (1,3) → (2,3)" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(1, 3));
    defer t.deinit(allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    t.cpuData()[2] = 3.0;

    var result = try broadcastTo(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);

    // Row 0: [1, 2, 3]
    try std.testing.expectEqual(@as(f32, 1.0), result.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 2.0), result.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 3.0), result.cpuData()[2]);
    // Row 1: [1, 2, 3] (repeated)
    try std.testing.expectEqual(@as(f32, 1.0), result.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 2.0), result.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 3.0), result.cpuData()[5]);
}

test "broadcastTo — (3,) → (2,3)" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init1D(3));
    defer t.deinit(allocator);
    t.cpuData()[0] = 4.0;
    t.cpuData()[1] = 5.0;
    t.cpuData()[2] = 6.0;

    var result = try broadcastTo(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);

    // Row 0: [4, 5, 6]
    try std.testing.expectEqual(@as(f32, 4.0), result.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 5.0), result.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 6.0), result.cpuData()[2]);
    // Row 1: [4, 5, 6]
    try std.testing.expectEqual(@as(f32, 4.0), result.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 5.0), result.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 6.0), result.cpuData()[5]);
}

test "broadcastTo — same shape returns owned copy" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);

    var result = try broadcastTo(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);
    try std.testing.expect(result.isOwned());
    try std.testing.expect(result.cpuData().ptr != t.cpuData().ptr);
    try std.testing.expectEqual(@as(f32, 0.0), result.cpuData()[0]);
}

test "broadcastTo — non-contiguous (transposed) view same shape" {
    const allocator = std.testing.allocator;
    // Build a 2x3 tensor, transpose it, then broadcastTo with the same
    // shape (3,2). This tests the fix for @memcpy-on-non-contiguous:
    // before the fix, @memcpy copied raw bytes ignoring strides,
    // silently producing the original (non-transposed) data.
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);
    // t = [[1, 2, 3], [4, 5, 6]]
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    t.cpuData()[2] = 3.0;
    t.cpuData()[3] = 4.0;
    t.cpuData()[4] = 5.0;
    t.cpuData()[5] = 6.0;

    // transpose2d returns a non-contiguous VIEW: shape (3,2), strides [1,3]
    // Logically: [[1,4], [2,5], [3,6]]
    const t_view = try t.transpose2d();
    try std.testing.expect(!t_view.isContiguous());

    var result = try broadcastTo(allocator, t_view, Shape.init2D(3, 2));
    defer result.deinit(allocator);

    // result must reflect the transposed (logical) ordering, not the
    // raw memory layout of the original tensor.
    try std.testing.expectEqual(@as(f32, 1.0), result.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 4.0), result.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 2.0), result.cpuData()[2]);
    try std.testing.expectEqual(@as(f32, 5.0), result.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 3.0), result.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 6.0), result.cpuData()[5]);
}

test "backward transpose2d — correct gradient through non-contiguous view" {
    const allocator = std.testing.allocator;
    // Simulate the exact scenario that was broken: a Linear layer's
    // weight (D_out, D_in) is transposed to (D_in, D_out) for matmul,
    // and the backward must produce a gradient of shape (D_out, D_in).
    // Before the fix, backwardTranspose2d called sumToShape on a
    // transposed view, which used @memcpy ignoring strides, producing
    // a silently transposed gradient.
    var w = try Tensor.init(allocator, Shape.init2D(3, 2));
    defer w.deinit(allocator);
    // w = [[1, 2], [3, 4], [5, 6]]  shape (3, 2)
    w.cpuData()[0] = 1.0;
    w.cpuData()[1] = 2.0;
    w.cpuData()[2] = 3.0;
    w.cpuData()[3] = 4.0;
    w.cpuData()[4] = 5.0;
    w.cpuData()[5] = 6.0;

    // Upstream gradient for w^T has shape (2, 3)
    var grad_wt = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer grad_wt.deinit(allocator);
    // grad_wt = [[10, 20, 30], [40, 50, 60]]
    grad_wt.cpuData()[0] = 10.0;
    grad_wt.cpuData()[1] = 20.0;
    grad_wt.cpuData()[2] = 30.0;
    grad_wt.cpuData()[3] = 40.0;
    grad_wt.cpuData()[4] = 50.0;
    grad_wt.cpuData()[5] = 60.0;

    const node = Node{
        .id = 0,
        .op = .transpose2d,
        .parents = .{ null, null },
        .n_parents = 1,
        .saved = .{ .tensor_ref = w },
    };

    const result = try backward(allocator, node, &grad_wt);
    const da = result[0] orelse unreachable;
    defer {
        da.deinit(allocator);
        allocator.destroy(da);
    }

    // da should have shape (3, 2) = w's shape
    // da = transpose(grad_wt) = [[10, 40], [20, 50], [30, 60]]
    try std.testing.expectEqual(@as(usize, 3), da.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), da.shape.dims[1]);
    try std.testing.expectEqual(@as(f32, 10.0), da.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 40.0), da.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 20.0), da.cpuData()[2]);
    try std.testing.expectEqual(@as(f32, 50.0), da.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, 30.0), da.cpuData()[4]);
    try std.testing.expectEqual(@as(f32, 60.0), da.cpuData()[5]);
}
