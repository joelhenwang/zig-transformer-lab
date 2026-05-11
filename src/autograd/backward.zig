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

const ops_elementwise = @import("../tensor/ops/elementwise.zig");
const ops_matmul = @import("../tensor/ops/matmul.zig");
const ops_shape = @import("../tensor/ops/shape_ops.zig");
const ops_reduce = @import("../tensor/ops/reduce.zig");
const ops_unary = @import("../tensor/ops/unary.zig");
const ops_softmax = @import("../tensor/ops/softmax.zig");
const ops_create = @import("../tensor/ops/create.zig");
// PR-iota: backward.broadcastTo routes to CUDA when grad is on GPU.
const cuda_dispatch = @import("../backend/cuda/dispatch.zig");

const max_parent_grads = 2;
pub const BackwardResult = [max_parent_grads]?*Tensor;

/// Heap-allocate a Tensor value and return a pointer to it.
///
/// BackwardResult entries are `?*Tensor`, so every Tensor value produced
/// by an op must be boxed on the heap before being stored in the result
/// array. This helper centralizes the alloc+store pattern to avoid
/// repeating allocator.create / ptr.* = ... at every call site.
///
/// Memory ownership:
///   The returned pointer is owned by the caller, who must eventually
///   call ptr.deinit(allocator) and allocator.destroy(ptr).
fn heapAlloc(allocator: std.mem.Allocator, tensor: Tensor) LabError!*Tensor {
    const ptr = try allocator.create(Tensor);
    ptr.* = tensor;
    return ptr;
}

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
        .add => try backwardAdd(allocator, node, grad_output, &result),
        .sub => try backwardSub(allocator, node, grad_output, &result),
        .mul => try backwardMul(allocator, node, grad_output, &result),
        .div => try backwardDiv(allocator, node, grad_output, &result),

        // ----- Elementwise scalar -----
        .add_scalar => try backwardAddScalar(allocator, node, grad_output, &result),
        .mul_scalar => try backwardMulScalar(allocator, node, grad_output, &result),

        // ----- Linear algebra -----
        .matmul => try backwardMatmul(allocator, node, grad_output, &result),
        .matmul_batch => try backwardMatmulBatch(allocator, node, grad_output, &result),

        // ----- Shape transforms -----
        .transpose2d => try backwardTranspose2d(allocator, node, grad_output, &result),
        .reshape => try backwardReshape(allocator, node, grad_output, &result),

        // ----- Reductions -----
        .sum => try backwardSum(allocator, node, grad_output, &result),
        .mean => try backwardMean(allocator, node, grad_output, &result),

        // ----- Unary activations -----
        .exp => try backwardExp(allocator, node, grad_output, &result),
        .log => try backwardLog(allocator, node, grad_output, &result),
        .neg => try backwardNeg(allocator, node, grad_output, &result),
        .relu => try backwardRelu(allocator, node, grad_output, &result),
        .gelu => try backwardGelu(allocator, node, grad_output, &result),

        // ----- Normalization -----
        .softmax => try backwardSoftmax(allocator, node, grad_output, &result),
        .log_softmax => try backwardLogSoftmax(allocator, node, grad_output, &result),

        // ----- Loss -----
        .cross_entropy => try backwardCrossEntropy(allocator, node, grad_output, &result),

        // ----- Unary math -----
        .sqrt => try backwardSqrt(allocator, node, grad_output, &result),

        // ----- Embedding (Stage 4) -----
        .embedding => try backwardEmbedding(allocator, node, grad_output, &result),

        // ----- 3D inner transpose -----
        .transpose_inner2d => try backwardTransposeInner2d(allocator, node, grad_output, &result),

        // ----- 4D axis-1/2 transpose -----
        .transpose_axes12_4d => try backwardTransposeAxes12_4d(allocator, node, grad_output, &result),
    }

    return result;
}

// ---------------------------------------------------------------------------
// Individual backward implementations
// ---------------------------------------------------------------------------

/// add(a, b) → c
/// dL/da = sumToShape(dL/dc, a.shape)
/// dL/db = sumToShape(dL/dc, b.shape)
///
/// Broadcasting backward: if a was broadcast from (1,3) to (2,3),
/// sumToShape reduces the gradient back along the broadcasted axes.
fn backwardAdd(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_pair => |tp| {
            const da = try ops_reduce.sumToShape(allocator, grad_output.*, tp.a.shape);
            result[0] = try heapAlloc(allocator, da);
            const db = try ops_reduce.sumToShape(allocator, grad_output.*, tp.b.shape);
            result[1] = try heapAlloc(allocator, db);
        },
        else => return error.InvalidArgument,
    }
}

/// sub(a, b) → c
/// dL/da = sumToShape(dL/dc, a.shape)
/// dL/db = sumToShape(-dL/dc, b.shape)
fn backwardSub(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_pair => |tp| {
            const da = try ops_reduce.sumToShape(allocator, grad_output.*, tp.a.shape);
            result[0] = try heapAlloc(allocator, da);
            var neg_grad = try ops_unary.neg(allocator, grad_output.*, null);
            defer neg_grad.deinit(allocator);
            const db = try ops_reduce.sumToShape(allocator, neg_grad, tp.b.shape);
            result[1] = try heapAlloc(allocator, db);
        },
        else => return error.InvalidArgument,
    }
}

/// mul(a, b) → c
/// dL/da = sumToShape(dL/dc * b, a.shape)
/// dL/db = sumToShape(dL/dc * a, b.shape)
fn backwardMul(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_pair => |tp| {
            var da_full = try ops_elementwise.mul(allocator, grad_output.*, tp.b, null);
            defer da_full.deinit(allocator);
            const da = try ops_reduce.sumToShape(allocator, da_full, tp.a.shape);
            result[0] = try heapAlloc(allocator, da);

            var db_full = try ops_elementwise.mul(allocator, grad_output.*, tp.a, null);
            defer db_full.deinit(allocator);
            const db = try ops_reduce.sumToShape(allocator, db_full, tp.b.shape);
            result[1] = try heapAlloc(allocator, db);
        },
        else => return error.InvalidArgument,
    }
}

/// div(a, b) → c = a / b
/// dL/da = sumToShape(dL/dc / b, a.shape)
/// dL/db = sumToShape(-dL/dc * a / b², b.shape)
fn backwardDiv(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_pair => |tp| {
            var da_full = try ops_elementwise.div(allocator, grad_output.*, tp.b, null);
            defer da_full.deinit(allocator);
            const da = try ops_reduce.sumToShape(allocator, da_full, tp.a.shape);
            result[0] = try heapAlloc(allocator, da);

            // -dL/dc * a / b²
            var neg_grad = try ops_unary.neg(allocator, grad_output.*, null);
            defer neg_grad.deinit(allocator);
            var neg_times_a = try ops_elementwise.mul(allocator, neg_grad, tp.a, null);
            defer neg_times_a.deinit(allocator);
            var b_sq = try ops_elementwise.mul(allocator, tp.b, tp.b, null);
            defer b_sq.deinit(allocator);
            var db_full = try ops_elementwise.div(allocator, neg_times_a, b_sq, null);
            defer db_full.deinit(allocator);
            const db = try ops_reduce.sumToShape(allocator, db_full, tp.b.shape);
            result[1] = try heapAlloc(allocator, db);
        },
        else => return error.InvalidArgument,
    }
}

/// add_scalar(a, s) → c = a + s
/// dL/da = dL/dc  (scalar has no gradient)
fn backwardAddScalar(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_scalar => |ts| {
            const da = try ops_reduce.sumToShape(allocator, grad_output.*, ts.shape);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// mul_scalar(a, s) → c = a * s
/// dL/da = dL/dc * s
fn backwardMulScalar(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_scalar => |ts| {
            // dL/da = dL/dc * s
            var scaled = try ops_elementwise.mulScalar(allocator, grad_output.*, ts.scalar, null);
            defer scaled.deinit(allocator);
            const da = try ops_reduce.sumToShape(allocator, scaled, ts.shape);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// matmul(A, B) → C = A @ B
/// dL/dA = dL/dC @ Bᵀ
/// dL/dB = Aᵀ @ dL/dC
///
/// This is the most important backward in the transformer — every
/// linear layer and attention score computation uses matmul.
fn backwardMatmul(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_pair => |tp| {
            // dL/dA = dL/dC @ Bᵀ
            const bt = try tp.b.transpose2d();
            // CUDA matmul requires contiguous inputs; the transpose
            // view is non-contiguous. Materialise a fresh contiguous
            // copy via bcast_copy (identity-shape broadcastTo) and
            // deinit it after the matmul consumes it.
            if (bt.device == .cuda) {
                var bt_c = try cuda_dispatch.broadcastTo(bt, bt.shape);
                defer bt_c.storage.deinit(allocator);
                const da_val = try ops_matmul.matmul(allocator, grad_output.*, bt_c, null);
                result[0] = try heapAlloc(allocator, da_val);
            } else {
                const da_val = try ops_matmul.matmul(allocator, grad_output.*, bt, null);
                result[0] = try heapAlloc(allocator, da_val);
            }

            // dL/dB = Aᵀ @ dL/dC
            const at = try tp.a.transpose2d();
            if (at.device == .cuda) {
                var at_c = try cuda_dispatch.broadcastTo(at, at.shape);
                defer at_c.storage.deinit(allocator);
                const db_val = try ops_matmul.matmul(allocator, at_c, grad_output.*, null);
                result[1] = try heapAlloc(allocator, db_val);
            } else {
                const db_val = try ops_matmul.matmul(allocator, at, grad_output.*, null);
                result[1] = try heapAlloc(allocator, db_val);
            }
        },
        else => return error.InvalidArgument,
    }
}

/// matmul_batch(A, B) → C  [batched]
/// dL/dA = dL/dC @ Bᵀ,  dL/dB = Aᵀ @ dL/dC
/// Same formula as 2D matmul, but with inner-dim transpose for 3D.
fn backwardMatmulBatch(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_pair => |tp| {
            // B is (B_batch, K, N) → Bᵀ is (B_batch, N, K). For CUDA
            // we must materialise the transpose view into a contiguous
            // buffer because cuBLAS matmulBatch rejects non-contig
            // inputs. For CPU the view is fine — the CPU matmul walks
            // via strides.
            const bt = try ops_shape.transposeInner2d(tp.b);
            if (bt.device == .cuda) {
                var bt_c = try cuda_dispatch.broadcastTo(bt, bt.shape);
                defer bt_c.storage.deinit(allocator);
                const da_val = try ops_matmul.matmulBatch(allocator, grad_output.*, bt_c, null);
                result[0] = try heapAlloc(allocator, da_val);
            } else {
                const da_val = try ops_matmul.matmulBatch(allocator, grad_output.*, bt, null);
                result[0] = try heapAlloc(allocator, da_val);
            }

            // A is (B_batch, M, K) → Aᵀ is (B_batch, K, M). Same fix.
            const at = try ops_shape.transposeInner2d(tp.a);
            if (at.device == .cuda) {
                var at_c = try cuda_dispatch.broadcastTo(at, at.shape);
                defer at_c.storage.deinit(allocator);
                const db_val = try ops_matmul.matmulBatch(allocator, at_c, grad_output.*, null);
                result[1] = try heapAlloc(allocator, db_val);
            } else {
                const db_val = try ops_matmul.matmulBatch(allocator, at, grad_output.*, null);
                result[1] = try heapAlloc(allocator, db_val);
            }
        },
        else => return error.InvalidArgument,
    }
}

/// transpose_inner2d(a: (B, M, K)) → (B, K, M)
/// dL/da = transpose_inner2d(dL/dc)
/// Transpose is its own inverse: applying it twice restores the original.
fn backwardTransposeInner2d(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            // Transpose the gradient's inner dims back.
            const grad_t = try ops_shape.transposeInner2d(grad_output.*);
            // Make contiguous copy matching the original input's shape.
            const da = try ops_shape.reshapeTracked(allocator, grad_t, a.shape, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// transpose_axes12_4d(a: (B, T, H, D)) → (B, H, T, D)
/// dL/da = transpose_axes12_4d(dL/dc)
/// Same permutation applied twice returns the identity, so the
/// backward is just another call to the same view-producing op.
fn backwardTransposeAxes12_4d(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            const grad_t = try ops_shape.transposeAxes12_4d(grad_output.*);
            // Materialise contiguous to match the original input's shape.
            const da = try ops_shape.reshapeTracked(allocator, grad_t, a.shape, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// transpose2d(a) → aᵀ
/// dL/da = transpose2d(dL/dc)
/// Transpose is its own inverse: (Aᵀ)ᵀ = A.
fn backwardTranspose2d(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            // Transpose the gradient back. transpose2d() returns a
            // non-contiguous VIEW (strides are swapped, no data copy).
            // We MUST use reshapeTracked (which does stride-aware
            // element-by-element copy) instead of sumToShape (which
            // uses @memcpy in its fast path and ignores strides).
            //
            // Without this fix, the gradient for Linear.weight is
            // silently transposed — sumToShape sees the view's shape
            // matches the target, does @memcpy of the raw buffer, and
            // copies data in the wrong (non-transposed) order.
            const grad_t = try grad_output.transpose2d();
            const da = try ops_shape.reshapeTracked(allocator, grad_t, a.shape, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// reshape(a) → a_reshaped
/// dL/da = reshape(dL/dc, a.shape)
/// Just reinterpret the gradient's memory layout back to the original shape.
fn backwardReshape(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            // Reshape's backward is another reshape — the gradient just
            // needs to be reshaped back to the original input's shape.
            // sumToShape is wrong here: it sums over broadcast dims,
            // which would e.g. collapse (B,T,D)→(T,D) instead of
            // correctly reshaping (B,T,D)→(B*T,D).
            const da = try ops_shape.reshapeTracked(allocator, grad_output.*, a.shape, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// sum(a, axis) → a_reduced
/// dL/da = broadcastTo(dL/dc, a.shape)
///
/// The gradient of a sum is just the broadcast/expand of the upstream
/// gradient back to the original shape. When you sum along an axis,
/// each element in that axis contributed equally, so the gradient
/// flows back equally to all of them.
fn backwardSum(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .reduce_info => |ri| {
            const da = try broadcastTo(allocator, grad_output.*, ri.shape);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// mean(a, axis) → a_meaned
/// dL/da = broadcastTo(dL/dc / axis_size, a.shape)
///
/// Mean = sum / N, so the gradient has the 1/N factor distributed
/// equally to all elements along the reduced axis.
fn backwardMean(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .reduce_info => |ri| {
            const axis_size = @as(f32, @floatFromInt(ri.shape.dims[ri.axis]));
            var scaled = try ops_elementwise.mulScalar(allocator, grad_output.*, 1.0 / axis_size, null);
            defer scaled.deinit(allocator);
            const da = try broadcastTo(allocator, scaled, ri.shape);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// exp(a) → c = exp(a)
/// dL/da = dL/dc * exp(a)
fn backwardExp(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            var exp_a = try ops_unary.exp(allocator, a, null);
            defer exp_a.deinit(allocator);
            const da = try ops_elementwise.mul(allocator, grad_output.*, exp_a, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// log(a) → c = log(a)
/// dL/da = dL/dc / a
fn backwardLog(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            const da = try ops_elementwise.div(allocator, grad_output.*, a, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// neg(a) → c = -a
/// dL/da = -dL/dc
fn backwardNeg(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    _ = node;
    const da = try ops_unary.neg(allocator, grad_output.*, null);
    result[0] = try heapAlloc(allocator, da);
}

/// relu(a) → c = max(0, a)
/// dL/da = dL/dc * (a > 0 ? 1 : 0)
fn backwardRelu(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            var mask = try Tensor.init(allocator, a.shape);
            for (a.data, 0..) |v, i| {
                mask.data[i] = if (v > 0.0) 1.0 else 0.0;
            }
            defer mask.deinit(allocator);
            const da = try ops_elementwise.mul(allocator, grad_output.*, mask, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// gelu(a) → c = GELU(a)
/// dL/da = dL/dc * GELU'(a)
fn backwardGelu(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            // CUDA fast path (Milestone 2): one fused kernel computes
            // grad_in = grad_out * (phi + x*pdf) using erff + expf
            // intrinsics. This avoids the host-side loop over a.data
            // (which is the empty compat alias on CUDA).
            if (a.device == .cuda) {
                const da = try cuda_dispatch.geluExactBackward(a, grad_output.*);
                result[0] = try heapAlloc(allocator, da);
                return;
            }
            var gelu_grad = try Tensor.init(allocator, a.shape);
            defer gelu_grad.deinit(allocator);
            for (a.data, 0..) |v, i| {
                gelu_grad.data[i] = geluDerivative(v);
            }
            const da = try ops_elementwise.mul(allocator, grad_output.*, gelu_grad, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// sqrt(a) → c = √a
/// dL/da = dL/dc * 1 / (2 * √a) = dL/dc / (2 * c)
///
/// The derivative of √x is 1 / (2√x). Since we saved the input,
/// we compute √a again (or equivalently, divide by 2*c where c=√a).
fn backwardSqrt(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |a| {
            // √a — recompute the forward value
            var sqrt_a = try ops_unary.sqrt(allocator, a, null);
            defer sqrt_a.deinit(allocator);
            // 2 * √a
            var two_sqrt_a = try ops_elementwise.mulScalar(allocator, sqrt_a, 2.0, null);
            defer two_sqrt_a.deinit(allocator);
            // dL/da = dL/dc / (2√a)
            const da = try ops_elementwise.div(allocator, grad_output.*, two_sqrt_a, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// softmax(x) → S
/// dL/dx = S * (dL/dS - (dL/dS · S) * 1)  [row-wise, last axis]
///
/// This is the "diag - outer" formula for softmax Jacobian:
///   ∂softmax_i/∂x_j = S_i(δ_ij - S_j)
/// So the VJP is: dL/dx_j = S_j * (dL/dS_j - Σ_i S_i * dL/dS_i)
fn backwardSoftmax(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |x| {
            var s = try ops_softmax.softmax(allocator, x, null);
            defer s.deinit(allocator);

            // dot = sum(dL/dS * S, last_axis, keepdims)
            var prod = try ops_elementwise.mul(allocator, grad_output.*, s, null);
            defer prod.deinit(allocator);
            const ndim = grad_output.shape.ndim();
            const last_axis: u2 = @intCast(ndim - 1);
            var dot = try ops_reduce.sum(allocator, prod, last_axis, null);
            defer dot.deinit(allocator);

            // dL/dS - dot (broadcast across last axis)
            var diff = try ops_elementwise.sub(allocator, grad_output.*, dot, null);
            defer diff.deinit(allocator);

            // S * diff
            const da = try ops_elementwise.mul(allocator, s, diff, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// log_softmax(x) → y
/// dL/dx = dL/dy - softmax(x) * sum(dL/dy, last_axis, keepdims)
///
/// More numerically stable than going through softmax backward.
fn backwardLogSoftmax(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .tensor_ref => |x| {
            var s = try ops_softmax.softmax(allocator, x, null);
            defer s.deinit(allocator);

            const ndim = grad_output.shape.ndim();
            const last_axis: u2 = @intCast(ndim - 1);
            var sum_dy = try ops_reduce.sum(allocator, grad_output.*, last_axis, null);
            defer sum_dy.deinit(allocator);

            // softmax * sum_dy (broadcast across last axis)
            var scaled = try ops_elementwise.mul(allocator, s, sum_dy, null);
            defer scaled.deinit(allocator);

            // dL/dy - scaled
            const da = try ops_elementwise.sub(allocator, grad_output.*, scaled, null);
            result[0] = try heapAlloc(allocator, da);
        },
        else => return error.InvalidArgument,
    }
}

/// cross_entropy(logits, targets) → loss
/// dL/dlogits = softmax(logits) - one_hot(targets)
///
/// This is the famous "softmax + 1" backward. The gradient of
/// cross-entropy loss w.r.t. the logits is simply the softmax
/// probabilities minus a one-hot vector at the target class.
/// This is incredibly efficient — no Jacobian needed!
fn backwardCrossEntropy(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .ce_info => |ci| {
            // Recompute softmax(logits)
            var s = try ops_softmax.softmax(allocator, ci.logits, null);
            defer s.deinit(allocator);

            // Build one-hot: one_hot[i, targets[i]] = 1
            // Then: dL/dlogits = (softmax - one_hot) / B
            // where B is the batch size (since loss is mean over batch)
            const B = s.shape.dims[0];
            const C = s.shape.dims[1];
            var grad_logits = try Tensor.init(allocator, s.shape);
            for (0..B) |b| {
                for (0..C) |c| {
                    const idx = b * C + c;
                    const target_idx: usize = @intFromFloat(@round(ci.targets[b]));
                    const one_hot_val: f32 = if (c == target_idx) 1.0 else 0.0;
                    // dL/dlogits[i,j] = (1/B) * (softmax[i,j] - one_hot[i,j])
                    // The (1/B) comes from the mean reduction in cross_entropy
                    grad_logits.data[idx] = (s.data[idx] - one_hot_val) / @as(f32, @floatFromInt(B));
                }
            }

            // Scale by the upstream gradient (which is 1.0 for scalar loss)
            // For a scalar loss, grad_output = [1.0], so just multiply by 1.0
            const grad_scale = grad_output.data[0];
            for (0..grad_logits.data.len) |i| {
                grad_logits.data[i] *= grad_scale;
            }

            result[0] = try heapAlloc(allocator, grad_logits);
        },
        .ce_cuda_grad => |saved_grad| {
            // CUDA fast path (Milestone 1). The fused forward kernel
            // already baked the `1/B` mean factor and the
            // `softmax - one_hot` subtraction into `saved_grad`, so
            // backward just returns a DtoD clone.
            //
            // We intentionally do NOT read grad_output.data[0] here —
            // CUDA tensors keep an empty CPU compat alias. The tape's
            // `backward` seeds the loss gradient with 1.0 via
            // ops_create.onesLike, and cross-entropy output is always
            // the root of backward in our autograd design. If a future
            // caller ever routes a non-unit gradient through CE, they
            // would need to fuse the scale into the clone (one extra
            // mulScalar launch). Not needed today.
            const g_copy = try cuda_dispatch.cloneDevice(saved_grad);
            result[0] = try heapAlloc(allocator, g_copy);
        },
        else => return error.InvalidArgument,
    }
}

/// embedding(weight, indices) → output
/// dL/dweight = scatter-add: grad_weight[idx[i], :] += grad_output[i, :]
///
/// The embedding forward gathers rows: output[i] = weight[idx[i]].
/// The backward scatters the output gradient back into the weight
/// gradient: each row's gradient is added to the corresponding
/// weight row. This is a "scatter-add" operation.
///
/// The gradient for the indices (parent 1) is always null — integer
/// indices don't have gradients.
fn backwardEmbedding(
    allocator: std.mem.Allocator,
    node: Node,
    grad_output: *Tensor,
    result: *BackwardResult,
) LabError!void {
    switch (node.saved) {
        .embedding_info => |ei| {
            // ei.weight: (vocab_size, d_model) — snapshot by value
            // ei.indices: []const f32 — borrowed slice of index values
            const vocab_size = ei.weight.shape.dims[0];
            const d_model = ei.weight.shape.dims[1];

            // Initialize weight gradient to zeros
            var grad_weight = try Tensor.init(allocator, ei.weight.shape);
            grad_weight.fill(0.0);

            // Scatter-add: for each position in the output, the gradient
            // flows back to the corresponding row in the weight table.
            // grad_output shape: (num_indices, d_model)
            const num_indices = ei.indices.len;
            for (0..num_indices) |i| {
                const idx: usize = @intFromFloat(ei.indices[i]);
                if (idx >= vocab_size) {
                    grad_weight.deinit(allocator);
                    return error.InvalidArgument;
                }
                // grad_weight[idx, :] += grad_output[i, :]
                for (0..d_model) |d| {
                    grad_weight.data[idx * d_model + d] += grad_output.data[i * d_model + d];
                }
            }

            result[0] = try heapAlloc(allocator, grad_weight);
            // result[1] remains null — indices don't require grad
        },
        .tensor_pair => |tp| {
            // PR-mu: CUDA embedding path. tp.a = weight (CUDA snapshot),
            // tp.b = ids (CUDA snapshot). grad_output is CUDA too.
            const V = tp.a.shape.dims[0];
            // Flatten grad_output's leading dims into (N, D). The
            // existing shape is (...ids.shape..., D) which the scatter
            // kernel treats as a flat (N, D); tape-snapshot of tp.b
            // already has the ids.shape we need.
            const grad_weight = try cuda_dispatch.embeddingBackward(tp.b, grad_output.*, V);
            result[0] = try heapAlloc(allocator, grad_weight);
        },
        else => return error.InvalidArgument,
    }
}

// ---------------------------------------------------------------------------
// broadcastTo — expand a tensor to a larger shape
// ---------------------------------------------------------------------------

/// Broadcast a tensor from a smaller shape to a larger (target) shape.
///
/// This is the INVERSE of sumToShape: instead of summing along broadcast
/// axes, we REPEAT the data along those axes.
///
/// Algorithm:
///   1. If shapes are equal → return a view (no-op).
///   2. Right-align the shapes.
///   3. For each axis where target has a larger dim:
///      - If tensor dim is 1 → repeat the values along this axis.
///      - If tensor has fewer dims → add a new axis (unsqueeze).
///   4. Return the broadcasted tensor.
///
/// Worked example:
///   broadcastTo(t: (1,3), target: (2,3))
///     → [[a,b,c], [a,b,c]]  (repeat row 0 twice)
///
///   broadcastTo(t: (3,), target: (2,3))
///     → [[a,b,c], [a,b,c]]  (unsqueeze, then repeat)
///
/// Implementation: iterate over the output tensor's elements, mapping
/// each multi-dim index back to the input tensor using broadcast rules
/// (if input dim is 1, use index 0; otherwise use the output index).
///
/// Memory ownership:
///   Returns a new owned tensor. Caller must deinit.
pub fn broadcastTo(allocator: std.mem.Allocator, tensor: Tensor, target: Shape) LabError!Tensor {
    if (tensor.device == .cuda) {
        return try cuda_dispatch.broadcastTo(tensor, target);
    }
    // Same shape → return an owned copy (not a view) because callers
    // may deinit the source tensor, which would invalidate a view.
    // CRITICAL: must check contiguity before @memcpy. If the tensor
    // is a non-contiguous view (e.g., from transpose2d), @memcpy
    // copies raw buffer bytes in memory order, ignoring strides,
    // producing silently wrong data. The stride-aware loop reads
    // elements in logical (shape) order using strides for correct
    // indexing.
    if (shape_equals(tensor.shape, target)) {
        const out = try Tensor.init(allocator, target);
        if (isContiguous(tensor.shape, tensor.strides)) {
            @memcpy(out.data, tensor.data[0..out.data.len]);
        } else {
            const n = totalElements(target);
            const ndim = tensor.shape.ndim();
            for (0..n) |flat| {
                var offset: usize = 0;
                var remaining: usize = flat;
                var axis: usize = 0;
                while (axis + 1 < ndim) : (axis += 1) {
                    var block: usize = 1;
                    var a2: usize = axis + 1;
                    while (a2 < ndim) : (a2 += 1) block *= tensor.shape.dims[a2];
                    const idx = remaining / block;
                    remaining %= block;
                    offset += idx * tensor.strides.values[axis];
                }
                offset += remaining * tensor.strides.values[axis];
                out.data[flat] = tensor.data[offset];
            }
        }
        return out;
    }

    var result = try Tensor.init(allocator, target);
    const out_n = totalElements(target);
    const ndim_out = target.ndim();
    const ndim_in = tensor.shape.ndim();

    // For each element in the output, compute the corresponding input index
    for (0..out_n) |out_i| {
        // Decompose flat output index into multi-dim coordinates
        var out_coords: [4]usize = .{ 0, 0, 0, 0 };
        var remaining = out_i;
        for (0..ndim_out) |d| {
            const rev_d = ndim_out - 1 - d;
            out_coords[rev_d] = remaining % target.dims[rev_d];
            remaining /= target.dims[rev_d];
        }

        // Map output coordinates to input coordinates (broadcast rules)
        var in_flat: usize = 0;
        const extra_dims = if (ndim_out > ndim_in) ndim_out - ndim_in else 0;
        for (0..ndim_in) |d| {
            const out_d = extra_dims + d;
            const in_dim = tensor.shape.dims[d];
            const coord = if (in_dim == 1) 0 else out_coords[out_d];
            in_flat += coord * tensor.strides.values[d];
        }

        result.data[out_i] = tensor.data[in_flat];
    }

    return result;
}

// ---------------------------------------------------------------------------
// Gradient accumulation
// ---------------------------------------------------------------------------

/// Accumulate src gradient into dst: dst += src.
///
/// This implements the multivariable chain rule: if a tensor is used
/// in multiple operations, its gradient is the SUM of contributions
/// from each operation.
///
/// Device routing (Session 2):
///   CUDA tensors keep an empty CPU compat alias (PR-δ), so the
///   CPU loop `for (0..n) |i| dst.data[i] += src.data[i]` over
///   `dst.data.len == 0` is a silent no-op — dropping every
///   multi-path gradient contribution (e.g. LayerNorm's path from
///   `a` through `mean(a)` back to `a`). The CUDA branch launches
///   the same fused add kernel used by elementwise.add, writing
///   the result back into `dst`'s existing buffer in place.
pub fn accumulateGrad(dst: *Tensor, src: *Tensor) void {
    std.debug.assert(shape_equals(dst.shape, src.shape));
    if (dst.device == .cuda) {
        // In-place accumulate via DtoD temp: tmp = dst + src, copy tmp -> dst.
        //
        // Why not a dedicated in-place kernel? The current elw_add kernel
        // writes to a third buffer (`c = a + b`) and we don't have an
        // `add_inplace` variant. Adding one is ~10 LOC but adds an API
        // surface the CPU path doesn't have. For now we allocate a temp
        // output, then DtoD-copy back. One extra allocation + one extra
        // copy per accumulate; negligible at our scale.
        var tmp = cuda_dispatch.add(dst.*, src.*) catch |err| {
            // The assertion above + same-device invariants guarantee this
            // cannot fail for shape reasons. Only OOM is plausible; we
            // must not silently drop the accumulation.
            std.log.warn("accumulateGrad CUDA failed: {}", .{err});
            return;
        };
        defer tmp.storage.deinit(undefined); // allocator unused on CUDA deinit
        // Copy tmp -> dst's existing DeviceBuffer. Both are same shape,
        // contiguous, same context. cuMemcpyDtoD_v2 handles it.
        const dst_buf = switch (dst.storage) {
            .cuda => |b| b,
            .cpu => unreachable, // device==.cuda invariant
        };
        const src_buf = switch (tmp.storage) {
            .cuda => |b| b,
            .cpu => unreachable,
        };
        dst_buf.copyFromDevice(src_buf) catch |err| {
            std.log.warn("accumulateGrad CUDA DtoD copy failed: {}", .{err});
        };
        return;
    }
    const n = dst.data.len;
    for (0..n) |i| {
        dst.data[i] += src.data[i];
    }
}

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
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;

    var grad_out = try Tensor.init(allocator, Shape.init1D(3));
    defer grad_out.deinit(allocator);
    grad_out.data[0] = 1.0;
    grad_out.data[1] = 1.0;
    grad_out.data[2] = 1.0;

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

    try std.testing.expectEqual(@as(f32, -1.0), da.data[0]);
    try std.testing.expectEqual(@as(f32, -1.0), da.data[1]);
    try std.testing.expectEqual(@as(f32, -1.0), da.data[2]);
}

test "backward relu — dL/da = dL/dc * (a > 0)" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, Shape.init1D(4));
    defer a.deinit(allocator);
    a.data[0] = 2.0; // positive
    a.data[1] = -1.0; // negative
    a.data[2] = 0.0; // zero
    a.data[3] = 5.0; // positive

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

    try std.testing.expectEqual(@as(f32, 1.0), da.data[0]); // a > 0 → grad passes
    try std.testing.expectEqual(@as(f32, 0.0), da.data[1]); // a < 0 → grad blocked
    try std.testing.expectEqual(@as(f32, 0.0), da.data[2]); // a == 0 → grad blocked
    try std.testing.expectEqual(@as(f32, 1.0), da.data[3]); // a > 0 → grad passes
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
    grad_out.data[0] = 1.0;
    grad_out.data[1] = 2.0;
    grad_out.data[2] = 3.0;
    grad_out.data[3] = 4.0;
    grad_out.data[4] = 5.0;
    grad_out.data[5] = 6.0;

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
    try std.testing.expectEqual(@as(f32, 5.0), da.data[0]);
    try std.testing.expectEqual(@as(f32, 7.0), da.data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), da.data[2]);

    // dL/db = sumToShape(grad, (2,3)) → same shape, no reduction: [1, 2, 3, 4, 5, 6]
    try std.testing.expectEqual(@as(f32, 1.0), db.data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), db.data[5]);
}

test "backward matmul — dL/dA = dL/dC @ Bᵀ, dL/dB = Aᵀ @ dL/dC" {
    const allocator = std.testing.allocator;
    // A: (2,3), B: (3,4), C: (2,4)
    var A = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer A.deinit(allocator);
    var B = try Tensor.init(allocator, Shape.init2D(3, 4));
    defer B.deinit(allocator);

    // Fill with small values
    A.data[0] = 1.0;
    A.data[1] = 2.0;
    A.data[2] = 3.0;
    A.data[3] = 4.0;
    A.data[4] = 5.0;
    A.data[5] = 6.0;
    B.data[0] = 0.1;
    B.data[1] = 0.2;
    B.data[2] = 0.3;
    B.data[3] = 0.4;
    B.data[4] = 0.5;
    B.data[5] = 0.6;
    B.data[6] = 0.7;
    B.data[7] = 0.8;
    B.data[8] = 0.9;
    B.data[9] = 1.0;
    B.data[10] = 1.1;
    B.data[11] = 1.2;

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
    a.data[0] = 0.0; // exp(0) = 1
    a.data[1] = 1.0; // exp(1) = e
    a.data[2] = 2.0; // exp(2) = e²

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

    try std.testing.expect(std.math.approxEqAbs(f32, da.data[0], 1.0, 1e-5));
    try std.testing.expect(std.math.approxEqAbs(f32, da.data[1], std.math.e, 1e-4));
    try std.testing.expect(std.math.approxEqAbs(f32, da.data[2], std.math.e * std.math.e, 1e-3));
}

test "accumulateGrad — dst += src" {
    const allocator = std.testing.allocator;
    var dst = try Tensor.init(allocator, Shape.init1D(3));
    defer dst.deinit(allocator);
    dst.data[0] = 1.0;
    dst.data[1] = 2.0;
    dst.data[2] = 3.0;

    var src = try Tensor.init(allocator, Shape.init1D(3));
    defer src.deinit(allocator);
    src.data[0] = 10.0;
    src.data[1] = 20.0;
    src.data[2] = 30.0;

    accumulateGrad(&dst, &src);

    try std.testing.expectEqual(@as(f32, 11.0), dst.data[0]);
    try std.testing.expectEqual(@as(f32, 22.0), dst.data[1]);
    try std.testing.expectEqual(@as(f32, 33.0), dst.data[2]);
}

test "broadcastTo — (1,3) → (2,3)" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(1, 3));
    defer t.deinit(allocator);
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;

    var result = try broadcastTo(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);

    // Row 0: [1, 2, 3]
    try std.testing.expectEqual(@as(f32, 1.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), result.data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), result.data[2]);
    // Row 1: [1, 2, 3] (repeated)
    try std.testing.expectEqual(@as(f32, 1.0), result.data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), result.data[4]);
    try std.testing.expectEqual(@as(f32, 3.0), result.data[5]);
}

test "broadcastTo — (3,) → (2,3)" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init1D(3));
    defer t.deinit(allocator);
    t.data[0] = 4.0;
    t.data[1] = 5.0;
    t.data[2] = 6.0;

    var result = try broadcastTo(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);

    // Row 0: [4, 5, 6]
    try std.testing.expectEqual(@as(f32, 4.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), result.data[1]);
    try std.testing.expectEqual(@as(f32, 6.0), result.data[2]);
    // Row 1: [4, 5, 6]
    try std.testing.expectEqual(@as(f32, 4.0), result.data[3]);
    try std.testing.expectEqual(@as(f32, 5.0), result.data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), result.data[5]);
}

test "broadcastTo — same shape returns owned copy" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer t.deinit(allocator);

    var result = try broadcastTo(allocator, t, Shape.init2D(2, 3));
    defer result.deinit(allocator);
    try std.testing.expect(result.owned);
    try std.testing.expect(result.data.ptr != t.data.ptr);
    try std.testing.expectEqual(@as(f32, 0.0), result.data[0]);
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
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;
    t.data[4] = 5.0;
    t.data[5] = 6.0;

    // transpose2d returns a non-contiguous VIEW: shape (3,2), strides [1,3]
    // Logically: [[1,4], [2,5], [3,6]]
    const t_view = try t.transpose2d();
    try std.testing.expect(!t_view.isContiguous());

    var result = try broadcastTo(allocator, t_view, Shape.init2D(3, 2));
    defer result.deinit(allocator);

    // result must reflect the transposed (logical) ordering, not the
    // raw memory layout of the original tensor.
    try std.testing.expectEqual(@as(f32, 1.0), result.data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), result.data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), result.data[2]);
    try std.testing.expectEqual(@as(f32, 5.0), result.data[3]);
    try std.testing.expectEqual(@as(f32, 3.0), result.data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), result.data[5]);
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
    w.data[0] = 1.0;
    w.data[1] = 2.0;
    w.data[2] = 3.0;
    w.data[3] = 4.0;
    w.data[4] = 5.0;
    w.data[5] = 6.0;

    // Upstream gradient for w^T has shape (2, 3)
    var grad_wt = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer grad_wt.deinit(allocator);
    // grad_wt = [[10, 20, 30], [40, 50, 60]]
    grad_wt.data[0] = 10.0;
    grad_wt.data[1] = 20.0;
    grad_wt.data[2] = 30.0;
    grad_wt.data[3] = 40.0;
    grad_wt.data[4] = 50.0;
    grad_wt.data[5] = 60.0;

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
    try std.testing.expectEqual(@as(f32, 10.0), da.data[0]);
    try std.testing.expectEqual(@as(f32, 40.0), da.data[1]);
    try std.testing.expectEqual(@as(f32, 20.0), da.data[2]);
    try std.testing.expectEqual(@as(f32, 50.0), da.data[3]);
    try std.testing.expectEqual(@as(f32, 30.0), da.data[4]);
    try std.testing.expectEqual(@as(f32, 60.0), da.data[5]);
}
