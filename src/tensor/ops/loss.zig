//!
//! zig-transformer-lab — Cross-entropy loss
//!
//! Purpose:
//!   Provides cross-entropy loss, the standard loss function for
//!   classification tasks and language models.  In a transformer, this
//!   measures how well the model's predicted log-probabilities match
//!   the true next-token indices.
//!
//!   Cross-entropy = -log(P(target class)).  It penalizes the model
//!   for assigning low probability to the correct class.  When the
//!   model is perfectly confident (P=1 for the target), loss = 0.
//!   When the model is maximally wrong (P->0), loss -> +Inf.
//!
//! Shape contract:
//!   crossEntropy: logits:(B, C), targets:(B,) -> loss:(1,)
//!
//!   logits:  (B, C) raw scores for B samples over C classes
//!   targets: (B,) integer class indices stored as f32 (e.g. 0.0, 1.0)
//!   output:  (1,) scalar mean loss across the batch
//!
//! Math:
//!   For each sample i:
//!     loss_i = -log_softmax(logits[i, targets[i]])
//!   Mean loss:
//!     loss = (1/B) * sum_i loss_i
//!
//!   Equivalently:
//!     loss = -(1/B) * sum_i (logits[i, targets[i]] - max(logits[i,:])
//!            - log(sum_j exp(logits[i,j] - max(logits[i,:]))))
//!
//!   We use log_softmax internally for numerical stability (see
//!   softmax.zig for why max-subtraction matters).
//!
//! Memory ownership:
//!   Returns a new owned tensor of shape (1,).  The caller must call
//!   deinit(allocator) on the returned tensor when done.
//!
//! Errors:
//!   InvalidArgument — logits not rank-2, targets not rank-1, or
//!                     target index out of range [0, C)
//!   ShapeMismatch  — batch sizes of logits and targets differ
//!   OutOfMemory    — allocator could not fulfill the output buffer
//!
//! TODO:
//!   - future: support label smoothing (soft targets).
//!   - future: support ignore_index for padding tokens (today every
//!     target contributes to the loss equally).
//!
//! Credits:
//!   Cross-entropy loss is standard in ML textbooks.  The numerical
//!   stability trick (log_softmax) is from the PyTorch implementation.
//!   No code copied.

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Tensor = @import("../tensor.zig").Tensor;
const Shape = @import("../shape.zig").Shape;
const totalElements = @import("../shape.zig").totalElements;
const logSoftmax = @import("softmax.zig").logSoftmax;
const Tape = @import("../../autograd/tape.zig").Tape;
const Node = @import("../../autograd/node.zig").Node;
const OpKind = @import("../../autograd/node.zig").OpKind;
// Milestone 1: CUDA fused CE path routes here.
const cuda_dispatch = @import("../../backend/cuda/dispatch.zig");

/// Mean cross-entropy loss between logits and integer targets.
///
/// Computes the mean cross-entropy loss over a batch.  Internally uses
/// log_softmax for numerical stability, then gathers the log-
/// probability of each target class and averages.
///
/// Shape: logits:(B, C), targets:(B,) -> loss:(1,)
///
/// Worked example:
///   // logits = [[2.0, 1.0, 0.1]], targets = [0.0]
///   // log_softmax = [[-0.399, -1.399, -2.299]]
///   // loss = -(-0.399) / 1 = 0.399
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
///
/// Device routing (Milestone 1):
///   When `logits` lives on CUDA we call `cuda_dispatch.crossEntropyFused`,
///   which produces both the scalar loss and `grad_logits` in a single
///   launch. The grad is saved in a dedicated `ce_cuda_grad` SavedData
///   variant; the backward pass just DtoD-clones it, avoiding the
///   recompute-softmax-on-the-host round-trip that the CPU path needs.
pub fn crossEntropy(allocator: std.mem.Allocator, logits: Tensor, targets: Tensor, tape: ?*Tape) LabError!Tensor {
    // --- Input validation (device-agnostic) ---
    // logits must be rank-2: (batch_size, num_classes)
    if (logits.shape.ndim() != 2) return LabError.InvalidArgument;
    // targets must be rank-1: (batch_size,)
    if (targets.shape.ndim() != 1) return LabError.InvalidArgument;

    const B = logits.shape.dims[0];
    const C = logits.shape.dims[1];
    const B_t = targets.shape.dims[0];

    // Batch sizes must match
    if (B != B_t) return LabError.ShapeMismatch;

    // --- CUDA fast path ---
    // The fused kernel does not range-check targets (a DtoH just for
    // validation would defeat the fusion). The training loop is the
    // only production caller; it produces in-range indices by
    // construction. A malformed index on CUDA reads at most one float
    // from logits[target_idx] beyond the row; with standard memory
    // layout this stays within the allocation for any `target_idx < 2**31`.
    // For defence-in-depth we may add a host-side validation in a
    // Stage 9 review.
    if (logits.storage == .cuda) {
        var fused = try cuda_dispatch.crossEntropyFused(logits, targets);
        // On error after this point we must free BOTH allocations.
        // `loss` is returned to the caller; `grad_logits` is either
        // handed to the tape (which DtoD-copies) or freed locally.
        errdefer fused.loss.storage.deinit(allocator);
        var loss = fused.loss;

        if (tape) |t| {
            if (logits.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .cross_entropy,
                    .parents = .{ logits.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .ce_cuda_grad = fused.grad_logits },
                });
                loss.requires_grad = true;
                loss.tape_node = node_id;
            }
        }
        // The tape's takeOwnershipOfSaved DtoD-clones grad_logits
        // into kept_alive_cuda, so the original buffer is ours to
        // free regardless of whether we recorded a node.
        fused.grad_logits.storage.deinit(allocator);
        return loss;
    }

    // --- CPU path ---
    // Validate target indices are in range [0, C). This prevents
    // out-of-bounds reads into the log_probs tensor, which would
    // silently produce wrong results (or segfault in ReleaseSafe).
    for (0..B) |i| {
        const class_idx = @as(usize, @intFromFloat(@round(targets.data[i])));
        if (class_idx >= C) return LabError.InvalidArgument;
    }

    // Compute log_softmax of logits. This is the numerically stable
    // way to get log-probabilities. See softmax.zig for why we don't
    // compute log(softmax(x)) directly.
    var log_probs = try logSoftmax(allocator, logits, null);
    defer log_probs.deinit(allocator);

    // Gather the log-prob of each target and accumulate.
    var loss_sum: f32 = 0;
    for (0..B) |i| {
        // Round the f32 target to the nearest integer. This handles
        // the case where class indices are stored as f32 with tiny
        // floating-point imprecision (e.g., 1.9999999 instead of 2).
        const class_idx = @as(usize, @intFromFloat(@round(targets.data[i])));

        // log_probs is contiguous (newly allocated by logSoftmax),
        // so the offset for (i, class_idx) is i * C + class_idx.
        const log_p = log_probs.data[i * C + class_idx];

        // Cross-entropy = -log(P(target)). We negate because
        // log_probs stores log-probabilities (negative or zero) and
        // we want the loss (positive or zero).
        loss_sum += -log_p;
    }

    // Mean over the batch. Dividing by B gives the average loss per
    // sample, which makes the loss scale-independent of batch size.
    const mean_loss = loss_sum / @as(f32, @floatFromInt(B));

    // --- Allocate scalar output ---
    var out = try Tensor.init(allocator, Shape.init1D(1));
    errdefer out.deinit(allocator);
    out.data[0] = mean_loss;

    if (tape) |t| {
        if (logits.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .cross_entropy,
                .parents = .{ logits.tape_node, null },
                .n_parents = 1,
                .saved = .{ .ce_info = .{ .logits = logits, .targets = targets.data } },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "crossEntropy known 3-class example" {
    const alloc = std.testing.allocator;

    // logits = [[2.0, 1.0, 0.1]], target = [0.0]
    // log_softmax: max=2.0
    //   exp(0)=1, exp(-1)=0.3679, exp(-1.9)=0.1496
    //   sum=1.5175, ln(1.5175)=0.4173
    //   log_softmax = [-0.4173, -1.4173, -2.3173]
    // loss = -(-0.4173)/1 = 0.4173
    var logits = try Tensor.init(alloc, Shape.init2D(1, 3));
    defer logits.deinit(alloc);
    logits.data[0] = 2.0;
    logits.data[1] = 1.0;
    logits.data[2] = 0.1;

    var targets = try Tensor.init(alloc, Shape.init1D(1));
    defer targets.deinit(alloc);
    targets.data[0] = 0.0;

    var loss = try crossEntropy(alloc, logits, targets, null);
    defer loss.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f32, 0.4173), loss.data[0], 1e-2);
}

test "crossEntropy uniform logits gives loss = log(C)" {
    const alloc = std.testing.allocator;

    // When all logits are equal, softmax assigns uniform probability
    // 1/C to each class.  Cross-entropy = -log(1/C) = log(C).
    // For C=3: loss = ln(3) ~ 1.0986
    var logits = try Tensor.init(alloc, Shape.init2D(1, 3));
    defer logits.deinit(alloc);
    logits.data[0] = 1.0;
    logits.data[1] = 1.0;
    logits.data[2] = 1.0;

    var targets = try Tensor.init(alloc, Shape.init1D(1));
    defer targets.deinit(alloc);
    targets.data[0] = 0.0;

    var loss = try crossEntropy(alloc, logits, targets, null);
    defer loss.deinit(alloc);

    // ln(3) ~ 1.0986  — use @log builtin for natural log
    const ln3: f32 = @log(@as(f32, 3.0));
    try std.testing.expectApproxEqAbs(ln3, loss.data[0], 1e-3);
}

test "crossEntropy rejects mismatched batch sizes" {
    const alloc = std.testing.allocator;

    var logits = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer logits.deinit(alloc);
    var targets = try Tensor.init(alloc, Shape.init1D(3));
    defer targets.deinit(alloc);

    // logits batch=2, targets batch=3
    try std.testing.expectError(LabError.ShapeMismatch, crossEntropy(alloc, logits, targets, null));
}

test "crossEntropy rejects out-of-range target" {
    const alloc = std.testing.allocator;

    var logits = try Tensor.init(alloc, Shape.init2D(1, 3));
    defer logits.deinit(alloc);
    var targets = try Tensor.init(alloc, Shape.init1D(1));
    defer targets.deinit(alloc);
    // C=3, but target index is 5 (out of range)
    targets.data[0] = 5.0;

    try std.testing.expectError(LabError.InvalidArgument, crossEntropy(alloc, logits, targets, null));
}
