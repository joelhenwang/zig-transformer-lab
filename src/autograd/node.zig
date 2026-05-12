//!
//! zig-transformer-lab — Autograd tape node and operation kinds
//!
//! Purpose:
//!   Defines the building blocks of the tape-based reverse-mode autograd
//!   engine. Every operation that participates in gradient computation
//!   records a Node on the Tape. During backward, the Tape traverses
//!   these nodes in reverse topological order, accumulating gradients.
//!
//!   This is the Zig equivalent of PyTorch's autograd graph: each Node
//!   is like a torch.autograd.Function's ctx, storing exactly the data
//!   needed to compute the backward pass for one operation.
//!
//! Shape contract:
//!   Nodes don't have shapes — they store NodeIds that reference Tensors
//!   in the tape. The backward functions (in backward.zig) are responsible
//!   for producing gradient tensors with the correct shapes.
//!
//! Math:
//!   Reverse-mode autograd computes vector-Jacobian products (VJPs).
//!   For a function f: R^n -> R^m and an upstream gradient v in R^m,
//!   the VJP is J_f^T v, where J_f is the m x n Jacobian.
//!
//!   For our specific ops:
//!     add(a, b):        dL/da = dL/dc (broadcast-reduced to a's shape)
//!     mul(a, b):        dL/da = dL/dc * b,  dL/db = dL/dc * a
//!     matmul(A, B):     dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
//!     softmax(x):       dL/dx = S * (dL/dS - (dL/dS · S) * 1)
//!     cross_entropy:    dL/dlogits = softmax(logits) - one_hot(targets)
//!
//!   These formulas are derived and explained in detail in backward.zig
//!   and docs/03_autograd.md.
//!
//! Memory ownership:
//!   Node is a value type (copied by value into the tape's ArrayList).
//!   The saved tensors in SavedData are BORROWED pointers — they point
//!   to tensors that must outlive the tape. In the training loop, this
//!   is guaranteed because:
//!     1. The tape is created fresh each training step.
//!     2. All intermediate tensors (saved data) live in the tape's
//!        node records or are held by the caller.
//!     3. The tape is deinited before the input tensors.
//!
//!   The grad_map (in tape.zig) OWNS the gradient tensors it creates
//!   during backward, and frees them in deinit().
//!
//! Errors:
//!   None in this file — Node construction is infallible.
//!
//! TODO:
//!   - future: compress saved tensors by recomputing the forward for
//!     unary ops instead of storing inputs. Classic activation
//!     checkpointing / gradient checkpointing tradeoff: saves memory
//!     at the cost of compute.
//!
//! Credits:
//!   The tape-based autograd design is inspired by micrograd (Andrej
//!   Karpathy) and PyTorch's autograd engine. No code copied.
//;

const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const NodeId = @import("../tensor/tensor.zig").NodeId;
const Shape = @import("../tensor/shape.zig").Shape;

// ---------------------------------------------------------------------------
// OpKind — enumeration of all operations that support autograd
// ---------------------------------------------------------------------------

/// Every operation that can appear on the autograd tape.
///
/// This enum is exhaustive: the backward dispatch in tape.zig does a
/// switch on OpKind, and the compiler will error if we add a new variant
/// here without adding a corresponding backward case. This is a major
/// safety advantage over function-pointer-based dispatch.
///
/// Grouped by category:
///   - Elementwise binary: add, sub, mul, div
///   - Elementwise scalar:  add_scalar, mul_scalar
///   - Linear algebra:     matmul, matmul_batch
///   - Shape transforms:   transpose2d, reshape
///   - Reductions:         sum, mean
///   - Unary activations:  exp, log, neg, relu, gelu
///   - Normalization:      softmax, log_softmax
///   - Loss:               cross_entropy
///   - Lookup:             embedding (stub — full backward in Stage 4)
pub const OpKind = enum {
    // Elementwise binary ops (two tensor inputs)
    add,
    sub,
    mul,
    div,

    // Elementwise scalar ops (one tensor + one scalar)
    add_scalar,
    mul_scalar,

    // Linear algebra
    matmul,
    matmul_batch,

    // Shape transforms (no data movement, just stride/shape changes)
    transpose2d,
    reshape,

    // Reductions
    sum,
    mean,

    // Unary activations
    exp,
    log,
    neg,
    relu,
    gelu,

    // Normalization
    softmax,
    log_softmax,

    // Loss functions
    cross_entropy,

    // Unary math (elementwise)
    sqrt,

    // Embedding lookup (Stage 4 will implement the backward)
    embedding,

    // 3D inner transpose: (B, M, K) → (B, K, M), swaps dims[1] and dims[2]
    transpose_inner2d,

    // 4D axis-1/2 transpose: (B, T, H, D) → (B, H, T, D), swaps
    // dims[1] and dims[2]. Added in Stage 8 M4 to support multi-head
    // attention, where Q/K/V are reshaped to (B, T, H, d_head) and
    // then permuted so head-axis can be folded into the batch
    // dimension before batched matmul.
    transpose_axes12_4d,
};

// ---------------------------------------------------------------------------
// SavedData — what a Node remembers from the forward pass
// ---------------------------------------------------------------------------

/// Data saved during the forward pass that the backward pass needs.
///
/// Different operations need different pieces of information. The
/// tagged union approach makes it clear what each backward consumes
/// and prevents accidentally saving too much.
///
/// CRITICAL DESIGN: tensor_ref and tensor_pair store Tensor structs
/// BY VALUE, not by pointer. When ops take Tensor by value, storing
/// a pointer to the by-value parameter would dangle after the op
/// returns (the parameter is a stack-local copy). By storing the
/// entire Tensor struct, we capture the data slice header (which
/// still points to the original heap buffer) and the shape/strides.
/// The original heap buffer is alive as long as the caller's tensor
/// outlives the tape — the same contract as PyTorch's autograd.
///
/// SavedData convention per op:
///   add, sub       → tensor_pair(a, b)  [need both shapes for sumToShape]
///   mul, div       → tensor_pair(a, b)  [need both values and shapes]
///   add_scalar     → tensor_scalar(shape, s) [need a's shape, scalar value]
///   mul_scalar     → tensor_scalar(shape, s)
///   matmul, batch  → tensor_pair(A, B)  [need both for dA, dB]
///   transpose2d    → tensor_ref(a)      [need a's shape]
///   reshape        → tensor_ref(a)      [need a's shape]
///   sum, mean      → reduce_info        [need input shape + axis]
///   exp, log, relu,
///   gelu, softmax,
///   log_softmax    → tensor_ref(a)      [need input values]
///   neg            → nothing            [backward is just negate]
///   cross_entropy  → ce_info            [need logits + targets]
///   embedding      → embedding_info     [need weight + indices]
pub const SavedData = union(enum) {
    /// No saved data needed (neg).
    nothing,

    /// A snapshot of a single tensor (stored by value).
    /// The `data` slice points to the original heap buffer, which
    /// must outlive the tape. Shape and strides are copies.
    /// Used by: transpose2d, reshape, exp, log, relu, gelu, softmax, log_softmax.
    tensor_ref: Tensor,

    /// Snapshots of two tensors (stored by value).
    /// Same lifetime contract as tensor_ref: the original tensors'
    /// data buffers must outlive the tape.
    /// Used by: add, sub, mul, div, matmul, matmul_batch.
    tensor_pair: struct { a: Tensor, b: Tensor },

    /// Input shape plus a scalar value (stored by value — no dangling pointer).
    /// Used by: add_scalar, mul_scalar.
    tensor_scalar: struct { shape: Shape, scalar: f32 },

    /// Reduction info: the input shape and the reduction axis.
    /// Used by: sum, mean.
    /// Shape is stored by value so backward doesn't need a dangling pointer.
    reduce_info: struct { shape: Shape, axis: u2 },

    /// Cross-entropy info: logits tensor snapshot + target indices.
    /// Used by: cross_entropy (CPU path only).
    /// logits is a snapshot (by value); targets is a borrowed data slice
    /// that must outlive the tape.
    ce_info: struct { logits: Tensor, targets: []const f32 },

    /// Cross-entropy CUDA fast path: the fused forward kernel already
    /// computes `grad_logits = (softmax(logits) - one_hot(targets)) / N`
    /// inside one launch, so the backward pass has nothing to recompute.
    /// We save `grad_logits` directly and `backwardCrossEntropy` returns
    /// a DtoD clone of it.
    ///
    /// Rationale for a dedicated variant (not an optional field on
    /// `ce_info`): `ce_info.targets` is a `[]const f32` slice borrowed
    /// from `targets.data`, which is `&.{}` for CUDA tensors (PR-δ
    /// invariant — CUDA tensors keep an empty compat alias). A shared
    /// variant with two conditionally-valid fields reads like a
    /// footgun; two disjoint variants keep the CPU and CUDA code paths
    /// cleanly separate.
    ce_cuda_grad: Tensor,

    /// Embedding info: weight tensor snapshot + index tensor data.
    /// Used by: embedding (stub for Stage 4).
    embedding_info: struct { weight: Tensor, indices: []const f32 },
};

// ---------------------------------------------------------------------------
// Node — one entry on the autograd tape
// ---------------------------------------------------------------------------

/// Maximum number of parent nodes an operation can have.
/// Binary ops (add, sub, mul, div, matmul) have 2 parents.
/// Unary ops (exp, relu, etc.) have 1 parent.
/// Some ops (cross_entropy) have 2 logical parents (logits + targets)
/// but targets typically don't require grad, so effectively 1.
const max_parents = 2;

/// A single node on the autograd tape.
///
/// Each node records:
///   1. A unique ID (assigned by the tape on recording).
///   2. The operation kind (determines which backward to run).
///   3. The IDs of the parent nodes (inputs to the forward op).
///   4. How many parents are actually used (1 or 2).
///   5. Saved data from the forward pass (tensors, scalars, etc.).
///
/// The tape stores an ArrayList of these nodes. During backward, we
/// walk this list in reverse topological order, and for each node we
/// look up the gradient of its output, compute the gradient of each
/// parent input, and accumulate into the grad map.
///
/// Worked example — building a tape for (a * b + c):
///
///   The tape only records COMPUTATION nodes, not leaf tensors.
///   Leaf tensors (a, b, c) have tape_node=null and requires_grad=true
///   until they first participate in a recorded op, at which point
///   trackLeaf() assigns them a placeholder node. The tape records
///   the actual operations that produce intermediate tensors:
///
///     Node 0: op=mul,  parents=[a.id, b.id], saved=tensor_pair  // a * b → d
///     Node 1: op=add,  parents=[d.id, c.id], saved=nothing      // d + c → loss
///
///   Where a.id, b.id, c.id are the placeholder NodeIds assigned by
///   trackLeaf, and d.id is the id of Node 0 (stored back onto the
///   mul output tensor as tensor.tape_node).
pub const Node = struct {
    /// Unique ID for this node, assigned by the tape.
    /// This is also the tape_node value stored on the output tensor.
    id: NodeId,

    /// What operation this node represents.
    /// Determines which backward function to call.
    op: OpKind,

    /// Parent node IDs (inputs to the forward operation).
    /// Only the first n_parents entries are meaningful.
    /// For a leaf (no operation), both are null and n_parents=0.
    parents: [max_parents]?NodeId,

    /// How many parent entries in `parents` are valid (0, 1, or 2).
    n_parents: u2,

    /// Data saved during the forward pass that the backward needs.
    /// See SavedData for what each variant stores.
    saved: SavedData,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "OpKind is exhaustive — all variants can be constructed" {
    // This test verifies that every OpKind variant exists and can be
    // instantiated. If we add a new variant, this test will still
    // compile (unlike the backward dispatch switch, which will error).
    const ops = [_]OpKind{
        .add,       .sub,         .mul,         .div,
        .add_scalar, .mul_scalar,
        .matmul,    .matmul_batch,
        .transpose2d, .reshape,
        .sum,       .mean,
        .exp,       .log,        .neg,         .relu, .gelu,
        .softmax,   .log_softmax,
        .cross_entropy,
        .sqrt,
        .embedding,
        .transpose_inner2d,
        .transpose_axes12_4d,
    };
    // Just verify we can create and compare them
    try std.testing.expect(ops[0] == .add);
    try std.testing.expect(ops[ops.len - 1] == .transpose_axes12_4d);
    try std.testing.expect(ops.len == 24);
}

test "Node construction — binary op" {
    // Simulate a node for: c = a + b
    // where a's tape_node = 10, b's tape_node = 20
    var a = try Tensor.init(std.testing.allocator, @import("../tensor/shape.zig").Shape.init1D(3));
    defer a.deinit(std.testing.allocator);
    var b = try Tensor.init(std.testing.allocator, @import("../tensor/shape.zig").Shape.init1D(3));
    defer b.deinit(std.testing.allocator);

    // SavedData stores Tensor by value — a snapshot of the struct.
    // The `data` slice in the snapshot still points to the original
    // heap buffer, which must outlive the tape.
    const node = Node{
        .id = 30,
        .op = .add,
        .parents = .{ 10, 20 },
        .n_parents = 2,
        .saved = .{ .tensor_pair = .{ .a = a, .b = b } },
    };
    try std.testing.expectEqual(@as(NodeId, 30), node.id);
    try std.testing.expect(node.op == .add);
    try std.testing.expectEqual(@as(NodeId, 10), node.parents[0].?);
    try std.testing.expectEqual(@as(NodeId, 20), node.parents[1].?);
    try std.testing.expectEqual(@as(u2, 2), node.n_parents);
}

test "Node construction — unary op with saved tensor" {
    // Simulate a node for: y = relu(x)
    // where x's tape_node = 5, and we save x for the backward mask
    var x = try Tensor.init(std.testing.allocator, @import("../tensor/shape.zig").Shape.init1D(3));
    defer x.deinit(std.testing.allocator);

    // SavedData stores the Tensor struct by value (snapshot).
    // The `data` slice in the snapshot shares the original heap buffer.
    const node = Node{
        .id = 6,
        .op = .relu,
        .parents = .{ 5, null },
        .n_parents = 1,
        .saved = .{ .tensor_ref = x },
    };
    try std.testing.expect(node.op == .relu);
    try std.testing.expectEqual(@as(u2, 1), node.n_parents);
    try std.testing.expect(node.parents[1] == null);

    // Verify the saved data snapshot shares the same data buffer
    switch (node.saved) {
        .tensor_ref => |t| {
            try std.testing.expect(t.cpuData().ptr == x.cpuData().ptr);
        },
        else => unreachable,
    }
}

test "SavedData variants — tensor_scalar" {
    const saved = SavedData{ .tensor_scalar = .{ .shape = @import("../tensor/shape.zig").Shape.init1D(1), .scalar = 3.14 } };
    switch (saved) {
        .tensor_scalar => |ts| {
            try std.testing.expectEqual(@as(f32, 3.14), ts.scalar);
            try std.testing.expectEqual(@as(usize, 1), ts.shape.dims[0]);
        },
        else => unreachable,
    }
}

test "SavedData variants — nothing" {
    const saved = SavedData.nothing;
    try std.testing.expect(saved == .nothing);
}
