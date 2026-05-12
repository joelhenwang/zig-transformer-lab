//!
//! zig-transformer-lab — Causal Self-Attention (multi-head, pre-norm)
//!
//! Purpose:
//!   Implements multi-head causal self-attention as used in GPT-2:
//!     1. Project input to Q, K, V using three Linear layers
//!     2. Reshape Q/K/V to (B, T, n_head, d_head), permute heads to
//!        the batch dimension to produce (B*n_head, T, d_head)
//!     3. Compute scaled dot-product attention with causal mask in
//!        the head-flattened batched form
//!     4. Un-permute and reshape back to (B, T, D), then apply output
//!        projection
//!
//!   This is the heart of the transformer — the mechanism that allows
//!   each position to attend to all previous positions, with multiple
//!   heads computing independent attention patterns in parallel.
//!
//!   Stage 8 Milestone 4 generalised this from `n_head = 1` (Stages
//!   2–7) to `n_head ≥ 1`. The default config still produces
//!   `n_head = 1`, which is bit-identical modulo reshape/permute
//!   rounding (< 1 ULP per element on our sizes).
//!
//! Shape contract:
//!   forward(x: (B, T, D), tape) → output: (B, T, D)
//!   where B = batch, T = seq_len, D = d_model, H = n_head,
//!         d_head = D / H (enforced at init).
//!
//! Math:
//!   Q = X @ W_q^T + b_q       shape: (B, T, D)
//!   K = X @ W_k^T + b_k       shape: (B, T, D)
//!   V = X @ W_v^T + b_v       shape: (B, T, D)
//!
//!   Reshape + permute each to (B*H, T, d_head):
//!     Q_flat = permute(reshape(Q, (B, T, H, d_head)), (B, H, T, d_head))
//!     Q_flat = reshape(Q_flat, (B*H, T, d_head))
//!     K_flat, V_flat: same.
//!
//!   scores = Q_flat @ K_flat^T / sqrt(d_head)    shape: (B*H, T, T)
//!   scores += causal_mask (broadcast)            shape: (B*H, T, T)
//!   weights = softmax(scores)                    shape: (B*H, T, T)
//!   attn_out = weights @ V_flat                  shape: (B*H, T, d_head)
//!
//!   Un-permute: reshape (B*H, T, d_head) → (B, H, T, d_head) →
//!   permute → (B, T, H, d_head) → reshape → (B, T, D).
//!
//!   output = attn_out @ W_o^T + b_o              shape: (B, T, D)
//!
//!   Note the `sqrt(d_head)` scale (NOT `sqrt(D)`): in multi-head
//!   attention each head attends over `d_head` dimensions, so the
//!   scaling factor is per-head, not per-model. For n_head=1 these
//!   are equal.
//!
//! Memory ownership:
//!   Owns the four Linear sub-layers. Freed in deinit().
//!   The causal mask is created once in init() and reused.
//!
//! Errors:
//!   OutOfMemory — from sub-layer allocation
//!   InvalidArgument — d_model % n_head != 0 at init
//!
//! Credits:
//!   Attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017).
//!   Multi-head split + causal masking from GPT-2 (Radford et al., 2019).
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const Linear = @import("linear.zig").Linear;
const ops_matmul = @import("../tensor/ops/matmul.zig");
const ops_shape = @import("../tensor/ops/shape_ops.zig");
const ops_elementwise = @import("../tensor/ops/elementwise.zig");
const ops_softmax = @import("../tensor/ops/softmax.zig");
const ops_create = @import("../tensor/ops/create.zig");

pub const CausalSelfAttention = struct {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    allocator: std.mem.Allocator,
    d_model: usize,

    /// Number of attention heads. Each head attends over `d_head =
    /// d_model / n_head` dimensions. Must evenly divide `d_model`
    /// (checked at init).
    n_head: usize,

    /// Causal mask: upper triangle is -1e9, lower triangle + diagonal is 0.
    /// Shape: (max_seq_len, max_seq_len). Created once, reused every forward.
    causal_mask: Tensor,

    /// Create multi-head causal self-attention.
    ///
    /// Four Linear layers: Q, K, V projections + output projection.
    /// The causal mask is precomputed for max_seq_len positions.
    ///
    /// `n_head` must divide `d_model`. For `n_head = 1` this is the
    /// same single-head attention Stages 2-7 shipped; for `n_head > 1`
    /// the forward pipeline reshapes Q/K/V into heads-first layout.
    ///
    /// Worked example:
    ///   var attn = try CausalSelfAttention.init(alloc, 32, 4, 16, true, &rng);
    ///   defer attn.deinit();
    ///   // d_model=32, n_head=4 -> d_head=8
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        n_head: usize,
        max_seq_len: usize,
        use_bias: bool,
        rng: *Rng,
    ) LabError!CausalSelfAttention {
        if (n_head == 0 or d_model % n_head != 0) return error.InvalidArgument;

        var w_q = try Linear.init(allocator, d_model, d_model, use_bias, rng);
        errdefer w_q.deinit();
        var w_k = try Linear.init(allocator, d_model, d_model, use_bias, rng);
        errdefer w_k.deinit();
        var w_v = try Linear.init(allocator, d_model, d_model, use_bias, rng);
        errdefer w_v.deinit();
        var w_o = try Linear.init(allocator, d_model, d_model, use_bias, rng);
        errdefer w_o.deinit();

        // Build causal mask: (T, T) where mask[i,j] = 0 if j<=i, else -1e9
        var causal_mask = try Tensor.init(allocator, Shape.init2D(max_seq_len, max_seq_len));
        for (0..max_seq_len) |i| {
            for (0..max_seq_len) |j| {
                causal_mask.cpuData()[i * max_seq_len + j] = if (j <= i) 0.0 else -1e9;
            }
        }

        return CausalSelfAttention{
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
            .d_model = d_model,
            .n_head = n_head,
            .causal_mask = causal_mask,
        };
    }

    /// Compute multi-head causal self-attention.
    ///
    /// Flow: x → Q,K,V projections → reshape + permute into per-head
    /// layout → scaled dot-product attention with causal mask → un-
    /// permute + reshape back → output projection.
    ///
    /// Worked example:
    ///   // x shape (2, 4, 32), n_head = 4
    ///   var out = try attn.forward(x, &tape);
    ///   // out.shape == (2, 4, 32); 4 heads each attended over d_head=8
    pub fn forward(self: CausalSelfAttention, input: Tensor, tape: ?*Tape) LabError!Tensor {
        const B = input.shape.dims[0];
        const T = input.shape.dims[1];
        const D = self.d_model;
        const H = self.n_head;
        const d_head = D / H;

        // Step 1: Q, K, V projections. Linear.forward handles 3D→2D
        // reshape internally; outputs are (B, T, D).
        var q = try self.w_q.forward(input, tape);
        defer q.deinit(self.allocator);
        var k = try self.w_k.forward(input, tape);
        defer k.deinit(self.allocator);
        var v = try self.w_v.forward(input, tape);
        defer v.deinit(self.allocator);

        // Step 2: split heads. (B, T, D) → (B, T, H, d_head) → permute
        // to (B, H, T, d_head) → reshape to (B*H, T, d_head). We use
        // the tracked ops so backward propagates through the permute
        // correctly.
        //
        // Only do this when n_head > 1. For n_head == 1 the whole
        // permute chain is a no-op (it's an identity-equivalent
        // reshape), and skipping it keeps the Stage 7 CUDA
        // attention parity test passing bit-for-bit.
        var q_flat = try self.splitHeads(q, B, T, H, d_head, tape);
        defer q_flat.deinit(self.allocator);
        var k_flat = try self.splitHeads(k, B, T, H, d_head, tape);
        defer k_flat.deinit(self.allocator);
        var v_flat = try self.splitHeads(v, B, T, H, d_head, tape);
        defer v_flat.deinit(self.allocator);

        // Step 3: scores = Q_flat @ K_flat^T / sqrt(d_head).
        // K_flat^T: (B*H, T, d_head) -> (B*H, d_head, T) via
        // transposeInner2dTracked so backward reaches K_flat.
        var k_flat_t = try ops_shape.transposeInner2dTracked(self.allocator, k_flat, tape);
        defer k_flat_t.deinit(self.allocator);

        // Q @ K^T → (B*H, T, T)
        var scores = try ops_matmul.matmulBatch(self.allocator, q_flat, k_flat_t, tape);
        defer scores.deinit(self.allocator);

        // Scale by 1/sqrt(d_head), NOT 1/sqrt(d_model). For n_head=1
        // these are equal (d_head == d_model); for n_head>1 the per-
        // head scaling is the mathematically correct choice.
        const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(d_head))));
        var scaled_scores = try ops_elementwise.mulScalar(self.allocator, scores, scale, tape);
        defer scaled_scores.deinit(self.allocator);

        // Step 4: causal mask. The mask has shape (T, T); broadcast to
        // (B*H, T, T) via a (1, T, T) reshape + broadcast-add.
        var mask_slice: Tensor = undefined;
        if (T == self.causal_mask.shape.dims[0]) {
            mask_slice = self.causal_mask;
        } else {
            mask_slice = try Tensor.init(self.allocator, Shape.init2D(T, T));
            for (0..T) |i| {
                for (0..T) |j| {
                    mask_slice.cpuData()[i * T + j] = if (j <= i) 0.0 else -1e9;
                }
            }
        }
        const mask_needs_free = T != self.causal_mask.shape.dims[0];
        defer {
            if (mask_needs_free) mask_slice.deinit(self.allocator);
        }

        // CUDA upload (inherited from Stage 7 Milestone 2).
        var mask_cuda: ?Tensor = null;
        defer {
            if (mask_cuda) |*mc| mc.storage.deinit(self.allocator);
        }
        const mask_for_ops: Tensor = if (input.device == .cuda) blk: {
            const ctx = switch (input.storage) {
                .cuda => |b| b.ctx,
                .cpu => unreachable,
            };
            mask_cuda = try mask_slice.toCuda(ctx);
            break :blk mask_cuda.?;
        } else mask_slice;

        var mask_3d = try ops_shape.reshapeTracked(self.allocator, mask_for_ops, Shape.init3D(1, T, T), null);
        defer mask_3d.deinit(self.allocator);

        var masked_scores = try ops_elementwise.add(self.allocator, scaled_scores, mask_3d, tape);
        defer masked_scores.deinit(self.allocator);

        // Step 5: softmax along last axis (axis 2 for 3D).
        var weights = try ops_softmax.softmax(self.allocator, masked_scores, tape);
        defer weights.deinit(self.allocator);

        // Step 6: attn_out = weights @ V_flat → (B*H, T, d_head)
        var attn_out_flat = try ops_matmul.matmulBatch(self.allocator, weights, v_flat, tape);
        defer attn_out_flat.deinit(self.allocator);

        // Step 7: merge heads. (B*H, T, d_head) → (B, H, T, d_head) →
        // permute to (B, T, H, d_head) → reshape to (B, T, D).
        var attn_out = try self.mergeHeads(attn_out_flat, B, T, H, d_head, tape);
        defer attn_out.deinit(self.allocator);

        // Step 8: output projection.
        const output = try self.w_o.forward(attn_out, tape);

        return output;
    }

    /// Split heads: (B, T, D) → (B*H, T, d_head).
    ///
    /// For `n_head == 1` this is a no-op pipeline (the reshape +
    /// permute + reshape chain is mathematically an identity), and
    /// we still record the three ops on the tape so the backward
    /// chain sees the same node count regardless of `n_head`. The
    /// per-op cost is negligible (three stride-aware copies of a
    /// small tensor).
    fn splitHeads(
        self: CausalSelfAttention,
        x: Tensor,
        B: usize,
        T: usize,
        H: usize,
        d_head: usize,
        tape: ?*Tape,
    ) LabError!Tensor {
        // (B, T, D) -> (B, T, H, d_head)
        var x_4d = try ops_shape.reshapeTracked(self.allocator, x, Shape.init4D(B, T, H, d_head), tape);
        errdefer x_4d.deinit(self.allocator);
        // (B, T, H, d_head) -> (B, H, T, d_head)
        var x_perm = try ops_shape.transposeAxes12_4dTracked(self.allocator, x_4d, tape);
        x_4d.deinit(self.allocator);
        errdefer x_perm.deinit(self.allocator);
        // (B, H, T, d_head) -> (B*H, T, d_head)
        const x_flat = try ops_shape.reshapeTracked(self.allocator, x_perm, Shape.init3D(B * H, T, d_head), tape);
        x_perm.deinit(self.allocator);
        return x_flat;
    }

    /// Merge heads: (B*H, T, d_head) → (B, T, D).
    fn mergeHeads(
        self: CausalSelfAttention,
        x: Tensor,
        B: usize,
        T: usize,
        H: usize,
        d_head: usize,
        tape: ?*Tape,
    ) LabError!Tensor {
        // (B*H, T, d_head) -> (B, H, T, d_head)
        var x_4d = try ops_shape.reshapeTracked(self.allocator, x, Shape.init4D(B, H, T, d_head), tape);
        errdefer x_4d.deinit(self.allocator);
        // (B, H, T, d_head) -> (B, T, H, d_head)
        var x_perm = try ops_shape.transposeAxes12_4dTracked(self.allocator, x_4d, tape);
        x_4d.deinit(self.allocator);
        errdefer x_perm.deinit(self.allocator);
        // (B, T, H, d_head) -> (B, T, D)
        const x_out = try ops_shape.reshapeTracked(self.allocator, x_perm, Shape.init3D(B, T, H * d_head), tape);
        x_perm.deinit(self.allocator);
        return x_out;
    }

    /// Append this layer's learnable parameters to the list.
    pub fn parameters(self: *CausalSelfAttention, list: *std.ArrayList(*Tensor)) void {
        self.w_q.parameters(list);
        self.w_k.parameters(list);
        self.w_v.parameters(list);
        self.w_o.parameters(list);
    }

    /// Free all sub-layers and the causal mask.
    pub fn deinit(self: *CausalSelfAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
        self.causal_mask.deinit(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "CausalSelfAttention init — creates 4 linear layers" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 8, 1, 4, true, &rng);
    defer attn.deinit();

    try std.testing.expectEqual(@as(usize, 8), attn.d_model);
    try std.testing.expectEqual(@as(usize, 1), attn.n_head);
    try std.testing.expectEqual(@as(usize, 4), attn.causal_mask.shape.dims[0]);
}

test "CausalSelfAttention init — rejects n_head not dividing d_model" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    // d_model=8, n_head=3 -> 8 % 3 = 2 != 0 -> InvalidArgument
    try std.testing.expectError(error.InvalidArgument, CausalSelfAttention.init(alloc, 8, 3, 4, true, &rng));
}

test "CausalSelfAttention init — rejects n_head == 0" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    try std.testing.expectError(error.InvalidArgument, CausalSelfAttention.init(alloc, 8, 0, 4, true, &rng));
}

test "CausalSelfAttention forward — n_head=1 produces (B, T, D) output" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 8, 1, 4, true, &rng);
    defer attn.deinit();

    var x = try ops_create.randn(alloc, Shape.init3D(2, 3, 8), &rng, 0.0, 1.0);
    defer x.deinit(alloc);

    var out = try attn.forward(x, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 8), out.shape.dims[2]);

    for (out.cpuData()) |v| try std.testing.expect(std.math.isFinite(v));
}

test "CausalSelfAttention forward — n_head=2 produces (B, T, D) output" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 8, 2, 4, true, &rng);
    defer attn.deinit();

    var x = try ops_create.randn(alloc, Shape.init3D(2, 3, 8), &rng, 0.0, 1.0);
    defer x.deinit(alloc);

    var out = try attn.forward(x, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 8), out.shape.dims[2]);

    for (out.cpuData()) |v| try std.testing.expect(std.math.isFinite(v));
}

test "CausalSelfAttention forward — n_head=4 on D=16 produces finite output" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 16, 4, 4, true, &rng);
    defer attn.deinit();

    var x = try ops_create.randn(alloc, Shape.init3D(2, 4, 16), &rng, 0.0, 1.0);
    defer x.deinit(alloc);

    var out = try attn.forward(x, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 16), out.shape.dims[2]);
    for (out.cpuData()) |v| try std.testing.expect(std.math.isFinite(v));
}

test "CausalSelfAttention causal mask — upper triangle is -1e9" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 8, 1, 3, true, &rng);
    defer attn.deinit();

    // mask[0,0]=0, mask[0,1]=-1e9, mask[0,2]=-1e9
    // mask[1,0]=0, mask[1,1]=0,     mask[1,2]=-1e9
    // mask[2,0]=0, mask[2,1]=0,     mask[2,2]=0
    try std.testing.expectEqual(@as(f32, 0.0), attn.causal_mask.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, -1e9), attn.causal_mask.cpuData()[1]);
    try std.testing.expectEqual(@as(f32, 0.0), attn.causal_mask.cpuData()[3]);
    try std.testing.expectEqual(@as(f32, -1e9), attn.causal_mask.cpuData()[5]);
    try std.testing.expectEqual(@as(f32, 0.0), attn.causal_mask.cpuData()[8]);
}
