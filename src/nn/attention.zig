//!
//! zig-transformer-lab — Causal Self-Attention (1 head, pre-norm)
//!
//! Purpose:
//!   Implements single-head causal self-attention as used in GPT-2:
//!     1. Project input to Q, K, V using three Linear layers
//!     2. Compute scaled dot-product attention with causal mask
//!     3. Project attention output back to d_model
//!
//!   This is the heart of the transformer — the mechanism that allows
//!   each position to attend to all previous positions.
//!
//!   num_heads=1 is hard-coded (per decision D5). Multi-head attention
//!   is deferred to a later stage.
//!
//! Shape contract:
//!   forward(x: (B, T, D), tape) → output: (B, T, D)
//!   where B=batch, T=seq_len, D=d_model
//!
//! Math:
//!   Q = X @ W_q^T + b_q    shape: (B*T, D) → reshape to (B, T, D)
//!   K = X @ W_k^T + b_k    shape: same
//!   V = X @ W_v^T + b_v    shape: same
//!
//!   scores = Q @ K^T / sqrt(D)    shape: (B, T, T)
//!   scores += causal_mask          shape: (B, T, T) [upper triangle = -1e9]
//!   weights = softmax(scores)      shape: (B, T, T)
//!   attn_out = weights @ V         shape: (B, T, D)
//!
//!   output = attn_out @ W_o^T + b_o  shape: (B, T, D)
//!
//! Memory ownership:
//!   Owns the four Linear sub-layers. Freed in deinit().
//!   The causal mask is created once in init() and reused.
//!
//! Errors:
//!   OutOfMemory — from sub-layer allocation
//!
//! Credits:
//!   Attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017).
//!   Causal masking from GPT-2 (Radford et al., 2019). No code copied.

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

    /// Causal mask: upper triangle is -1e9, lower triangle + diagonal is 0.
    /// Shape: (max_seq_len, max_seq_len). Created once, reused every forward.
    causal_mask: Tensor,

    /// Create single-head causal self-attention.
    ///
    /// Four Linear layers: Q, K, V projections + output projection.
    /// The causal mask is precomputed for max_seq_len positions.
    ///
    /// Worked example:
    ///   var attn = try CausalSelfAttention.init(alloc, 32, 16, true, &rng);
    ///   defer attn.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        max_seq_len: usize,
        use_bias: bool,
        rng: *Rng,
    ) LabError!CausalSelfAttention {
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
                causal_mask.data[i * max_seq_len + j] = if (j <= i) 0.0 else -1e9;
            }
        }

        return CausalSelfAttention{
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
            .d_model = d_model,
            .causal_mask = causal_mask,
        };
    }

    /// Compute causal self-attention.
    ///
    /// Flow: x → Q,K,V projections → scaled dot-product attention
    /// with causal mask → output projection.
    ///
    /// Worked example:
    ///   // x shape (2, 4, 32)
    ///   var out = try attn.forward(x, &tape);
    ///   // out.shape == (2, 4, 32)
    pub fn forward(self: CausalSelfAttention, input: Tensor, tape: ?*Tape) LabError!Tensor {
        const B = input.shape.dims[0];
        const T = input.shape.dims[1];
        _ = B;

        // Step 1: Q, K, V projections
        // Linear.forward handles 3D→2D reshape internally
        var q = try self.w_q.forward(input, tape);
        defer q.deinit(self.allocator);
        var k = try self.w_k.forward(input, tape);
        defer k.deinit(self.allocator);
        var v = try self.w_v.forward(input, tape);
        defer v.deinit(self.allocator);

        // q, k, v are (B, T, D). We need them as (B, T, D) for batched matmul.

        // Step 2: scores = Q @ K^T / sqrt(D)
        // K^T: swap last two dims → (B, D, T)
        // Use transposeInner2dTracked so the tape records the transpose —
        // backward can then transpose the matmulBatch's gradient for K^T
        // back to match K's shape (B, T, D). Without tracking, the
        // matmulBatch backward produces a gradient with K^T's shape,
        // which doesn't match K's shape.
        var k_t = try ops_shape.transposeInner2dTracked(self.allocator, k, tape);
        defer k_t.deinit(self.allocator);

        // Q @ K^T → (B, T, T)
        var scores = try ops_matmul.matmulBatch(self.allocator, q, k_t, tape);
        defer scores.deinit(self.allocator);

        // Scale by 1/sqrt(D)
        const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(self.d_model))));
        var scaled_scores = try ops_elementwise.mulScalar(self.allocator, scores, scale, tape);
        defer scaled_scores.deinit(self.allocator);

        // Step 3: Add causal mask
        // Slice the mask to the actual sequence length T
        var mask_slice: Tensor = undefined;
        if (T == self.causal_mask.shape.dims[0]) {
            mask_slice = self.causal_mask;
        } else {
            // Create a T×T slice of the mask
            mask_slice = try Tensor.init(self.allocator, Shape.init2D(T, T));
            for (0..T) |i| {
                for (0..T) |j| {
                    mask_slice.data[i * T + j] = if (j <= i) 0.0 else -1e9;
                }
            }
        }
        const mask_needs_free = T != self.causal_mask.shape.dims[0];
        defer {
            if (mask_needs_free) mask_slice.deinit(self.allocator);
        }

        // Session 2: when input lives on CUDA, upload the mask
        // on-demand. The causal mask is built on CPU in
        // CausalSelfAttention.init (hand-computed), but the ops
        // below need it to share a device with scaled_scores. We
        // re-upload per-forward (tiny: at most max_seq_len^2
        // floats) rather than caching — caching would require
        // mut-self on forward, which the current pre-norm block
        // doesn't thread through.
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

        // Broadcast mask from (T,T) to (B,T,T) and add
        // We need to expand mask to (B, T, T) for elementwise add
        var mask_3d = try ops_shape.reshapeTracked(self.allocator, mask_for_ops, Shape.init3D(1, T, T), null);
        defer mask_3d.deinit(self.allocator);

        var masked_scores = try ops_elementwise.add(self.allocator, scaled_scores, mask_3d, tape);
        defer masked_scores.deinit(self.allocator);

        // Step 4: softmax along last axis (axis 2 for 3D)
        // Our softmax operates along the last axis, which is what we want
        var weights = try ops_softmax.softmax(self.allocator, masked_scores, tape);
        defer weights.deinit(self.allocator);

        // Step 5: attn_out = weights @ V → (B, T, D)
        var attn_out = try ops_matmul.matmulBatch(self.allocator, weights, v, tape);
        defer attn_out.deinit(self.allocator);

        // Step 6: Output projection
        const output = try self.w_o.forward(attn_out, tape);

        return output;
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

    var attn = try CausalSelfAttention.init(alloc, 8, 4, true, &rng);
    defer attn.deinit();

    try std.testing.expectEqual(@as(usize, 8), attn.d_model);
    try std.testing.expectEqual(@as(usize, 4), attn.causal_mask.shape.dims[0]);
}

test "CausalSelfAttention forward — correct output shape" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 8, 4, true, &rng);
    defer attn.deinit();

    var x = try ops_create.randn(alloc, Shape.init3D(2, 3, 8), &rng, 0.0, 1.0);
    defer x.deinit(alloc);

    var out = try attn.forward(x, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 8), out.shape.dims[2]);

    // All values should be finite
    const n = out.data.len;
    for (0..n) |i| {
        try std.testing.expect(std.math.isFinite(out.data[i]));
    }
}

test "CausalSelfAttention causal mask — upper triangle is -1e9" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var attn = try CausalSelfAttention.init(alloc, 8, 3, true, &rng);
    defer attn.deinit();

    // mask[0,0]=0, mask[0,1]=-1e9, mask[0,2]=-1e9
    // mask[1,0]=0, mask[1,1]=0,     mask[1,2]=-1e9
    // mask[2,0]=0, mask[2,1]=0,     mask[2,2]=0
    try std.testing.expectEqual(@as(f32, 0.0), attn.causal_mask.data[0]);
    try std.testing.expectEqual(@as(f32, -1e9), attn.causal_mask.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), attn.causal_mask.data[3]);
    try std.testing.expectEqual(@as(f32, -1e9), attn.causal_mask.data[5]);
    try std.testing.expectEqual(@as(f32, 0.0), attn.causal_mask.data[8]);
}
