//!
//! zig-transformer-lab — Transformer Block (pre-norm)
//!
//! Purpose:
//!   A single transformer block implementing the pre-norm architecture:
//!     x = x + Attention(LayerNorm(x))
//!     x = x + MLP(LayerNorm(x))
//!
//!   Pre-norm (LayerNorm before attention/MLP) is used in GPT-2 and
//!   is more stable for training than post-norm (per decision D13).
//!
//!   The residual connections (+ x) are critical for training deep
//!   networks — they allow gradients to flow directly through the
//!   skip connection, preventing vanishing gradients.
//!
//! Shape contract:
//!   forward(x: (B, T, D), tape) → output: (B, T, D)
//!
//! Math:
//!   h = x + Attention(LN_1(x))
//!   out = h + MLP(LN_2(h))
//!
//!   The residual add is recorded on the tape so gradients flow
//!   back through both the skip path (identity) and the
//!   transformation path (attention/MLP).
//!
//! Memory ownership:
//!   Owns ln1, attn, ln2, mlp sub-layers. Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from sub-layer allocation
//!
//! Credits:
//!   Pre-norm architecture from GPT-2 (Radford et al., 2019).
//!   Residual connections from "Deep Residual Learning" (He et al., 2016).
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const LayerNorm = @import("layernorm.zig").LayerNorm;
const CausalSelfAttention = @import("attention.zig").CausalSelfAttention;
const MLP = @import("mlp.zig").MLP;
const TransformerConfig = @import("module.zig").TransformerConfig;
const ops_elementwise = @import("../tensor/ops/elementwise.zig");

pub const TransformerBlock = struct {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: MLP,
    allocator: std.mem.Allocator,

    /// Create a transformer block from a config.
    ///
    /// Worked example:
    ///   const cfg = TransformerConfig{ .d_model = 32, .max_seq_len = 16 };
    ///   var block = try TransformerBlock.init(alloc, cfg, &rng);
    ///   defer block.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: TransformerConfig,
        rng: *Rng,
    ) LabError!TransformerBlock {
        var ln1 = try LayerNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
        errdefer ln1.deinit();
        var attn = try CausalSelfAttention.init(allocator, cfg.d_model, cfg.n_head, cfg.max_seq_len, cfg.bias, rng);
        errdefer attn.deinit();
        var ln2 = try LayerNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
        errdefer ln2.deinit();
        var mlp = try MLP.init(allocator, cfg.d_model, cfg.d_ff, cfg.bias, rng);
        errdefer mlp.deinit();

        return TransformerBlock{
            .ln1 = ln1,
            .attn = attn,
            .ln2 = ln2,
            .mlp = mlp,
            .allocator = allocator,
        };
    }

    /// Compute one transformer block: pre-norm + residual.
    ///
    /// h = x + Attention(LN_1(x))
    /// out = h + MLP(LN_2(h))
    ///
    /// Worked example:
    ///   // x shape (2, 4, 32)
    ///   var out = try block.forward(x, &tape);
    ///   // out.shape == (2, 4, 32)
    pub fn forward(self: TransformerBlock, input: Tensor, tape: ?*Tape) LabError!Tensor {
        // --- Attention sub-block ---
        // LN_1(x)
        var ln1_out = try self.ln1.forward(input, tape);
        defer ln1_out.deinit(self.allocator);

        // Attention(LN_1(x))
        var attn_out = try self.attn.forward(ln1_out, tape);
        defer attn_out.deinit(self.allocator);

        // x + Attention(LN_1(x))  — residual connection
        var h = try ops_elementwise.add(self.allocator, input, attn_out, tape);
        defer h.deinit(self.allocator);

        // --- MLP sub-block ---
        // LN_2(h)
        var ln2_out = try self.ln2.forward(h, tape);
        defer ln2_out.deinit(self.allocator);

        // MLP(LN_2(h))
        var mlp_out = try self.mlp.forward(ln2_out, tape);
        defer mlp_out.deinit(self.allocator);

        // h + MLP(LN_2(h))  — residual connection
        const output = try ops_elementwise.add(self.allocator, h, mlp_out, tape);

        return output;
    }

    /// Append this block's learnable parameters to the list.
    pub fn parameters(self: *TransformerBlock, list: *std.ArrayList(*Tensor)) void {
        self.ln1.parameters(list);
        self.attn.parameters(list);
        self.ln2.parameters(list);
        self.mlp.parameters(list);
    }

    /// Free all sub-layers.
    pub fn deinit(self: *TransformerBlock) void {
        self.ln1.deinit();
        self.attn.deinit();
        self.ln2.deinit();
        self.mlp.deinit();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "TransformerBlock forward — correct output shape" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .d_model = 8,
        .d_ff = 32,
        .max_seq_len = 4,
        .vocab_size = 16,
    };

    var block = try TransformerBlock.init(alloc, cfg, &rng);
    defer block.deinit();

    const ops_create = @import("../tensor/ops/create.zig");
    var x = try ops_create.randn(alloc, Shape.init3D(2, 3, 8), &rng, 0.0, 1.0);
    defer x.deinit(alloc);

    var out = try block.forward(x, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 8), out.shape.dims[2]);

    // Check all values finite
    for (0..out.data.len) |i| {
        try std.testing.expect(std.math.isFinite(out.data[i]));
    }
}

test "TransformerBlock parameters — collects from all sub-layers" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .d_model = 8,
        .d_ff = 32,
        .max_seq_len = 4,
        .vocab_size = 16,
    };

    var block = try TransformerBlock.init(alloc, cfg, &rng);
    defer block.deinit();

    var param_list: std.ArrayList(*Tensor) = .empty;
    defer param_list.deinit(alloc);

    // Pre-count: LN1(2) + Attn(4*2=8) + LN2(2) + MLP(2*2=4) = 16
    try param_list.ensureTotalCapacity(alloc, 16);
    block.parameters(&param_list);

    // Should have params from all sub-layers
    try std.testing.expect(param_list.items.len > 0);
}
