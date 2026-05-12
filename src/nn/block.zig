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
const RMSNorm = @import("rmsnorm.zig").RMSNorm;
const CausalSelfAttention = @import("attention.zig").CausalSelfAttention;
const MLP = @import("mlp.zig").MLP;
const SwiGLU = @import("swiglu.zig").SwiGLU;
const TransformerConfig = @import("module.zig").TransformerConfig;
const ops_elementwise = @import("../tensor/ops/elementwise.zig");

pub const TransformerBlock = struct {
    // Normalization (one of these is active based on config)
    ln1: ?LayerNorm,
    ln2: ?LayerNorm,
    rms1: ?RMSNorm,
    rms2: ?RMSNorm,

    attn: CausalSelfAttention,

    // MLP (one of these is active based on config)
    mlp: ?MLP,
    swiglu: ?SwiGLU,

    allocator: std.mem.Allocator,
    use_rms_norm: bool,
    use_swiglu: bool,

    /// Create a transformer block from a config.
    ///
    /// When config.use_rms_norm=true, uses RMSNorm instead of LayerNorm.
    /// When config.use_swiglu=true, uses SwiGLU instead of GELU MLP.
    ///
    /// Worked example:
    ///   const cfg = TransformerConfig{ .d_model = 32, .use_rms_norm = true };
    ///   var block = try TransformerBlock.init(alloc, cfg, &rng);
    ///   defer block.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: TransformerConfig,
        rng: *Rng,
    ) LabError!TransformerBlock {
        // --- Normalization layers ---
        var ln1: ?LayerNorm = null;
        var ln2: ?LayerNorm = null;
        var rms1: ?RMSNorm = null;
        var rms2: ?RMSNorm = null;

        if (cfg.use_rms_norm) {
            rms1 = try RMSNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
            errdefer rms1.?.deinit();
            rms2 = try RMSNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
            errdefer rms2.?.deinit();
        } else {
            ln1 = try LayerNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
            errdefer ln1.?.deinit();
            ln2 = try LayerNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
            errdefer ln2.?.deinit();
        }

        // --- Attention ---
        var attn = try CausalSelfAttention.init(allocator, cfg.d_model, cfg.n_head, cfg.max_seq_len, cfg.bias, rng);
        errdefer attn.deinit();

        // --- Feed-forward ---
        var mlp: ?MLP = null;
        var swiglu: ?SwiGLU = null;

        if (cfg.use_swiglu) {
            swiglu = try SwiGLU.init(allocator, cfg.d_model, cfg.d_ff, cfg.bias, rng);
            errdefer swiglu.?.deinit();
        } else {
            mlp = try MLP.init(allocator, cfg.d_model, cfg.d_ff, cfg.bias, rng);
            errdefer mlp.?.deinit();
        }

        return TransformerBlock{
            .ln1 = ln1,
            .ln2 = ln2,
            .rms1 = rms1,
            .rms2 = rms2,
            .attn = attn,
            .mlp = mlp,
            .swiglu = swiglu,
            .allocator = allocator,
            .use_rms_norm = cfg.use_rms_norm,
            .use_swiglu = cfg.use_swiglu,
        };
    }

    /// Compute one transformer block: pre-norm + residual.
    ///
    /// h = x + Attention(Norm_1(x))
    /// out = h + FFN(Norm_2(h))
    ///
    /// Norm is LayerNorm or RMSNorm based on config.
    /// FFN is MLP or SwiGLU based on config.
    pub fn forward(self: TransformerBlock, input: Tensor, tape: ?*Tape) LabError!Tensor {
        // --- Attention sub-block ---
        // Norm_1(x)
        var norm1_out = if (self.use_rms_norm)
            try self.rms1.?.forward(input, tape)
        else
            try self.ln1.?.forward(input, tape);
        defer norm1_out.deinit(self.allocator);

        // Attention(Norm_1(x))
        var attn_out = try self.attn.forward(norm1_out, tape);
        defer attn_out.deinit(self.allocator);

        // x + Attention(Norm_1(x))  — residual connection
        var h = try ops_elementwise.add(self.allocator, input, attn_out, tape);
        defer h.deinit(self.allocator);

        // --- FFN sub-block ---
        // Norm_2(h)
        var norm2_out = if (self.use_rms_norm)
            try self.rms2.?.forward(h, tape)
        else
            try self.ln2.?.forward(h, tape);
        defer norm2_out.deinit(self.allocator);

        // FFN(Norm_2(h))
        var ffn_out = if (self.use_swiglu)
            try self.swiglu.?.forward(norm2_out, tape)
        else
            try self.mlp.?.forward(norm2_out, tape);
        defer ffn_out.deinit(self.allocator);

        // h + FFN(Norm_2(h))  — residual connection
        const output = try ops_elementwise.add(self.allocator, h, ffn_out, tape);
        return output;
    }

    /// Append this block's learnable parameters to the list.
    pub fn parameters(self: *TransformerBlock, list: *std.ArrayList(*Tensor)) void {
        if (self.use_rms_norm) {
            (@constCast(&self.rms1.?)).parameters(list);
        } else {
            (@constCast(&self.ln1.?)).parameters(list);
        }
        self.attn.parameters(list);
        if (self.use_rms_norm) {
            (@constCast(&self.rms2.?)).parameters(list);
        } else {
            (@constCast(&self.ln2.?)).parameters(list);
        }
        if (self.use_swiglu) {
            (@constCast(&self.swiglu.?)).parameters(list);
        } else {
            (@constCast(&self.mlp.?)).parameters(list);
        }
    }

    /// Free all sub-layers.
    pub fn deinit(self: *TransformerBlock) void {
        if (self.ln1) |*ln| ln.deinit();
        if (self.ln2) |*ln| ln.deinit();
        if (self.rms1) |*rn| rn.deinit();
        if (self.rms2) |*rn| rn.deinit();
        self.attn.deinit();
        if (self.mlp) |*m| m.deinit();
        if (self.swiglu) |*sg| sg.deinit();
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
    for (0..out.cpuData().len) |i| {
        try std.testing.expect(std.math.isFinite(out.cpuData()[i]));
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
