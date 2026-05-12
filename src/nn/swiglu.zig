//!
//! zig-transformer-lab — SwiGLU MLP (Gated Linear Unit with SiLU)
//!
//! Purpose:
//!   Implements the SwiGLU feed-forward network used in Llama, Mistral,
//!   and most modern LLMs. Replaces the standard GELU MLP.
//!
//!   SwiGLU uses a gate mechanism: one projection (W_gate) produces a
//!   gate signal that modulates another projection (W_up). This gives
//!   the network finer control over which information flows through.
//!
//! Shape contract:
//!   forward(x: (B, T, D), tape) -> output: (B, T, D)
//!
//! Math:
//!   SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
//!   SiLU(x) = x * sigmoid(x)
//!
//!   Three weight matrices:
//!     W_gate: (D, d_ff) — gate projection
//!     W_up:   (D, d_ff) — up projection
//!     W_down: (d_ff, D) — down projection
//!
//!   Standard MLP uses d_ff = 4*D. SwiGLU uses d_ff = (8/3)*D to
//!   match parameter count (3 matrices vs 2).
//!
//! Memory ownership:
//!   Owns w_gate, w_up, w_down Linear sub-layers. Freed in deinit().
//!
//! Credits:
//!   Shazeer (2020), "GLU Variants Improve Transformer."
//!   Dauphin et al. (2017), "Language Modeling with Gated Convolutional Networks."
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const Linear = @import("linear.zig").Linear;
const ops_elementwise = @import("../tensor/ops/elementwise.zig");

pub const SwiGLU = struct {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
    allocator: std.mem.Allocator,

    /// Create a SwiGLU MLP layer.
    ///
    /// d_ff should be (8*d_model)/3 rounded up to nearest multiple of 8
    /// to match standard MLP parameter count. The caller passes d_ff
    /// directly (computed in TransformerBlock or model config).
    ///
    /// Worked example:
    ///   var sg = try SwiGLU.init(alloc, 64, 170, true, &rng);
    ///   defer sg.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        use_bias: bool,
        rng: *Rng,
    ) LabError!SwiGLU {
        var w_gate = try Linear.init(allocator, d_model, d_ff, use_bias, rng);
        errdefer w_gate.deinit();
        var w_up = try Linear.init(allocator, d_model, d_ff, use_bias, rng);
        errdefer w_up.deinit();
        var w_down = try Linear.init(allocator, d_ff, d_model, use_bias, rng);
        errdefer w_down.deinit();

        return SwiGLU{
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .allocator = allocator,
        };
    }

    /// Compute SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    ///
    /// SiLU(x) = x * sigmoid(x). We implement this as:
    ///   gate_linear = x @ W_gate           (linear projection)
    ///   gate_sigmoid = sigmoid(gate_linear) (computed via: 1/(1+exp(-x)))
    ///   gate = gate_linear * gate_sigmoid   (SiLU = x * sigmoid(x))
    ///   up = x @ W_up                      (linear projection)
    ///   hidden = gate * up                  (element-wise gating)
    ///   output = hidden @ W_down            (down projection)
    ///
    /// Worked example:
    ///   // x shape (2, 4, 64)
    ///   var out = try sg.forward(x, &tape);
    ///   // out.shape == (2, 4, 64)
    pub fn forward(self: SwiGLU, input: Tensor, tape: ?*Tape) LabError!Tensor {
        const alloc = self.allocator;

        // gate_linear = input @ W_gate
        var gate_linear = try self.w_gate.forward(input, tape);
        defer gate_linear.deinit(alloc);

        // Compute SiLU(gate_linear) = gate_linear * sigmoid(gate_linear)
        // sigmoid(x) = 1 / (1 + exp(-x))
        // We compose: neg -> exp -> addScalar(1) -> div(gate_linear, _)
        // This gives: gate_linear / (1 + exp(-gate_linear)) = gate_linear * sigmoid(gate_linear)
        var neg_gate = try ops_elementwise.neg(alloc, gate_linear, tape);
        defer neg_gate.deinit(alloc);
        var exp_neg = try @import("../tensor/ops/unary.zig").exp(alloc, neg_gate, tape);
        defer exp_neg.deinit(alloc);
        var one_plus_exp = try ops_elementwise.addScalar(alloc, exp_neg, 1.0, tape);
        defer one_plus_exp.deinit(alloc);
        var silu_gate = try ops_elementwise.div(alloc, gate_linear, one_plus_exp, tape);
        defer silu_gate.deinit(alloc);

        // up = input @ W_up
        var up = try self.w_up.forward(input, tape);
        defer up.deinit(alloc);

        // hidden = silu_gate * up (element-wise gating)
        var hidden = try ops_elementwise.mul(alloc, silu_gate, up, tape);
        defer hidden.deinit(alloc);

        // output = hidden @ W_down
        const output = try self.w_down.forward(hidden, tape);
        return output;
    }

    /// Append learnable parameters to the list.
    pub fn parameters(self: *SwiGLU, list: *std.ArrayList(*Tensor)) void {
        self.w_gate.parameters(list);
        self.w_up.parameters(list);
        self.w_down.parameters(list);
    }

    /// Free owned sub-layers.
    pub fn deinit(self: *SwiGLU) void {
        self.w_gate.deinit();
        self.w_up.deinit();
        self.w_down.deinit();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "SwiGLU forward produces correct shape" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);
    var sg = try SwiGLU.init(alloc, 8, 16, true, &rng);
    defer sg.deinit();

    var input = try Tensor.init(alloc, Shape.init2D(3, 8));
    defer input.deinit(alloc);
    // Fill with small values
    for (input.cpuData(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;

    var out = try sg.forward(input, null);
    defer out.deinit(alloc);

    // Output should have same shape as input
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 8), out.shape.dims[1]);
}

test "SwiGLU parameter count matches expectation" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);
    // d_model=8, d_ff=16, bias=true
    // w_gate: 8*16 + 16 = 144
    // w_up: 8*16 + 16 = 144
    // w_down: 16*8 + 8 = 136
    // total = 424
    var sg = try SwiGLU.init(alloc, 8, 16, true, &rng);
    defer sg.deinit();

    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(alloc);
    try params.ensureTotalCapacity(alloc, 8);
    sg.parameters(&params);
    // 3 weights + 3 biases = 6 parameters
    try std.testing.expectEqual(@as(usize, 6), params.items.len);
}
