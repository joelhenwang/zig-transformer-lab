//!
//! zig-transformer-lab — Root Mean Square Layer Normalization
//!
//! Purpose:
//!   Implements RMSNorm: normalizes by the root-mean-square of the input
//!   along the last dimension, then scales by a learnable gamma.
//!   Simpler and faster than LayerNorm (no mean subtraction, no beta).
//!
//!   Used by Llama, Mistral, Gemma, and most modern LLMs since 2023.
//!
//! Shape contract:
//!   forward(x: (B, T, D), tape) -> y: (B, T, D)
//!   gamma: (D,) — learnable scale parameter
//!
//! Math:
//!   RMSNorm(x) = x / sqrt(mean(x^2, axis=-1) + eps) * gamma
//!
//!   Compared to LayerNorm:
//!     LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
//!     RMSNorm:    x            / sqrt(mean(x^2) + eps) * gamma
//!
//!   ~30% fewer FLOPs (no mean, no subtraction, no beta).
//!
//! Memory ownership:
//!   Owns gamma tensor. Freed in deinit().
//!
//! Credits:
//!   Zhang & Sennrich (2019), "Root Mean Square Layer Normalization."
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const ops_reduce = @import("../tensor/ops/reduce.zig");
const ops_elementwise = @import("../tensor/ops/elementwise.zig");
const ops_unary = @import("../tensor/ops/unary.zig");
const ops_create = @import("../tensor/ops/create.zig");
const module = @import("module.zig");

pub const RMSNorm = struct {
    gamma: Tensor,
    allocator: std.mem.Allocator,
    d_model: usize,
    eps: f32,

    /// Create an RMSNorm layer.
    ///
    /// Gamma initialized to ones (scale=1) so the initial transform
    /// is identity — the network learns to deviate as needed.
    ///
    /// Worked example:
    ///   var rn = try RMSNorm.init(alloc, 32, 1e-6, &rng);
    ///   defer rn.deinit();
    ///   // rn.gamma.shape == (32,), all ones
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        eps: f32,
        _: *Rng,
    ) LabError!RMSNorm {
        const gamma = try ops_create.ones(allocator, Shape.init1D(d_model));
        module.assignParamId(&@constCast(&gamma).*);

        return RMSNorm{
            .gamma = gamma,
            .allocator = allocator,
            .d_model = d_model,
            .eps = eps,
        };
    }

    /// Compute RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    ///
    /// All ops are tape-tracked so gradients flow through.
    ///
    /// Worked example:
    ///   // x shape (2, 4, 32)
    ///   var out = try rn.forward(x, &tape);
    ///   // out.shape == (2, 4, 32)
    pub fn forward(self: RMSNorm, input: Tensor, tape: ?*Tape) LabError!Tensor {
        const alloc = self.allocator;

        // Step 1: x^2
        var x_sq = try ops_elementwise.mul(alloc, input, input, tape);
        defer x_sq.deinit(alloc);

        // Step 2: mean(x^2, last_axis)
        const ndim = input.shape.ndim();
        const last_axis: u2 = @intCast(ndim - 1);
        var mean_sq = try ops_reduce.mean(alloc, x_sq, last_axis, tape);
        defer mean_sq.deinit(alloc);

        // Step 3: mean(x^2) + eps
        var mean_eps = try ops_elementwise.addScalar(alloc, mean_sq, self.eps, tape);
        defer mean_eps.deinit(alloc);

        // Step 4: sqrt(mean(x^2) + eps)
        var rms = try ops_unary.sqrt(alloc, mean_eps, tape);
        defer rms.deinit(alloc);

        // Step 5: x / rms (broadcast division: rms is reduced along last axis)
        var x_norm = try ops_elementwise.div(alloc, input, rms, tape);
        defer x_norm.deinit(alloc);

        // Step 6: x_norm * gamma (broadcast: gamma is (D,))
        const out = try ops_elementwise.mul(alloc, x_norm, self.gamma, tape);
        return out;
    }

    /// Append learnable parameters (gamma only) to the list.
    pub fn parameters(self: *RMSNorm, list: *std.ArrayList(*Tensor)) void {
        list.append(self.allocator, &self.gamma) catch {};
    }

    /// Free owned tensors.
    pub fn deinit(self: *RMSNorm) void {
        self.gamma.deinit(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "RMSNorm forward produces correct shape" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);
    var rn = try RMSNorm.init(alloc, 4, 1e-6, &rng);
    defer rn.deinit();

    var input = try Tensor.init(alloc, Shape.init2D(2, 4));
    defer input.deinit(alloc);
    input.cpuData()[0] = 1.0;
    input.cpuData()[1] = 2.0;
    input.cpuData()[2] = 3.0;
    input.cpuData()[3] = 4.0;
    input.cpuData()[4] = -1.0;
    input.cpuData()[5] = -2.0;
    input.cpuData()[6] = -3.0;
    input.cpuData()[7] = -4.0;

    var out = try rn.forward(input, null);
    defer out.deinit(alloc);

    // Output should have same shape
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), out.shape.dims[1]);

    // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // x_norm[0] = 1/2.7386 ≈ 0.3651 (times gamma=1.0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.3651), out.cpuData()[0], 0.01);
}
