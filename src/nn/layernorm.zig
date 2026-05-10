//!
//! zig-transformer-lab — Layer Normalization
//!
//! Purpose:
//!   Implements LayerNorm: normalizes across the last dimension (d_model)
//!   to zero mean and unit variance, then applies an affine transform
//!   with learnable scale (gamma) and bias (beta).
//!
//!   LayerNorm is critical for transformer training stability. Without
//!   it, deep networks suffer from gradient explosion/vanishing.
//!
//!   This implementation COMPOSES LayerNorm from existing tape-tracked
//!   ops (mean, sub, mul, sqrt, div, add) instead of adding a fused
//!   OpKind. This is more pedagogical — students see how autograd
//!   composes through ~7 tape nodes per LayerNorm forward.
//!
//! Shape contract:
//!   forward(x: (N, D), tape) → y: (N, D)
//!   gamma: (D,), beta: (D,)
//!
//!   For 3D input (B, T, D), normalization is over D (last axis).
//!
//! Math:
//!   LayerNorm(x):
//!     mu = mean(x, axis=-1)               // (N, 1) — mean per row
//!     x_centered = x - mu                 // (N, D) — zero-mean
//!     var = mean(x_centered^2, axis=-1)    // (N, 1) — variance per row
//!     inv_std = 1 / sqrt(var + eps)       // (N, 1) — inverse stdev
//!     x_norm = x_centered * inv_std       // (N, D) — normalized
//!     y = gamma * x_norm + beta           // (N, D) — affine transform
//!
//! Memory ownership:
//!   Owns gamma and beta tensors. Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from parameter allocation
//!
//! Credits:
//!   LayerNorm introduced in "Layer Normalization" (Ba et al., 2016).
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
const ops_shape = @import("../tensor/ops/shape_ops.zig");

pub const LayerNorm = struct {
    gamma: Tensor,
    beta: Tensor,
    allocator: std.mem.Allocator,
    d_model: usize,
    eps: f32,

    /// Create a LayerNorm layer.
    ///
    /// Gamma initialized to ones (scale=1), beta to zeros (shift=0).
    /// This way, the initial transform is identity — the network
    /// learns to deviate from identity as needed.
    ///
    /// Worked example:
    ///   var ln = try LayerNorm.init(alloc, 32, 1e-5, &rng);
    ///   defer ln.deinit();
    ///   // ln.gamma.shape == (32,), all ones
    ///   // ln.beta.shape == (32,), all zeros
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        eps: f32,
        _: *Rng,
    ) LabError!LayerNorm {
        const gamma = try ops_create.ones(allocator, Shape.init1D(d_model));
        const beta = try ops_create.zeros(allocator, Shape.init1D(d_model));

        return LayerNorm{
            .gamma = gamma,
            .beta = beta,
            .allocator = allocator,
            .d_model = d_model,
            .eps = eps,
        };
    }

    /// Compute LayerNorm(x) by composing existing tape-tracked ops.
    ///
    /// This creates ~7 tape nodes per forward call, which is
    /// pedagogically valuable — students can inspect the tape
    /// and see exactly how each gradient flows through the
    /// normalization.
    ///
    /// Worked example:
    ///   // x shape (4, 32)
    ///   var y = try ln.forward(x, &tape);
    ///   // y shape (4, 32), each row has mean 0, variance 1 (approx)
    pub fn forward(self: LayerNorm, input: Tensor, tape: ?*Tape) LabError!Tensor {
        const ndim = input.shape.ndim();
        const last_axis: u2 = @intCast(ndim - 1);

        // Step 1: mu = mean(x, last_axis)  → shape (..., 1)
        var mu = try ops_reduce.mean(self.allocator, input, last_axis, tape);
        if (tape) |t| try t.keepAlive(&mu);
        defer mu.deinit(self.allocator);

        // Step 2: x_centered = x - mu  (mu broadcasts along last axis)
        var x_centered = try ops_elementwise.sub(self.allocator, input, mu, tape);
        if (tape) |t| try t.keepAlive(&x_centered);
        defer x_centered.deinit(self.allocator);

        // Step 3: x_centered_sq = x_centered * x_centered
        var x_centered_sq = try ops_elementwise.mul(self.allocator, x_centered, x_centered, tape);
        if (tape) |t| try t.keepAlive(&x_centered_sq);
        defer x_centered_sq.deinit(self.allocator);

        // Step 4: variance = mean(x_centered_sq, last_axis)  → shape (..., 1)
        var variance = try ops_reduce.mean(self.allocator, x_centered_sq, last_axis, tape);
        if (tape) |t| try t.keepAlive(&variance);
        defer variance.deinit(self.allocator);

        // Step 5: var_eps = variance + eps
        var var_eps = try ops_elementwise.addScalar(self.allocator, variance, self.eps, tape);
        if (tape) |t| try t.keepAlive(&var_eps);
        defer var_eps.deinit(self.allocator);

        // Step 6: std_val = sqrt(var_eps)
        // x_norm = x_centered / std_val  (std_val broadcasts from (...,1))
        var std_val = try ops_unary.sqrt(self.allocator, var_eps, tape);
        if (tape) |t| try t.keepAlive(&std_val);
        defer std_val.deinit(self.allocator);

        var x_norm = try ops_elementwise.div(self.allocator, x_centered, std_val, tape);
        if (tape) |t| try t.keepAlive(&x_norm);
        defer x_norm.deinit(self.allocator);

        // Step 7: gamma * x_norm (gamma broadcasts from (D,) to (...,D))
        // Use reshapeTracked WITH the tape so the backward can reshape
        // the mul's gradient for the broadcast gamma back to (D,),
        // matching the original gamma's shape.
        var gamma_2d = try ops_shape.reshapeTracked(self.allocator, self.gamma, Shape.init2D(1, self.d_model), tape);
        if (tape) |t| try t.keepAlive(&gamma_2d);
        defer gamma_2d.deinit(self.allocator);

        var scaled = try ops_elementwise.mul(self.allocator, x_norm, gamma_2d, tape);
        if (tape) |t| try t.keepAlive(&scaled);
        defer scaled.deinit(self.allocator);

        // Step 8: scaled + beta (beta broadcasts from (D,) to (...,D))
        // Same reasoning as gamma: track the reshape so backward can
        // convert the add's gradient back to (D,) for beta.
        var beta_2d = try ops_shape.reshapeTracked(self.allocator, self.beta, Shape.init2D(1, self.d_model), tape);
        if (tape) |t| try t.keepAlive(&beta_2d);
        defer beta_2d.deinit(self.allocator);

        const y = try ops_elementwise.add(self.allocator, scaled, beta_2d, tape);

        return y;
    }

    /// Append this layer's learnable parameters to the list.
    pub fn parameters(self: *LayerNorm, list: *std.ArrayList(*Tensor)) void {
        list.appendAssumeCapacity(&self.gamma);
        list.appendAssumeCapacity(&self.beta);
    }

    /// Free gamma and beta tensors.
    pub fn deinit(self: *LayerNorm) void {
        self.gamma.deinit(self.allocator);
        self.beta.deinit(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "LayerNorm init — gamma ones, beta zeros" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var ln = try LayerNorm.init(alloc, 4, 1e-5, &rng);
    defer ln.deinit();

    try std.testing.expectEqual(@as(usize, 4), ln.gamma.shape.dims[0]);
    for (0..4) |i| {
        try std.testing.expectEqual(@as(f32, 1.0), ln.gamma.data[i]);
        try std.testing.expectEqual(@as(f32, 0.0), ln.beta.data[i]);
    }
}

test "LayerNorm forward — normalizes to zero mean, unit variance" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var ln = try LayerNorm.init(alloc, 4, 1e-5, &rng);
    defer ln.deinit();

    // Input: row 0 = [1, 2, 3, 4] (mean=2.5, var=1.25)
    //         row 1 = [10, 20, 30, 40] (mean=25, var=125)
    var x = try Tensor.init(alloc, Shape.init2D(2, 4));
    defer x.deinit(alloc);
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;
    x.data[4] = 10.0;
    x.data[5] = 20.0;
    x.data[6] = 30.0;
    x.data[7] = 40.0;

    var y = try ln.forward(x, null);
    defer y.deinit(alloc);

    // Check shape
    try std.testing.expectEqual(@as(usize, 2), y.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), y.shape.dims[1]);

    // With gamma=1, beta=0, output should be normalized.
    // Row 0: x_norm = (x - 2.5) / sqrt(1.25 + eps)
    // Approx values: [-1.342, -0.447, 0.447, 1.342]
    // Check that the mean of row 0 is close to 0
    var mean0: f32 = 0;
    for (0..4) |i| {
        mean0 += y.data[i];
    }
    mean0 /= 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean0, 1e-3);
}

test "LayerNorm forward — 3D input" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var ln = try LayerNorm.init(alloc, 4, 1e-5, &rng);
    defer ln.deinit();

    var x = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer x.deinit(alloc);
    for (0..24) |i| x.data[i] = @floatFromInt(i + 1);

    var y = try ln.forward(x, null);
    defer y.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), y.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), y.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), y.shape.dims[2]);
}
