//!
//! zig-transformer-lab — AdamW optimizer with bias correction and
//!                        decoupled weight decay
//!
//! Purpose:
//!   Implements AdamW, the optimizer of choice for transformer training.
//!   Key difference from "vanilla Adam": weight decay is applied
//!   directly to parameters (decoupled), not added to the gradient.
//!   This produces better regularization and is the standard for
//!   GPT-2 and later models.
//!
//! Shape contract:
//!   step(params) — updates each param.data in-place
//!
//! Math:
//!   m = β₁ * m + (1 - β₁) * grad          (1st moment estimate)
//!   v = β₂ * v + (1 - β₂) * grad²         (2nd moment estimate)
//!   m̂ = m / (1 - β₁ᵗ)                     (bias-corrected 1st moment)
//!   v̂ = v / (1 - β₂ᵗ)                     (bias-corrected 2nd moment)
//!   param -= lr * (m̂ / (√v̂ + ε) + wd * param)
//!
//!   The bias correction (1 - βᵗ) compensates for the fact that m
//!   and v are initialized at zero, which causes them to be biased
//!   towards zero in early steps. t is the step counter.
//!
//!   Weight decay is DECOUPLED: it's subtracted directly from the
//!   parameter, not added to the gradient. This is the "W" in AdamW.
//!
//! Memory ownership:
//!   Owns m and v buffers (HashMaps). Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from moment buffer allocation
//!
//! Credits:
//!   AdamW from "Decoupled Weight Decay Regularization" (Loshchilov
//!   & Hutter, 2017). No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Optimizer = @import("optimizer.zig").Optimizer;

pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.1,
};

const ParamState = struct {
    m: Tensor,
    v: Tensor,
};

pub const AdamW = struct {
    allocator: std.mem.Allocator,
    config: AdamWConfig,

    /// Per-parameter state: first and second moment estimates.
    state: std.AutoHashMap(usize, ParamState),

    /// Step counter for bias correction.
    t: usize,

    /// Create an AdamW optimizer.
    ///
    /// Worked example:
    ///   var adam = try AdamW.init(alloc, .{ .lr = 3e-4 });
    ///   defer adam.deinit(alloc);
    ///   var opt = adam.optimizer();
    pub fn init(allocator: std.mem.Allocator, config: AdamWConfig) LabError!AdamW {
        return AdamW{
            .allocator = allocator,
            .config = config,
            .state = std.AutoHashMap(usize, ParamState).init(allocator),
            .t = 0,
        };
    }

    /// Perform one AdamW update step.
    pub fn step(ctx: *anyopaque, params: []const *Tensor) anyerror!void {
        const self: *AdamW = @ptrCast(@alignCast(ctx));
        self.t += 1;

        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const eps = self.config.eps;
        const lr = self.config.lr;
        const wd = self.config.weight_decay;

        // Bias correction factors: 1 / (1 - βᵗ)
        // Compute β₁ᵗ and β₂ᵗ by repeated multiplication (t is small)
        var beta1_t: f64 = 1.0;
        var beta2_t: f64 = 1.0;
        for (0..self.t) |_| {
            beta1_t *= beta1;
            beta2_t *= beta2;
        }
        const bc1: f32 = @floatCast(1.0 / (1.0 - beta1_t));
        const bc2: f32 = @floatCast(1.0 / (1.0 - beta2_t));

        for (params) |param| {
            const grad = param.grad orelse continue;
            const key = @intFromPtr(param.data.ptr);

            // Get or create state
            if (!self.state.contains(key)) {
                var m = try Tensor.init(self.allocator, param.shape);
                m.fill(0.0);
                var v = try Tensor.init(self.allocator, param.shape);
                v.fill(0.0);
                try self.state.put(key, .{ .m = m, .v = v });
            }
            const s = self.state.getPtr(key).?;

            // Update moments: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g²
            for (param.data, grad.data, s.m.data, s.v.data) |*p, g, *m, *vel| {
                m.* = beta1 * m.* + (1.0 - beta1) * g;
                vel.* = beta2 * vel.* + (1.0 - beta2) * g * g;

                // Bias-corrected moments
                const m_hat = m.* * bc1;
                const v_hat = vel.* * bc2;

                // AdamW update: param -= lr * (m̂ / (√v̂ + ε) + wd * param)
                p.* -= lr * (m_hat / (@sqrt(v_hat) + eps) + wd * p.*);
            }
        }
    }

    /// Zero all parameter gradients.
    pub fn zeroGrad(_: *anyopaque, params: []const *Tensor) void {
        for (params) |param| {
            if (param.grad) |g| {
                g.fill(0.0);
            }
        }
    }

    /// Free moment buffers and the AdamW struct (called by Optimizer.deinit).
    pub fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *AdamW = @ptrCast(@alignCast(ctx));
        var iter = self.state.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.m.deinit(allocator);
            entry.value_ptr.*.v.deinit(allocator);
        }
        self.state.deinit();
        allocator.destroy(self);
    }

    /// Free moment buffers (standalone, for direct use outside Optimizer wrapper).
    pub fn deinit(self: *AdamW, allocator: std.mem.Allocator) void {
        var iter = self.state.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.m.deinit(allocator);
            entry.value_ptr.*.v.deinit(allocator);
        }
        self.state.deinit();
    }

    /// Return an Optimizer wrapper for this AdamW instance.
    pub fn optimizer(self: *AdamW) Optimizer {
        return Optimizer{
            .ctx = @ptrCast(self),
            .step_fn = AdamW.step,
            .zero_grad_fn = AdamW.zeroGrad,
            .deinit_fn = AdamW.deinitImpl,
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "AdamW step — decreases parameter magnitude" {
    const alloc = std.testing.allocator;

    var adam = try AdamW.init(alloc, .{ .lr = 0.01, .weight_decay = 0.0 });
    defer adam.deinit(alloc);

    var param = try Tensor.init(alloc, @import("../tensor/shape.zig").Shape.init1D(3));
    defer param.deinit(alloc);
    param.data[0] = 5.0;
    param.data[1] = -3.0;
    param.data[2] = 0.0;

    var grad = try Tensor.init(alloc, @import("../tensor/shape.zig").Shape.init1D(3));
    defer grad.deinit(alloc);
    grad.data[0] = 1.0;
    grad.data[1] = -1.0;
    grad.data[2] = 0.0;

    param.grad = &grad;

    var opt = adam.optimizer();
    const params = [_]*Tensor{&param};
    try opt.step(&params);

    // Update should move param in the direction of -grad
    try std.testing.expect(param.data[0] < 5.0);
    try std.testing.expect(param.data[1] > -3.0);
}
