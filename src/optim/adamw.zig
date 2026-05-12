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
const ParamId = @import("../nn/module.zig").ParamId;
// PR-mu: AdamW.step routes CUDA params to a single-kernel update
// (backend.cuda.dispatch.adamwStep). Moment state (m, v) is
// allocated on the same device as the param on first encounter.
const ops_create = @import("../tensor/ops/create.zig");
const cuda_dispatch = @import("../backend/cuda/dispatch.zig");

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
    ///
    /// PR-ζ: keyed by the parameter's stable `ParamId` rather than by
    /// `@intFromPtr(param.cpuData().ptr)`. The pointer key was a latent bug:
    /// checkpoint loads that allocate a fresh buffer (or future device
    /// transfers that move the tensor to CUDA) would silently change
    /// the key, and the optimizer would start again with zero moments
    /// for that parameter. A `ParamId` is assigned once at layer init
    /// and persists across buffer replacements.
    state: std.AutoHashMap(ParamId, ParamState),

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
            .state = std.AutoHashMap(ParamId, ParamState).init(allocator),
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
            // PR-ζ: every parameter must carry a stable ParamId. Nil
            // would mean the caller passed a non-parameter tensor (or
            // forgot to call `nn.module.assignParamId`); fail loudly
            // rather than silently key on an unstable pointer.
            const key = param.param_id orelse return error.InvalidArgument;

            // Get or create state. Moment tensors live on the same
            // device as the parameter (PR-mu) — zerosLike routes to
            // cuMemsetD32_v2 for CUDA params.
            if (!self.state.contains(key)) {
                const m = try ops_create.zerosLike(self.allocator, param.*);
                const v = try ops_create.zerosLike(self.allocator, param.*);
                try self.state.put(key, .{ .m = m, .v = v });
            }
            const s = self.state.getPtr(key).?;

            // PR-mu: CUDA path uses a single kernel launch per param.
            if (param.device == .cuda) {
                try cuda_dispatch.adamwStep(param.*, grad.*, s.m, s.v, .{
                    .lr = lr,
                    .beta1 = beta1,
                    .beta2 = beta2,
                    .eps = eps,
                    .weight_decay = wd,
                    .bc1 = bc1,
                    .bc2 = bc2,
                });
                continue;
            }

            // CPU path.
            for (param.cpuData(), grad.cpuData(), s.m.cpuData(), s.v.cpuData()) |*p, g, *m, *vel| {
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
                if (g.device == .cuda) {
                    // PR-mu: zero CUDA gradient buffer via
                    // cuMemsetD32_v2 (bit pattern 0 = f32 0.0).
                    // Errors from memset are unreachable in practice
                    // (no new alloc); we log and continue.
                    cuda_dispatch.fillZeros(g.*) catch |err| {
                        std.log.warn("AdamW.zeroGrad: fillZeros failed: {s}", .{@errorName(err)});
                    };
                } else {
                    g.fill(0.0);
                }
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
    param.cpuData()[0] = 5.0;
    param.cpuData()[1] = -3.0;
    param.cpuData()[2] = 0.0;
    // PR-ζ: assign a ParamId so the optimizer can key state.
    @import("../nn/module.zig").assignParamId(&param);

    var grad = try Tensor.init(alloc, @import("../tensor/shape.zig").Shape.init1D(3));
    defer grad.deinit(alloc);
    grad.cpuData()[0] = 1.0;
    grad.cpuData()[1] = -1.0;
    grad.cpuData()[2] = 0.0;

    param.grad = &grad;

    var opt = adam.optimizer();
    const params = [_]*Tensor{&param};
    try opt.step(&params);

    // Update should move param in the direction of -grad
    try std.testing.expect(param.cpuData()[0] < 5.0);
    try std.testing.expect(param.cpuData()[1] > -3.0);
}

test "AdamW step — rejects parameter without ParamId" {
    const alloc = std.testing.allocator;

    var adam = try AdamW.init(alloc, .{ .lr = 0.01 });
    defer adam.deinit(alloc);

    var param = try Tensor.init(alloc, @import("../tensor/shape.zig").Shape.init1D(2));
    defer param.deinit(alloc);
    var grad = try Tensor.init(alloc, @import("../tensor/shape.zig").Shape.init1D(2));
    defer grad.deinit(alloc);
    param.grad = &grad;
    // Intentionally do NOT call assignParamId.

    var opt = adam.optimizer();
    const params = [_]*Tensor{&param};
    try std.testing.expectError(error.InvalidArgument, opt.step(&params));
}

test "AdamW state persists across buffer replacement (same ParamId)" {
    // PR-ζ guarantee: if a parameter keeps its ParamId but its
    // backing buffer is replaced (e.g. by a checkpoint load that
    // allocates a fresh Tensor), the optimizer must NOT lose state.
    // A pointer-keyed optimizer would silently start fresh; an ID-
    // keyed one continues with the existing m/v moments.
    const alloc = std.testing.allocator;

    var adam = try AdamW.init(alloc, .{ .lr = 0.01 });
    defer adam.deinit(alloc);
    var opt = adam.optimizer();

    const Shape = @import("../tensor/shape.zig").Shape;
    const assignParamId = @import("../nn/module.zig").assignParamId;

    // First parameter lifetime.
    var p1 = try Tensor.init(alloc, Shape.init1D(2));
    assignParamId(&p1);
    const id = p1.param_id.?;

    var g1 = try Tensor.init(alloc, Shape.init1D(2));
    g1.cpuData()[0] = 1.0;
    g1.cpuData()[1] = -0.5;
    p1.grad = &g1;
    try opt.step(&.{&p1});
    try std.testing.expect(adam.state.contains(id));

    p1.deinit(alloc);
    g1.deinit(alloc);

    // Replacement parameter with the same ID (simulating checkpoint
    // reload). The buffer is different; if we were keying by
    // `@intFromPtr(p2.cpuData().ptr)` the new key wouldn't match and the
    // optimizer would allocate new m/v buffers. With ParamId keying,
    // the existing state is reused.
    var p2 = try Tensor.init(alloc, Shape.init1D(2));
    defer p2.deinit(alloc);
    p2.param_id = id;
    var g2 = try Tensor.init(alloc, Shape.init1D(2));
    defer g2.deinit(alloc);
    g2.cpuData()[0] = 0.3;
    g2.cpuData()[1] = 0.1;
    p2.grad = &g2;

    const state_before = adam.state.count();
    try opt.step(&.{&p2});
    const state_after = adam.state.count();
    try std.testing.expectEqual(state_before, state_after);
}
