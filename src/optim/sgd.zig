//!
//! zig-transformer-lab — SGD optimizer with momentum and weight decay
//!
//! Purpose:
//!   Implements Stochastic Gradient Descent with optional momentum
//!   and L2 weight decay. The simplest optimizer — good baseline
//!   and pedagogically clear.
//!
//! Shape contract:
//!   step(params) — updates each param.data in-place
//!
//! Math:
//!   Without momentum:
//!     param -= lr * (grad + weight_decay * param)
//!
//!   With momentum (classical):
//!     v = momentum * v + grad + weight_decay * param
//!     param -= lr * v
//!
//!   Weight decay is "coupled" (added to gradient before momentum),
//!   which is the standard SGD convention (different from AdamW's
//!   decoupled weight decay).
//!
//! Memory ownership:
//!   Owns the velocity buffer HashMap. Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from velocity buffer allocation
//!
//! Credits:
//!   Standard SGD with momentum (Polyak, 1964). No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Optimizer = @import("optimizer.zig").Optimizer;

pub const SGDConfig = struct {
    lr: f32 = 0.01,
    momentum: f32 = 0.0,
    weight_decay: f32 = 0.0,
};

pub const SGD = struct {
    allocator: std.mem.Allocator,
    config: SGDConfig,

    /// Velocity buffers: keyed by param data pointer (as usize).
    /// v[i] = momentum * v[i] + grad[i] + weight_decay * param[i]
    velocity: std.AutoHashMap(usize, Tensor),

    /// Create an SGD optimizer.
    ///
    /// Worked example:
    ///   var sgd = try SGD.init(alloc, .{ .lr = 0.01, .momentum = 0.9 });
    ///   defer sgd.deinit(alloc);
    ///   var opt = sgd.optimizer();
    pub fn init(allocator: std.mem.Allocator, config: SGDConfig) LabError!SGD {
        return SGD{
            .allocator = allocator,
            .config = config,
            .velocity = std.AutoHashMap(usize, Tensor).init(allocator),
        };
    }

    /// Perform one SGD update step.
    ///
    /// For each parameter:
    ///   v = momentum * v + grad + weight_decay * param
    ///   param -= lr * v
    pub fn step(ctx: *anyopaque, params: []const *Tensor) anyerror!void {
        const self: *SGD = @ptrCast(@alignCast(ctx));
        for (params) |param| {
            const grad = param.grad orelse continue;
            const key = @intFromPtr(param.data.ptr);

            // Get or create velocity buffer
            if (!self.velocity.contains(key)) {
                var v = try Tensor.init(self.allocator, param.shape);
                v.fill(0.0);
                try self.velocity.put(key, v);
            }
            const v = self.velocity.getPtr(key).?;

            for (param.data, grad.data, v.data) |*p, g, *vel| {
                vel.* = self.config.momentum * vel.* + g + self.config.weight_decay * p.*;
                p.* -= self.config.lr * vel.*;
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

    /// Free velocity buffers and the SGD struct (called by Optimizer.deinit).
    pub fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *SGD = @ptrCast(@alignCast(ctx));
        var iter = self.velocity.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
        }
        self.velocity.deinit();
        allocator.destroy(self);
    }

    /// Free velocity buffers (standalone, for direct use outside Optimizer wrapper).
    /// Does NOT free the SGD struct itself.
    pub fn deinit(self: *SGD, allocator: std.mem.Allocator) void {
        var iter = self.velocity.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
        }
        self.velocity.deinit();
    }

    /// Return an Optimizer wrapper for this SGD instance.
    ///
    /// The wrapper owns the SGD struct — deinit frees both the
    /// velocity buffers AND the SGD struct itself.
    pub fn optimizer(self: *SGD) Optimizer {
        return Optimizer{
            .ctx = @ptrCast(self),
            .step_fn = SGD.step,
            .zero_grad_fn = SGD.zeroGrad,
            .deinit_fn = SGD.deinitImpl,
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "SGD step — decreases parameter magnitude" {
    const alloc = std.testing.allocator;

    var sgd = try SGD.init(alloc, .{ .lr = 0.1 });
    defer sgd.deinit(alloc);

    // Create a parameter with gradient
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

    // Use the Optimizer vtable to call step
    var opt = sgd.optimizer();
    const params = [_]*Tensor{&param};
    try opt.step(&params);

    // param -= 0.1 * grad: [5-0.1, -3+0.1, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 4.9), param.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.9), param.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), param.data[2], 1e-4);
}
