//!
//! zig-transformer-lab — Optimizer protocol (vtable-based polymorphism)
//!
//! Purpose:
//!   Defines the Optimizer interface that SGD and AdamW implement.
//!   Uses a vtable (*anyopaque + function pointers) so that training
//!   code can swap optimizers at runtime without changing the loop.
//!
//!   This mirrors PyTorch's torch.optim.Optimizer base class, but
//!   implemented with Zig's explicit vtable pattern (no inheritance).
//!
//! Shape contract:
//!   step() and zeroGrad() operate on a list of parameter pointers.
//!   Each parameter is a *Tensor with a .grad field.
//!
//! Math:
//!   See sgd.zig and adamw.zig for per-optimizer update rules.
//!
//! Memory ownership:
//!   The optimizer owns its internal state (momentum buffers, etc.)
//!   and frees them in deinit(). It does NOT own the parameter tensors
//!   — those are borrowed pointers from the model.
//!
//! Errors:
//!   OutOfMemory — from state allocation (AdamW's m/v buffers)
//!
//! Credits:
//!   Inspired by PyTorch's torch.optim.Optimizer. No code copied.

const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Function signature for optimizer step.
/// ctx is the concrete optimizer (SGD, AdamW, etc.) cast to *anyopaque.
const StepFn = *const fn (ctx: *anyopaque, params: []const *Tensor) anyerror!void;

/// Function signature for zeroing gradients.
const ZeroGradFn = *const fn (ctx: *anyopaque, params: []const *Tensor) void;

/// Function signature for freeing optimizer state.
const DeinitFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void;

/// The Optimizer vtable — allows swapping SGD ↔ AdamW at runtime.
///
/// Usage:
///   var sgd = try SGD.init(allocator, .{ .lr = 0.01 });
///   var opt = sgd.optimizer();  // returns Optimizer wrapping SGD
///   try opt.step(params);       // calls SGD.step internally
///   opt.zeroGrad(params);
///   opt.deinit(allocator);      // calls SGD.deinit internally
pub const Optimizer = struct {
    /// Pointer to the concrete optimizer (SGD, AdamW, etc.)
    ctx: *anyopaque,

    /// Virtual function table
    step_fn: StepFn,
    zero_grad_fn: ZeroGradFn,
    deinit_fn: DeinitFn,

    /// Perform one optimization step: update all parameters using
    /// their gradients. Each optimizer implements its own rule.
    ///
    /// Worked example:
    ///   try opt.step(&model_params);
    ///   // All param.data values have been updated
    pub fn step(self: Optimizer, params: []const *Tensor) !void {
        try self.step_fn(self.ctx, params);
    }

    /// Zero all parameter gradients in preparation for the next
    /// forward/backward pass.
    ///
    /// Worked example:
    ///   opt.zeroGrad(&model_params);
    ///   // All param.grad tensors now contain zeros
    pub fn zeroGrad(self: Optimizer, params: []const *Tensor) void {
        self.zero_grad_fn(self.ctx, params);
    }

    /// Free optimizer-internal state (momentum buffers, etc.).
    /// Does NOT free the parameter tensors — those are owned by the model.
    pub fn deinit(self: Optimizer, allocator: std.mem.Allocator) void {
        self.deinit_fn(self.ctx, allocator);
    }
};
