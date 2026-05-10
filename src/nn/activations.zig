//!
//! zig-transformer-lab — Activation functions as nn layers
//!
//! Purpose:
//!   Wraps activation functions (GELU) as stateless nn layers that
//!   conform to the module protocol. Stateless means no learnable
//!   parameters — the layer just delegates to the underlying op.
//!
//!   Why wrap a function as a layer? So that composite layers (MLP,
//!   TransformerBlock) can treat all sub-components uniformly via
//!   the module protocol, even though some have parameters and
//!   others don't.
//!
//! Shape contract:
//!   forward(x: (N, D), tape) → y: (N, D)  [same shape]
//!
//! Math:
//!   GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
//!
//! Memory ownership:
//!   No owned state. deinit() is a no-op.
//!
//! Errors:
//!   OutOfMemory — from the underlying op's output allocation
//!
//! Credits:
//!   GELU introduced in "Gaussian Error Linear Units" (Hendrycks
//!   & Gimpel, 2016). No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Tape = @import("../autograd/tape.zig").Tape;
const ops_unary = @import("../tensor/ops/unary.zig");

/// Stateless GELU activation layer.
///
/// Wraps ops.unary.geluExact as an nn layer. No parameters,
/// no internal state — just a function call with a tape-aware
/// interface.
///
/// The allocator is stored so forward() matches the module protocol
/// (self, input, tape) without an extra allocator argument.
///
/// Worked example:
///   var gelu = GELU{ .allocator = gpa };
///   var y = try gelu.forward(x, &tape);
///   // y.shape == x.shape, y = GELU(x)
pub const GELU = struct {
    allocator: std.mem.Allocator,

    /// Compute GELU(x).
    ///
    /// Delegates to ops.unary.geluExact. The tape is passed through
    /// so gradients flow back to x.
    ///
    /// Worked example:
    ///   // x shape (4, 16)
    ///   var y = try gelu.forward(x, &tape);
    ///   // y shape (4, 16), y[i,j] = GELU(x[i,j])
    pub fn forward(self: GELU, input: Tensor, tape: ?*Tape) LabError!Tensor {
        return ops_unary.geluExact(self.allocator, input, tape);
    }

    /// No parameters — does nothing.
    pub fn parameters(_: GELU, _: *std.ArrayList(*Tensor)) void {}

    /// No state — nothing to free.
    pub fn deinit(_: *GELU) void {}
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "GELU forward — output shape matches input" {
    const alloc = std.testing.allocator;
    var gelu = GELU{ .allocator = alloc };

    var x = try Tensor.init(alloc, @import("../tensor/shape.zig").Shape.init2D(2, 4));
    defer x.deinit(alloc);
    for (0..8) |i| x.data[i] = @floatFromInt(i);

    var y = try gelu.forward(x, null);
    defer y.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), y.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), y.shape.dims[1]);

    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y.data[0], 1e-4);
}
