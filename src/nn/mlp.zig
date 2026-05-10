//!
//! zig-transformer-lab — MLP (Feed-Forward Network)
//!
//! Purpose:
//!   Implements the position-wise feed-forward network used in
//!   transformer blocks: Linear → GELU → Linear.
//!
//!   The MLP is where most of the transformer's parameters live.
//!   The hidden dimension d_ff is typically 4x the model dimension,
//!   giving the network capacity to learn complex transformations.
//!
//! Shape contract:
//!   forward(x: (B, T, D), tape) → output: (B, T, D)
//!
//! Math:
//!   MLP(x) = W_2 @ GELU(W_1 @ x + b_1) + b_2
//!   where W_1: (d_ff, D), W_2: (D, d_ff)
//!
//! Memory ownership:
//!   Owns fc1 and fc2 Linear sub-layers. Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from sub-layer allocation
//!
//! Credits:
//!   Standard feed-forward network from "Attention Is All You Need"
//!   (Vaswani et al., 2017). No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const Linear = @import("linear.zig").Linear;
const GELU = @import("activations.zig").GELU;

pub const MLP = struct {
    fc1: Linear,
    fc2: Linear,
    gelu: GELU,
    allocator: std.mem.Allocator,

    /// Create an MLP with two Linear layers and GELU activation.
    ///
    /// fc1: D → d_ff (expand), fc2: d_ff → D (project back)
    ///
    /// Worked example:
    ///   var mlp = try MLP.init(alloc, 32, 128, true, &rng);
    ///   defer mlp.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        use_bias: bool,
        rng: *Rng,
    ) LabError!MLP {
        var fc1 = try Linear.init(allocator, d_model, d_ff, use_bias, rng);
        errdefer fc1.deinit();
        var fc2 = try Linear.init(allocator, d_ff, d_model, use_bias, rng);
        errdefer fc2.deinit();

        return MLP{
            .fc1 = fc1,
            .fc2 = fc2,
            .gelu = GELU{ .allocator = allocator },
            .allocator = allocator,
        };
    }

    /// Compute MLP(x) = fc2(GELU(fc1(x))).
    ///
    /// Worked example:
    ///   // x shape (2, 4, 32)
    ///   var out = try mlp.forward(x, &tape);
    ///   // out.shape == (2, 4, 32)
    pub fn forward(self: MLP, input: Tensor, tape: ?*Tape) LabError!Tensor {
        // fc1: (B, T, D) → (B, T, d_ff)
        var h = try self.fc1.forward(input, tape);
        errdefer h.deinit(self.allocator);

        // GELU: (B, T, d_ff) → (B, T, d_ff)
        var activated = try self.gelu.forward(h, tape);
        if (tape) |t| try t.keepAlive(&h);
        defer h.deinit(self.allocator);
        errdefer activated.deinit(self.allocator);

        // fc2: (B, T, d_ff) → (B, T, D)
        const output = try self.fc2.forward(activated, tape);
        if (tape) |t| try t.keepAlive(&activated);
        defer activated.deinit(self.allocator);

        return output;
    }

    /// Append this layer's learnable parameters to the list.
    pub fn parameters(self: *MLP, list: *std.ArrayList(*Tensor)) void {
        self.fc1.parameters(list);
        self.fc2.parameters(list);
    }

    /// Free sub-layers.
    pub fn deinit(self: *MLP) void {
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "MLP init — correct sub-layer dimensions" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var mlp = try MLP.init(alloc, 8, 32, true, &rng);
    defer mlp.deinit();

    // fc1: (32, 8), fc2: (8, 32)
    try std.testing.expectEqual(@as(usize, 32), mlp.fc1.weight.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 8), mlp.fc1.weight.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 8), mlp.fc2.weight.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 32), mlp.fc2.weight.shape.dims[1]);
}

test "MLP forward — correct output shape" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);

    var mlp = try MLP.init(alloc, 8, 32, true, &rng);
    defer mlp.deinit();

    const ops_create = @import("../tensor/ops/create.zig");
    var x = try ops_create.randn(alloc, Shape.init3D(2, 3, 8), &rng, 0.0, 1.0);
    defer x.deinit(alloc);

    var out = try mlp.forward(x, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 8), out.shape.dims[2]);
}
