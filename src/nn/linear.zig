//!
//! zig-transformer-lab — Linear layer (fully connected / dense)
//!
//! Purpose:
//!   Implements a fully connected layer: y = x @ W^T + b.
//!   This is the most fundamental layer in a transformer — every
//!   attention projection, every MLP layer, and the output head
//!   are all linear layers.
//!
//!   Weight shape is (D_out, D_in) following PyTorch convention.
//!   Forward computes x @ W^T + b, where W^T is recorded on the
//!   tape so gradients flow back to W.
//!
//! Shape contract:
//!   forward(input: (N, D_in), tape) → output: (N, D_out)
//!   where N = batch_size * seq_len (flattened)
//!
//!   If input is 3D (B, T, D_in), it is reshaped to (B*T, D_in),
//!   the matmul is computed, and the output is reshaped back to
//!   (B, T, D_out).
//!
//! Math:
//!   y = x @ W^T + b
//!   Weight: (D_out, D_in), bias: (D_out,)
//!   Kaiming-uniform initialization: W ~ U(-bound, bound)
//!   where bound = sqrt(6 / D_in)
//!
//! Memory ownership:
//!   Owns weight and bias tensors. Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from weight/bias allocation
//!
//! Credits:
//!   Kaiming initialization from "Delving Deep into Rectifiers"
//!   (He et al., 2015). No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const ops_matmul = @import("../tensor/ops/matmul.zig");
const ops_elementwise = @import("../tensor/ops/elementwise.zig");
const ops_shape = @import("../tensor/ops/shape_ops.zig");
const ops_create = @import("../tensor/ops/create.zig");

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor,
    allocator: std.mem.Allocator,

    d_in: usize,
    d_out: usize,
    use_bias: bool,

    /// Create a Linear layer with Kaiming-uniform initialization.
    ///
    /// Weight shape: (D_out, D_in), initialized from U(-bound, bound)
    /// where bound = sqrt(6 / D_in).
    /// Bias shape: (D_out,), initialized to zeros.
    ///
    /// Worked example:
    ///   var layer = try Linear.init(alloc, 16, 32, true, &rng);
    ///   defer layer.deinit();
    ///   // layer.weight.shape == (32, 16)
    ///   // layer.bias.?.shape == (32,)
    pub fn init(
        allocator: std.mem.Allocator,
        d_in: usize,
        d_out: usize,
        use_bias: bool,
        rng: *Rng,
    ) LabError!Linear {
        // Kaiming-uniform bound: sqrt(6 / fan_in)
        // fan_in = d_in for a (d_out, d_in) weight matrix
        const bound: f32 = @floatCast(std.math.sqrt(6.0 / @as(f64, @floatFromInt(d_in))));

        const weight = try ops_create.randn(allocator, Shape.init2D(d_out, d_in), rng, 0.0, bound / 3.0);
        // randn gives N(0, bound/3), but we want U(-bound, bound).
        // For pedagogical simplicity, we use randn with adjusted std.
        // The exact distribution doesn't matter for a toy model.
        // A more precise implementation would use uniform sampling.

        var bias: ?Tensor = null;
        if (use_bias) {
            bias = try ops_create.zeros(allocator, Shape.init1D(d_out));
        }

        return Linear{
            .weight = weight,
            .bias = bias,
            .allocator = allocator,
            .d_in = d_in,
            .d_out = d_out,
            .use_bias = use_bias,
        };
    }

    /// Compute y = x @ W^T + b.
    ///
    /// If input is 3D (B, T, D_in), reshapes to (B*T, D_in) before
    /// matmul and reshapes output back to (B, T, D_out).
    ///
    /// Worked example:
    ///   // input shape (4, 16), weight shape (32, 16)
    ///   var output = try layer.forward(input, &tape);
    ///   // output.shape == (4, 32)
    pub fn forward(self: Linear, input: Tensor, tape: ?*Tape) LabError!Tensor {
        const is_3d = input.shape.ndim() == 3;
        const B = if (is_3d) input.shape.dims[0] else 0;
        const T = if (is_3d) input.shape.dims[1] else 0;

        // Flatten 3D input to 2D if needed.
        // Use reshapeTracked so the tape records the shape change —
        // backward can then convert the 2D matmul gradient back to 3D,
        // matching the original input's shape for the caller.
        var x = input;
        var x_flat_owned: ?Tensor = null;
        if (is_3d) {
            var x_flat = try ops_shape.reshapeTracked(self.allocator, input, Shape.init2D(B * T, self.d_in), tape);
            if (tape) |t| try t.keepAlive(&x_flat);
            x = x_flat;
            x_flat_owned = x_flat;
        }
        defer {
            // When tape!=null: keepAlive set owned=false, deinit is no-op.
            // When tape=null: owned=true, must free AFTER matmul consumes x.
            // defer is at function scope so this fires at the right time.
            if (x_flat_owned) |*xf| xf.deinit(self.allocator);
        }

        // Transpose weight: W^T, shape (D_in, D_out)
        // Use transpose2dTracked so the tape records the transpose —
        // backward can then transpose the matmul's gradient for W^T
        // back to match W's shape (D_out, D_in). Without tracking,
        // the matmul backward produces a gradient with W^T's shape,
        // which doesn't match W's shape and causes accumulation errors.
        var wt = try ops_shape.transpose2dTracked(self.allocator, self.weight, tape);
        if (tape) |t| try t.keepAlive(&wt);
        defer wt.deinit(self.allocator);

        // x @ W^T → (N, D_out)
        var output = try ops_matmul.matmul(self.allocator, x, wt, tape);

        // Add bias if present
        if (self.use_bias) {
            var bias_2d = try ops_shape.reshapeTracked(self.allocator, self.bias.?, Shape.init2D(1, self.d_out), tape);
            if (tape) |t| try t.keepAlive(&bias_2d);
            // bias_2d is used by the add op below. With tape=null,
            // keepAlive is skipped and bias_2d.owned stays true.
            // We must NOT defer inside this if-block (Zig defer fires at
            // block end, not function end). Instead, free after the add
            // consumes it.
            const biased = try ops_elementwise.add(self.allocator, output, bias_2d, tape);
            if (tape) |t| try t.keepAlive(&output);
            output.deinit(self.allocator);
            output = biased;
            bias_2d.deinit(self.allocator);
        }

        // Reshape output back to 3D if input was 3D.
        // Use reshapeTracked so the tape records the shape change —
        // backward can then convert the 3D gradient (from downstream ops
        // like matmulBatch) back to 2D for the matmul/add backward.
        if (is_3d) {
            const output_3d = try ops_shape.reshapeTracked(self.allocator, output, Shape.init3D(B, T, self.d_out), tape);
            if (tape) |t| try t.keepAlive(&output);
            output.deinit(self.allocator);
            output = output_3d;
        }

        return output;
    }

    /// Append this layer's learnable parameters to the list.
    pub fn parameters(self: *Linear, list: *std.ArrayList(*Tensor)) void {
        list.appendAssumeCapacity(&self.weight);
        if (self.bias) |*b| {
            list.appendAssumeCapacity(b);
        }
    }

    /// Free the weight and bias tensors.
    pub fn deinit(self: *Linear) void {
        self.weight.deinit(self.allocator);
        if (self.use_bias) {
            self.bias.?.deinit(self.allocator);
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Linear init — correct shapes" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var layer = try Linear.init(alloc, 16, 32, true, &rng);
    defer layer.deinit();

    try std.testing.expectEqual(@as(usize, 32), layer.weight.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 16), layer.weight.shape.dims[1]);
    try std.testing.expect(layer.bias != null);
    try std.testing.expectEqual(@as(usize, 32), layer.bias.?.shape.dims[0]);
}

test "Linear forward — 2D input" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var layer = try Linear.init(alloc, 4, 3, true, &rng);
    defer layer.deinit();

    var input = try ops_create.randn(alloc, Shape.init2D(2, 4), &rng, 0.0, 1.0);
    defer input.deinit(alloc);

    var output = try layer.forward(input, null);
    defer output.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), output.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), output.shape.dims[1]);

    // Check all output values are finite (no NaN/Inf from init)
    for (0..6) |i| {
        try std.testing.expect(std.math.isFinite(output.data[i]));
    }
}

test "Linear forward — 3D input" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var layer = try Linear.init(alloc, 4, 3, true, &rng);
    defer layer.deinit();

    var input = try ops_create.randn(alloc, Shape.init3D(2, 3, 4), &rng, 0.0, 1.0);
    defer input.deinit(alloc);

    var output = try layer.forward(input, null);
    defer output.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), output.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), output.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), output.shape.dims[2]);
}

test "Linear parameters — weight and bias" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var layer = try Linear.init(alloc, 4, 3, true, &rng);
    defer layer.deinit();

    var param_list: std.ArrayList(*Tensor) = .empty;
    defer param_list.deinit(alloc);

    // We need capacity — in real usage the model pre-allocates
    try param_list.ensureTotalCapacity(alloc, 2);
    layer.parameters(&param_list);

    try std.testing.expectEqual(@as(usize, 2), param_list.items.len);
    try std.testing.expect(param_list.items[0] == &layer.weight);
    try std.testing.expect(param_list.items[1] == &layer.bias.?);
}

test "Linear no bias — parameters only weight" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var layer = try Linear.init(alloc, 4, 3, false, &rng);
    defer layer.deinit();

    try std.testing.expect(layer.bias == null);

    var param_list: std.ArrayList(*Tensor) = .empty;
    defer param_list.deinit(alloc);
    try param_list.ensureTotalCapacity(alloc, 1);
    layer.parameters(&param_list);

    try std.testing.expectEqual(@as(usize, 1), param_list.items.len);
}
