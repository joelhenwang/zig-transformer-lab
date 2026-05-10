//!
//! zig-transformer-lab — Embedding lookup layer
//!
//! Purpose:
//!   Maps integer token IDs to dense vectors (rows of a weight matrix).
//!   This is how a transformer converts discrete tokens into continuous
//!   representations that the rest of the model can process.
//!
//!   Forward gathers rows: output[i] = weight[idx[i]].
//!   Backward scatter-adds: grad_weight[idx[i]] += grad_output[i].
//!
//! Shape contract:
//!   forward(ids: (N,), tape) → output: (N, d_model)
//!   where N = number of tokens (B*T in a batch)
//!
//!   For 3D ids (B, T), the output is (B, T, d_model).
//!
//! Math:
//!   Embedding is a lookup table — no arithmetic, just indexing.
//!   The gradient is a scatter-add: each output position's gradient
//!   flows back to exactly one row in the weight matrix.
//!
//! Memory ownership:
//!   Owns the weight tensor (vocab_size, d_model). Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from weight allocation
//!   InvalidArgument — index out of vocabulary range
//!
//! Credits:
//!   Standard embedding layer. No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Node = @import("../autograd/node.zig").Node;
const OpKind = @import("../autograd/node.zig").OpKind;
const Rng = @import("../core/rng.zig").Rng;
const ops_create = @import("../tensor/ops/create.zig");
const ops_shape = @import("../tensor/ops/shape_ops.zig");
const totalElements = @import("../tensor/shape.zig").totalElements;
const module = @import("module.zig");
// PR-mu: embedding routes CUDA inputs to GPU gather / scatter-add
// kernels via backend.cuda.dispatch.
const cuda_dispatch = @import("../backend/cuda/dispatch.zig");

pub const Embedding = struct {
    weight: Tensor,
    allocator: std.mem.Allocator,
    vocab_size: usize,
    d_model: usize,

    /// Create an Embedding layer with random initialization.
    ///
    /// Weight shape: (vocab_size, d_model), initialized from N(0, 1/sqrt(d_model)).
    ///
    /// Worked example:
    ///   var embed = try Embedding.init(alloc, 64, 32, &rng);
    ///   defer embed.deinit();
    ///   // embed.weight.shape == (64, 32)
    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: usize,
        d_model: usize,
        rng: *Rng,
    ) LabError!Embedding {
        const std_dev: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(d_model))));
        const weight = try ops_create.randn(allocator, Shape.init2D(vocab_size, d_model), rng, 0.0, std_dev);

        var layer = Embedding{
            .weight = weight,
            .allocator = allocator,
            .vocab_size = vocab_size,
            .d_model = d_model,
        };
        module.assignParamId(&layer.weight);
        return layer;
    }

    /// Look up embeddings for a batch of token IDs.
    ///
    /// ids can be 1D (N,) or 2D (B, T). Output is always the ids shape
    /// with an extra d_model dimension appended.
    ///
    /// Worked example:
    ///   // ids shape (2, 4), d_model = 32
    ///   var output = try embed.forward(ids, &tape);
    ///   // output.shape == (2, 4, 32)
    pub fn forward(self: Embedding, ids: Tensor, tape: ?*Tape) LabError!Tensor {
        // PR-mu: route to CUDA when both tensors live on GPU. ids is
        // an integer lookup key stored as f32 (our Tensor type is
        // f32-only); the GPU kernel rounds to int with rintf.
        if (self.weight.device == .cuda) {
            var output = try cuda_dispatch.embeddingForward(self.weight, ids);
            if (tape) |t| {
                if (self.weight.requires_grad) {
                    // Embedding backward uses the indices and the
                    // weight's SHAPE (not its values). We store the
                    // weight snapshot so tape cloneTensorData makes a
                    // DtoD copy; the backward reads only shape + ctx
                    // off it. For the indices we need to snapshot the
                    // ids TENSOR (CUDA) because its host slice is
                    // empty. The existing .embedding_info variant
                    // stores indices as []const f32 which works for
                    // CPU only; for CUDA we store the ids tensor in
                    // a .tensor_pair together with weight so
                    // backward can see both device buffers.
                    const node_id = try t.record(Node{
                        .id = undefined,
                        .op = .embedding,
                        .parents = .{ self.weight.tape_node, null },
                        .n_parents = 1,
                        .saved = .{ .tensor_pair = .{ .a = self.weight, .b = ids } },
                    });
                    output.requires_grad = true;
                    output.tape_node = node_id;
                }
            }
            return output;
        }
        const n_ids = totalElements(ids.shape);
        const output_shape = switch (ids.shape.ndim()) {
            1 => Shape.init2D(n_ids, self.d_model),
            2 => Shape.init3D(ids.shape.dims[0], ids.shape.dims[1], self.d_model),
            else => return error.InvalidArgument,
        };

        var output = try Tensor.init(self.allocator, output_shape);
        errdefer output.deinit(self.allocator);

        // Gather rows: output[i, :] = weight[idx[i], :]
        for (0..n_ids) |i| {
            const idx: usize = @intFromFloat(ids.data[i]);
            if (idx >= self.vocab_size) return error.InvalidArgument;

            // Copy the row from weight to output
            const src_offset = idx * self.d_model;
            const dst_offset = i * self.d_model;
            @memcpy(output.data[dst_offset .. dst_offset + self.d_model], self.weight.data[src_offset .. src_offset + self.d_model]);
        }

        // Record on the tape for backward (scatter-add)
        if (tape) |t| {
            if (self.weight.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .embedding,
                    .parents = .{ self.weight.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .embedding_info = .{
                        .weight = self.weight,
                        .indices = ids.data,
                    } },
                });
                output.requires_grad = true;
                output.tape_node = node_id;
            }
        }

        return output;
    }

    /// Append this layer's learnable parameters to the list.
    pub fn parameters(self: *Embedding, list: *std.ArrayList(*Tensor)) void {
        list.appendAssumeCapacity(&self.weight);
    }

    /// Free the weight tensor.
    pub fn deinit(self: *Embedding) void {
        self.weight.deinit(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Embedding init — correct shape" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var embed = try Embedding.init(alloc, 64, 32, &rng);
    defer embed.deinit();

    try std.testing.expectEqual(@as(usize, 64), embed.weight.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 32), embed.weight.shape.dims[1]);
}

test "Embedding forward — 1D ids" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var embed = try Embedding.init(alloc, 64, 32, &rng);
    defer embed.deinit();

    var ids = try Tensor.init(alloc, Shape.init1D(4));
    defer ids.deinit(alloc);
    ids.data[0] = 0.0;
    ids.data[1] = 1.0;
    ids.data[2] = 2.0;
    ids.data[3] = 3.0;

    var output = try embed.forward(ids, null);
    defer output.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 4), output.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 32), output.shape.dims[1]);
}

test "Embedding forward — 2D ids" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var embed = try Embedding.init(alloc, 64, 32, &rng);
    defer embed.deinit();

    var ids = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer ids.deinit(alloc);
    for (0..6) |i| ids.data[i] = @floatFromInt(i);

    var output = try embed.forward(ids, null);
    defer output.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), output.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), output.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 32), output.shape.dims[2]);
}

test "Embedding forward — matches weight rows" {
    const alloc = std.testing.allocator;
    var rng = @import("../core/rng.zig").Rng.init(42);

    var embed = try Embedding.init(alloc, 4, 3, &rng);
    defer embed.deinit();

    // Set specific weight values
    embed.weight.data[0 * 3 + 0] = 10.0;
    embed.weight.data[0 * 3 + 1] = 20.0;
    embed.weight.data[0 * 3 + 2] = 30.0;
    embed.weight.data[2 * 3 + 0] = 40.0;
    embed.weight.data[2 * 3 + 1] = 50.0;
    embed.weight.data[2 * 3 + 2] = 60.0;

    var ids = try Tensor.init(alloc, Shape.init1D(2));
    defer ids.deinit(alloc);
    ids.data[0] = 0.0; // row 0
    ids.data[1] = 2.0; // row 2

    var output = try embed.forward(ids, null);
    defer output.deinit(alloc);

    // output[0] should be weight[0] = [10, 20, 30]
    try std.testing.expectEqual(@as(f32, 10.0), output.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), output.data[1]);
    try std.testing.expectEqual(@as(f32, 30.0), output.data[2]);
    // output[1] should be weight[2] = [40, 50, 60]
    try std.testing.expectEqual(@as(f32, 40.0), output.data[3]);
    try std.testing.expectEqual(@as(f32, 50.0), output.data[4]);
    try std.testing.expectEqual(@as(f32, 60.0), output.data[5]);
}
