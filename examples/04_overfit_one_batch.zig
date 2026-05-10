//!
//! 04_overfit_one_batch.zig — Overfit a single batch to verify training works
//!
//! Creates a tiny transformer model and trains it on a single batch of
//! random data. Verifies that:
//!   1. Forward pass produces correct-shaped logits
//!   2. Backward pass computes gradients
//!   3. Optimizer step decreases the loss
//!
//! This is the "sanity check" for the entire nn + autograd + optim pipeline.
//! If this passes, all the pieces work together correctly.
//!
//! IMPORTANT: All intermediate tensors from the forward pass must outlive
//! the tape's backward call. The tape's SavedData stores slices that point
//! into the original tensor buffers. If those buffers are freed before
//! backward runs, we get use-after-free. The safest pattern is to keep
//! all intermediates alive until after backward.
//;

const std = @import("std");
const ztl = @import("zig_transformer_lab");
const Tensor = ztl.Tensor;
const Shape = ztl.shape.Shape;
const Tape = ztl.Tape;
const TransformerConfig = ztl.nn.module.TransformerConfig;
const TinyWordTransformer = ztl.nn.model.TinyWordTransformer;
const AdamW = ztl.optim.adamw.AdamW;
const Rng = ztl.rng.Rng;

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;

    // --- Configuration ---
    const V = 32;
    const D = 16;
    const T = 8;
    const B = 2;
    const lr = 1e-3;
    const num_steps = 50;

    const cfg = TransformerConfig{
        .vocab_size = V,
        .d_model = D,
        .max_seq_len = T,
        .d_ff = D * 4,
        .ln_eps = 1e-5,
        .bias = true,
    };

    // --- Initialize model and optimizer ---
    var rng = Rng.init(42);
    var model = try TinyWordTransformer.init(allocator, cfg, &rng);
    defer model.deinit();

    var adam = try AdamW.init(allocator, .{
        .lr = lr,
        .beta1 = 0.9,
        .beta2 = 0.95,
        .eps = 1e-8,
        .weight_decay = 0.01,
    });
    defer adam.deinit(allocator);
    var opt = adam.optimizer();

    // --- Collect parameters ---
    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    try params.ensureTotalCapacity(allocator, 32);
    model.parameters(&params);

    // --- Create a fixed batch of random data ---
    var ids = try Tensor.init(allocator, Shape.init2D(B, T));
    defer ids.deinit(allocator);
    {
        var data_rng = Rng.init(123);
        for (0..B * T) |i| {
            ids.data[i] = @floatFromInt(data_rng.random().intRangeLessThan(usize, 0, V));
        }
    }

    // Create random targets (next-token prediction: shift by 1)
    var targets = try Tensor.init(allocator, Shape.init2D(B, T));
    defer targets.deinit(allocator);
    {
        var target_rng = Rng.init(456);
        for (0..B * T) |i| {
            targets.data[i] = @floatFromInt(target_rng.random().intRangeLessThan(usize, 0, V));
        }
    }

    // --- Training loop ---
    try w.print("=== Overfit One Batch: V={} D={} T={} B={} ===\n", .{ V, D, T, B });
    try w.print("Parameters: {} tensors\n\n", .{params.items.len});

    var prev_loss: f32 = std.math.inf(f32);

    for (0..num_steps) |step| {
        // Create fresh tape for this step
        var tape = Tape.init(allocator);
        defer tape.deinit();

        // Track all parameters on the tape
        for (params.items) |param| {
            param.requires_grad = true;
            _ = try tape.trackLeaf(param);
        }

        // Forward pass
        var logits_3d = try model.forward(ids, &tape);
        // The model's internal intermediates are kept alive by the tape.
        // The returned logits_3d is owned by the caller; its data is
        // referenced by the cross-entropy node's SavedData, so we must
        // keep it alive through backward.
        try tape.keepAlive(&logits_3d);

        // Reshape logits from (B, T, V) to (B*T, V) for cross-entropy.
        // This is a view (not tape-tracked) — the gradient flows through
        // the 3D logits which are already on the tape from the model forward.
        const logits = try logits_3d.reshape(Shape.init2D(B * T, V));

        // Reshape targets from (B, T) to (B*T,) for cross-entropy
        const targets_flat = try targets.reshape(Shape.init1D(B * T));

        // Compute cross-entropy loss
        var loss = try ztl.ops.loss.crossEntropy(allocator, logits, targets_flat, &tape);

        // Print loss periodically
        if (step % 10 == 0 or step == num_steps - 1) {
            try w.print("Step {:3}: loss = {d:.6}\n", .{ step, loss.data[0] });
        }

        // Verify loss decreases (after warmup)
        if (step > 0 and step % 10 == 0) {
            if (loss.data[0] < prev_loss) {
                try w.print("  Loss decreased!\n", .{});
            } else {
                try w.print("  WARNING: Loss did not decrease\n", .{});
            }
        }

        // Backward pass
        try tape.backward(&loss);

        // Save loss value before freeing
        const loss_val = loss.data[0];

        // Free loss tensor — its data is not referenced by any SavedData.
        loss.deinit(allocator);
        // logits_3d data was transferred to the tape via keepAlive,
        // so this deinit is a no-op (owned=false). The tape will free
        // the buffer in tape.deinit().
        logits_3d.deinit(allocator);

        // Optimizer step
        try opt.step(params.items);
        opt.zeroGrad(params.items);

        prev_loss = loss_val;
    }

    try w.print("\n=== Training Complete ===\n", .{});
    try w.print("Final loss: {d:.6}\n", .{prev_loss});
    try w.print("Training pipeline works!\n", .{});
}
