//!
//! 05_train_tiny.zig — Train on real text data (data/tiny.txt)
//!
//! This is the first example that trains a transformer on actual text data
//! instead of random token IDs. It demonstrates the full data pipeline:
//!   1. Load text file → build vocabulary → encode to token IDs
//!   2. Create sliding-window (input, target) pairs
//!   3. Shuffle and batch the windows
//!   4. Train the model with cross-entropy loss and AdamW
//!
//! This example verifies that the tokenizer, dataset, windowing, and
//! batcher all work correctly end-to-end.
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
const Dataset = ztl.data.dataset.Dataset;
const Windowing = ztl.data.windowing.Windowing;
const Batcher = ztl.data.batcher.Batcher;

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;
    const io = init.io;

    // --- Configuration ---
    const max_vocab: usize = 256;
    const T: usize = 8; // sequence length (context window)
    const B: usize = 4; // batch size
    const D: usize = 32; // model dimension
    const lr = 1e-3;
    const num_steps: usize = 100;

    // --- Load dataset ---
    try w.print("=== 05: Train on data/tiny.txt ===\n", .{});
    try w.print("Loading data/tiny.txt (max_vocab={})...\n", .{max_vocab});

    var ds = try Dataset.init(allocator, io, "data/tiny.txt", max_vocab, true);
    defer ds.deinit();

    try w.print("  Tokens: {}\n", .{ds.tokens.len});
    try w.print("  Vocab size: {}\n", .{ds.vocab.size()});

    // --- Windowing ---
    const windowing = Windowing.init(ds.tokens, T);
    try w.print("  Windows (T={}): {}\n", .{ T, windowing.count() });

    // --- Batcher ---
    var rng = Rng.init(42);
    var batcher = try Batcher.init(allocator, windowing, B, &rng);
    defer batcher.deinit();
    try w.print("  Batches (B={}): {}\n", .{ B, batcher.count() });

    // --- Model ---
    const cfg = TransformerConfig{
        .vocab_size = ds.vocab.size(),
        .d_model = D,
        .max_seq_len = T,
        .d_ff = D * 4,
        .ln_eps = 1e-5,
        .bias = true,
    };

    var model_rng = Rng.init(1337);
    var model = try TinyWordTransformer.init(allocator, cfg, &model_rng);
    defer model.deinit();

    try w.print("  Model: V={} D={} T={}\n\n", .{ ds.vocab.size(), D, T });

    // --- Optimizer ---
    var adam = try AdamW.init(allocator, .{
        .lr = lr,
        .beta1 = 0.9,
        .beta2 = 0.999,
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

    // --- Training loop ---
    var prev_loss: f32 = std.math.inf(f32);

    for (0..num_steps) |step| {
        // Reset batcher at each epoch
        if (step % batcher.count() == 0) {
            batcher.reset(&rng);
        }

        // Get next batch (or reset if exhausted)
        const batch = batcher.next() orelse blk: {
            batcher.reset(&rng);
            break :blk batcher.next().?;
        };
        defer batch.deinit();

        // Create input tensor (B, T) from batch.input
        var ids = try Tensor.init(allocator, Shape.init2D(B, T));
        defer ids.deinit(allocator);
        for (0..B * T) |i| {
            ids.data[i] = @floatFromInt(batch.input[i]);
        }

        // Create target tensor (B*T,) from batch.target
        var targets = try Tensor.init(allocator, Shape.init1D(B * T));
        defer targets.deinit(allocator);
        for (0..B * T) |i| {
            targets.data[i] = @floatFromInt(batch.target[i]);
        }

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
        try tape.keepAlive(&logits_3d);

        // Reshape logits from (B, T, V) to (B*T, V) — tape-tracked
        var logits = try ztl.ops.shape_ops.reshapeTracked(allocator, logits_3d, Shape.init2D(B * T, ds.vocab.size()), &tape);
        try tape.keepAlive(&logits);
        defer logits.deinit(allocator);

        // Compute cross-entropy loss
        var loss = try ztl.ops.loss.crossEntropy(allocator, logits, targets, &tape);

        // Print loss periodically
        if (step % 10 == 0 or step == num_steps - 1) {
            try w.print("Step {:3}: loss = {d:.4}\n", .{ step, loss.data[0] });
        }

        // Backward pass
        try tape.backward(&loss);

        // Save loss value before freeing
        const loss_val = loss.data[0];
        loss.deinit(allocator);
        logits_3d.deinit(allocator);

        // Optimizer step
        try opt.step(params.items);
        opt.zeroGrad(params.items);

        prev_loss = loss_val;
    }

    // --- Print sample decoded batch ---
    try w.print("\n=== Sample Batch Decode ===\n", .{});
    batcher.reset(&rng);
    if (batcher.next()) |sample_batch| {
        defer sample_batch.deinit();
        // Decode first 3 positions of first sample
        for (0..@min(3, T)) |t| {
            const id = sample_batch.input[t];
            const word = ds.vocab.decodeId(id);
            try w.print("  input[{d}]: {} ({s})\n", .{ t, id, word });
        }
    }

    try w.print("\n=== Training Complete ===\n", .{});
    try w.print("Final loss: {d:.4}\n", .{prev_loss});
    try w.print("Vocab size: {}\n", .{ds.vocab.size()});
}
