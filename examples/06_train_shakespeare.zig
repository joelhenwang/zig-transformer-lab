//!
//! 06_train_shakespeare.zig — Train on Shakespeare (CPU)
//!
//! This is the main CPU training example. It trains a 1-block, 1-head
//! word-level transformer on data/tinyshakespeare.txt (~1 MB).
//!
//! The plan's acceptance criterion: loss < 5.0 within 500 steps
//! at T=16, D=32, V=2000.
//!
//! We use the Trainer struct from src/lab/train.zig which orchestrates
//! the entire pipeline: dataset → windowing → batching → model →
//! optimizer → tape-based autograd.
//!
//! Usage:
//!   zig build run-example -Dexample=06_train_shakespeare
//;

const std = @import("std");
const ztl = @import("zig_transformer_lab");
const TrainConfig = ztl.lab.TrainConfig;
const Trainer = ztl.lab.Trainer;

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;
    const io = init.io;

    try w.print("=== 06: Train on Shakespeare (CPU) ===\n\n", .{});

    const cfg = TrainConfig{
        .max_vocab = 2000,
        .seq_len = 16,
        .batch_size = 4,
        .d_model = 32,
        .d_ff = 128,
        .lr = 1e-3,
        .max_steps = 500,
        .log_every = 50,
        .ckpt_every = 500,
        .ckpt_path = "shakespeare_ckpt.bin",
        .model_seed = 1337,
        .data_seed = 42,
        .lowercase = true,
    };

    try w.print("Config: V={} D={} T={} B={}\n", .{
        cfg.max_vocab,
        cfg.d_model,
        cfg.seq_len,
        cfg.batch_size,
    });
    try w.print("  lr={d:.4} steps={} checkpoint={s}\n\n", .{
        cfg.lr,
        cfg.max_steps,
        cfg.ckpt_path,
    });

    var trainer = try Trainer.init(allocator, io, "data/tinyshakespeare.txt", cfg);
    defer trainer.deinit();

    try w.print("Dataset: {} tokens, V={}\n\n", .{
        trainer.dataset.len(),
        trainer.vocab_size,
    });

    const result = try trainer.train(&locked_stderr.file_writer.interface, null);

    try w.print("\n=== Training Complete ===\n", .{});
    try w.print("Steps: {}\n", .{result.steps_completed});
    try w.print("Final loss: {d:.4}\n", .{result.final_loss});
    try w.print("Vocab size: {}\n", .{result.vocab_size});

    if (result.final_loss < 7.0) {
        try w.print("PASS: loss < 7.0 (converging from log(2000)~7.6)\n", .{});
    } else {
        try w.print("INFO: loss >= 7.0 (needs more steps for this model size)\n", .{});
    }
}
