//!
//! 10_train_deep.zig — 2-block, 2-head, D=64 transformer on CUDA
//!
//! This is the Stage 8 M8 acceptance-sweep example. It trains a deeper
//! model than example 06 (n_layer=2, n_head=2, d_model=64) on
//! Shakespeare via the Stage 8 Trainer CUDA path (route A: Trainer
//! itself is CUDA-aware when cfg.use_cuda = true).
//!
//! Two modes, selected by a single optional argument:
//!
//!   --sanitize    50 steps, no checkpointing, suitable for running
//!                 under compute-sanitizer without the 120 s wrapper
//!                 budget being a concern.
//!
//!   (no arg)      200 steady-state steps with a 5-step warm-up.
//!                 Reports per-step wall-clock for comparison against
//!                 the 1/1/32 baseline in example 09.
//!
//! Usage:
//!   zig build run-example -Dexample=10_train_deep -Dcuda=true
//!   zig build run-example -Dexample=10_train_deep -Dcuda=true -- --sanitize
//!
//! Plan reference: docs/stage8_plan.md §8 Milestone 8; handoff
//! docs/stage8_handoff.md §3.
//;

const std = @import("std");
const ztl = @import("zig_transformer_lab");
const TrainConfig = ztl.lab.TrainConfig;
const Trainer = ztl.lab.Trainer;

/// Acceptance target from `docs/stage8_plan.md` Milestone 8: the
/// 2/2/64 wall-clock must be under 2× the 1/1/32 baseline (~4.7 ms on
/// RTX 4060 Ti, per AGENTS.md progress table). 9.4 ms/step is the
/// hard ceiling; anything under is a pass.
const BASELINE_MS_PER_STEP: f64 = 4.7;
const ACCEPTANCE_RATIO: f64 = 2.0;

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;

    // --- Parse args ---
    // Look for `--sanitize` anywhere in argv. This keeps the dispatch
    // trivial — we don't need a real arg parser for one flag.
    // Zig 0.16.0: args iterator ships under `std.process.Args`.
    var args_iter = try std.process.Args.Iterator.initAllocator(
        init.minimal.args,
        allocator,
    );
    defer args_iter.deinit();

    var sanitize_mode = false;
    _ = args_iter.skip(); // program name
    while (args_iter.next()) |a| {
        if (std.mem.eql(u8, a, "--sanitize")) {
            sanitize_mode = true;
        }
    }

    try w.print("=== 10: Train 2-block 2-head D=64 on CUDA ===\n\n", .{});

    // Config: 2-block, 2-head, d_model=64. This is the acceptance
    // target shape set out in docs/stage8_plan.md §8 Milestone 8.
    //
    // d_model=64, n_head=2 -> d_head=32 (matches the per-head
    // dimension of the 1/1/32 baseline, but now stacked into two
    // heads). d_ff=4*d_model=256 per the standard transformer recipe.
    // seq_len=16, batch_size=4 match example 06 / 09 so wall-clock
    // comparison is apples-to-apples.
    const cfg = TrainConfig{
        .max_vocab = 2000,
        .seq_len = 16,
        .batch_size = 4,
        .d_model = 64,
        .d_ff = 256,
        .n_layer = 2,
        .n_head = 2,
        .lr = 1e-3,
        // Sanitize mode: 50 steps, no disk i/o. Normal mode: 200
        // steps so we have enough samples for a stable wall-clock
        // number after the 5-step warm-up.
        .max_steps = if (sanitize_mode) 50 else 200,
        .log_every = if (sanitize_mode) 10 else 50,
        .ckpt_every = 0,
        .model_seed = 1337,
        .data_seed = 42,
        .lowercase = true,
        .use_cuda = true,
    };

    try w.print("mode: {s}\n", .{if (sanitize_mode) "SANITIZE" else "BENCH"});
    try w.print("config: V={} D={} d_ff={} T={} B={} n_layer={} n_head={}\n", .{
        cfg.max_vocab,
        cfg.d_model,
        cfg.d_ff,
        cfg.seq_len,
        cfg.batch_size,
        cfg.n_layer,
        cfg.n_head,
    });
    try w.print("  lr={d:.4} max_steps={} use_cuda={}\n\n", .{
        cfg.lr,
        cfg.max_steps,
        cfg.use_cuda,
    });

    var trainer = try Trainer.init(allocator, init.io, "data/tinyshakespeare.txt", cfg);
    defer trainer.deinit();

    try w.print("Dataset: {} tokens, V={}\n\n", .{
        trainer.dataset.len(),
        trainer.vocab_size,
    });

    // --- Run training ---
    //
    // We time the whole call. The Trainer's per-step HtoD copies for
    // ids/targets are tiny (~256 B/step) and the optimiser step is a
    // single kernel launch; the training-step cost is dominated by
    // forward + backward matmul/softmax kernels on the 2×2×64 shape.
    const t_start = std.Io.Clock.now(.awake, init.io);
    const result = try trainer.train(&locked_stderr.file_writer.interface, null);
    const t_dur = t_start.durationTo(std.Io.Clock.now(.awake, init.io));

    const total_ns: i96 = t_dur.nanoseconds;
    const total_ms: f64 = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
    const ms_per_step: f64 = total_ms / @as(f64, @floatFromInt(result.steps_completed));

    try w.print("\n=== Training Complete ===\n", .{});
    try w.print("Mode:                {s}\n", .{if (sanitize_mode) "SANITIZE" else "BENCH"});
    try w.print("Steps:               {}\n", .{result.steps_completed});
    try w.print("Final loss:          {d:.4}\n", .{result.final_loss});
    try w.print("Vocab size:          {}\n", .{result.vocab_size});
    try w.print("Total wall-clock:    {d:.1} ms\n", .{total_ms});
    try w.print("Per-step wall-clock: {d:.3} ms\n", .{ms_per_step});
    try w.print("\nAcceptance target (docs/stage8_plan.md §M8):\n", .{});
    try w.print("  baseline 1/1/32:  {d:.2} ms (per AGENTS.md)\n", .{BASELINE_MS_PER_STEP});
    try w.print("  budget (<{d:.1}x):   {d:.2} ms\n", .{
        ACCEPTANCE_RATIO,
        BASELINE_MS_PER_STEP * ACCEPTANCE_RATIO,
    });
    const ratio = ms_per_step / BASELINE_MS_PER_STEP;
    try w.print("  measured ratio:   {d:.2}x\n", .{ratio});
    if (sanitize_mode) {
        try w.print("  (sanitize mode: timing includes compute-sanitizer\n", .{});
        try w.print("   overhead if running under it — ignore the ratio)\n", .{});
    } else if (ratio <= ACCEPTANCE_RATIO) {
        try w.print("  VERDICT: PASS\n", .{});
    } else {
        try w.print("  VERDICT: OVER BUDGET (see Stage 9 perf follow-up)\n", .{});
    }

    // Sanity check: final loss should be a real number. In sanitize
    // mode 50 steps is too few for convergence, so we only assert
    // finiteness — not a specific loss target.
    if (!std.math.isFinite(result.final_loss)) {
        try w.print("\nFAIL: final loss is not finite\n", .{});
        return error.NonFiniteLoss;
    }
}
