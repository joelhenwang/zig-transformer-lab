//!
//! 09_cuda_benchmark.zig — CUDA vs CPU training-step speed benchmark
//!
//! Purpose:
//!   Measures wall-clock time per `forward + loss + backward + step`
//!   iteration on CPU and CUDA using the same TinyWordTransformer
//!   config as Shakespeare training (V=2000, D=32, T=16, B=4).
//!   Reports the speedup ratio. Playbook target: >= 30x.
//!
//!   The benchmark discards a warm-up period (first 10 iterations)
//!   and measures the next 90 iterations, synchronising after each
//!   CUDA step so wall-clock reflects actual kernel work rather than
//!   the driver's queued-launch optimism.
//!
//! Usage:
//!   zig build run-example -Dexample=09_cuda_benchmark -Dcuda=true
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
const CudaContext = ztl.backend.cuda.context.CudaContext;
const module = ztl.backend.cuda.module;

const WARMUP: usize = 10;
const MEASURED: usize = 90;

fn runOnce(
    allocator: std.mem.Allocator,
    model: *TinyWordTransformer,
    ids: Tensor,
    targets: Tensor,
    V: usize,
    BT: usize,
    opt: *AdamW,
) !f32 {
    var tape = Tape.init(allocator);
    defer tape.deinit();

    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    try params.ensureTotalCapacity(allocator, 32);
    model.parameters(&params);
    for (params.items) |p| _ = try tape.trackLeaf(p);

    var logits = try model.forward(ids, &tape);
    defer logits.storage.deinit(allocator);
    var logits_2d = try ztl.ops.shape_ops.reshapeTracked(
        allocator,
        logits,
        Shape.init2D(BT, V),
        &tape,
    );
    defer logits_2d.storage.deinit(allocator);
    var loss = try ztl.ops.loss.crossEntropy(allocator, logits_2d, targets, &tape);
    defer loss.storage.deinit(allocator);

    try tape.backward(&loss);
    try AdamW.step(opt, params.items);
    AdamW.zeroGrad(opt, params.items);

    // Read back scalar loss (HtoD round-trip for CUDA; cheap for CPU).
    if (loss.device == .cuda) {
        var loss_cpu = try loss.toCpu(allocator);
        defer loss_cpu.deinit(allocator);
        return loss_cpu.data[0];
    }
    return loss.data[0];
}

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;

    // Shakespeare-matched config.
    const cfg = TransformerConfig{
        .vocab_size = 2000,
        .d_model = 32,
        .d_ff = 128,
        .max_seq_len = 16,
    };
    const B: usize = 4;
    const T: usize = 16;
    const BT = B * T;

    try w.print("config: V={d}, D={d}, d_ff={d}, T={d}, B={d}\n", .{
        cfg.vocab_size, cfg.d_model, cfg.d_ff, T, B,
    });
    try w.print("measuring {d} iterations after {d} warm-up steps\n\n", .{ MEASURED, WARMUP });

    // ----- CUDA setup + CUDA model -----
    var ctx = try CudaContext.init(allocator);
    defer ctx.deinit();
    const io = init.io;
    inline for (&[_][:0]const u8{
        "elementwise", "reduce", "softmax", "unary", "embedding", "ce_loss", "adamw",
    }) |name| {
        _ = try module.loadPtxFromFile(&ctx, io, name);
    }

    var rng = Rng.init(42);
    var model_cpu = try TinyWordTransformer.init(allocator, cfg, &rng);
    defer model_cpu.deinit();
    var model_gpu = try TinyWordTransformer.init(allocator, cfg, &rng);
    defer model_gpu.deinit();
    {
        var p_cpu: std.ArrayList(*Tensor) = .empty;
        defer p_cpu.deinit(allocator);
        try p_cpu.ensureTotalCapacity(allocator, 32);
        model_cpu.parameters(&p_cpu);
        var p_gpu: std.ArrayList(*Tensor) = .empty;
        defer p_gpu.deinit(allocator);
        try p_gpu.ensureTotalCapacity(allocator, 32);
        model_gpu.parameters(&p_gpu);
        for (0..p_cpu.items.len) |i| {
            @memcpy(p_gpu.items[i].data, p_cpu.items[i].data);
            p_cpu.items[i].requires_grad = true;
            p_gpu.items[i].requires_grad = true;
        }
    }
    try model_gpu.moveToCuda(&ctx);

    var ids_cpu = try Tensor.init(allocator, Shape.init2D(B, T));
    defer ids_cpu.deinit(allocator);
    var id_rng = Rng.init(123);
    for (0..BT) |i| ids_cpu.data[i] = @floatFromInt(id_rng.random().intRangeLessThan(usize, 0, cfg.vocab_size));
    var ids_gpu = try ids_cpu.toCuda(&ctx);
    defer ids_gpu.storage.deinit(allocator);

    var targets_cpu = try Tensor.init(allocator, Shape.init1D(BT));
    defer targets_cpu.deinit(allocator);
    var tgt_rng = Rng.init(456);
    for (0..BT) |i| targets_cpu.data[i] = @floatFromInt(tgt_rng.random().intRangeLessThan(usize, 0, cfg.vocab_size));
    var targets_gpu = try targets_cpu.toCuda(&ctx);
    defer targets_gpu.storage.deinit(allocator);

    var opt_cpu = try AdamW.init(allocator, .{});
    defer opt_cpu.deinit(allocator);
    var opt_gpu = try AdamW.init(allocator, .{});
    defer opt_gpu.deinit(allocator);

    // Warm-up both paths (discard timing). CUDA warm-up hides the
    // PTX-cache-miss cost on the first launch of each kernel.
    try w.print("warming up CPU...\n", .{});
    for (0..WARMUP) |_| _ = try runOnce(allocator, &model_cpu, ids_cpu, targets_cpu, cfg.vocab_size, BT, &opt_cpu);
    try w.print("warming up CUDA...\n", .{});
    for (0..WARMUP) |_| {
        _ = try runOnce(allocator, &model_gpu, ids_gpu, targets_gpu, cfg.vocab_size, BT, &opt_gpu);
        try ctx.synchronize();
    }

    // ----- Measured loops -----
    try w.print("measuring CPU...\n", .{});
    const cpu_start = std.Io.Clock.now(.awake, io);
    var last_cpu_loss: f32 = 0.0;
    for (0..MEASURED) |_| {
        last_cpu_loss = try runOnce(allocator, &model_cpu, ids_cpu, targets_cpu, cfg.vocab_size, BT, &opt_cpu);
    }
    const cpu_dur = cpu_start.durationTo(std.Io.Clock.now(.awake, io));

    try w.print("measuring CUDA...\n", .{});
    const gpu_start = std.Io.Clock.now(.awake, io);
    var last_gpu_loss: f32 = 0.0;
    for (0..MEASURED) |_| {
        last_gpu_loss = try runOnce(allocator, &model_gpu, ids_gpu, targets_gpu, cfg.vocab_size, BT, &opt_gpu);
        try ctx.synchronize();
    }
    const gpu_dur = gpu_start.durationTo(std.Io.Clock.now(.awake, io));

    const cpu_ns: i96 = cpu_dur.nanoseconds;
    const gpu_ns: i96 = gpu_dur.nanoseconds;
    const cpu_ms = @as(f64, @floatFromInt(cpu_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(MEASURED));
    const gpu_ms = @as(f64, @floatFromInt(gpu_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(MEASURED));
    const speedup = cpu_ms / gpu_ms;

    try w.print("\nper-step wall-clock (averaged over {d} iters):\n", .{MEASURED});
    try w.print("  cpu:     {d:.3} ms\n", .{cpu_ms});
    try w.print("  cuda:    {d:.3} ms\n", .{gpu_ms});
    try w.print("  speedup: {d:.2}x\n", .{speedup});
    try w.print("\nfinal loss: cpu={d:.4}  cuda={d:.4}\n", .{ last_cpu_loss, last_gpu_loss });
    try w.print("playbook target: >= 30x.  {s}.\n", .{
        if (speedup >= 30.0) "TARGET MET" else "under target (see stage7 endgame doc §Risk Register)",
    });
}
