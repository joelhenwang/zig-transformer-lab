//!
//! 08_cuda_vs_cpu.zig — Full-model forward + backward + AdamW parity demo
//!
//! Purpose:
//!   Demonstrates that TinyWordTransformer produces matching outputs on
//!   CPU and CUDA across a complete training step. Mirrors the
//!   tests/integration_cuda.zig `TinyWordTransformer one training step`
//!   test, but runs as an example so a human can eyeball the printed
//!   diffs.
//!
//! Usage:
//!   zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true
//!
//! Requires:
//!   - CUDA-capable GPU visible on device 0
//!   - Kernels pre-built: `zig build kernels -Dcuda=true`
//!   - Linux host (Windows build compiles but runtime dlopen fails)
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

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;

    // --- Setup: CUDA context + PTX modules --------------------------------

    var ctx = try CudaContext.init(allocator);
    defer ctx.deinit();
    try w.print("CUDA context ready on device 0.\n", .{});

    const io = init.io;
    // Preload every PTX module the forward + backward chain might touch.
    inline for (&[_][:0]const u8{
        "elementwise", "reduce", "softmax", "unary", "embedding", "ce_loss", "adamw",
    }) |name| {
        _ = try module.loadPtxFromFile(&ctx, io, name);
    }

    // --- Two models with identical weights ---------------------------------

    const cfg = TransformerConfig{
        .vocab_size = 32,
        .d_model = 16,
        .d_ff = 64,
        .max_seq_len = 8,
    };

    var rng = Rng.init(42);
    var model_cpu = try TinyWordTransformer.init(allocator, cfg, &rng);
    defer model_cpu.deinit();
    var model_gpu = try TinyWordTransformer.init(allocator, cfg, &rng);
    defer model_gpu.deinit();

    // Copy every parameter from CPU model into GPU-destined model.
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

    // --- Deterministic inputs / targets -----------------------------------

    const B: usize = 2;
    const T: usize = 4;
    var ids_cpu = try Tensor.init(allocator, Shape.init2D(B, T));
    defer ids_cpu.deinit(allocator);
    var id_rng = Rng.init(123);
    for (0..B * T) |i| {
        ids_cpu.cpuData()[i] = @floatFromInt(id_rng.random().intRangeLessThan(usize, 0, cfg.vocab_size));
    }
    var ids_gpu = try ids_cpu.toCuda(&ctx);
    defer ids_gpu.storage.deinit(allocator);

    var targets_cpu = try Tensor.init(allocator, Shape.init1D(B * T));
    defer targets_cpu.deinit(allocator);
    var tgt_rng = Rng.init(456);
    for (0..B * T) |i| {
        targets_cpu.cpuData()[i] = @floatFromInt(tgt_rng.random().intRangeLessThan(usize, 0, cfg.vocab_size));
    }
    var targets_gpu = try targets_cpu.toCuda(&ctx);
    defer targets_gpu.storage.deinit(allocator);

    // --- CPU: forward + loss + backward + step ----------------------------

    var tape_cpu = Tape.init(allocator);
    defer tape_cpu.deinit();
    {
        var p: std.ArrayList(*Tensor) = .empty;
        defer p.deinit(allocator);
        try p.ensureTotalCapacity(allocator, 32);
        model_cpu.parameters(&p);
        for (p.items) |x| _ = try tape_cpu.trackLeaf(x);
    }

    var logits_cpu = try model_cpu.forward(ids_cpu, &tape_cpu);
    defer logits_cpu.deinit(allocator);
    var logits_2d_cpu = try ztl.ops.shape_ops.reshapeTracked(
        allocator,
        logits_cpu,
        Shape.init2D(B * T, cfg.vocab_size),
        &tape_cpu,
    );
    defer logits_2d_cpu.deinit(allocator);
    var loss_cpu = try ztl.ops.loss.crossEntropy(allocator, logits_2d_cpu, targets_cpu, &tape_cpu);
    defer loss_cpu.deinit(allocator);
    try tape_cpu.backward(&loss_cpu);

    var opt_cpu = try AdamW.init(allocator, .{});
    defer opt_cpu.deinit(allocator);
    var params_cpu: std.ArrayList(*Tensor) = .empty;
    defer params_cpu.deinit(allocator);
    try params_cpu.ensureTotalCapacity(allocator, 32);
    model_cpu.parameters(&params_cpu);
    try AdamW.step(&opt_cpu, params_cpu.items);

    // --- CUDA: same pipeline ----------------------------------------------

    var tape_gpu = Tape.init(allocator);
    defer tape_gpu.deinit();
    {
        var p: std.ArrayList(*Tensor) = .empty;
        defer p.deinit(allocator);
        try p.ensureTotalCapacity(allocator, 32);
        model_gpu.parameters(&p);
        for (p.items) |x| _ = try tape_gpu.trackLeaf(x);
    }

    var logits_gpu = try model_gpu.forward(ids_gpu, &tape_gpu);
    defer logits_gpu.storage.deinit(allocator);
    var logits_2d_gpu = try ztl.ops.shape_ops.reshapeTracked(
        allocator,
        logits_gpu,
        Shape.init2D(B * T, cfg.vocab_size),
        &tape_gpu,
    );
    defer logits_2d_gpu.storage.deinit(allocator);
    var loss_gpu = try ztl.ops.loss.crossEntropy(allocator, logits_2d_gpu, targets_gpu, &tape_gpu);
    defer loss_gpu.storage.deinit(allocator);
    try tape_gpu.backward(&loss_gpu);

    var opt_gpu = try AdamW.init(allocator, .{});
    defer opt_gpu.deinit(allocator);
    var params_gpu: std.ArrayList(*Tensor) = .empty;
    defer params_gpu.deinit(allocator);
    try params_gpu.ensureTotalCapacity(allocator, 32);
    model_gpu.parameters(&params_gpu);
    try AdamW.step(&opt_gpu, params_gpu.items);
    try ctx.synchronize();

    // --- Compare -----------------------------------------------------------

    var loss_back = try loss_gpu.toCpu(allocator);
    defer loss_back.deinit(allocator);
    try w.print("\nLoss    cpu={d:.6}  cuda={d:.6}  diff={d:.6}\n", .{
        loss_cpu.cpuData()[0],
        loss_back.cpuData()[0],
        @abs(loss_cpu.cpuData()[0] - loss_back.cpuData()[0]),
    });

    var worst_abs: f32 = 0.0;
    var worst_idx: usize = 0;
    for (0..params_cpu.items.len) |i| {
        var p_gpu_host = try params_gpu.items[i].*.toCpu(allocator);
        defer p_gpu_host.deinit(allocator);
        const n = params_cpu.items[i].cpuData().len;
        var local: f32 = 0.0;
        for (0..n) |k| {
            const d = @abs(params_cpu.items[i].cpuData()[k] - p_gpu_host.cpuData()[k]);
            if (d > local) local = d;
        }
        if (local > worst_abs) {
            worst_abs = local;
            worst_idx = i;
        }
    }
    try w.print("Post-step  worst_abs_diff={d:.6}  (param {d})\n", .{ worst_abs, worst_idx });
    try w.print("\nPlaybook budget: post-one-step abs < 2e-3.  Result: {s}.\n", .{
        if (worst_abs < 2e-3) "PASS" else "FAIL",
    });
}
