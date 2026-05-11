//!
//! zig-transformer-lab — CUDA integration tests (Stage 7)
//!
//! Scope:
//!   Linux-only. Compiled when `zig build test -Dcuda=true`. Each test
//!   exercises one layer of the CUDA backend against a real GPU on the
//!   remote RTX machine (joelwang-rtx@192.168.1.197). Tests are
//!   `error.SkipZigTest` on non-Linux hosts so the same `-Dcuda=true`
//!   build command still produces a clean test run on the Windows dev
//!   host (useful for detecting compile errors in the CUDA layer
//!   without needing to push to remote).
//!
//! Pattern:
//!   - bindings.load() is idempotent across tests; we do not unload
//!     between them. dlclose + re-dlopen can interact with CUDA's
//!     internal static state in subtle ways, so tests share the
//!     loaded library handles.
//!   - Each test creates and destroys its own CUDA resources (context,
//!     cublas handle, device buffers) with `defer` so
//!     `compute-sanitizer --leak-check=full` reports a clean run.
//!
//! Organisation:
//!   Tests are grouped by PR. PR-alpha (this PR) provides the first
//!   three; PR-beta and PR-gamma will extend this file with context-
//!   lifecycle and DeviceBuffer roundtrip tests.
//!

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const lab = @import("zig_transformer_lab");
const bindings = lab.backend.cuda.bindings;
const context = lab.backend.cuda.context;
const mem = lab.backend.cuda.mem;
const module = lab.backend.cuda.module;
const dispatch = lab.backend.cuda.dispatch;
const gemm = lab.backend.cuda.gemm;
const CudaContext = context.CudaContext;
const DeviceBuffer = mem.DeviceBuffer;
const Tensor = lab.Tensor;
const Storage = lab.Storage;
const Shape = lab.shape.Shape;
const computeStrides = lab.shape.computeStrides;
const oracle = lab.testing_utils.oracle;
const ops_elementwise = lab.ops.elementwise;
const ops_reduce = lab.ops.reduce;
const ops_matmul = lab.ops.matmul;
const ops_softmax = lab.ops.softmax;
const ops_loss = lab.ops.loss;
const Tape = lab.Tape;
const Embedding = lab.nn.embedding.Embedding;

// -- Io plumbing (shared with oracle tests; see tests/integration_oracle.zig) -----
//
// PR-zeta and later need file I/O to read zig-out/ptx/<stem>.ptx at
// runtime. The file-reading API in Zig 0.16 takes a `std.Io`
// parameter; we construct a single `std.Io.Threaded` instance per
// test process and reuse it across tests.

fn testIo() !std.Io {
    const T = struct {
        var instance: ?std.Io.Threaded = null;
    };
    if (T.instance == null) {
        T.instance = std.Io.Threaded.init(testing.allocator, .{});
    }
    return T.instance.?.io();
}

// ---------------------------------------------------------------------------
// PR-alpha: dynamic loader smoke tests
// ---------------------------------------------------------------------------

test "cuda bindings: dlopen libcuda.so.1 and libcublas.so resolves every symbol" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    try bindings.load();
    try testing.expect(bindings.loader != null);
}

test "cuda bindings: cuInit + device enumeration reports at least one NVIDIA GPU" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    try bindings.load();
    const L = bindings.loader.?;

    // cuInit is idempotent; calling it once per test keeps each test
    // independent in case test order changes.
    try bindings.check(L.cuInit(0));

    var driver_version: c_int = 0;
    try bindings.check(L.cuDriverGetVersion(&driver_version));
    std.debug.print("cuda driver version: {d}\n", .{driver_version});
    // CUDA 12.x driver version is 12000+, 13.x is 13000+. We just
    // assert non-zero here; the exact version lives in AGENTS.md.
    try testing.expect(driver_version > 0);

    var count: c_int = 0;
    try bindings.check(L.cuDeviceGetCount(&count));
    try testing.expect(count >= 1);

    var dev: bindings.CUdevice = 0;
    try bindings.check(L.cuDeviceGet(&dev, 0));

    var name_buf: [128]u8 = [_]u8{0} ** 128;
    try bindings.check(L.cuDeviceGetName(&name_buf, @intCast(name_buf.len), dev));
    const name = std.mem.sliceTo(&name_buf, 0);
    std.debug.print("cuda device 0: {s} (count={d})\n", .{ name, count });

    // Every GPU we target reports "NVIDIA" somewhere in its name.
    // "GeForce" is an additional acceptable substring (RTX 4060 Ti
    // reports "NVIDIA GeForce RTX 4060 Ti"). Anything else likely
    // means we ended up on a non-NVIDIA OpenCL fallback, which is
    // a configuration bug worth failing on.
    try testing.expect(
        std.mem.indexOf(u8, name, "NVIDIA") != null or
            std.mem.indexOf(u8, name, "GeForce") != null,
    );
}

test "cuda bindings: cuCtxCreate_v2 + cublasCreate_v2 round-trip without leaks" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    try bindings.load();
    const L = bindings.loader.?;
    try bindings.check(L.cuInit(0));

    var dev: bindings.CUdevice = 0;
    try bindings.check(L.cuDeviceGet(&dev, 0));

    var ctx: bindings.CUcontext = null;
    try bindings.check(L.cuCtxCreate_v2(&ctx, 0, dev));
    // Destroy on any exit path. The return code is discarded in the
    // cleanup path (we already know the test outcome), but in debug
    // builds CUDA would surface issues via the subsequent test's
    // cuInit call if the destroy silently failed.
    defer _ = L.cuCtxDestroy_v2(ctx);

    var handle: bindings.cublasHandle_t = null;
    try bindings.checkCublas(L.cublasCreate_v2(&handle));
    defer _ = L.cublasDestroy_v2(handle);

    // At this point: context is current, cublas handle is valid.
    // Later PRs exercise stream creation, memory allocation, and
    // kernel launch from this same setup; PR-alpha's job is simply
    // to prove the lifecycle pair works.
}

// ---------------------------------------------------------------------------
// PR-beta: CudaContext lifecycle
// ---------------------------------------------------------------------------

test "cuda context: init + deinit round-trip on device 0" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // Fresh context: device, ctx, stream, cublas all non-null /
    // valid; PTX module cache is empty.
    try testing.expect(ctx.ctx != null);
    try testing.expect(ctx.stream != null);
    try testing.expect(ctx.cublas != null);
    try testing.expectEqual(@as(usize, 0), ctx.ptx_modules.count());
}

test "cuda context: synchronize on empty stream is a no-op and returns success" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // No work submitted; synchronize is just a round-trip to the
    // driver. Should succeed immediately with no CUDA errors
    // remaining sticky afterward.
    try ctx.synchronize();
    try ctx.synchronize();
}

test "cuda context: two sequential init/deinit cycles leave no state behind" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    {
        var ctx = try CudaContext.init(testing.allocator);
        defer ctx.deinit();
        try ctx.synchronize();
    }
    // Second construction — would fail with CUDA_ERROR_INVALID_CONTEXT
    // or OUT_OF_MEMORY if the prior deinit leaked a context or stream.
    {
        var ctx = try CudaContext.init(testing.allocator);
        defer ctx.deinit();
        try ctx.synchronize();
    }
}

// ---------------------------------------------------------------------------
// PR-gamma: DeviceBuffer alloc / copy / free
// ---------------------------------------------------------------------------

test "cuda DeviceBuffer: alloc + deinit on an empty context" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    var buf = try DeviceBuffer.alloc(&ctx, 1024);
    defer buf.deinit();

    try testing.expectEqual(@as(usize, 1024), buf.len);
    try testing.expect(buf.owned);
    try testing.expect(buf.ptr != 0);
}

test "cuda DeviceBuffer: alloc(0) returns a valid, non-owning, zero-length handle" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    var buf = try DeviceBuffer.alloc(&ctx, 0);
    defer buf.deinit();

    try testing.expectEqual(@as(usize, 0), buf.len);
    try testing.expectEqual(@as(bindings.CUdeviceptr, 0), buf.ptr);
    try testing.expect(!buf.owned);
}

test "cuda DeviceBuffer: HtoD + DtoH round-trip is byte-identical (NaN included)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // NaN is the interesting case: a float memcpy preserves the exact
    // bit pattern, but `f == f` is false for any NaN. We compare via
    // @bitCast to u32 so bit-exactness is visible even for NaN payloads.
    // Other values stress the range: finite, subnormal-adjacent, and
    // extremes. f32 round-trip under memcpy is always bit-identical.
    const nan_val = std.math.nan(f32);
    const src = [_]f32{
        1.0,
        -2.5,
        1.0e6,
        1.0e-6,
        nan_val,
    };

    var buf = try DeviceBuffer.fromHost(&ctx, &src);
    defer buf.deinit();

    var dst: [5]f32 = undefined;
    try buf.copyToHost(&dst);

    for (src, dst, 0..) |s, d, i| {
        const sb: u32 = @bitCast(s);
        const db: u32 = @bitCast(d);
        if (sb != db) {
            std.debug.print(
                "mismatch at [{d}]: src=0x{x:0>8} ({e}) dst=0x{x:0>8} ({e})\n",
                .{ i, sb, s, db, d },
            );
            return error.TestUnexpectedResult;
        }
    }
}

test "cuda DeviceBuffer: toHost() allocates + copies" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    const src = [_]f32{ 3.14, 2.71, -1.0, 0.0, 42.0 };
    var buf = try DeviceBuffer.fromHost(&ctx, &src);
    defer buf.deinit();

    const host_copy = try buf.toHost(testing.allocator);
    defer testing.allocator.free(host_copy);

    try testing.expectEqual(src.len, host_copy.len);
    for (src, host_copy) |s, d| {
        try testing.expectEqual(s, d);
    }
}

test "cuda DeviceBuffer: DtoD copy preserves bytes between two buffers" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    const src_vals = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var a = try DeviceBuffer.fromHost(&ctx, &src_vals);
    defer a.deinit();

    var b = try DeviceBuffer.alloc(&ctx, src_vals.len);
    defer b.deinit();

    try b.copyFromDevice(a);

    var dst: [8]f32 = undefined;
    try b.copyToHost(&dst);
    try testing.expectEqualSlices(f32, &src_vals, &dst);
}

test "cuda DeviceBuffer: mismatched sizes return ShapeMismatch" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    var buf = try DeviceBuffer.alloc(&ctx, 10);
    defer buf.deinit();

    const too_short = [_]f32{0} ** 5;
    const too_long = [_]f32{0} ** 20;
    try testing.expectError(error.ShapeMismatch, buf.copyFromHost(&too_short));
    try testing.expectError(error.ShapeMismatch, buf.copyFromHost(&too_long));

    var mismatch_dst: [3]f32 = undefined;
    try testing.expectError(error.ShapeMismatch, buf.copyToHost(&mismatch_dst));
}

// ---------------------------------------------------------------------------
// PR-delta: Tensor with real Storage.cuda backed by DeviceBuffer
// ---------------------------------------------------------------------------

test "cuda Tensor: Storage.cuda wraps a real DeviceBuffer and passes invariants" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // Allocate 12 f32s on the device and wrap them in a Storage.cuda.
    // The DeviceBuffer is owned; transferring ownership into Storage
    // means the Tensor's storage.deinit must free it.
    const src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const dbuf = try DeviceBuffer.fromHost(&ctx, &src);
    // Do NOT defer dbuf.deinit(); ownership moves into the Tensor below.

    const s = Shape.init3D(2, 3, 2);
    var t = Tensor{
        .data = &.{}, // CPU compat alias is intentionally empty on CUDA
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cuda,
        .owned = false, // ownership of CUDA memory is inside DeviceBuffer
        .storage = .{ .cuda = dbuf },
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };

    // Structural invariants hold.
    try t.checkInvariants();

    // Storage.len reports the device-buffer element count.
    try testing.expectEqual(@as(usize, 12), t.storage.len());

    // Device tag and storage union tag agree.
    try testing.expect(t.device == .cuda);
    switch (t.storage) {
        .cuda => |b| {
            try testing.expect(b.owned);
            try testing.expect(b.ptr != 0);
            try testing.expectEqual(@as(usize, 12), b.len);
        },
        .cpu => unreachable,
    }

    // Deinit drives the CUDA free path. The arg allocator is ignored
    // on the CUDA branch; we pass testing.allocator for uniformity.
    t.storage.deinit(testing.allocator);
}

test "cuda Tensor: nonOwningStorage view over Storage.cuda preserves ctx/ptr/len, flips owned" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    const src = [_]f32{ 10, 20, 30, 40 };
    var dbuf = try DeviceBuffer.fromHost(&ctx, &src);
    defer dbuf.deinit(); // the owner lives here, not in the storage below

    // Build a Storage.cuda that is already a non-owning "view" of dbuf,
    // then exercise nonOwningStorage (which must preserve ctx/ptr/len
    // and clear any owned bit). This mirrors how view()/reshape()/
    // transpose2d() will construct child storages from CUDA parents
    // once PR-epsilon wires Tensor.toCuda.
    const owner_storage = Storage{ .cuda = .{
        .ctx = dbuf.ctx,
        .ptr = dbuf.ptr,
        .len = dbuf.len,
        .owned = false,
    } };
    const view_storage = lab.nonOwningStorage(owner_storage);

    switch (view_storage) {
        .cuda => |b| {
            try testing.expectEqual(dbuf.ctx, b.ctx);
            try testing.expectEqual(dbuf.ptr, b.ptr);
            try testing.expectEqual(dbuf.len, b.len);
            try testing.expect(!b.owned);
        },
        .cpu => unreachable,
    }
}

// ---------------------------------------------------------------------------
// PR-epsilon: Tensor.toCuda / toCpu round-trip
// ---------------------------------------------------------------------------

test "cuda Tensor.toCuda rejects a CUDA source" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // Build a real CUDA tensor first, then assert toCuda refuses to
    // re-transfer it. This surfaces caller mistakes instead of quietly
    // doubling GPU memory use.
    const src = [_]f32{ 1, 2, 3, 4 };
    const dbuf = try DeviceBuffer.fromHost(&ctx, &src);
    var gpu = Tensor{
        .data = &.{},
        .shape = Shape.init1D(4),
        .strides = computeStrides(Shape.init1D(4)),
        .dtype = .f32,
        .device = .cuda,
        .owned = false,
        .storage = .{ .cuda = dbuf },
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    defer gpu.storage.deinit(testing.allocator);

    try testing.expectError(error.DeviceMismatch, gpu.toCuda(&ctx));
}

test "cuda Tensor.toCpu rejects a CPU source" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;

    var cpu = try Tensor.init(testing.allocator, Shape.init1D(4));
    defer cpu.deinit(testing.allocator);

    try testing.expectError(error.DeviceMismatch, cpu.toCpu(testing.allocator));
}

test "cuda Tensor.toCuda / toCpu round-trip is byte-identical for a contiguous tensor" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // Fill a (2, 3) CPU tensor with known values, round-trip via CUDA,
    // expect byte-identical result (memcpy on f32 preserves bit
    // patterns — any mismatch here is a hard bug, not f32 rounding).
    var cpu = try Tensor.init(testing.allocator, Shape.init2D(2, 3));
    defer cpu.deinit(testing.allocator);
    for (cpu.data, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var gpu = try cpu.toCuda(&ctx);
    defer gpu.storage.deinit(testing.allocator);

    // Top-level invariants on the GPU tensor: CPU compat alias empty,
    // device tag cuda, storage.len matches the full source buffer.
    try testing.expectEqual(@as(usize, 0), gpu.data.len);
    try testing.expect(gpu.device == .cuda);
    try testing.expectEqual(cpu.data.len, gpu.storage.len());
    try testing.expectEqual(cpu.offset, gpu.offset);
    try testing.expect(lab.shape.equals(cpu.shape, gpu.shape));

    var back = try gpu.toCpu(testing.allocator);
    defer back.deinit(testing.allocator);

    try testing.expectEqualSlices(f32, cpu.data, back.data);
    try testing.expect(lab.shape.equals(cpu.shape, back.shape));
    try testing.expectEqual(cpu.strides.values[0], back.strides.values[0]);
    try testing.expectEqual(cpu.strides.values[1], back.strides.values[1]);
    try testing.expectEqual(cpu.offset, back.offset);
}

test "cuda Tensor.toCuda preserves transposed (non-contiguous) strides" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // Build a (3, 2) tensor with known values, transpose to (2, 3)
    // view, round-trip via CUDA. The view's strides are (1, 2) — NOT
    // row-major — so this test is the one the playbook flagged: copy
    // the whole backing buffer, not just totalElements(shape), or
    // CUDA-side stride walks go off the end.
    var parent = try Tensor.init(testing.allocator, Shape.init2D(3, 2));
    defer parent.deinit(testing.allocator);
    for (parent.data, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    const transposed = try parent.transpose2d();
    // transposed: shape (2, 3), strides (1, 2), storage.len == 6,
    //            owned=false (view into parent).

    const gpu_view = try transposed.toCuda(&ctx);
    // gpu_view: shape (2, 3), strides (1, 2), storage.cuda.len == 6,
    //           owned=true on the embedded DeviceBuffer, but top-level
    //           owned=false per the PR-delta CUDA invariant.
    var gpu_view_mut = gpu_view;
    defer gpu_view_mut.storage.deinit(testing.allocator);

    try testing.expect(lab.shape.equals(transposed.shape, gpu_view.shape));
    try testing.expectEqual(transposed.strides.values[0], gpu_view.strides.values[0]);
    try testing.expectEqual(transposed.strides.values[1], gpu_view.strides.values[1]);
    try testing.expectEqual(@as(usize, 6), gpu_view.storage.len());

    // Round-trip back to CPU and confirm element-wise equality at each
    // logical index. Because strides were preserved, iterating (i, j)
    // on either side reads the same element.
    var back = try gpu_view.toCpu(testing.allocator);
    defer back.deinit(testing.allocator);

    for (0..2) |i| {
        for (0..3) |j| {
            const expected = transposed.at(&[_]usize{ i, j });
            const actual = back.at(&[_]usize{ i, j });
            try testing.expectEqual(expected, actual);
        }
    }

    // Parent is untouched (transfer is a copy, not a move).
    for (parent.data, 0..) |v, i| {
        try testing.expectEqual(@as(f32, @floatFromInt(i + 1)), v);
    }
}

test "cuda Tensor.toCuda preserves requires_grad and param_id" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    // A parameter-like CPU tensor: requires_grad=true, with a param_id
    // set. toCuda must carry both through so AdamW state keyed by
    // ParamId keeps matching across the device transfer.
    var cpu = try Tensor.init(testing.allocator, Shape.init1D(4));
    defer cpu.deinit(testing.allocator);
    cpu.requires_grad = true;
    cpu.param_id = 42;
    for (cpu.data, 0..) |*v, i| v.* = @floatFromInt(i);

    var gpu = try cpu.toCuda(&ctx);
    defer gpu.storage.deinit(testing.allocator);

    try testing.expect(gpu.requires_grad);
    try testing.expectEqual(@as(?u32, 42), gpu.param_id);
    // grad and tape_node are reset on transfer (see toCuda doc comment).
    try testing.expect(gpu.grad == null);
    try testing.expect(gpu.tape_node == null);
}

// ---------------------------------------------------------------------------
// PR-zeta: PTX loader + vector_add smoke kernel
// ---------------------------------------------------------------------------

test "cuda PTX loader: loadPtxFromFile caches modules by stem" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    // First call reads from disk + cuModuleLoadData; second call must
    // hit the cache (identical handle, no extra allocation in the map).
    const m1 = try module.loadPtxFromFile(&ctx, io, "vector_add");
    try testing.expectEqual(@as(usize, 1), ctx.ptx_modules.count());
    const m2 = try module.loadPtxFromFile(&ctx, io, "vector_add");
    try testing.expectEqual(m1, m2);
    try testing.expectEqual(@as(usize, 1), ctx.ptx_modules.count());
}

test "cuda PTX loader: getKernel resolves extern C symbol" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    _ = try module.loadPtxFromFile(&ctx, io, "vector_add");
    const kfn = try module.getKernel(&ctx, "vector_add", "vector_add");
    try testing.expect(kfn != null);
}

test "cuda PTX loader: getKernel on a missing kernel name returns CudaError" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    _ = try module.loadPtxFromFile(&ctx, io, "vector_add");
    try testing.expectError(
        error.CudaError,
        module.getKernel(&ctx, "vector_add", "nonexistent_kernel_xyz"),
    );
}

test "cuda vector_add kernel: 1024 f32 elementwise add matches CPU reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    _ = try module.loadPtxFromFile(&ctx, io, "vector_add");
    const kfn = try module.getKernel(&ctx, "vector_add", "vector_add");

    const N: usize = 1024;
    const a_host = try testing.allocator.alloc(f32, N);
    defer testing.allocator.free(a_host);
    const b_host = try testing.allocator.alloc(f32, N);
    defer testing.allocator.free(b_host);

    // Seeded deterministic inputs. Mix of positive, negative, and
    // zero values to exercise sign handling; avoid values near the
    // f32 precision edge so the CPU reference is bit-identical to
    // the GPU result (FADD on sm_89 is IEEE-compliant without
    // --use_fast_math).
    var rng = std.Random.DefaultPrng.init(0xC0FFEE);
    const r = rng.random();
    for (0..N) |i| {
        a_host[i] = (r.float(f32) - 0.5) * 100.0;
        b_host[i] = (r.float(f32) - 0.5) * 100.0;
    }

    var buf_a = try DeviceBuffer.fromHost(&ctx, a_host);
    defer buf_a.deinit();
    var buf_b = try DeviceBuffer.fromHost(&ctx, b_host);
    defer buf_b.deinit();
    var buf_c = try DeviceBuffer.alloc(&ctx, N);
    defer buf_c.deinit();

    // Kernel argument packing: cuLaunchKernel wants an array of
    // pointers, one per argument. Each pointer must outlive the call.
    // We pack locals here and pass addresses into the params array.
    //
    // The CUdeviceptr values must be passed BY POINTER even though
    // they are already integer handles, because cuLaunchKernel reads
    // each parameter's *bytes* by dereferencing the pointer in the
    // kernelParams array.
    const n_arg: c_uint = @intCast(N);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&buf_a.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&buf_b.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&buf_c.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
    };

    // 1D launch: block size 256, grid size ceil(N / 256). Matches the
    // standard pattern documented in the module.zig header.
    const block_x: c_uint = 256;
    const grid_x: c_uint = @intCast((N + 255) / 256);
    try module.launch(
        &ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ block_x, 1, 1 },
        0,
        &args,
    );
    // Sync so any async kernel error surfaces here, at the launch site.
    try ctx.synchronize();

    // Copy result back and compare element-wise with CPU reference.
    const c_host = try testing.allocator.alloc(f32, N);
    defer testing.allocator.free(c_host);
    try buf_c.copyToHost(c_host);

    for (0..N) |i| {
        const expected = a_host[i] + b_host[i];
        try testing.expectEqual(expected, c_host[i]);
    }
}

test "cuda vector_add kernel: bounds check prevents OOB for N not divisible by block size" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    _ = try module.loadPtxFromFile(&ctx, io, "vector_add");
    const kfn = try module.getKernel(&ctx, "vector_add", "vector_add");

    // N = 1000 gives grid_x = ceil(1000/256) = 4 blocks of 256 threads
    // each, covering 1024 threads — 24 of which have i >= N and must
    // early-return without touching c[i]. This test is the PR-zeta
    // gotcha backstop: a kernel without the bounds check would
    // corrupt 24 bytes past c's end, which compute-sanitizer would
    // catch but the test would otherwise miss.
    const N: usize = 1000;
    const a_host = try testing.allocator.alloc(f32, N);
    defer testing.allocator.free(a_host);
    const b_host = try testing.allocator.alloc(f32, N);
    defer testing.allocator.free(b_host);
    for (0..N) |i| {
        a_host[i] = @floatFromInt(i);
        b_host[i] = @floatFromInt(i * 2);
    }

    var buf_a = try DeviceBuffer.fromHost(&ctx, a_host);
    defer buf_a.deinit();
    var buf_b = try DeviceBuffer.fromHost(&ctx, b_host);
    defer buf_b.deinit();
    var buf_c = try DeviceBuffer.alloc(&ctx, N);
    defer buf_c.deinit();

    const n_arg: c_uint = @intCast(N);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&buf_a.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&buf_b.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&buf_c.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
    };

    const block_x: c_uint = 256;
    const grid_x: c_uint = @intCast((N + 255) / 256);
    try module.launch(
        &ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ block_x, 1, 1 },
        0,
        &args,
    );
    try ctx.synchronize();

    const c_host = try testing.allocator.alloc(f32, N);
    defer testing.allocator.free(c_host);
    try buf_c.copyToHost(c_host);

    for (0..N) |i| {
        const expected = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i * 2));
        try testing.expectEqual(expected, c_host[i]);
    }
}

// ---------------------------------------------------------------------------
// PR-eta: elementwise CUDA ops (forward-only)
// ---------------------------------------------------------------------------
//
// These tests exercise src/backend/cuda/dispatch.zig. Every test
// pre-loads the "elementwise" PTX module (dispatch does not auto-load
// per the design decision in dispatch.zig's header) and then runs a
// CUDA op against small tensors, comparing against either a PyTorch
// oracle fixture (for the add_2d case) or a CPU-computed reference
// (for the other six ops, which do not have dedicated oracle
// fixtures yet — those land in a follow-up oracle expansion).

const FIXTURE_ROOT = "tests/fixtures";

fn fixturePath(comptime case: []const u8, comptime file: []const u8) []const u8 {
    return FIXTURE_ROOT ++ "/" ++ case ++ "/" ++ file;
}

/// Helper: build a small contiguous CPU tensor from a fixed slice,
/// push it to CUDA, and return the CUDA tensor. Caller must
/// `storage.deinit(alloc)` the returned tensor.
fn cudaFromSlice(
    ctx: *const CudaContext,
    s: Shape,
    values: []const f32,
) !Tensor {
    var cpu = try Tensor.init(testing.allocator, s);
    defer cpu.deinit(testing.allocator);
    std.debug.assert(cpu.data.len == values.len);
    @memcpy(cpu.data, values);
    return try cpu.toCuda(ctx);
}

test "cuda dispatch add: oracle add_2d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;

    // Load the PyTorch fixtures (same inputs as the CPU oracle test
    // "oracle add_2d: forward and backward parity" in
    // tests/integration_oracle.zig).
    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var b_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_1.ztlt"));
    defer b_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);

    // Transfer inputs to CUDA.
    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    var b_gpu = try b_cpu.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);

    // Run CUDA add, bring result back to host.
    var y_gpu = try dispatch.add(a_gpu, b_gpu);
    defer y_gpu.storage.deinit(alloc);
    // Debug: synchronise before reading back so any async launch
    // error surfaces at this line. Release builds would skip this.
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);

    // Elementwise parity vs oracle. FADD on sm_89 without fast-math
    // is IEEE compliant; tolerance mirrors docs/stage7_plan.md
    // Section 7.2 "Elementwise" row.
    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

test "cuda dispatch sub/mul/div: small fixed inputs match CPU reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const s = Shape.init2D(2, 3);
    const a_vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_vals = [_]f32{ 10, 20, 30, 40, 50, 60 };

    // sub: a - b
    {
        var a = try cudaFromSlice(&ctx, s, &a_vals);
        defer a.storage.deinit(testing.allocator);
        var b = try cudaFromSlice(&ctx, s, &b_vals);
        defer b.storage.deinit(testing.allocator);
        var r = try dispatch.sub(a, b);
        defer r.storage.deinit(testing.allocator);
        var r_cpu = try r.toCpu(testing.allocator);
        defer r_cpu.deinit(testing.allocator);
        for (0..6) |i| {
            try testing.expectEqual(a_vals[i] - b_vals[i], r_cpu.data[i]);
        }
    }
    // mul: a * b
    {
        var a = try cudaFromSlice(&ctx, s, &a_vals);
        defer a.storage.deinit(testing.allocator);
        var b = try cudaFromSlice(&ctx, s, &b_vals);
        defer b.storage.deinit(testing.allocator);
        var r = try dispatch.mul(a, b);
        defer r.storage.deinit(testing.allocator);
        var r_cpu = try r.toCpu(testing.allocator);
        defer r_cpu.deinit(testing.allocator);
        for (0..6) |i| {
            try testing.expectEqual(a_vals[i] * b_vals[i], r_cpu.data[i]);
        }
    }
    // div: a / b
    {
        var a = try cudaFromSlice(&ctx, s, &a_vals);
        defer a.storage.deinit(testing.allocator);
        var b = try cudaFromSlice(&ctx, s, &b_vals);
        defer b.storage.deinit(testing.allocator);
        var r = try dispatch.div(a, b);
        defer r.storage.deinit(testing.allocator);
        var r_cpu = try r.toCpu(testing.allocator);
        defer r_cpu.deinit(testing.allocator);
        for (0..6) |i| {
            // FDIV is not guaranteed bit-identical between hardware
            // FPUs; allow a tiny ulp slack. The values we picked
            // (1/10, 2/20, 3/30, ...) all simplify to 0.1 which
            // round-trips the same on any IEEE implementation, but
            // we leave headroom for robustness.
            const expected = a_vals[i] / b_vals[i];
            try testing.expectApproxEqAbs(expected, r_cpu.data[i], 1e-6);
        }
    }
}

test "cuda dispatch neg / addScalar / mulScalar: small fixed inputs" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const s = Shape.init1D(5);
    const a_vals = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };

    {
        var a = try cudaFromSlice(&ctx, s, &a_vals);
        defer a.storage.deinit(testing.allocator);
        var r = try dispatch.neg(a);
        defer r.storage.deinit(testing.allocator);
        var r_cpu = try r.toCpu(testing.allocator);
        defer r_cpu.deinit(testing.allocator);
        for (0..5) |i| try testing.expectEqual(-a_vals[i], r_cpu.data[i]);
    }
    {
        var a = try cudaFromSlice(&ctx, s, &a_vals);
        defer a.storage.deinit(testing.allocator);
        var r = try dispatch.addScalar(a, 3.5);
        defer r.storage.deinit(testing.allocator);
        var r_cpu = try r.toCpu(testing.allocator);
        defer r_cpu.deinit(testing.allocator);
        for (0..5) |i| try testing.expectEqual(a_vals[i] + 3.5, r_cpu.data[i]);
    }
    {
        var a = try cudaFromSlice(&ctx, s, &a_vals);
        defer a.storage.deinit(testing.allocator);
        var r = try dispatch.mulScalar(a, -0.5);
        defer r.storage.deinit(testing.allocator);
        var r_cpu = try r.toCpu(testing.allocator);
        defer r_cpu.deinit(testing.allocator);
        for (0..5) |i| try testing.expectEqual(a_vals[i] * -0.5, r_cpu.data[i]);
    }
}

test "cuda dispatch add: shape mismatch returns ShapeMismatch" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    var a = try cudaFromSlice(&ctx, Shape.init1D(3), &[_]f32{ 1, 2, 3 });
    defer a.storage.deinit(testing.allocator);
    var b = try cudaFromSlice(&ctx, Shape.init1D(4), &[_]f32{ 1, 2, 3, 4 });
    defer b.storage.deinit(testing.allocator);

    try testing.expectError(error.ShapeMismatch, dispatch.add(a, b));
}

test "cuda dispatch add: CPU input returns DeviceMismatch" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    // a lives on CPU, b on CUDA. The mismatch check fires before any
    // shape / layout checks.
    var a = try Tensor.init(testing.allocator, Shape.init1D(3));
    defer a.deinit(testing.allocator);
    var b = try cudaFromSlice(&ctx, Shape.init1D(3), &[_]f32{ 1, 2, 3 });
    defer b.storage.deinit(testing.allocator);

    try testing.expectError(error.DeviceMismatch, dispatch.add(a, b));
}

// ---------------------------------------------------------------------------
// PR-eta.2: ops_elementwise.add routes to CUDA dispatch + tape records
// ---------------------------------------------------------------------------
//
// PR-eta shipped CUDA kernels behind a separate dispatch module
// (backend.cuda.dispatch). PR-eta.2 wires the CPU-facing op surface
// (src/tensor/ops/elementwise.zig) so existing call sites get CUDA
// execution transparently when they pass CUDA tensors. It also
// extends tape.cloneTensorData to DtoD-copy CUDA saved tensors into
// tape-owned DeviceBuffers (so backward can read them after the
// caller frees the sources).
//
// Full backward parity vs oracle add_2d grad_input_*.ztlt requires
// (a) a CUDA code path through ops_reduce.sumToShape for the
// identity case and (b) either a CUDA sumAll kernel for the
// standard backwardThroughSum idiom or a manual grad_seed injection
// helper. Both are scoped to a later PR; this test asserts only the
// forward-with-tape plumbing.

test "cuda ops_elementwise.add: routes to CUDA and records on tape" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;

    // Build two CUDA tensors with requires_grad=true so ops_elementwise.add
    // records on the tape.
    var a_gpu = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer a_gpu.storage.deinit(alloc);
    a_gpu.requires_grad = true;
    var b_gpu = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 10, 20, 30, 40, 50, 60 });
    defer b_gpu.storage.deinit(alloc);
    b_gpu.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a_gpu);
    _ = try tape.trackLeaf(&b_gpu);

    // The call routes to cuda_dispatch.add (PR-eta.2) and then
    // recordBinaryOp clones the inputs into tape-owned DeviceBuffers
    // via cloneTensorData's new .cuda branch.
    var out = try ops_elementwise.add(alloc, a_gpu, b_gpu, &tape);
    defer out.storage.deinit(alloc);

    // Output is a CUDA tensor of the same shape.
    try testing.expect(out.device == .cuda);
    try testing.expect(out.requires_grad);
    try testing.expect(out.tape_node != null);

    // Sanity-check forward math by round-tripping to host.
    try ctx.synchronize();
    var out_cpu = try out.toCpu(alloc);
    defer out_cpu.deinit(alloc);
    for (0..6) |i| {
        const a_vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
        const b_vals = [_]f32{ 10, 20, 30, 40, 50, 60 };
        try testing.expectEqual(a_vals[i] + b_vals[i], out_cpu.data[i]);
    }

    // Tape state: 2 leaf nodes + 1 add node = 3 total.
    try testing.expectEqual(@as(usize, 3), tape.nodes.items.len);
    // The add node owns two kept-alive CUDA buffers (copies of a, b).
    try testing.expectEqual(@as(usize, 2), tape.kept_alive_cuda.items.len);
    // No CPU buffers kept alive on this purely-CUDA path.
    try testing.expectEqual(@as(usize, 0), tape.kept_alive.items.len);
}

test "cuda ops_elementwise.add: tape keeps CUDA buffers alive after source deinit" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;

    var tape = Tape.init(alloc);
    defer tape.deinit();

    // Scope: build sources, add with tape, then destroy sources. The
    // tape-owned snapshots in kept_alive_cuda must still be readable
    // after the scope exits (compute-sanitizer catches the
    // use-after-free otherwise).
    {
        var a_gpu = try cudaFromSlice(&ctx, Shape.init1D(4), &[_]f32{ 1, 2, 3, 4 });
        defer a_gpu.storage.deinit(alloc);
        a_gpu.requires_grad = true;
        var b_gpu = try cudaFromSlice(&ctx, Shape.init1D(4), &[_]f32{ 10, 20, 30, 40 });
        defer b_gpu.storage.deinit(alloc);
        b_gpu.requires_grad = true;

        _ = try tape.trackLeaf(&a_gpu);
        _ = try tape.trackLeaf(&b_gpu);

        var out = try ops_elementwise.add(alloc, a_gpu, b_gpu, &tape);
        out.storage.deinit(alloc); // free the output eagerly too
    }

    // After the inner scope returns:
    //  - a_gpu and b_gpu are deinited (their device buffers freed).
    //  - The tape's kept_alive_cuda still owns copies. If
    //    cloneTensorData had forgotten to allocate a fresh buffer
    //    and instead aliased the caller's, the next line would
    //    dereference a freed CUdeviceptr.
    try testing.expectEqual(@as(usize, 2), tape.kept_alive_cuda.items.len);

    // Read one of the kept-alive buffers back via a non-owning
    // DeviceBuffer view to confirm it still holds the original bytes.
    const saved = tape.kept_alive_cuda.items[0];
    var host_back: [4]f32 = undefined;
    const view = DeviceBuffer{
        .ctx = saved.ctx,
        .ptr = saved.ptr,
        .len = saved.len,
        .owned = false,
    };
    try view.copyToHost(&host_back);
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &host_back);
}

// ---------------------------------------------------------------------------
// PR-theta: broadcasting elementwise ops on CUDA (forward)
// ---------------------------------------------------------------------------

test "cuda dispatch addBroadcast: oracle add_broadcast_2d_1d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;

    // Oracle fixture: (2,3) + (3,) -> (2,3) broadcast add
    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var b_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "input_1.ztlt"));
    defer b_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "output.ztlt"));
    defer expect_y.deinit(alloc);

    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    var b_gpu = try b_cpu.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.addBroadcast(a_gpu, b_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);

    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

test "cuda ops_elementwise.add: picks fast path vs broadcast path" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;

    // Same-shape contiguous: fast path.
    {
        var a = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
        defer a.storage.deinit(alloc);
        var b = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 10, 20, 30, 40, 50, 60 });
        defer b.storage.deinit(alloc);
        var y = try ops_elementwise.add(alloc, a, b, null);
        defer y.storage.deinit(alloc);
        var y_cpu = try y.toCpu(alloc);
        defer y_cpu.deinit(alloc);
        for (0..6) |i| {
            const a_vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
            const b_vals = [_]f32{ 10, 20, 30, 40, 50, 60 };
            try testing.expectEqual(a_vals[i] + b_vals[i], y_cpu.data[i]);
        }
    }

    // (2,3) + (3,) broadcast: stride-aware path.
    {
        var a = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
        defer a.storage.deinit(alloc);
        var b = try cudaFromSlice(&ctx, Shape.init1D(3), &[_]f32{ 100, 200, 300 });
        defer b.storage.deinit(alloc);
        var y = try ops_elementwise.add(alloc, a, b, null);
        defer y.storage.deinit(alloc);
        var y_cpu = try y.toCpu(alloc);
        defer y_cpu.deinit(alloc);
        // Expected: [1+100, 2+200, 3+300, 4+100, 5+200, 6+300]
        const expected = [_]f32{ 101, 202, 303, 104, 205, 306 };
        for (0..6) |i| try testing.expectEqual(expected[i], y_cpu.data[i]);
    }
}

test "cuda dispatch mulBroadcast: (1,3) * (2,3) expands the size-1 axis" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;
    var a = try cudaFromSlice(&ctx, Shape.init2D(1, 3), &[_]f32{ 2, 3, 5 });
    defer a.storage.deinit(alloc);
    var b = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 10, 100, 1000, 10000, 100000 });
    defer b.storage.deinit(alloc);

    var y = try dispatch.mulBroadcast(a, b);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    // Expected: [2*1, 3*10, 5*100, 2*1000, 3*10000, 5*100000]
    const expected = [_]f32{ 2, 30, 500, 2000, 30000, 500000 };
    for (0..6) |i| try testing.expectEqual(expected[i], y_cpu.data[i]);
}

// ---------------------------------------------------------------------------
// PR-iota: reductions (sumAll, sumAxis, broadcastTo, sumToShape)
// ---------------------------------------------------------------------------

test "cuda dispatch sumAll: contiguous input sums to scalar (oracle add_2d input_0)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;

    // Reuse the add_2d input_0 fixture for a realistic sum check.
    var x_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_0.ztlt"));
    defer x_cpu.deinit(alloc);

    // CPU reference.
    var cpu_sum: f32 = 0;
    for (x_cpu.data) |v| cpu_sum += v;

    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);

    var s_gpu = try dispatch.sumAll(x_gpu);
    defer s_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var s_cpu = try s_gpu.toCpu(alloc);
    defer s_cpu.deinit(alloc);
    // atomicAdd ordering means we accept a small relative error.
    try testing.expectApproxEqRel(cpu_sum, s_cpu.data[0], 1e-4);
}

test "cuda dispatch sumAxis: (2,3) axis=0 gives (1,3)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;
    var x = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer x.storage.deinit(alloc);

    var s = try dispatch.sumAxis(x, 0);
    defer s.storage.deinit(alloc);
    try ctx.synchronize();

    try testing.expectEqual(@as(usize, 2), s.shape.ndim());
    try testing.expectEqual(@as(usize, 1), s.shape.dims[0]);
    try testing.expectEqual(@as(usize, 3), s.shape.dims[1]);

    var s_cpu = try s.toCpu(alloc);
    defer s_cpu.deinit(alloc);
    // columns: [1+4, 2+5, 3+6] = [5, 7, 9]
    try testing.expectEqual(@as(f32, 5), s_cpu.data[0]);
    try testing.expectEqual(@as(f32, 7), s_cpu.data[1]);
    try testing.expectEqual(@as(f32, 9), s_cpu.data[2]);
}

test "cuda dispatch sumAxis: (2,3) axis=1 gives (2,1)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;
    var x = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer x.storage.deinit(alloc);

    var s = try dispatch.sumAxis(x, 1);
    defer s.storage.deinit(alloc);
    try ctx.synchronize();

    var s_cpu = try s.toCpu(alloc);
    defer s_cpu.deinit(alloc);
    // rows: [1+2+3, 4+5+6] = [6, 15]
    try testing.expectEqual(@as(f32, 6), s_cpu.data[0]);
    try testing.expectEqual(@as(f32, 15), s_cpu.data[1]);
}

test "cuda dispatch broadcastTo: scalar -> (2,3) produces an all-ones tensor" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;
    var scalar = try cudaFromSlice(&ctx, Shape.init1D(1), &[_]f32{7.5});
    defer scalar.storage.deinit(alloc);

    var out = try dispatch.broadcastTo(scalar, Shape.init2D(2, 3));
    defer out.storage.deinit(alloc);
    try ctx.synchronize();

    var out_cpu = try out.toCpu(alloc);
    defer out_cpu.deinit(alloc);
    for (0..6) |i| try testing.expectEqual(@as(f32, 7.5), out_cpu.data[i]);
}

test "cuda dispatch sumToShape: (2,3) -> (3,) collapses leading broadcasted dim" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;
    var grad = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer grad.storage.deinit(alloc);

    var r = try dispatch.sumToShape(grad, Shape.init1D(3));
    defer r.storage.deinit(alloc);
    try ctx.synchronize();

    var r_cpu = try r.toCpu(alloc);
    defer r_cpu.deinit(alloc);
    try testing.expectEqual(@as(usize, 3), r_cpu.data.len);
    // columns: [1+4, 2+5, 3+6] = [5, 7, 9]
    try testing.expectEqual(@as(f32, 5), r_cpu.data[0]);
    try testing.expectEqual(@as(f32, 7), r_cpu.data[1]);
    try testing.expectEqual(@as(f32, 9), r_cpu.data[2]);
}

test "cuda oracle add_2d: forward + backward parity via ops_elementwise.add" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;

    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var b_cpu = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_1.ztlt"));
    defer b_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    // Transfer to CUDA and mark requires_grad.
    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    a_gpu.requires_grad = true;
    var b_gpu = try b_cpu.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);
    b_gpu.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a_gpu);
    _ = try tape.trackLeaf(&b_gpu);

    // Forward.
    var y = try ops_elementwise.add(alloc, a_gpu, b_gpu, &tape);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    // Forward parity vs oracle.
    {
        var y_cpu = try y.toCpu(alloc);
        defer y_cpu.deinit(alloc);
        try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    }

    // Loss = sumAll(y). Registers a reduction node on the tape.
    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try ctx.synchronize();

    // Backward seeds grad[loss] = ones() on CUDA (via onesLike),
    // walks back through the sum node -> broadcastTo -> grad_y =
    // ones(y.shape), then through the add node -> sumToShape
    // (identity for same-shape) -> grad_a = grad_b = ones(a.shape).
    try tape.backward(&loss);
    try ctx.synchronize();

    // Read gradients back to host for parity.
    var da_cpu = try a_gpu.grad.?.*.toCpu(alloc);
    defer da_cpu.deinit(alloc);
    var db_cpu = try b_gpu.grad.?.*.toCpu(alloc);
    defer db_cpu.deinit(alloc);

    try oracle.expectClose(da_cpu, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try oracle.expectClose(db_cpu, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// ---------------------------------------------------------------------------
// PR-kappa: cuBLAS row-major GEMM (matmul, matmulBatch)
// ---------------------------------------------------------------------------
//
// Every test here uses deliberately asymmetric M/N/K dimensions.
// A wrong transa/transb/operand-swap in the row-major -> col-major
// translation would yield either the wrong output shape (caught
// at the dims check) or the wrong values on the first call. See
// docs/08_backends_cuda.md §3.5, §3.7, §11 for the derivation and
// test-shape rationale.

test "cuda gemm matmul: asymmetric (4,5) @ (5,3) hand-computed reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    const alloc = testing.allocator;

    // Simple integer-valued inputs so the CPU reference computation
    // is exact under f32 (no rounding).
    //   a : (4, 5)  row-major: each row is 1,2,3,4,5
    //   b : (5, 3)  row-major: column i has value i+1 in every row
    // Expected:  a @ b  where (a@b)[i,j] = sum_k a[i,k] * b[k,j]
    //           = (1+2+3+4+5) * b[:,j] summed appropriately
    var a_vals: [20]f32 = undefined;
    for (0..4) |i| {
        for (0..5) |k| a_vals[i * 5 + k] = @floatFromInt(k + 1); // rows = 1,2,3,4,5
    }
    var b_vals: [15]f32 = undefined;
    for (0..5) |k| {
        for (0..3) |j| b_vals[k * 3 + j] = @floatFromInt(j + 1); // col j = j+1 everywhere
    }
    var a = try cudaFromSlice(&ctx, Shape.init2D(4, 5), &a_vals);
    defer a.storage.deinit(alloc);
    var b = try cudaFromSlice(&ctx, Shape.init2D(5, 3), &b_vals);
    defer b.storage.deinit(alloc);

    var c = try gemm.matmul(a, b);
    defer c.storage.deinit(alloc);
    try ctx.synchronize();

    var c_cpu = try c.toCpu(alloc);
    defer c_cpu.deinit(alloc);
    try testing.expectEqual(@as(usize, 2), c_cpu.shape.ndim());
    try testing.expectEqual(@as(usize, 4), c_cpu.shape.dims[0]);
    try testing.expectEqual(@as(usize, 3), c_cpu.shape.dims[1]);

    // Hand reference: for every (i, j) we have
    //   c[i,j] = sum_k (k+1) * (j+1) = (j+1) * sum_k (k+1) = (j+1) * 15
    for (0..4) |i| {
        for (0..3) |j| {
            const expected: f32 = @as(f32, @floatFromInt(j + 1)) * 15.0;
            const got = c_cpu.at(&[_]usize{ i, j });
            try testing.expectEqual(expected, got);
        }
    }
}

test "cuda gemm matmul: oracle matmul_2d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    const alloc = testing.allocator;

    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var b_cpu = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "input_1.ztlt"));
    defer b_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);

    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    var b_gpu = try b_cpu.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);

    var y_gpu = try gemm.matmul(a_gpu, b_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
}

test "cuda gemm matmul: shape/layout validation" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();

    const alloc = testing.allocator;
    var a = try cudaFromSlice(&ctx, Shape.init2D(2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer a.storage.deinit(alloc);
    // K mismatch: a is (2,3), b is (4,5) — can't multiply.
    var bad = try cudaFromSlice(&ctx, Shape.init2D(4, 5), &[_]f32{0} ** 20);
    defer bad.storage.deinit(alloc);
    try testing.expectError(error.ShapeMismatch, gemm.matmul(a, bad));

    // Rank mismatch.
    var three_d = try cudaFromSlice(&ctx, Shape.init3D(1, 2, 3), &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer three_d.storage.deinit(alloc);
    try testing.expectError(error.ShapeMismatch, gemm.matmul(a, three_d));
}

test "cuda gemm matmulBatch: oracle matmul_batch_3d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();

    const alloc = testing.allocator;

    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var b_cpu = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "input_1.ztlt"));
    defer b_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);

    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    var b_gpu = try b_cpu.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);

    var y_gpu = try gemm.matmulBatch(a_gpu, b_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
}

test "cuda ops_matmul.matmul: oracle matmul_2d forward + backward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;

    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var b_cpu = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "input_1.ztlt"));
    defer b_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    a_gpu.requires_grad = true;
    var b_gpu = try b_cpu.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);
    b_gpu.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a_gpu);
    _ = try tape.trackLeaf(&b_gpu);

    var y = try ops_matmul.matmul(alloc, a_gpu, b_gpu, &tape);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    {
        var y_cpu = try y.toCpu(alloc);
        defer y_cpu.deinit(alloc);
        try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
    }

    // Loss = sumAll(y); backward computes dL/dA = dL/dC @ B^T,
    // dL/dB = A^T @ dL/dC via more matmuls that also route through
    // the CUDA dispatch.
    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try tape.backward(&loss);
    try ctx.synchronize();

    var da_cpu = try a_gpu.grad.?.*.toCpu(alloc);
    defer da_cpu.deinit(alloc);
    var db_cpu = try b_gpu.grad.?.*.toCpu(alloc);
    defer db_cpu.deinit(alloc);

    try oracle.expectClose(da_cpu, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
    try oracle.expectClose(db_cpu, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
}

// ---------------------------------------------------------------------------
// PR-lambda: softmax / log-softmax over the last axis
// ---------------------------------------------------------------------------

test "cuda dispatch softmaxLastAxis: oracle softmax_3d_last_axis forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "softmax");

    const alloc = testing.allocator;
    var x_cpu = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "input_0.ztlt"));
    defer x_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "output.ztlt"));
    defer expect_y.deinit(alloc);

    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.softmaxLastAxis(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
}

test "cuda dispatch logSoftmaxLastAxis: oracle log_softmax_3d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "softmax");

    const alloc = testing.allocator;
    var x_cpu = try oracle.loadTensor(alloc, io, fixturePath("log_softmax_3d", "input_0.ztlt"));
    defer x_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("log_softmax_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);

    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.logSoftmaxLastAxis(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
}

test "cuda dispatch softmaxLastAxis: large-C stress test (D=64) sums to 1.0 per row" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "softmax");

    const alloc = testing.allocator;
    // Shape (3, 64): catches shared-memory sizing bugs that the oracle's
    // D=4 fixture wouldn't expose. Values span ~[-5, +5] so the
    // max-subtraction path is real (not a vacuous shift).
    const N: usize = 3;
    const C: usize = 64;
    var x_host = try alloc.alloc(f32, N * C);
    defer alloc.free(x_host);
    var rng = std.Random.DefaultPrng.init(0xC01D);
    const r = rng.random();
    for (0..N * C) |i| x_host[i] = (r.float(f32) - 0.5) * 10.0;

    var x_gpu = try cudaFromSlice(&ctx, Shape.init2D(N, C), x_host);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.softmaxLastAxis(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    for (0..N) |row| {
        var row_sum: f32 = 0;
        for (0..C) |c| row_sum += y_cpu.data[row * C + c];
        try testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 1e-5);
    }
}

test "cuda ops_softmax.softmax: oracle softmax_3d_last_axis forward + backward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");
    _ = try module.loadPtxFromFile(&ctx, io, "softmax");

    const alloc = testing.allocator;

    var x_cpu = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "input_0.ztlt"));
    defer x_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_dx = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "grad_input_0.ztlt"));
    defer expect_dx.deinit(alloc);

    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);
    x_gpu.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&x_gpu);

    var y = try ops_softmax.softmax(alloc, x_gpu, &tape);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    {
        var y_cpu = try y.toCpu(alloc);
        defer y_cpu.deinit(alloc);
        try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
    }

    // Backward: loss = sumAll(y) -> backwardSoftmax uses
    // diag - outer product formula with internal mul / sum / sub
    // which all route to CUDA.
    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try tape.backward(&loss);
    try ctx.synchronize();

    var dx_cpu = try x_gpu.grad.?.*.toCpu(alloc);
    defer dx_cpu.deinit(alloc);
    try oracle.expectClose(dx_cpu, expect_dx, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// ---------------------------------------------------------------------------
// PR-mu: embedding (forward gather + backward scatter-add) and AdamW
// ---------------------------------------------------------------------------

test "cuda dispatch embeddingForward: oracle embedding_3d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "embedding");

    const alloc = testing.allocator;

    // Oracle fixture: weight (6, 4), ids (2, 3), output (2, 3, 4).
    // Note: input_0 is the weight; input_1 is the ids.
    var weight_cpu = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "input_0.ztlt"));
    defer weight_cpu.deinit(alloc);
    var ids_cpu = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "input_1.ztlt"));
    defer ids_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);

    var weight_gpu = try weight_cpu.toCuda(&ctx);
    defer weight_gpu.storage.deinit(alloc);
    var ids_gpu = try ids_cpu.toCuda(&ctx);
    defer ids_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.embeddingForward(weight_gpu, ids_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);
    // embedding is a plain gather; bit-exact under f32. Use a small
    // epsilon so the OR condition `max_abs_diff < abs_tol` passes
    // when diff == 0 (expectClose uses strict inequality).
    try oracle.expectClose(y_cpu, expect_y, .{ .rel_tol = 1e-6, .abs_tol = 1e-6 });
}

test "cuda dispatch embeddingBackward: scatter-add with repeated ids" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "embedding");

    const alloc = testing.allocator;

    // Hand-crafted: ids = [1, 0, 1] (id 1 appears twice — stresses atomicAdd).
    // grad_out[0] = [1, 2, 3, 4]   -> adds to grad_weight[1]
    // grad_out[1] = [10, 20, 30, 40] -> adds to grad_weight[0]
    // grad_out[2] = [100, 200, 300, 400] -> adds to grad_weight[1]
    // Expected grad_weight:
    //   row 0 = [10, 20, 30, 40]
    //   row 1 = [101, 202, 303, 404]
    //   row 2 = [0, 0, 0, 0]
    const V: usize = 3;
    const D: usize = 4;

    var ids = try cudaFromSlice(&ctx, Shape.init1D(3), &[_]f32{ 1, 0, 1 });
    defer ids.storage.deinit(alloc);
    var grad_out = try cudaFromSlice(&ctx, Shape.init2D(3, 4), &[_]f32{
        1, 2, 3, 4,
        10, 20, 30, 40,
        100, 200, 300, 400,
    });
    defer grad_out.storage.deinit(alloc);

    var grad_weight = try dispatch.embeddingBackward(ids, grad_out, V);
    defer grad_weight.storage.deinit(alloc);
    try ctx.synchronize();

    var gw_cpu = try grad_weight.toCpu(alloc);
    defer gw_cpu.deinit(alloc);
    const expected = [_]f32{
        10, 20, 30, 40,
        101, 202, 303, 404,
        0, 0, 0, 0,
    };
    for (0..V * D) |i| try testing.expectEqual(expected[i], gw_cpu.data[i]);
}

test "cuda Embedding.forward: oracle embedding_3d forward + backward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");
    _ = try module.loadPtxFromFile(&ctx, io, "embedding");

    const alloc = testing.allocator;

    var weight_cpu = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "input_0.ztlt"));
    defer weight_cpu.deinit(alloc);
    var ids_cpu = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "input_1.ztlt"));
    defer ids_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_dw = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "grad_input_0.ztlt"));
    defer expect_dw.deinit(alloc);

    // Build an Embedding by hand with CUDA weight (init does CPU randn).
    var weight_gpu = try weight_cpu.toCuda(&ctx);
    defer weight_gpu.storage.deinit(alloc);
    weight_gpu.requires_grad = true;
    weight_gpu.param_id = 7;

    var ids_gpu = try ids_cpu.toCuda(&ctx);
    defer ids_gpu.storage.deinit(alloc);

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&weight_gpu);

    const embed = Embedding{
        .weight = weight_gpu,
        .allocator = alloc,
        .vocab_size = weight_gpu.shape.dims[0],
        .d_model = weight_gpu.shape.dims[1],
    };

    var y = try embed.forward(ids_gpu, &tape);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    {
        var y_cpu_back = try y.toCpu(alloc);
        defer y_cpu_back.deinit(alloc);
        try oracle.expectClose(y_cpu_back, expect_y, .{ .rel_tol = 1e-6, .abs_tol = 1e-6 });
    }

    // Backward via sumAll loss.
    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try tape.backward(&loss);
    try ctx.synchronize();

    var dw_cpu = try weight_gpu.grad.?.*.toCpu(alloc);
    defer dw_cpu.deinit(alloc);
    try oracle.expectClose(dw_cpu, expect_dw, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

test "cuda dispatch adamwStep: one update matches CPU reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "adamw");

    const alloc = testing.allocator;

    // Deterministic inputs.
    const N: usize = 8;
    const param_init = [_]f32{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 };
    const grad_vals = [_]f32{ 0.01, 0.02, -0.03, 0.04, -0.05, 0.06, -0.07, 0.08 };

    // GPU step.
    var p_gpu = try cudaFromSlice(&ctx, Shape.init1D(N), &param_init);
    defer p_gpu.storage.deinit(alloc);
    var g_gpu = try cudaFromSlice(&ctx, Shape.init1D(N), &grad_vals);
    defer g_gpu.storage.deinit(alloc);
    var m_gpu = try dispatch.zerosOn(&ctx, Shape.init1D(N));
    defer m_gpu.storage.deinit(alloc);
    var v_gpu = try dispatch.zerosOn(&ctx, Shape.init1D(N));
    defer v_gpu.storage.deinit(alloc);

    const lr: f32 = 1e-3;
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;
    const wd: f32 = 0.1;
    const t: u32 = 1;
    const bc1: f32 = @floatCast(1.0 / (1.0 - std.math.pow(f64, @floatCast(beta1), @floatFromInt(t))));
    const bc2: f32 = @floatCast(1.0 / (1.0 - std.math.pow(f64, @floatCast(beta2), @floatFromInt(t))));

    try dispatch.adamwStep(p_gpu, g_gpu, m_gpu, v_gpu, .{
        .lr = lr,
        .beta1 = beta1,
        .beta2 = beta2,
        .eps = eps,
        .weight_decay = wd,
        .bc1 = bc1,
        .bc2 = bc2,
    });
    try ctx.synchronize();

    // CPU reference (same formula as AdamW.step's CPU branch).
    var p_ref: [N]f32 = param_init;
    var m_ref: [N]f32 = [_]f32{0} ** N;
    var v_ref: [N]f32 = [_]f32{0} ** N;
    for (0..N) |i| {
        const g = grad_vals[i];
        m_ref[i] = beta1 * m_ref[i] + (1.0 - beta1) * g;
        v_ref[i] = beta2 * v_ref[i] + (1.0 - beta2) * g * g;
        const m_hat = m_ref[i] * bc1;
        const v_hat = v_ref[i] * bc2;
        p_ref[i] -= lr * (m_hat / (@sqrt(v_hat) + eps) + wd * p_ref[i]);
    }

    var p_back = try p_gpu.toCpu(alloc);
    defer p_back.deinit(alloc);
    for (0..N) |i| try testing.expectApproxEqAbs(p_ref[i], p_back.data[i], 1e-6);
}

// ---------------------------------------------------------------------------
// Milestone 1: Cross-entropy fused forward + grad (completes PR-mu)
// ---------------------------------------------------------------------------

test "cuda dispatch crossEntropyFused: oracle cross_entropy_3d forward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "ce_loss");

    const alloc = testing.allocator;

    // Oracle: logits (B=2, T=3, V=5), targets (B, T) f32, loss (1,).
    // Our CE kernel operates on flat (N, V) with targets (N,), matching
    // how the trainer reshapes the 3D tensors before loss. We do the
    // same reshape host-side: mutate shape + strides before toCuda, so
    // the CUDA copy sees a 2D contiguous view of the same bytes.
    var logits_3d = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "input_0.ztlt"));
    defer logits_3d.deinit(alloc);
    var targets_2d = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "input_1.ztlt"));
    defer targets_2d.deinit(alloc);
    var expect_loss = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "output.ztlt"));
    defer expect_loss.deinit(alloc);

    const B = logits_3d.shape.dims[0];
    const T = logits_3d.shape.dims[1];
    const V = logits_3d.shape.dims[2];
    logits_3d.shape = Shape.init2D(B * T, V);
    logits_3d.strides = computeStrides(logits_3d.shape);
    targets_2d.shape = Shape.init1D(B * T);
    targets_2d.strides = computeStrides(targets_2d.shape);

    var logits_gpu = try logits_3d.toCuda(&ctx);
    defer logits_gpu.storage.deinit(alloc);
    var targets_gpu = try targets_2d.toCuda(&ctx);
    defer targets_gpu.storage.deinit(alloc);

    var fused = try dispatch.crossEntropyFused(logits_gpu, targets_gpu);
    defer fused.loss.storage.deinit(alloc);
    defer fused.grad_logits.storage.deinit(alloc);
    try ctx.synchronize();

    var loss_cpu = try fused.loss.toCpu(alloc);
    defer loss_cpu.deinit(alloc);
    // rel/abs both at 1e-4: CE involves max-subtract + sum-of-exps +
    // log + atomicAdd across rows; drift from strict f32 evaluation
    // order accumulates but stays well inside this tolerance on our
    // (B*T=6) row count.
    try oracle.expectClose(loss_cpu, expect_loss, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

test "cuda ops_loss.crossEntropy via tape: oracle cross_entropy_3d forward+backward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    // CE kernel for the fused forward; reduce/elementwise aren't
    // needed on the backward path because we DtoD-clone the saved
    // grad via cuMemcpyDtoD_v2 (no kernel launch).
    _ = try module.loadPtxFromFile(&ctx, io, "ce_loss");

    const alloc = testing.allocator;

    var logits_3d = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "input_0.ztlt"));
    defer logits_3d.deinit(alloc);
    var targets_2d = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "input_1.ztlt"));
    defer targets_2d.deinit(alloc);
    var expect_loss = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "output.ztlt"));
    defer expect_loss.deinit(alloc);
    var expect_grad_3d = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "grad_input_0.ztlt"));
    defer expect_grad_3d.deinit(alloc);

    const B = logits_3d.shape.dims[0];
    const T = logits_3d.shape.dims[1];
    const V = logits_3d.shape.dims[2];
    logits_3d.shape = Shape.init2D(B * T, V);
    logits_3d.strides = computeStrides(logits_3d.shape);
    targets_2d.shape = Shape.init1D(B * T);
    targets_2d.strides = computeStrides(targets_2d.shape);

    var logits_gpu = try logits_3d.toCuda(&ctx);
    defer logits_gpu.storage.deinit(alloc);
    logits_gpu.requires_grad = true;
    logits_gpu.param_id = 42;

    var targets_gpu = try targets_2d.toCuda(&ctx);
    defer targets_gpu.storage.deinit(alloc);

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&logits_gpu);

    var loss = try ops_loss.crossEntropy(alloc, logits_gpu, targets_gpu, &tape);
    defer loss.storage.deinit(alloc);
    try ctx.synchronize();

    // Forward parity.
    {
        var loss_cpu = try loss.toCpu(alloc);
        defer loss_cpu.deinit(alloc);
        try oracle.expectClose(loss_cpu, expect_loss, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
    }

    // Backward. Tape seeds the scalar loss with 1.0 via onesLike,
    // backwardCrossEntropy DtoD-clones the saved grad from
    // `.ce_cuda_grad`.
    try tape.backward(&loss);
    try ctx.synchronize();

    var grad_cpu_flat = try logits_gpu.grad.?.*.toCpu(alloc);
    defer grad_cpu_flat.deinit(alloc);
    // Expected grad is stored as (B, T, V); we produced a flat (B*T, V)
    // buffer. Compare row-major: they must match bit-level since both
    // are contiguous with the same element order.
    expect_grad_3d.shape = Shape.init2D(B * T, V);
    expect_grad_3d.strides = computeStrides(expect_grad_3d.shape);
    try oracle.expectClose(grad_cpu_flat, expect_grad_3d, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// ---------------------------------------------------------------------------
// Milestone 2: Unary CUDA kernels (GELU, sqrt, exp, log)
// ---------------------------------------------------------------------------

const ops_unary = lab.ops.unary;

test "cuda dispatch geluExact: random input matches CPU reference within 1e-5" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "unary");

    const alloc = testing.allocator;

    // Handpicked values that span the sensitive region around 0 and
    // the saturating tails. CPU uses an A&S polynomial erf; CUDA uses
    // the `erff` intrinsic. Drift is typically <= 2 ULP per element,
    // well inside the 1e-5 absolute tolerance.
    const x_host = [_]f32{ -2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 4.0 };
    var x_gpu = try cudaFromSlice(&ctx, Shape.init1D(x_host.len), &x_host);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.geluExact(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();
    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);

    // Reference: our own CPU geluExact (which uses the A&S polynomial).
    var x_ref = try Tensor.init(alloc, Shape.init1D(x_host.len));
    defer x_ref.deinit(alloc);
    @memcpy(x_ref.data, &x_host);
    var y_ref = try ops_unary.geluExact(alloc, x_ref, null);
    defer y_ref.deinit(alloc);

    try oracle.expectClose(y_cpu, y_ref, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

test "cuda dispatch geluExact: tape records and backward matches CPU within 1e-4" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "unary");
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;

    const x_host = [_]f32{ -2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 4.0 };
    var x_gpu = try cudaFromSlice(&ctx, Shape.init1D(x_host.len), &x_host);
    defer x_gpu.storage.deinit(alloc);
    x_gpu.requires_grad = true;
    x_gpu.param_id = 11;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&x_gpu);

    var y = try ops_unary.geluExact(alloc, x_gpu, &tape);
    defer y.storage.deinit(alloc);
    // Loss = sumAll(y) so every element of y gets grad 1.0 — backward
    // then reads gelu'(x) directly.
    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try tape.backward(&loss);
    try ctx.synchronize();

    var dx_cpu = try x_gpu.grad.?.*.toCpu(alloc);
    defer dx_cpu.deinit(alloc);

    // Reference: compute CPU gelu backward via the same tape API.
    var x_ref = try Tensor.init(alloc, Shape.init1D(x_host.len));
    defer x_ref.deinit(alloc);
    @memcpy(x_ref.data, &x_host);
    x_ref.requires_grad = true;
    x_ref.param_id = 11;
    var tape_cpu = Tape.init(alloc);
    defer tape_cpu.deinit();
    _ = try tape_cpu.trackLeaf(&x_ref);
    var y_ref = try ops_unary.geluExact(alloc, x_ref, &tape_cpu);
    defer y_ref.deinit(alloc);
    var loss_ref = try ops_reduce.sumAll(alloc, y_ref, &tape_cpu);
    defer loss_ref.deinit(alloc);
    try tape_cpu.backward(&loss_ref);

    try oracle.expectClose(dx_cpu, x_ref.grad.?.*, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

test "cuda dispatch sqrt: positive random input matches CPU" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "unary");

    const alloc = testing.allocator;
    const x_host = [_]f32{ 0.01, 0.25, 1.0, 2.0, 9.0, 100.0 };
    var x_gpu = try cudaFromSlice(&ctx, Shape.init1D(x_host.len), &x_host);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.sqrt(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();
    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);

    // Expected: IEEE f32 sqrt.
    for (0..x_host.len) |i| {
        const expected: f32 = @sqrt(x_host[i]);
        try testing.expectApproxEqRel(expected, y_cpu.data[i], 1e-6);
    }
}

test "cuda dispatch exp: small-range input matches CPU" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "unary");

    const alloc = testing.allocator;
    // Keep inputs moderate so expf doesn't overflow f32 (e^88 ~ 1e38).
    const x_host = [_]f32{ -3.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    var x_gpu = try cudaFromSlice(&ctx, Shape.init1D(x_host.len), &x_host);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.exp(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();
    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);

    for (0..x_host.len) |i| {
        const expected: f32 = @exp(x_host[i]);
        try testing.expectApproxEqRel(expected, y_cpu.data[i], 1e-5);
    }
}

test "cuda dispatch log: positive input matches CPU" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "unary");

    const alloc = testing.allocator;
    const x_host = [_]f32{ 0.1, 0.5, 1.0, 2.718281828, 10.0, 100.0 };
    var x_gpu = try cudaFromSlice(&ctx, Shape.init1D(x_host.len), &x_host);
    defer x_gpu.storage.deinit(alloc);

    var y_gpu = try dispatch.log(x_gpu);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();
    var y_cpu = try y_gpu.toCpu(alloc);
    defer y_cpu.deinit(alloc);

    for (0..x_host.len) |i| {
        const expected: f32 = @log(x_host[i]);
        try testing.expectApproxEqAbs(expected, y_cpu.data[i], 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Milestone 2 (cont.): meanAxis composition + oracle gelu_2d parity
// ---------------------------------------------------------------------------

test "cuda ops_reduce.mean: (2,3,4) axis=1 matches CPU reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");

    const alloc = testing.allocator;

    // Deterministic 24-element input arranged as (2,3,4).
    var x_host: [24]f32 = undefined;
    for (0..24) |i| x_host[i] = @floatFromInt(i);

    var x_cpu = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer x_cpu.deinit(alloc);
    @memcpy(x_cpu.data, &x_host);

    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);

    // CPU reference: same tape-free call.
    var y_ref = try ops_reduce.mean(alloc, x_cpu, 1, null);
    defer y_ref.deinit(alloc);

    var y_gpu = try ops_reduce.mean(alloc, x_gpu, 1, null);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_back = try y_gpu.toCpu(alloc);
    defer y_back.deinit(alloc);

    try oracle.expectClose(y_back, y_ref, .{ .rel_tol = 1e-5, .abs_tol = 1e-6 });
}

test "cuda ops_unary.geluExact: oracle gelu_2d forward + backward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "unary");
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;

    // Oracle fixture: a (3, 4) tensor, forward is exact erf-based GELU,
    // loss = sumAll(y), grad w.r.t. a is stored in grad_input_0.ztlt.
    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("gelu_2d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("gelu_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("gelu_2d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);

    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    a_gpu.requires_grad = true;
    a_gpu.param_id = 77;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a_gpu);

    var y = try ops_unary.geluExact(alloc, a_gpu, &tape);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    {
        var y_back = try y.toCpu(alloc);
        defer y_back.deinit(alloc);
        // GELU drift between CUDA erff and CPU A&S polynomial is a
        // few ULP per element — rel=1e-4 / abs=1e-5 absorbs it.
        try oracle.expectClose(y_back, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    }

    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try tape.backward(&loss);
    try ctx.synchronize();

    var da_back = try a_gpu.grad.?.*.toCpu(alloc);
    defer da_back.deinit(alloc);
    // Backward composes gelu'(x) = phi(x) + x*pdf(x); same tolerance
    // as forward since the backward is a straight elementwise apply
    // of a smooth function. For `sumAll` loss, grad_out is all ones,
    // so drift here is purely from gelu' vs the CPU reference.
    try oracle.expectClose(da_back, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// ---------------------------------------------------------------------------
// Session 2: LayerNorm oracle parity + Linear smoke tests
// ---------------------------------------------------------------------------

const LayerNorm = lab.nn.layernorm.LayerNorm;
const Linear = lab.nn.linear.Linear;

test "cuda LayerNorm: oracle layernorm_3d forward + backward parity" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    // LayerNorm forward composes mean -> sub -> mul -> addScalar ->
    // sqrt -> div -> reshape -> mul -> reshape -> add. Every module
    // below is pulled in at least once. Backward needs broadcastTo
    // (reduce) and elementwise mulScalar/div.
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");
    _ = try module.loadPtxFromFile(&ctx, io, "unary");

    const alloc = testing.allocator;

    // Oracle: input (2,3,4), gamma (4,), beta (4,), eps=1e-5.
    var a_cpu = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "input_0.ztlt"));
    defer a_cpu.deinit(alloc);
    var gamma_cpu = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "input_1.ztlt"));
    defer gamma_cpu.deinit(alloc);
    var beta_cpu = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "input_2.ztlt"));
    defer beta_cpu.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_dgamma = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "grad_input_1.ztlt"));
    defer expect_dgamma.deinit(alloc);
    var expect_dbeta = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "grad_input_2.ztlt"));
    defer expect_dbeta.deinit(alloc);

    const D = a_cpu.shape.dims[2];

    // Build LayerNorm by hand with CUDA gamma/beta. (LayerNorm.init
    // would create CPU ones/zeros; we want the oracle values.)
    var gamma_gpu = try gamma_cpu.toCuda(&ctx);
    defer gamma_gpu.storage.deinit(alloc);
    gamma_gpu.requires_grad = true;
    gamma_gpu.param_id = 101;

    var beta_gpu = try beta_cpu.toCuda(&ctx);
    defer beta_gpu.storage.deinit(alloc);
    beta_gpu.requires_grad = true;
    beta_gpu.param_id = 102;

    var a_gpu = try a_cpu.toCuda(&ctx);
    defer a_gpu.storage.deinit(alloc);
    a_gpu.requires_grad = true;
    a_gpu.param_id = 103;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a_gpu);
    _ = try tape.trackLeaf(&gamma_gpu);
    _ = try tape.trackLeaf(&beta_gpu);

    const ln = LayerNorm{
        .gamma = gamma_gpu,
        .beta = beta_gpu,
        .allocator = alloc,
        .d_model = D,
        .eps = 1e-5,
    };

    var y = try ln.forward(a_gpu, &tape);
    defer y.storage.deinit(alloc);
    try ctx.synchronize();

    // Forward parity. LayerNorm composes 8+ ops; each kernel drifts a
    // few ULP from the CPU reference. The oracle tolerance rel=1e-4,
    // abs=1e-4 is generous enough to absorb the compound drift on a
    // (2,3,4) shape.
    {
        var y_back = try y.toCpu(alloc);
        defer y_back.deinit(alloc);
        try oracle.expectClose(y_back, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
    }

    // Backward via sumAll loss (same as oracle).
    var loss = try ops_reduce.sumAll(alloc, y, &tape);
    defer loss.storage.deinit(alloc);
    try tape.backward(&loss);
    try ctx.synchronize();

    // Per-input grad parity.
    {
        var da_back = try a_gpu.grad.?.*.toCpu(alloc);
        defer da_back.deinit(alloc);
        const abs_diff = try oracle.maxAbsDiff(da_back, expect_da);
        const rel_err = try oracle.maxRelErr(da_back, expect_da, 1e-8);
        std.debug.print("  LN da  abs_diff={d:.6} rel_err={d:.6}\n", .{ abs_diff, rel_err });
        try oracle.expectClose(da_back, expect_da, .{ .rel_tol = 1e-3, .abs_tol = 1e-4 });
    }
    {
        var dgamma_back = try gamma_gpu.grad.?.*.toCpu(alloc);
        defer dgamma_back.deinit(alloc);
        const abs_diff = try oracle.maxAbsDiff(dgamma_back, expect_dgamma);
        const rel_err = try oracle.maxRelErr(dgamma_back, expect_dgamma, 1e-8);
        std.debug.print("  LN dg  abs_diff={d:.6} rel_err={d:.6}\n", .{ abs_diff, rel_err });
        try oracle.expectClose(dgamma_back, expect_dgamma, .{ .rel_tol = 1e-3, .abs_tol = 1e-4 });
    }
    {
        var dbeta_back = try beta_gpu.grad.?.*.toCpu(alloc);
        defer dbeta_back.deinit(alloc);
        const abs_diff = try oracle.maxAbsDiff(dbeta_back, expect_dbeta);
        const rel_err = try oracle.maxRelErr(dbeta_back, expect_dbeta, 1e-8);
        std.debug.print("  LN db  abs_diff={d:.6} rel_err={d:.6}\n", .{ abs_diff, rel_err });
        try oracle.expectClose(dbeta_back, expect_dbeta, .{ .rel_tol = 1e-3, .abs_tol = 1e-4 });
    }
}

test "cuda Linear.forward: 2D input matches CPU reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;
    var rng = lab.rng.Rng.init(42);

    // Build a Linear on CPU, then replicate its weights on CUDA.
    var layer_cpu = try Linear.init(alloc, 4, 3, true, &rng);
    defer layer_cpu.deinit();

    // Input on both devices.
    var x_cpu = try lab.ops.create.randn(alloc, Shape.init2D(2, 4), &rng, 0.0, 1.0);
    defer x_cpu.deinit(alloc);
    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);

    // Clone weights to CUDA and wrap in a CUDA-weight Linear.
    var w_gpu = try layer_cpu.weight.toCuda(&ctx);
    defer w_gpu.storage.deinit(alloc);
    w_gpu.param_id = layer_cpu.weight.param_id;
    var b_gpu = try layer_cpu.bias.?.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);
    b_gpu.param_id = layer_cpu.bias.?.param_id;
    const layer_gpu = Linear{
        .weight = w_gpu,
        .bias = b_gpu,
        .allocator = alloc,
        .d_in = 4,
        .d_out = 3,
        .use_bias = true,
    };

    var y_cpu = try layer_cpu.forward(x_cpu, null);
    defer y_cpu.deinit(alloc);
    var y_gpu = try layer_gpu.forward(x_gpu, null);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    var y_back = try y_gpu.toCpu(alloc);
    defer y_back.deinit(alloc);
    // Linear = matmul + bias-broadcast-add. cuBLAS matmul vs our CPU
    // triple-loop matmul drifts up to ~5 ULP per accumulator slot on
    // K=4. rel=1e-4 / abs=1e-5 is comfortable on this shape.
    try oracle.expectClose(y_back, y_cpu, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

test "cuda Linear.forward: 3D input matches CPU reference" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var ctx = try CudaContext.init(testing.allocator);
    defer ctx.deinit();
    const io = try testIo();
    _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
    _ = try module.loadPtxFromFile(&ctx, io, "reduce");

    const alloc = testing.allocator;
    var rng = lab.rng.Rng.init(42);

    var layer_cpu = try Linear.init(alloc, 4, 3, true, &rng);
    defer layer_cpu.deinit();

    var x_cpu = try lab.ops.create.randn(alloc, Shape.init3D(2, 3, 4), &rng, 0.0, 1.0);
    defer x_cpu.deinit(alloc);
    var x_gpu = try x_cpu.toCuda(&ctx);
    defer x_gpu.storage.deinit(alloc);

    var w_gpu = try layer_cpu.weight.toCuda(&ctx);
    defer w_gpu.storage.deinit(alloc);
    w_gpu.param_id = layer_cpu.weight.param_id;
    var b_gpu = try layer_cpu.bias.?.toCuda(&ctx);
    defer b_gpu.storage.deinit(alloc);
    b_gpu.param_id = layer_cpu.bias.?.param_id;
    const layer_gpu = Linear{
        .weight = w_gpu,
        .bias = b_gpu,
        .allocator = alloc,
        .d_in = 4,
        .d_out = 3,
        .use_bias = true,
    };

    var y_cpu = try layer_cpu.forward(x_cpu, null);
    defer y_cpu.deinit(alloc);
    var y_gpu = try layer_gpu.forward(x_gpu, null);
    defer y_gpu.storage.deinit(alloc);
    try ctx.synchronize();

    try testing.expectEqual(@as(usize, 3), y_gpu.shape.ndim());
    try testing.expectEqual(@as(usize, 2), y_gpu.shape.dims[0]);
    try testing.expectEqual(@as(usize, 3), y_gpu.shape.dims[1]);
    try testing.expectEqual(@as(usize, 3), y_gpu.shape.dims[2]);

    var y_back = try y_gpu.toCpu(alloc);
    defer y_back.deinit(alloc);
    try oracle.expectClose(y_back, y_cpu, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}
