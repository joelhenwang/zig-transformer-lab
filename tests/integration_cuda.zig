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
const CudaContext = context.CudaContext;
const DeviceBuffer = mem.DeviceBuffer;
const Tensor = lab.Tensor;
const Storage = lab.Storage;
const Shape = lab.shape.Shape;
const computeStrides = lab.shape.computeStrides;

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
