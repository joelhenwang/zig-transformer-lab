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
const CudaContext = context.CudaContext;

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
