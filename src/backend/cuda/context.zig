//!
//! zig-transformer-lab — CudaContext lifecycle (Stage 7, PR-beta)
//!
//! Purpose:
//!   Wrap the "create device + context + stream + cuBLAS handle"
//!   boilerplate into one struct with a clearly-ordered init and
//!   deinit. Every later Stage 7 PR takes `*CudaContext` instead of
//!   rebuilding the full CUDA lane.
//!
//! Scope (intentional omissions):
//!   - No tensor or memory types. DeviceBuffer ships in PR-gamma.
//!   - No PTX module loading. PR-zeta does that, populating
//!     `ptx_modules` for the first time.
//!   - Single GPU only — device 0 is hard-coded per the Stage 7
//!     session plan (multi-GPU is a Stage 9 concern).
//!
//! Ownership:
//!   `CudaContext` owns four resources, released in strict reverse
//!   order on deinit:
//!       1. Any loaded .ptx modules (cuModuleUnload)
//!       2. cuBLAS handle             (cublasDestroy_v2)
//!       3. CUDA stream               (cuStreamDestroy_v2)
//!       4. CUDA context              (cuCtxDestroy_v2)
//!   The device handle is just a small integer; nothing to free.
//!
//!   `ptx_modules` stores its keys in memory owned by this struct:
//!   when PR-zeta adds modules, it must `allocator.dupe(u8, key)` on
//!   insert, and `deinit` frees the dup'd strings before tearing the
//!   HashMap down.
//!
//!   Important: callers MUST deinit any CUDA resources (tensors, PTX
//!   modules loaded externally) that reference this context BEFORE
//!   deiniting the context itself. Freeing a DeviceBuffer whose
//!   context has already been destroyed is undefined behaviour on
//!   the CUDA side.
//!
//! Errors:
//!   - error.CudaError: any Driver or cuBLAS call failed; cause is
//!     logged via bindings.check()/checkCublas().
//!   - error.OutOfMemory: the process-global allocator could not
//!     service the HashMap initialisation. Very unlikely.
//!
//! Lifetime contract for Stage 7:
//!   One CudaContext per process, held by the training loop (or by
//!   each test) and threaded explicitly through every CUDA dispatch
//!   call. We deliberately do NOT expose a global singleton: a
//!   forgotten deinit in a test would poison subsequent tests with
//!   CUDA_ERROR_INVALID_CONTEXT, and that failure mode is much harder
//!   to diagnose than an "explicit init required" API.
//!
//! Credits:
//!   Structure mirrors the "context bundle" pattern used in many
//!   CUDA wrappers (e.g. the CUDA samples' cudaSetup helper). No
//!   third-party code copied.
//!

const std = @import("std");
const builtin = @import("builtin");
const errors = @import("../../core/errors.zig");
const bindings = @import("bindings.zig");

const LabError = errors.LabError;

/// Device index we always target. Multi-GPU is a Stage 9 concern;
/// in Stage 7 we assume a single GPU (the RTX 4060 Ti on the remote).
pub const DEFAULT_DEVICE: c_int = 0;

/// Bundled CUDA lifecycle: one device + context + stream + cuBLAS
/// handle + (initially empty) PTX module cache.
pub const CudaContext = struct {
    allocator: std.mem.Allocator,
    device: bindings.CUdevice,
    ctx: bindings.CUcontext,
    stream: bindings.CUstream,
    cublas: bindings.cublasHandle_t,
    /// Cache of loaded .ptx modules keyed by their file stem (e.g.
    /// "elementwise", "softmax"). Populated by PR-zeta's PTX loader;
    /// empty in PR-beta. Keys are owned by this struct — each insert
    /// must `allocator.dupe` the stem, and deinit frees every key.
    ptx_modules: std.StringHashMapUnmanaged(bindings.CUmodule),

    /// Open libcuda/libcublas (idempotent), cuInit, pick device 0,
    /// create context, create stream, create cuBLAS handle, bind
    /// stream to handle. Order matches the lifecycle sequence
    /// documented in docs/stage7_plan.md §6 PR-beta.
    ///
    /// On any failure, prior successful resources are torn down in
    /// reverse order before returning the error — so a failed init
    /// leaves no partially-constructed CUDA state.
    pub fn init(allocator: std.mem.Allocator) LabError!CudaContext {
        try bindings.load();
        const L = bindings.loader.?;

        try bindings.check(L.cuInit(0));

        var device: bindings.CUdevice = DEFAULT_DEVICE;
        try bindings.check(L.cuDeviceGet(&device, DEFAULT_DEVICE));

        var ctx: bindings.CUcontext = null;
        try bindings.check(L.cuCtxCreate_v2(&ctx, 0, device));
        errdefer _ = L.cuCtxDestroy_v2(ctx);

        // Flags = 0: default stream behaviour (synchronous w.r.t. the
        // null stream on this device). Non-blocking streams are a
        // potential Stage 9 optimisation; correctness comes first.
        var stream: bindings.CUstream = null;
        try bindings.check(L.cuStreamCreate(&stream, 0));
        errdefer _ = L.cuStreamDestroy_v2(stream);

        var cublas: bindings.cublasHandle_t = null;
        try bindings.checkCublas(L.cublasCreate_v2(&cublas));
        errdefer _ = L.cublasDestroy_v2(cublas);

        // Route every cuBLAS call onto our stream. Otherwise cuBLAS
        // would use the null stream, which would not overlap with
        // any kernel we launched on our own stream (a correctness-
        // irrelevant but diagnostically-annoying behaviour).
        try bindings.checkCublas(L.cublasSetStream_v2(cublas, stream));

        return .{
            .allocator = allocator,
            .device = device,
            .ctx = ctx,
            .stream = stream,
            .cublas = cublas,
            .ptx_modules = .empty,
        };
    }

    /// Tear down every owned resource in strict reverse order of
    /// init. Non-success return codes from the destructors are
    /// logged via std.log.warn but NOT returned: destructors are
    /// infallible by convention so callers can rely on `defer
    /// ctx.deinit();` behaving predictably.
    ///
    /// Safety: after this returns, every field of `self` is
    /// undefined. Do not reuse the context.
    pub fn deinit(self: *CudaContext) void {
        // If the loader never initialised, there is no CUDA state
        // to tear down. This can happen only if init() returned an
        // error before reaching the CUDA calls, which would mean the
        // caller has a code bug (we'd never build a CudaContext
        // struct in that case) — but we guard anyway for robustness.
        const L = bindings.loader orelse return;

        // 1. Unload every PTX module and free the dup'd key strings.
        //    This is a no-op in PR-beta (the map is always empty)
        //    but the PR-zeta contract requires this order: modules
        //    must be unloaded before the context they were loaded
        //    into is destroyed.
        var it = self.ptx_modules.iterator();
        while (it.next()) |entry| {
            const r = L.cuModuleUnload(entry.value_ptr.*);
            if (r != bindings.CUDA_SUCCESS) {
                std.log.warn(
                    "CudaContext.deinit: cuModuleUnload({s}) -> {d} (ignored)",
                    .{ entry.key_ptr.*, r },
                );
            }
            self.allocator.free(entry.key_ptr.*);
        }
        self.ptx_modules.deinit(self.allocator);

        // 2. cuBLAS handle.
        const s = L.cublasDestroy_v2(self.cublas);
        if (s != bindings.CUBLAS_STATUS_SUCCESS) {
            std.log.warn("CudaContext.deinit: cublasDestroy_v2 -> {d} (ignored)", .{s});
        }

        // 3. Stream.
        const rs = L.cuStreamDestroy_v2(self.stream);
        if (rs != bindings.CUDA_SUCCESS) {
            std.log.warn("CudaContext.deinit: cuStreamDestroy_v2 -> {d} (ignored)", .{rs});
        }

        // 4. Context.
        const rc = L.cuCtxDestroy_v2(self.ctx);
        if (rc != bindings.CUDA_SUCCESS) {
            std.log.warn("CudaContext.deinit: cuCtxDestroy_v2 -> {d} (ignored)", .{rc});
        }

        self.* = undefined;
    }

    /// Block until every previously-submitted operation on this
    /// stream has finished. Use this sparingly — it is a hard
    /// barrier and defeats any potential kernel/memcpy overlap.
    ///
    /// Debug builds of later PRs will call this after every kernel
    /// launch to surface async errors (illegal addresses, bad
    /// launch configs) at the exact launch that caused them. Release
    /// builds skip the sync and only pay the cost on `toCpu` or
    /// other explicit host-visible boundaries.
    pub fn synchronize(self: *CudaContext) LabError!void {
        const L = bindings.loader.?;
        try bindings.check(L.cuStreamSynchronize(self.stream));
    }
};
