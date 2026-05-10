//!
//! zig-transformer-lab — CUDA dynamic bindings (Stage 7, PR-alpha)
//!
//! Purpose:
//!   Runtime-loaded bindings for the CUDA 12 / 13 Driver API and cuBLAS.
//!   The build does NOT link against libcuda or libcublas; instead we
//!   dlopen libcuda.so.1 and libcublas.so.{13,12} via std.DynLib, dlsym
//!   every symbol we will ever need across Stage 7 into a `Loader`
//!   struct of function pointers, and route every later CUDA call
//!   through that struct.
//!
//!   Why dlopen instead of linkSystemLibrary?
//!     Locked decision D1 — hand-written bindings, no external Zig
//!     CUDA dependency. Dlopen lets the binary run on machines without
//!     CUDA installed (the load() call fails cleanly at runtime, not
//!     at link time) and keeps build.zig portable across our Windows
//!     development host and the Linux RTX remote.
//!
//! Scope (what this file does NOT do):
//!   - No tensor, context, or memory abstractions — those come in
//!     later Stage 7 PRs (b, c, d, e). This file is a flat, typed
//!     view of the CUDA Driver / cuBLAS ABI.
//!   - No kernel launches or PTX handling yet — PR-f wires that up.
//!   - No device-side code; no C source files.
//!
//! Ownership:
//!   The process-global `loader: ?Loader` owns two std.DynLib handles
//!   after a successful `load()`. `unload()` closes both; calling it
//!   is optional since normal processes can leak the handles at exit.
//!   `load()` is idempotent — redundant calls are O(1).
//!
//! Platform support:
//!   Linux only. On other OSes, `load()` returns error.CudaError with
//!   a diagnostic log line. The file still *compiles* everywhere
//!   (std.DynLib is cross-platform), so a Windows build of
//!   `zig build test -Dcuda=true` fails at runtime with a clean
//!   message instead of failing to link.
//!
//! Error contract:
//!   Every public function returns LabError. The only variant ever
//!   returned is LabError.CudaError. Failure modes:
//!     - libcuda.so.1 or libcublas.so.{13,12} not found on the linker
//!       search path.
//!     - dlsym returns null for a symbol that should exist (logged
//!       with the symbol name).
//!     - A Driver or cuBLAS call returns a non-success code; the
//!       numeric code is logged via std.log.err, together with the
//!       human-readable string from cuGetErrorString /
//!       cublasGetStatusString when the loader is available.
//!
//! Symbol list decisions:
//!   - Driver API only. We do NOT dlopen libcudart. cuCtxSynchronize
//!     and cuStreamSynchronize cover our synchronisation needs.
//!   - The `_v2` suffix is load-bearing for cuCtxCreate_v2,
//!     cuCtxDestroy_v2, cuMemAlloc_v2, cuMemFree_v2, cuMemcpy*_v2,
//!     cuStreamDestroy_v2, cublasCreate_v2, cublasDestroy_v2,
//!     cublasSgemm_v2, cublasSetStream_v2. The unversioned names
//!     either do not exist or have an incompatible ABI on CUDA 12+.
//!   - cuDeviceGetName, cuInit, cuDriverGetVersion, cuStreamCreate,
//!     cuStreamSynchronize, cuGetErrorString, cuModuleLoadData,
//!     cuModuleUnload, cuModuleGetFunction, cuLaunchKernel, and
//!     cuMemsetD32_v2 keep their canonical names.
//!   - All ~30 symbols are resolved in a single load() call even
//!     though many are only *used* in later PRs; this keeps future
//!     PRs from having to touch this file except when adding support
//!     for a new ABI.
//!
//! Credits:
//!   Dlopen + function-pointer struct layout is a standard pattern for
//!   runtime-loaded ABIs (see e.g. Vulkan loader, OpenGL extension
//!   resolution). No third-party code copied. CUDA type names match
//!   NVIDIA's cuda.h / cublas_v2.h by deliberate choice so that
//!   readers comparing against the CUDA reference see the exact
//!   symbols they expect.
//!

const std = @import("std");
const builtin = @import("builtin");
const errors = @import("../../core/errors.zig");
const LabError = errors.LabError;

// ============================================================================
// Opaque handle types
// ============================================================================
//
// These mirror the CUDA headers but are never introspected on the Zig
// side. They're passed around as opaque handles/integers.
//
// A note on CUdeviceptr: it is defined by NVIDIA as an unsigned
// integer (64-bit on 64-bit platforms), NOT a pointer. It is a device
// address. Dereferencing it from host code is undefined behaviour. We
// deliberately keep its type as `u64` on the Zig side so the type
// system stops callers from accidentally treating it as `*f32`.

pub const CUdevice = c_int;
pub const CUcontext = ?*opaque {};
pub const CUstream = ?*opaque {};
pub const CUmodule = ?*opaque {};
pub const CUfunction = ?*opaque {};
pub const CUdeviceptr = u64;
pub const cublasHandle_t = ?*opaque {};

// ============================================================================
// Return codes
// ============================================================================

pub const CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

/// Selected CUresult codes we reference by name in later PRs.
/// The full list (500+ codes) is not replicated here; check() logs the
/// numeric code and the string from cuGetErrorString.
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;

pub const cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// cuBLAS transpose flags (used from PR-k onward; declared here so
// call sites can pass symbolic names without pulling in another file).
pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;
pub const CUBLAS_OP_C: c_int = 2;

// ============================================================================
// Loader — one function pointer per CUDA / cuBLAS symbol we'll ever call
// ============================================================================

/// Every CUDA / cuBLAS symbol resolved by load().
///
/// Fields are grouped by subsystem (init / device / context / stream /
/// memory / module / cuBLAS) to match the order they're called in a
/// typical lifecycle.
///
/// Field names match the exported C symbol names exactly, including
/// the `_v2` suffix where applicable. A call site reads
/// `L.cuCtxCreate_v2(...)` which maps directly to the CUDA
/// documentation without any renaming layer in between.
pub const Loader = struct {
    libcuda: std.DynLib,
    libcublas: std.DynLib,

    // --- Initialisation / device queries ---
    cuInit: *const fn (c_uint) callconv(.c) CUresult,
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGet: *const fn (*CUdevice, c_int) callconv(.c) CUresult,
    cuDeviceGetName: *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult,
    cuGetErrorString: *const fn (CUresult, *[*:0]const u8) callconv(.c) CUresult,

    // --- Context ---
    cuCtxCreate_v2: *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult,
    cuCtxDestroy_v2: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSetCurrent: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSynchronize: *const fn () callconv(.c) CUresult,

    // --- Stream ---
    cuStreamCreate: *const fn (*CUstream, c_uint) callconv(.c) CUresult,
    cuStreamDestroy_v2: *const fn (CUstream) callconv(.c) CUresult,
    cuStreamSynchronize: *const fn (CUstream) callconv(.c) CUresult,

    // --- Device memory ---
    cuMemAlloc_v2: *const fn (*CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemFree_v2: *const fn (CUdeviceptr) callconv(.c) CUresult,
    cuMemcpyHtoD_v2: *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult,
    cuMemcpyDtoH_v2: *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemcpyDtoD_v2: *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemsetD32_v2: *const fn (CUdeviceptr, c_uint, usize) callconv(.c) CUresult,

    // --- Module / kernel (used from PR-f) ---
    cuModuleLoadData: *const fn (*CUmodule, *const anyopaque) callconv(.c) CUresult,
    cuModuleUnload: *const fn (CUmodule) callconv(.c) CUresult,
    cuModuleGetFunction: *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult,
    /// cuLaunchKernel signature:
    ///   f: kernel handle
    ///   gridDim x/y/z, blockDim x/y/z: launch geometry
    ///   sharedMemBytes: dynamic shared memory per block
    ///   hStream: target stream (null = default stream)
    ///   kernelParams: array of ?*anyopaque, one pointer per kernel argument
    ///   extra: opaque extra-params array, usually null
    cuLaunchKernel: *const fn (
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        [*c]?*anyopaque,
        [*c]?*anyopaque,
    ) callconv(.c) CUresult,

    // --- cuBLAS (used from PR-k onward; loaded up front so this
    //     file stays frozen across later PRs) ---
    cublasCreate_v2: *const fn (*cublasHandle_t) callconv(.c) cublasStatus_t,
    cublasDestroy_v2: *const fn (cublasHandle_t) callconv(.c) cublasStatus_t,
    cublasSetStream_v2: *const fn (cublasHandle_t, CUstream) callconv(.c) cublasStatus_t,
    cublasSgemm_v2: *const fn (
        cublasHandle_t,
        c_int, // transa
        c_int, // transb
        c_int, // m
        c_int, // n
        c_int, // k
        *const f32, // alpha
        CUdeviceptr, // A
        c_int, // lda
        CUdeviceptr, // B
        c_int, // ldb
        *const f32, // beta
        CUdeviceptr, // C
        c_int, // ldc
    ) callconv(.c) cublasStatus_t,
    cublasSgemmStridedBatched: *const fn (
        cublasHandle_t,
        c_int, // transa
        c_int, // transb
        c_int, // m
        c_int, // n
        c_int, // k
        *const f32, // alpha
        CUdeviceptr, // A
        c_int, // lda
        i64, // strideA
        CUdeviceptr, // B
        c_int, // ldb
        i64, // strideB
        *const f32, // beta
        CUdeviceptr, // C
        c_int, // ldc
        i64, // strideC
        c_int, // batchCount
    ) callconv(.c) cublasStatus_t,
    cublasGetStatusString: *const fn (cublasStatus_t) callconv(.c) [*:0]const u8,
};

// ============================================================================
// Process-global loader state
// ============================================================================

/// Populated by `load()` on first successful call. `null` otherwise.
/// Callers should test `loader` before dereferencing, or use the
/// idiomatic `try load(); const L = loader.?;` pattern.
pub var loader: ?Loader = null;

// ============================================================================
// Internal helpers
// ============================================================================

/// dlsym a single symbol. On failure, logs the symbol name and returns
/// error.CudaError. The comptime generic `T` lets the caller pass the
/// exact function-pointer type declared on `Loader`, so the return
/// value is statically typed at the call site.
fn sym(comptime T: type, lib: *std.DynLib, name: [:0]const u8) LabError!T {
    if (lib.lookup(T, name)) |f| return f;
    std.log.err("cuda bindings: dlsym failed for symbol '{s}'", .{name});
    return error.CudaError;
}

/// Open libcublas, trying `.so.13` first (CUDA 13) then `.so.12`
/// (CUDA 12). We log every dlopen attempt so future sessions can
/// diagnose deployment drift by reading stderr alone.
fn openCublas() LabError!std.DynLib {
    const primary = "libcublas.so.13";
    const fallback = "libcublas.so.12";

    if (std.DynLib.open(primary)) |h| {
        return h;
    } else |err13| {
        std.log.warn("cuda bindings: dlopen({s}) failed: {s}; falling back to {s}", .{
            primary,
            @errorName(err13),
            fallback,
        });
        if (std.DynLib.open(fallback)) |h| {
            return h;
        } else |err12| {
            std.log.err("cuda bindings: dlopen(libcublas): {s} -> {s}, {s} -> {s}", .{
                primary,
                @errorName(err13),
                fallback,
                @errorName(err12),
            });
            return error.CudaError;
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Open libcuda.so.1 and libcublas.so.{13,12}, resolve every symbol
/// into the process-global `loader`. Idempotent.
///
/// On non-Linux platforms this is an immediate failure by design —
/// the library names are Linux-specific, and Stage 7 targets the
/// Linux RTX remote.
pub fn load() LabError!void {
    if (loader != null) return;

    if (comptime builtin.os.tag != .linux) {
        std.log.err(
            "cuda bindings: load() is only supported on Linux (current target: {s})",
            .{@tagName(builtin.os.tag)},
        );
        return error.CudaError;
    }

    var libcuda = std.DynLib.open("libcuda.so.1") catch |err| {
        std.log.err("cuda bindings: dlopen(libcuda.so.1) failed: {s}", .{@errorName(err)});
        return error.CudaError;
    };
    errdefer libcuda.close();

    var libcublas = try openCublas();
    errdefer libcublas.close();

    // Resolve every symbol. If any dlsym fails the errdefer closes
    // both DynLib handles before this function returns. Field names
    // are repeated as the third argument to sym() because that is the
    // actual C symbol name — there is deliberately no string-mapping
    // layer between our struct and the ABI.
    const L = Loader{
        .libcuda = libcuda,
        .libcublas = libcublas,

        .cuInit = try sym(@FieldType(Loader, "cuInit"), &libcuda, "cuInit"),
        .cuDriverGetVersion = try sym(@FieldType(Loader, "cuDriverGetVersion"), &libcuda, "cuDriverGetVersion"),
        .cuDeviceGetCount = try sym(@FieldType(Loader, "cuDeviceGetCount"), &libcuda, "cuDeviceGetCount"),
        .cuDeviceGet = try sym(@FieldType(Loader, "cuDeviceGet"), &libcuda, "cuDeviceGet"),
        .cuDeviceGetName = try sym(@FieldType(Loader, "cuDeviceGetName"), &libcuda, "cuDeviceGetName"),
        .cuGetErrorString = try sym(@FieldType(Loader, "cuGetErrorString"), &libcuda, "cuGetErrorString"),

        .cuCtxCreate_v2 = try sym(@FieldType(Loader, "cuCtxCreate_v2"), &libcuda, "cuCtxCreate_v2"),
        .cuCtxDestroy_v2 = try sym(@FieldType(Loader, "cuCtxDestroy_v2"), &libcuda, "cuCtxDestroy_v2"),
        .cuCtxSetCurrent = try sym(@FieldType(Loader, "cuCtxSetCurrent"), &libcuda, "cuCtxSetCurrent"),
        .cuCtxSynchronize = try sym(@FieldType(Loader, "cuCtxSynchronize"), &libcuda, "cuCtxSynchronize"),

        .cuStreamCreate = try sym(@FieldType(Loader, "cuStreamCreate"), &libcuda, "cuStreamCreate"),
        .cuStreamDestroy_v2 = try sym(@FieldType(Loader, "cuStreamDestroy_v2"), &libcuda, "cuStreamDestroy_v2"),
        .cuStreamSynchronize = try sym(@FieldType(Loader, "cuStreamSynchronize"), &libcuda, "cuStreamSynchronize"),

        .cuMemAlloc_v2 = try sym(@FieldType(Loader, "cuMemAlloc_v2"), &libcuda, "cuMemAlloc_v2"),
        .cuMemFree_v2 = try sym(@FieldType(Loader, "cuMemFree_v2"), &libcuda, "cuMemFree_v2"),
        .cuMemcpyHtoD_v2 = try sym(@FieldType(Loader, "cuMemcpyHtoD_v2"), &libcuda, "cuMemcpyHtoD_v2"),
        .cuMemcpyDtoH_v2 = try sym(@FieldType(Loader, "cuMemcpyDtoH_v2"), &libcuda, "cuMemcpyDtoH_v2"),
        .cuMemcpyDtoD_v2 = try sym(@FieldType(Loader, "cuMemcpyDtoD_v2"), &libcuda, "cuMemcpyDtoD_v2"),
        .cuMemsetD32_v2 = try sym(@FieldType(Loader, "cuMemsetD32_v2"), &libcuda, "cuMemsetD32_v2"),

        .cuModuleLoadData = try sym(@FieldType(Loader, "cuModuleLoadData"), &libcuda, "cuModuleLoadData"),
        .cuModuleUnload = try sym(@FieldType(Loader, "cuModuleUnload"), &libcuda, "cuModuleUnload"),
        .cuModuleGetFunction = try sym(@FieldType(Loader, "cuModuleGetFunction"), &libcuda, "cuModuleGetFunction"),
        .cuLaunchKernel = try sym(@FieldType(Loader, "cuLaunchKernel"), &libcuda, "cuLaunchKernel"),

        .cublasCreate_v2 = try sym(@FieldType(Loader, "cublasCreate_v2"), &libcublas, "cublasCreate_v2"),
        .cublasDestroy_v2 = try sym(@FieldType(Loader, "cublasDestroy_v2"), &libcublas, "cublasDestroy_v2"),
        .cublasSetStream_v2 = try sym(@FieldType(Loader, "cublasSetStream_v2"), &libcublas, "cublasSetStream_v2"),
        .cublasSgemm_v2 = try sym(@FieldType(Loader, "cublasSgemm_v2"), &libcublas, "cublasSgemm_v2"),
        .cublasSgemmStridedBatched = try sym(@FieldType(Loader, "cublasSgemmStridedBatched"), &libcublas, "cublasSgemmStridedBatched"),
        .cublasGetStatusString = try sym(@FieldType(Loader, "cublasGetStatusString"), &libcublas, "cublasGetStatusString"),
    };

    loader = L;
}

/// Close both library handles and clear the global. Optional: normal
/// processes can skip this and let exit reclaim the handles.
///
/// Test suites that care about rigorous cleanup (e.g. under
/// compute-sanitizer) can call unload() in a defer. Note that dlopen
/// followed by dlclose followed by another dlopen of the same library
/// on Linux typically reuses a cached handle, so the second load()
/// will succeed; CUDA's internal state also persists across dlclose.
pub fn unload() void {
    if (loader) |*L| {
        L.libcuda.close();
        L.libcublas.close();
        loader = null;
    }
}

/// Convert a CUresult into LabError. On failure the numeric code and
/// the string from cuGetErrorString (when the loader is available)
/// are logged via std.log.err so the test runner surfaces them.
pub fn check(r: CUresult) LabError!void {
    if (r == CUDA_SUCCESS) return;
    if (loader) |L| {
        var msg: [*:0]const u8 = "(no error string)";
        _ = L.cuGetErrorString(r, &msg);
        std.log.err("cuda driver error {d}: {s}", .{ r, std.mem.span(msg) });
    } else {
        std.log.err("cuda driver error {d} (loader not initialised)", .{r});
    }
    return error.CudaError;
}

/// Convert a cublasStatus_t into LabError. Same pattern as check().
pub fn checkCublas(s: cublasStatus_t) LabError!void {
    if (s == CUBLAS_STATUS_SUCCESS) return;
    if (loader) |L| {
        const msg = L.cublasGetStatusString(s);
        std.log.err("cublas error {d}: {s}", .{ s, std.mem.span(msg) });
    } else {
        std.log.err("cublas error {d} (loader not initialised)", .{s});
    }
    return error.CudaError;
}
