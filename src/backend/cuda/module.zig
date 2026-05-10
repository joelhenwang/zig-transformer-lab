//!
//! zig-transformer-lab — CUDA PTX module loader and kernel launcher
//!                        (Stage 7, PR-zeta)
//!
//! Purpose:
//!   Provides the three functions every later CUDA op needs:
//!     * loadPtxFromFile(ctx, stem) — reads zig-out/ptx/<stem>.ptx
//!       into memory, calls cuModuleLoadData, and caches the
//!       resulting CUmodule in ctx.ptx_modules keyed by `stem`.
//!       Idempotent: a second call for the same stem returns the
//!       cached module without re-reading from disk.
//!     * getKernel(ctx, stem, name) — resolves a kernel symbol by
//!       name inside a previously loaded module.
//!     * launch(...) — packs Zig values into the `kernelParams`
//!       array cuLaunchKernel expects, then submits the launch on
//!       ctx.stream.
//!
//! Ownership:
//!   Loaded modules are owned by CudaContext; the map is freed in
//!   CudaContext.deinit via cuModuleUnload. A successful
//!   loadPtxFromFile transfers ownership of the CUmodule and the
//!   dup'd key string into the context.
//!
//! Errors:
//!   - error.IoError      — the .ptx file was missing or unreadable.
//!                          Most common cause: `zig build kernels
//!                          -Dcuda=true` was not run after a checkout.
//!   - error.CudaError    — cuModuleLoadData returned non-success
//!                          (usually CUDA_ERROR_NO_BINARY_FOR_GPU,
//!                          0x209, when the .ptx was compiled for a
//!                          different -arch=) or cuLaunchKernel
//!                          failed (illegal memory access, bad launch
//!                          config, etc.).
//!   - error.OutOfMemory  — the map-key allocator.dupe failed, or
//!                          the file-read allocator failed.
//!
//! Launch contract:
//!   Every launch goes on ctx.stream. In debug builds the caller
//!   typically invokes ctx.synchronize() after a launch to surface
//!   async failures at the site that caused them; in release the
//!   sync is skipped and only host-visible boundaries (toCpu, etc.)
//!   pay the barrier cost. We do NOT force-sync inside launch()
//!   itself — that would defeat stream overlap in the release path.
//!
//! PTX path convention:
//!   The `stem` argument is the file stem (no extension, no
//!   directory). Example: loadPtxFromFile(&ctx, "vector_add")
//!   reads `zig-out/ptx/vector_add.ptx`. The path is relative to
//!   the process's current working directory; test runners start
//!   from the repository root via build.zig's run-step setCwd, so
//!   the relative path resolves correctly in the normal workflow.
//!
//! Credits:
//!   Structure (file-stem-keyed cache, launch helper that packs
//!   pointer arrays) is original; the cuLaunchKernel ABI is
//!   documented in NVIDIA's CUDA Driver API reference. No
//!   third-party code copied.
//!

const std = @import("std");
const errors = @import("../../core/errors.zig");
const bindings = @import("bindings.zig");
const context_mod = @import("context.zig");

const LabError = errors.LabError;
const CudaContext = context_mod.CudaContext;

/// Maximum PTX file size we will read into memory. PTX is plain text
/// and our kernels are tiny; 16 MiB is ~1000x headroom for a single
/// transformer kernel. A larger file triggers `error.IoError` —
/// refuse rather than risk a runaway allocation.
const MAX_PTX_BYTES: usize = 16 * 1024 * 1024;

/// Read `zig-out/ptx/<stem>.ptx` from disk and register the resulting
/// module with the CUDA driver via `cuModuleLoadData`.
///
/// Idempotent: returns the cached handle on the second and subsequent
/// calls for the same stem. The map owns both the duplicated stem
/// string and the CUmodule; both are freed in CudaContext.deinit.
///
/// The `io: std.Io` parameter is the Zig 0.16 convention for file
/// operations (see AGENTS.md gotchas #21 and #28). Tests construct a
/// `std.Io.Threaded` once per test module and pass `threaded.io()`
/// here.
pub fn loadPtxFromFile(
    ctx: *CudaContext,
    io: std.Io,
    stem: []const u8,
) LabError!bindings.CUmodule {
    // Cache lookup. StringHashMap.get returns ?V; on hit, short-circuit.
    if (ctx.ptx_modules.get(stem)) |module| return module;

    // Compose the path `zig-out/ptx/<stem>.ptx`. A 256-byte buffer is
    // more than enough for any plausible kernel stem.
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(
        &path_buf,
        "zig-out/ptx/{s}.ptx",
        .{stem},
    ) catch return error.InvalidArgument;

    const cwd = std.Io.Dir.cwd();
    const ptx_bytes = cwd.readFileAlloc(
        io,
        path,
        ctx.allocator,
        .limited(MAX_PTX_BYTES),
    ) catch {
        std.log.err(
            "cuda module: failed to read PTX at '{s}' — did you run `zig build kernels -Dcuda=true`?",
            .{path},
        );
        return error.IoError;
    };
    defer ctx.allocator.free(ptx_bytes);

    // cuModuleLoadData requires a null-terminated PTX text blob. The
    // driver scans from the given pointer until it hits a NUL byte to
    // determine where the PTX ends. Without the explicit terminator
    // the driver reads past our slice end (returning error 218,
    // "PTX JIT compilation failed", when the trailing garbage fails
    // to parse as PTX) — exactly the first-kernel-launch failure we
    // hit on the remote before this fix landed.
    //
    // We allocate a separate buffer rather than relying on ambient
    // null bytes in the allocator's slack: Zig's GPA deliberately
    // poisons unused bytes in safety mode, and even benign allocators
    // give no guarantee about the byte past `ptr[len-1]`.
    const nt = ctx.allocator.alloc(u8, ptx_bytes.len + 1) catch return error.OutOfMemory;
    defer ctx.allocator.free(nt);
    @memcpy(nt[0..ptx_bytes.len], ptx_bytes);
    nt[ptx_bytes.len] = 0;

    // cuModuleLoadData takes a `const void *`. The driver parses the
    // PTX text, compiles it (PTX JIT) to SASS for the current device,
    // and internally retains whatever it needs — we can safely free
    // `nt` after this call returns.
    const L = bindings.loader.?;
    var module: bindings.CUmodule = null;
    try bindings.check(L.cuModuleLoadData(
        &module,
        @ptrCast(nt.ptr),
    ));
    errdefer _ = L.cuModuleUnload(module);

    // Dup the stem into context-owned memory so the map key outlives
    // the caller's slice (which typically lives on the stack).
    const key_owned = ctx.allocator.dupe(u8, stem) catch return error.OutOfMemory;
    errdefer ctx.allocator.free(key_owned);

    try ctx.ptx_modules.put(ctx.allocator, key_owned, module);
    return module;
}

/// Resolve a kernel function pointer inside a previously-loaded PTX
/// module. `name` is the `extern "C"` symbol name, passed verbatim
/// to `cuModuleGetFunction`.
///
/// Error cases:
///   - error.InvalidArgument: the module hasn't been loaded yet.
///   - error.CudaError:       the driver could not find a matching
///     symbol (usually the kernel lacked `extern "C"` and was
///     C++-mangled, or the name is misspelled).
pub fn getKernel(
    ctx: *CudaContext,
    module_stem: []const u8,
    name: [:0]const u8,
) LabError!bindings.CUfunction {
    const module = ctx.ptx_modules.get(module_stem) orelse {
        std.log.err(
            "cuda module: kernel '{s}' requested from module '{s}', but module is not loaded",
            .{ name, module_stem },
        );
        return error.InvalidArgument;
    };

    const L = bindings.loader.?;
    var f: bindings.CUfunction = null;
    try bindings.check(L.cuModuleGetFunction(&f, module, name.ptr));
    return f;
}

/// Launch a kernel on `ctx.stream`.
///
/// `params` is a list of pointers, one per kernel argument. Each
/// pointer must outlive the `cuLaunchKernel` call (the driver copies
/// the values before returning). The most common pattern is a
/// stack-local array of `?*anyopaque`:
///
///   const n: c_uint = @intCast(buf.len);
///   const args = [_]?*anyopaque{
///       @constCast(@as(*const anyopaque, @ptrCast(&a.ptr))),
///       @constCast(@as(*const anyopaque, @ptrCast(&b.ptr))),
///       @constCast(@as(*const anyopaque, @ptrCast(&c.ptr))),
///       @constCast(@as(*const anyopaque, @ptrCast(&n))),
///   };
///   try launch(&ctx, kfn, .{grid, 1, 1}, .{block, 1, 1}, 0, &args);
///
/// We do NOT synchronise after the launch. Debug callers typically
/// follow with `ctx.synchronize()` to catch async errors; release
/// callers let the work overlap with other stream operations.
pub fn launch(
    ctx: *CudaContext,
    f: bindings.CUfunction,
    grid: [3]c_uint,
    block: [3]c_uint,
    shared_bytes: c_uint,
    params: []const ?*anyopaque,
) LabError!void {
    const L = bindings.loader.?;
    // cuLaunchKernel's kernelParams expects a mutable array of
    // ?*anyopaque. Zig's const-slice cast is safe because the driver
    // only reads from the array (writes to memory via the pointed-to
    // values, not to the array itself). @ptrCast moves from
    // `[*]const ?*anyopaque` to `[*c]?*anyopaque` (the C ABI type).
    try bindings.check(L.cuLaunchKernel(
        f,
        grid[0],
        grid[1],
        grid[2],
        block[0],
        block[1],
        block[2],
        shared_bytes,
        ctx.stream,
        @constCast(@ptrCast(params.ptr)),
        null, // extra: unused
    ));
}
