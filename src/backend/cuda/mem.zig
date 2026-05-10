//!
//! zig-transformer-lab — CUDA DeviceBuffer (Stage 7, PR-gamma)
//!
//! Purpose:
//!   RAII wrapper around `cuMemAlloc_v2` / `cuMemFree_v2`. Stores a
//!   length in `f32` elements together with its `CUdeviceptr`, and
//!   offers the four memcpy variants we'll need from PR-epsilon
//!   onward: host -> device, device -> host, device -> device, and a
//!   `toHost` convenience that pairs alloc + DtoH for callers who
//!   just want a `[]f32` back from the GPU.
//!
//! Scope (intentional omissions):
//!   - No Tensor integration. PR-delta extends Storage to carry a
//!     DeviceBuffer; PR-epsilon adds `Tensor.toCuda` / `toCpu`.
//!   - No stream-aware async copies yet. All memcpys use the
//!     synchronous v2 Driver-API variants. Stream overlap is a
//!     later-stage performance concern.
//!   - No pinned (page-locked) host memory. We use plain allocator
//!     allocations for host-side buffers. Pinned memory is cheap to
//!     add later if profiling shows transfer dominance.
//!
//! Ownership:
//!   A `DeviceBuffer` with `owned = true` will call
//!   `cuMemFree_v2(ptr)` on `deinit()`. A view over an existing
//!   buffer (e.g. a sub-slice; not used yet) would set
//!   `owned = false`. In PR-gamma every DeviceBuffer we construct
//!   is owning.
//!
//!   The buffer holds a `*const CudaContext` pointer so the
//!   destructor knows which Driver API to call against. **Callers
//!   must deinit all DeviceBuffers before deiniting the context
//!   they were allocated under.** `cuMemFree_v2` against a
//!   destroyed context is undefined behaviour on the CUDA side.
//!
//! Errors:
//!   - error.CudaError: allocation failed (usually
//!     CUDA_ERROR_OUT_OF_MEMORY), host/device sizes don't match, or
//!     a memcpy returned a non-success code.
//!   - error.OutOfMemory: only from `toHost` when the host-side
//!     allocator fails before any CUDA call is made.
//!
//! Units:
//!   Throughout this file, `len` and loop bounds are measured in
//!   `f32` elements, never bytes. Byte counts are computed at the
//!   API boundary via `len * @sizeOf(f32)` and stay local to one
//!   function. This keeps call sites (and later Storage code)
//!   independent of the element type.
//!
//! Credits:
//!   The alloc/view/free shape matches what zigrad, candle, and
//!   tinygrad all converge on; the implementation is original.
//!

const std = @import("std");
const errors = @import("../../core/errors.zig");
const bindings = @import("bindings.zig");
const context_mod = @import("context.zig");

const LabError = errors.LabError;
const CudaContext = context_mod.CudaContext;

/// Typed (in the sense that `len` counts `f32` elements) RAII handle
/// over a `cuMemAlloc_v2` allocation.
///
/// Invariants maintained across every method:
///   - `ptr` is a valid CUDA device address or 0 (only after deinit).
///   - `len` is in f32 elements; total bytes = `len * 4`.
///   - The context pointed to by `ctx` is live.
///   - `owned = true` means this struct will free `ptr` on deinit.
pub const DeviceBuffer = struct {
    ctx: *const CudaContext,
    ptr: bindings.CUdeviceptr,
    /// Element count in f32. Multiply by `@sizeOf(f32)` for bytes.
    len: usize,
    owned: bool,

    /// Allocate `n` f32 elements on the device. Returned buffer is
    /// owning — caller must `deinit` before the context is destroyed.
    ///
    /// `n == 0` is allowed and returns a DeviceBuffer with `ptr = 0`,
    /// `owned = false`, matching the "empty slice" convention used
    /// by the CPU storage. Calling memcpy on such a buffer is a
    /// zero-byte no-op.
    pub fn alloc(ctx: *const CudaContext, n: usize) LabError!DeviceBuffer {
        if (n == 0) {
            return .{ .ctx = ctx, .ptr = 0, .len = 0, .owned = false };
        }
        const L = bindings.loader.?;
        var ptr: bindings.CUdeviceptr = 0;
        try bindings.check(L.cuMemAlloc_v2(&ptr, n * @sizeOf(f32)));
        return .{ .ctx = ctx, .ptr = ptr, .len = n, .owned = true };
    }

    /// Allocate + copy host slice onto the device in one shot. Equivalent to
    /// `try alloc(ctx, src.len)` followed by `copyFromHost(src)`, but the
    /// combined form is what callers actually want and avoids a partially-
    /// initialised DeviceBuffer on HtoD failure (errdefer rewinds alloc).
    pub fn fromHost(ctx: *const CudaContext, src: []const f32) LabError!DeviceBuffer {
        var buf = try alloc(ctx, src.len);
        errdefer buf.deinit();
        try buf.copyFromHost(src);
        return buf;
    }

    /// Copy device contents back into a fresh host `[]f32` owned by the
    /// caller (freed with the same allocator that was passed in).
    pub fn toHost(self: DeviceBuffer, host_alloc: std.mem.Allocator) LabError![]f32 {
        const out = try host_alloc.alloc(f32, self.len);
        errdefer host_alloc.free(out);
        try self.copyToHost(out);
        return out;
    }

    /// HtoD: copy `src` into this buffer. Sizes must match exactly
    /// in element count; mismatch is a programmer error surfaced as
    /// `error.ShapeMismatch` rather than truncation.
    pub fn copyFromHost(self: DeviceBuffer, src: []const f32) LabError!void {
        if (src.len != self.len) {
            // Caller-side mistake: the error return is the semantic
            // signal; the log is a diagnostic breadcrumb. Use warn
            // rather than err so a test that deliberately triggers
            // this path (expectError) does not fail the test runner.
            std.log.warn(
                "DeviceBuffer.copyFromHost: host.len={d} != device.len={d}",
                .{ src.len, self.len },
            );
            return error.ShapeMismatch;
        }
        if (self.len == 0) return;
        const L = bindings.loader.?;
        try bindings.check(L.cuMemcpyHtoD_v2(
            self.ptr,
            @ptrCast(src.ptr),
            self.len * @sizeOf(f32),
        ));
    }

    /// DtoH: copy this buffer into `dst`. Sizes must match exactly.
    pub fn copyToHost(self: DeviceBuffer, dst: []f32) LabError!void {
        if (dst.len != self.len) {
            std.log.warn(
                "DeviceBuffer.copyToHost: host.len={d} != device.len={d}",
                .{ dst.len, self.len },
            );
            return error.ShapeMismatch;
        }
        if (self.len == 0) return;
        const L = bindings.loader.?;
        try bindings.check(L.cuMemcpyDtoH_v2(
            @ptrCast(dst.ptr),
            self.ptr,
            self.len * @sizeOf(f32),
        ));
    }

    /// DtoD: copy `src` into this buffer. Both buffers must belong
    /// to the same context and have identical element counts.
    pub fn copyFromDevice(self: DeviceBuffer, src: DeviceBuffer) LabError!void {
        if (src.len != self.len) {
            std.log.warn(
                "DeviceBuffer.copyFromDevice: src.len={d} != dst.len={d}",
                .{ src.len, self.len },
            );
            return error.ShapeMismatch;
        }
        if (self.ctx != src.ctx) {
            std.log.warn("DeviceBuffer.copyFromDevice: cross-context copy is not supported", .{});
            return error.DeviceMismatch;
        }
        if (self.len == 0) return;
        const L = bindings.loader.?;
        try bindings.check(L.cuMemcpyDtoD_v2(
            self.ptr,
            src.ptr,
            self.len * @sizeOf(f32),
        ));
    }

    /// Release the allocation if owned. Callers may safely call
    /// deinit on non-owning views; it is a no-op.
    ///
    /// Precondition: the owning CudaContext must still be alive.
    /// Freeing a buffer whose context has already been destroyed is
    /// undefined behaviour per the CUDA Driver API. Tests exercise
    /// this by deiniting buffers inside the same lexical scope as
    /// the context, via `defer` in reverse order of construction.
    pub fn deinit(self: *DeviceBuffer) void {
        if (self.owned and self.ptr != 0) {
            const L = bindings.loader.?;
            const r = L.cuMemFree_v2(self.ptr);
            if (r != bindings.CUDA_SUCCESS) {
                std.log.warn(
                    "DeviceBuffer.deinit: cuMemFree_v2(ptr=0x{x}, len={d}) -> {d} (ignored)",
                    .{ self.ptr, self.len, r },
                );
            }
        }
        self.* = undefined;
    }
};
