//!
//! zig-transformer-lab — CUDA elementwise op dispatch (Stage 7, PR-eta)
//!
//! Purpose:
//!   Forward-only elementwise operations on CUDA tensors. Each
//!   function validates inputs, allocates a fresh CUDA output buffer,
//!   launches the corresponding kernel from the "elementwise" PTX
//!   module, and returns a new CUDA Tensor.
//!
//!   PR-eta deliberately keeps this module **forward-only**. Tape
//!   recording, backward support, and the CPU/CUDA routing in
//!   src/tensor/ops/elementwise.zig all land in a follow-up PR.
//!   Splitting the work this way gives us a self-contained unit
//!   that proves forward correctness + oracle parity before we
//!   touch the shared ops entry points or the autograd backward
//!   path.
//!
//! Layout contract:
//!   All inputs must be same-shape, contiguous, and CUDA-device.
//!   Non-contiguous inputs (e.g. transpose2d views) are rejected
//!   with `error.InvalidLayout`: the flat-index kernel would read
//!   the wrong elements otherwise. Stride-aware kernels can land in
//!   a later PR if an op needs them.
//!
//! Module prerequisite:
//!   Callers MUST load the "elementwise" PTX module before calling
//!   any dispatch function:
//!
//!     _ = try module.loadPtxFromFile(&ctx, io, "elementwise");
//!
//!   Failing this, getKernel returns error.InvalidArgument from
//!   module.zig. We do NOT auto-load here because loadPtxFromFile
//!   needs an `io: std.Io` parameter, and threading io through
//!   every op is API creep we want to avoid. Tests pre-load once
//!   at context setup; real applications do the same.
//!
//! CudaContext source:
//!   The ctx pointer is NOT threaded through the op signatures.
//!   Instead we pick it up from the input DeviceBuffer's `ctx`
//!   field — every CUDA tensor already records which context owns
//!   its allocation. This keeps the op API identical to the CPU
//!   path (`allocator, a, b, tape`) and removes a source of
//!   get-the-ctx-wrong bugs. requireSameDevice already guarantees
//!   both inputs share the same context.
//!
//! Memory ownership:
//!   Each dispatch function returns an owning CUDA Tensor (new
//!   DeviceBuffer with `owned = true`). Caller must call
//!   `out.storage.deinit(alloc)` — the allocator argument is
//!   unused on the CUDA branch, passed for signature uniformity.
//!
//! Output metadata:
//!   - shape == input shape
//!   - strides == computeStrides(shape) (output is always contiguous)
//!   - offset == 0
//!   - device == .cuda
//!   - owned == false (top-level alias; ownership in DeviceBuffer)
//!   - requires_grad carried from inputs (OR for binary ops; unused
//!     today because no tape)
//!   - grad / tape_node reset to null
//!

const std = @import("std");
const errors = @import("../../core/errors.zig");
const bindings = @import("bindings.zig");
const context_mod = @import("context.zig");
const mem = @import("mem.zig");
const module = @import("module.zig");

const tensor_mod = @import("../../tensor/tensor.zig");
const shape_mod = @import("../../tensor/shape.zig");

const LabError = errors.LabError;
const CudaContext = context_mod.CudaContext;
const DeviceBuffer = mem.DeviceBuffer;
const Tensor = tensor_mod.Tensor;
const Storage = tensor_mod.Storage;
const debugCheckInvariants = tensor_mod.debugCheckInvariants;
const requireSameDevice = tensor_mod.requireSameDevice;
const Shape = shape_mod.Shape;
const computeStrides = shape_mod.computeStrides;
const shape_equals = shape_mod.equals;
const shape_isContiguous = shape_mod.isContiguous;
const totalElements = shape_mod.totalElements;

/// Standard 1D launch block size. 256 threads per block is a
/// well-behaved default on every sm_89-class device: fits within
/// the 1024-thread-per-block hardware limit with plenty of room for
/// occupancy, matches the cache line of typical memory accesses,
/// and leaves warp count (8 warps/block) in a convenient range for
/// the scheduler.
const BLOCK_X: c_uint = 256;

/// Extract the DeviceBuffer from a CUDA tensor's storage, or return
/// `error.DeviceMismatch` if the tensor is not CUDA-backed. Used at
/// the top of every binary / unary op.
fn requireCudaBuffer(t: Tensor) LabError!DeviceBuffer {
    return switch (t.storage) {
        .cuda => |b| b,
        .cpu => error.DeviceMismatch,
    };
}

/// Shared preamble for every same-shape binary op: device + shape +
/// layout checks plus a freshly-allocated CUDA output DeviceBuffer
/// sized to match the inputs. Returns (ctx, a_buf, b_buf, out_buf).
///
/// The returned DeviceBuffer is ALREADY OWNED — on error from any
/// subsequent step the caller must `errdefer out_buf.deinit()` to
/// avoid leaking the allocation.
fn binaryPreamble(
    a: Tensor,
    b: Tensor,
) LabError!struct { ctx: *const CudaContext, a_buf: DeviceBuffer, b_buf: DeviceBuffer, out_buf: DeviceBuffer } {
    try requireSameDevice(a, b);
    if (!shape_equals(a.shape, b.shape)) return error.ShapeMismatch;

    // PR-eta policy: same-shape, same-layout, contiguous inputs only.
    // A transposed view would have non-contiguous strides, and our
    // flat-index kernel would read / write the wrong physical
    // positions. Reject loudly; PR-theta adds broadcast / stride
    // support.
    if (!shape_isContiguous(a.shape, a.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(b.shape, b.strides)) return error.InvalidLayout;

    const a_buf = try requireCudaBuffer(a);
    const b_buf = try requireCudaBuffer(b);
    // requireSameDevice + the CUDA storage check together imply the
    // two buffers came from the same CudaContext. We pick the ctx off
    // `a` since either would do.
    const ctx = a_buf.ctx;

    // Allocate the output on the same context. Size matches the
    // logical tensor (not the full source buffer) because output is
    // freshly contiguous — views on output don't share parent
    // storage the way toCuda copies do.
    const n = totalElements(a.shape);
    const out_buf = try DeviceBuffer.alloc(ctx, n);
    return .{ .ctx = ctx, .a_buf = a_buf, .b_buf = b_buf, .out_buf = out_buf };
}

/// Package an owning CUDA DeviceBuffer into a fresh CUDA Tensor with
/// contiguous strides over the given shape. The `.owned` compat alias
/// is false (PR-δ invariant: CUDA tensors never claim ownership on
/// the top-level field), `.data` is intentionally empty, `.grad` and
/// `.tape_node` reset.
fn wrapContiguousOutput(
    out_buf: DeviceBuffer,
    s: Shape,
    requires_grad: bool,
) Tensor {
    const t = Tensor{
        .data = &.{},
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cuda,
        .owned = false,
        .storage = .{ .cuda = out_buf },
        .offset = 0,
        .requires_grad = requires_grad,
        .grad = null,
        .tape_node = null,
    };
    debugCheckInvariants(t);
    return t;
}

/// Launch a binary elementwise kernel over n f32 elements on
/// `ctx.stream`. Kernel signature is assumed to be
/// `(const float*, const float*, float*, unsigned int)`.
fn launchBinary(
    ctx: *const CudaContext,
    kernel_name: [:0]const u8,
    a: DeviceBuffer,
    b: DeviceBuffer,
    out: DeviceBuffer,
    n: usize,
) LabError!void {
    // CudaContext.ptx_modules is stored on the struct; getKernel
    // looks up by stem. We cast away const on ctx so the signature
    // of `module.getKernel` (which takes *CudaContext) matches.
    // The cast is safe because getKernel only reads from
    // ptx_modules — it never mutates context state.
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try module.getKernel(mut_ctx, "elementwise", kernel_name);

    const n_arg: c_uint = @intCast(n);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&a.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&b.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
    };
    const grid_x: c_uint = @intCast((n + BLOCK_X - 1) / BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ BLOCK_X, 1, 1 },
        0,
        &args,
    );
}

/// Launch a unary elementwise kernel over n f32 elements. Kernel
/// signature is assumed to be `(const float*, float*, unsigned int)`.
fn launchUnary(
    ctx: *const CudaContext,
    kernel_name: [:0]const u8,
    a: DeviceBuffer,
    out: DeviceBuffer,
    n: usize,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try module.getKernel(mut_ctx, "elementwise", kernel_name);

    const n_arg: c_uint = @intCast(n);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&a.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
    };
    const grid_x: c_uint = @intCast((n + BLOCK_X - 1) / BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ BLOCK_X, 1, 1 },
        0,
        &args,
    );
}

/// Launch a scalar elementwise kernel (unary input plus an f32
/// scalar passed by value). Kernel signature:
/// `(const float*, float, float*, unsigned int)`.
fn launchScalar(
    ctx: *const CudaContext,
    kernel_name: [:0]const u8,
    a: DeviceBuffer,
    scalar: f32,
    out: DeviceBuffer,
    n: usize,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try module.getKernel(mut_ctx, "elementwise", kernel_name);

    // The scalar slot expects an f32 by value. Put it in a local so
    // we can take its address for the kernelParams array.
    const s_val = scalar;
    const n_arg: c_uint = @intCast(n);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&a.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&s_val))),
        @constCast(@as(*const anyopaque, @ptrCast(&out.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
    };
    const grid_x: c_uint = @intCast((n + BLOCK_X - 1) / BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ BLOCK_X, 1, 1 },
        0,
        &args,
    );
}

// ---------------------------------------------------------------------------
// Public ops
// ---------------------------------------------------------------------------

/// out = a + b. Same-shape, contiguous, CUDA inputs only.
pub fn add(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try binaryPreamble(a, b);
    errdefer p.out_buf.deinit();
    const n = totalElements(a.shape);
    try launchBinary(p.ctx, "elw_add", p.a_buf, p.b_buf, p.out_buf, n);
    return wrapContiguousOutput(p.out_buf, a.shape, a.requires_grad or b.requires_grad);
}

/// out = a - b.
pub fn sub(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try binaryPreamble(a, b);
    errdefer p.out_buf.deinit();
    const n = totalElements(a.shape);
    try launchBinary(p.ctx, "elw_sub", p.a_buf, p.b_buf, p.out_buf, n);
    return wrapContiguousOutput(p.out_buf, a.shape, a.requires_grad or b.requires_grad);
}

/// out = a * b (elementwise, NOT matmul).
pub fn mul(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try binaryPreamble(a, b);
    errdefer p.out_buf.deinit();
    const n = totalElements(a.shape);
    try launchBinary(p.ctx, "elw_mul", p.a_buf, p.b_buf, p.out_buf, n);
    return wrapContiguousOutput(p.out_buf, a.shape, a.requires_grad or b.requires_grad);
}

/// out = a / b (elementwise).
pub fn div(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try binaryPreamble(a, b);
    errdefer p.out_buf.deinit();
    const n = totalElements(a.shape);
    try launchBinary(p.ctx, "elw_div", p.a_buf, p.b_buf, p.out_buf, n);
    return wrapContiguousOutput(p.out_buf, a.shape, a.requires_grad or b.requires_grad);
}

/// out = -a. Unary op; no second operand to device-check against.
pub fn neg(a: Tensor) LabError!Tensor {
    if (!shape_isContiguous(a.shape, a.strides)) return error.InvalidLayout;
    const a_buf = try requireCudaBuffer(a);
    const ctx = a_buf.ctx;

    const n = totalElements(a.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchUnary(ctx, "elw_neg", a_buf, out_buf, n);
    return wrapContiguousOutput(out_buf, a.shape, a.requires_grad);
}

/// out = a + s.
pub fn addScalar(a: Tensor, s: f32) LabError!Tensor {
    if (!shape_isContiguous(a.shape, a.strides)) return error.InvalidLayout;
    const a_buf = try requireCudaBuffer(a);
    const ctx = a_buf.ctx;

    const n = totalElements(a.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchScalar(ctx, "elw_add_scalar", a_buf, s, out_buf, n);
    return wrapContiguousOutput(out_buf, a.shape, a.requires_grad);
}

/// out = a * s.
pub fn mulScalar(a: Tensor, s: f32) LabError!Tensor {
    if (!shape_isContiguous(a.shape, a.strides)) return error.InvalidLayout;
    const a_buf = try requireCudaBuffer(a);
    const ctx = a_buf.ctx;

    const n = totalElements(a.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchScalar(ctx, "elw_mul_scalar", a_buf, s, out_buf, n);
    return wrapContiguousOutput(out_buf, a.shape, a.requires_grad);
}

// ---------------------------------------------------------------------------
// Broadcast dispatch (PR-theta)
// ---------------------------------------------------------------------------
//
// Rank-4 stride-aware broadcast for add / sub / mul / div. Host
// computes output shape via broadcastShapes, pads both inputs' axis
// strides to rank 4 (left-aligned with 0s for missing axes and
// size-1 broadcast axes), and launches the general kernel.
//
// Non-contiguous input strides are fine here — the kernel walks
// inputs through their own strides and output through contiguous
// strides. The only remaining constraint is that output offset = 0
// and input layouts pack within their buffers; Storage invariants
// already guarantee that.

/// Effective stride of input `t` at output axis `out_axis`, where the
/// output has rank `out_ndim`. Input axes are right-aligned; any
/// leading output axis that has no counterpart, or a size-1 input
/// axis that gets broadcast, contributes zero stride.
fn effectiveStride(t: Tensor, out_ndim: usize, out_axis: usize) i32 {
    const in_ndim = t.shape.ndim();
    if (out_axis + in_ndim < out_ndim) return 0; // output has leading extra dims
    const in_axis = out_axis + in_ndim - out_ndim;
    if (t.shape.dims[in_axis] == 1) return 0; // broadcast size-1 axis
    return @intCast(t.strides.values[in_axis]);
}

/// Output shape padded / strides aligned for the rank-4 broadcast
/// kernel. `d0..d3` are output dims (left-padded with 1s), `a_s*`
/// and `b_s*` are effective strides (zero for broadcasted / missing
/// axes).
const BroadcastLayout = struct {
    d: [4]c_uint,
    a_s: [4]c_int,
    b_s: [4]c_int,
    n: usize,
};

fn computeBroadcastLayout(a: Tensor, b: Tensor, out_shape: Shape) BroadcastLayout {
    const out_ndim = out_shape.ndim();
    const pad = 4 - out_ndim;
    var L: BroadcastLayout = .{
        .d = .{ 1, 1, 1, 1 },
        .a_s = .{ 0, 0, 0, 0 },
        .b_s = .{ 0, 0, 0, 0 },
        .n = totalElements(out_shape),
    };
    for (0..out_ndim) |i| {
        L.d[i + pad] = @intCast(out_shape.dims[i]);
        L.a_s[i + pad] = effectiveStride(a, out_ndim, i);
        L.b_s[i + pad] = effectiveStride(b, out_ndim, i);
    }
    return L;
}

/// Launch a broadcast binary kernel. The kernel signature is
/// `(const float*, const float*, float*, unsigned int,
///   unsigned int x4 output dims, int x4 a strides, int x4 b strides)`.
fn launchBroadcastBinary(
    ctx: *const CudaContext,
    kernel_name: [:0]const u8,
    a: DeviceBuffer,
    b: DeviceBuffer,
    out: DeviceBuffer,
    L: BroadcastLayout,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try module.getKernel(mut_ctx, "elementwise", kernel_name);

    const n_arg: c_uint = @intCast(L.n);
    const d = L.d;
    const asv = L.a_s;
    const bsv = L.b_s;
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&a.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&b.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[0]))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[1]))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[2]))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[3]))),
        @constCast(@as(*const anyopaque, @ptrCast(&asv[0]))),
        @constCast(@as(*const anyopaque, @ptrCast(&asv[1]))),
        @constCast(@as(*const anyopaque, @ptrCast(&asv[2]))),
        @constCast(@as(*const anyopaque, @ptrCast(&asv[3]))),
        @constCast(@as(*const anyopaque, @ptrCast(&bsv[0]))),
        @constCast(@as(*const anyopaque, @ptrCast(&bsv[1]))),
        @constCast(@as(*const anyopaque, @ptrCast(&bsv[2]))),
        @constCast(@as(*const anyopaque, @ptrCast(&bsv[3]))),
    };
    const grid_x: c_uint = @intCast((L.n + BLOCK_X - 1) / BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ BLOCK_X, 1, 1 },
        0,
        &args,
    );
}

/// Shared preamble for every broadcast binary op: device check,
/// broadcast-shape derivation, output buffer alloc, stride
/// computation. Returns (ctx, a_buf, b_buf, out_buf, layout).
fn broadcastPreamble(
    a: Tensor,
    b: Tensor,
) LabError!struct { ctx: *const CudaContext, a_buf: DeviceBuffer, b_buf: DeviceBuffer, out_buf: DeviceBuffer, layout: BroadcastLayout, out_shape: Shape } {
    try requireSameDevice(a, b);
    const out_shape = shape_mod.broadcastShapes(a.shape, b.shape) catch return error.ShapeMismatch;

    const a_buf = try requireCudaBuffer(a);
    const b_buf = try requireCudaBuffer(b);
    const ctx = a_buf.ctx;

    const layout = computeBroadcastLayout(a, b, out_shape);
    const out_buf = try DeviceBuffer.alloc(ctx, layout.n);
    return .{
        .ctx = ctx,
        .a_buf = a_buf,
        .b_buf = b_buf,
        .out_buf = out_buf,
        .layout = layout,
        .out_shape = out_shape,
    };
}

/// out = a + b with NumPy broadcasting, rank <= 4.
pub fn addBroadcast(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try broadcastPreamble(a, b);
    errdefer p.out_buf.deinit();
    try launchBroadcastBinary(p.ctx, "elw_broadcast_add", p.a_buf, p.b_buf, p.out_buf, p.layout);
    return wrapContiguousOutput(p.out_buf, p.out_shape, a.requires_grad or b.requires_grad);
}

/// out = a - b with broadcasting.
pub fn subBroadcast(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try broadcastPreamble(a, b);
    errdefer p.out_buf.deinit();
    try launchBroadcastBinary(p.ctx, "elw_broadcast_sub", p.a_buf, p.b_buf, p.out_buf, p.layout);
    return wrapContiguousOutput(p.out_buf, p.out_shape, a.requires_grad or b.requires_grad);
}

/// out = a * b with broadcasting.
pub fn mulBroadcast(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try broadcastPreamble(a, b);
    errdefer p.out_buf.deinit();
    try launchBroadcastBinary(p.ctx, "elw_broadcast_mul", p.a_buf, p.b_buf, p.out_buf, p.layout);
    return wrapContiguousOutput(p.out_buf, p.out_shape, a.requires_grad or b.requires_grad);
}

/// out = a / b with broadcasting.
pub fn divBroadcast(a: Tensor, b: Tensor) LabError!Tensor {
    var p = try broadcastPreamble(a, b);
    errdefer p.out_buf.deinit();
    try launchBroadcastBinary(p.ctx, "elw_broadcast_div", p.a_buf, p.b_buf, p.out_buf, p.layout);
    return wrapContiguousOutput(p.out_buf, p.out_shape, a.requires_grad or b.requires_grad);
}
