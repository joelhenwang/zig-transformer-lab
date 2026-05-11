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

// ---------------------------------------------------------------------------
// Reductions + broadcast-copy (PR-iota)
// ---------------------------------------------------------------------------

/// Fill `buf` with `n` copies of the 32-bit pattern `bits`. Used to
/// zero-init reduction accumulators (bits=0) and to initialise
/// onesLike tensors (bits=0x3f800000 for 1.0f).
fn fillBits(buf: DeviceBuffer, bits: u32, n: usize) LabError!void {
    if (n == 0) return;
    const L = bindings.loader.?;
    try bindings.check(L.cuMemsetD32_v2(buf.ptr, bits, n));
}

/// Allocate a fresh CUDA Tensor of the given shape on `ctx`, initialised
/// to zero via cuMemsetD32_v2.
pub fn zerosOn(ctx: *const CudaContext, s: Shape) LabError!Tensor {
    const n = totalElements(s);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try fillBits(out_buf, 0, n);
    return wrapContiguousOutput(out_buf, s, false);
}

/// Fill an existing CUDA tensor with zeros in-place. Intended for
/// gradient buffers between optimizer steps (AdamW.zeroGrad).
pub fn fillZeros(t: Tensor) LabError!void {
    const buf = try requireCudaBuffer(t);
    try fillBits(buf, 0, buf.len);
}

/// Allocate a fresh CUDA Tensor of the given shape on `ctx`, initialised
/// to 1.0. The bit pattern of f32 1.0 is 0x3f800000; cuMemsetD32_v2
/// writes the same 4-byte pattern to every element, which is a valid
/// (and fast) way to fill a float buffer with a constant.
pub fn onesOn(ctx: *const CudaContext, s: Shape) LabError!Tensor {
    const n = totalElements(s);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    const one_bits: u32 = @bitCast(@as(f32, 1.0));
    try fillBits(out_buf, one_bits, n);
    return wrapContiguousOutput(out_buf, s, false);
}

/// Whole-tensor sum to a 1-element Tensor. Uses atomicAdd; result is
/// within a few ULPs of a deterministic tree sum, which is well
/// inside oracle tolerances.
///
/// Only contiguous inputs are supported (which matches our backward
/// call sites: gradients are freshly-contiguous).
pub fn sumAll(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    const n = totalElements(x.shape);
    // Output is a 1-element scalar tensor. Pre-zero via memset so
    // atomicAdd accumulates into 0, not undefined memory.
    var out_buf = try DeviceBuffer.alloc(ctx, 1);
    errdefer out_buf.deinit();
    try fillBits(out_buf, 0, 1);

    const kfn = try module.getKernel(mut_ctx, "reduce", "reduce_sum_all");
    const n_arg: c_uint = @intCast(n);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&x_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out_buf.ptr))),
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
    return wrapContiguousOutput(out_buf, Shape.init1D(1), x.requires_grad);
}

/// Sum along a single axis of a contiguous input. Output shape
/// matches input with `dims[axis] = 1`.
pub fn sumAxis(x: Tensor, axis: u2) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const axis_u: usize = @intCast(axis);
    if (axis_u >= x.shape.ndim()) return error.InvalidArgument;
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    const ndim = x.shape.ndim();
    const axis_size = x.shape.dims[axis_u];
    var inner_count: usize = 1;
    {
        var i = axis_u + 1;
        while (i < ndim) : (i += 1) inner_count *= x.shape.dims[i];
    }

    var out_dims = x.shape.dims;
    out_dims[axis_u] = 1;
    const out_shape = Shape{ .dims = out_dims, .rank = x.shape.rank };
    const out_n = totalElements(out_shape);

    var out_buf = try DeviceBuffer.alloc(ctx, out_n);
    errdefer out_buf.deinit();

    const kfn = try module.getKernel(mut_ctx, "reduce", "reduce_sum_axis");
    const out_n_arg: c_uint = @intCast(out_n);
    const axis_size_arg: c_uint = @intCast(axis_size);
    const inner_arg: c_uint = @intCast(inner_count);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&x_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out_n_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&axis_size_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&inner_arg))),
    };
    const grid_x: c_uint = @intCast((out_n + BLOCK_X - 1) / BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ BLOCK_X, 1, 1 },
        0,
        &args,
    );
    return wrapContiguousOutput(out_buf, out_shape, x.requires_grad);
}

/// Rank-4 stride-aware broadcast copy: project input `x` onto
/// output shape `target` using NumPy broadcasting rules. Handles
/// scalar-to-tensor expansion, identity copy, and everything
/// between.
pub fn broadcastTo(x: Tensor, target: Shape) LabError!Tensor {
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    // Layout: 4D output dims (left-padded with 1s), effective strides
    // for x that account for broadcast (size-1 or missing axis -> 0).
    const out_ndim = target.ndim();
    const pad = 4 - out_ndim;
    var d: [4]c_uint = .{ 1, 1, 1, 1 };
    var s: [4]c_int = .{ 0, 0, 0, 0 };
    for (0..out_ndim) |i| {
        d[i + pad] = @intCast(target.dims[i]);
        s[i + pad] = effectiveStride(x, out_ndim, i);
    }
    const n = totalElements(target);

    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();

    const kfn = try module.getKernel(mut_ctx, "reduce", "bcast_copy");
    const n_arg: c_uint = @intCast(n);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&x_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&n_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[0]))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[1]))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[2]))),
        @constCast(@as(*const anyopaque, @ptrCast(&d[3]))),
        @constCast(@as(*const anyopaque, @ptrCast(&s[0]))),
        @constCast(@as(*const anyopaque, @ptrCast(&s[1]))),
        @constCast(@as(*const anyopaque, @ptrCast(&s[2]))),
        @constCast(@as(*const anyopaque, @ptrCast(&s[3]))),
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
    return wrapContiguousOutput(out_buf, target, x.requires_grad);
}

/// Reduce `grad` to shape `target` by summing along axes where
/// target was broadcast to grad's shape. Mirrors the CPU
/// `ops_reduce.sumToShape` contract. Supports the three common
/// cases used by backward:
///   - grad.shape == target: identity DtoD copy via bcast_copy with
///     all strides 0'd except for the real axes (effectively a
///     contiguous gather).
///   - grad has extra leading dims / size-1 target dims: sum each
///     broadcast axis in rightmost-to-leftmost order.
pub fn sumToShape(grad: Tensor, target: Shape) LabError!Tensor {
    if (shape_equals(grad.shape, target)) {
        // Identity case: use broadcastTo with matching shape (which
        // produces a clean contiguous copy using the real strides).
        return try broadcastTo(grad, target);
    }

    const grad_ndim = grad.shape.ndim();
    const target_ndim = target.ndim();
    const extra_dims = if (grad_ndim > target_ndim) grad_ndim - target_ndim else 0;

    // Build a list of axes to sum, rightmost first so later axis
    // indices stay valid as earlier axes collapse.
    var axes_to_sum: [4]u2 = .{ 0, 0, 0, 0 };
    var n_axes: usize = 0;
    // Right-aligned size-1 target dims correspond to broadcast axes.
    for (0..target_ndim) |ti| {
        const grad_axis = extra_dims + ti;
        const grad_dim = grad.shape.dims[grad_axis];
        const target_dim = target.dims[ti];
        if (target_dim == 1 and grad_dim > 1) {
            axes_to_sum[n_axes] = @intCast(grad_axis);
            n_axes += 1;
        }
    }
    // Extra leading dims: always summed away.
    for (0..extra_dims) |i| {
        axes_to_sum[n_axes] = @intCast(i);
        n_axes += 1;
    }

    // Sum rightmost axis first so our axis indices remain valid as
    // we move. sumAxis keeps rank; we reshape at the end if the
    // target has lower rank than grad.
    var current = grad;
    var owned = false;
    var i: usize = n_axes;
    while (i > 0) : (i -= 1) {
        const axis = axes_to_sum[i - 1];
        const summed = try sumAxis(current, axis);
        if (owned) current.storage.deinit(undefined); // allocator unused on CUDA
        current = summed;
        owned = true;
    }

    // If target has lower rank than current, we need to reshape
    // (squeeze the size-1 axes). For the add_broadcast_2d_1d case
    // this means grad (2,3) -> sumAxis(0) -> (1,3) -> reshape to (3,).
    if (!shape_equals(current.shape, target)) {
        errdefer if (owned) current.storage.deinit(undefined);
        const view = try reshapedView(current, target);
        const reshaped = try broadcastTo(view, target);
        if (owned) current.storage.deinit(undefined);
        return reshaped;
    }
    return current;
}

/// View `x` with a new shape, preserving the contiguous element
/// order. x must be contiguous and totalElements must match.
fn reshapedView(x: Tensor, target: Shape) LabError!Tensor {
    if (totalElements(x.shape) != totalElements(target)) return error.ShapeMismatch;
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    var out = x;
    out.shape = target;
    out.strides = computeStrides(target);
    out.storage = switch (x.storage) {
        .cuda => |b| .{ .cuda = .{
            .ctx = b.ctx,
            .ptr = b.ptr,
            .len = b.len,
            .owned = false,
        } },
        .cpu => unreachable, // only called on CUDA tensors in sumToShape
    };
    out.owned = false;
    return out;
}

// ---------------------------------------------------------------------------
// Softmax / log-softmax (PR-lambda)
// ---------------------------------------------------------------------------
//
// Row-wise last-axis softmax. The host reshapes the logical
// (..., C) tensor into a flat (num_rows, C) layout by flattening
// all leading dims: num_rows = totalElements / C. Kernel uses one
// block per row, BLOCK_SIZE=256 threads, three passes (max, sum,
// normalise). Contiguous inputs only — broadcasting / non-contig
// layouts would need additional stride parameters.

const SOFTMAX_BLOCK_X: c_uint = 256;

fn launchSoftmaxKernel(
    ctx: *const CudaContext,
    kernel_name: [:0]const u8,
    x: DeviceBuffer,
    out: DeviceBuffer,
    num_rows: usize,
    C: usize,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try module.getKernel(mut_ctx, "softmax", kernel_name);

    const num_rows_arg: c_uint = @intCast(num_rows);
    const C_arg: c_uint = @intCast(C);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&x.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&num_rows_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&C_arg))),
    };
    try module.launch(
        mut_ctx,
        kfn,
        .{ @intCast(num_rows), 1, 1 }, // grid: one block per row
        .{ SOFTMAX_BLOCK_X, 1, 1 },
        0,
        &args,
    );
}

/// Row-wise softmax over the last axis of `x`. Input must be
/// contiguous; output has the same shape.
pub fn softmaxLastAxis(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const ndim = x.shape.ndim();
    if (ndim == 0) return error.ShapeMismatch;
    const C = x.shape.dims[ndim - 1];
    if (C == 0) return error.ShapeMismatch;
    const total = totalElements(x.shape);
    const num_rows = total / C;

    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;

    var out_buf = try DeviceBuffer.alloc(ctx, total);
    errdefer out_buf.deinit();

    try launchSoftmaxKernel(ctx, "softmax_last", x_buf, out_buf, num_rows, C);
    return wrapContiguousOutput(out_buf, x.shape, x.requires_grad);
}

/// Row-wise log-softmax over the last axis of `x`.
pub fn logSoftmaxLastAxis(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const ndim = x.shape.ndim();
    if (ndim == 0) return error.ShapeMismatch;
    const C = x.shape.dims[ndim - 1];
    if (C == 0) return error.ShapeMismatch;
    const total = totalElements(x.shape);
    const num_rows = total / C;

    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;

    var out_buf = try DeviceBuffer.alloc(ctx, total);
    errdefer out_buf.deinit();

    try launchSoftmaxKernel(ctx, "log_softmax_last", x_buf, out_buf, num_rows, C);
    return wrapContiguousOutput(out_buf, x.shape, x.requires_grad);
}

// ---------------------------------------------------------------------------
// Embedding (PR-mu)
// ---------------------------------------------------------------------------
//
// Forward: out[i, j] = weight[round(ids[i]), j]
// Backward: grad_weight[round(ids[i]), j] += grad_out[i, j]   (atomicAdd)
//
// Inputs must be contiguous. `ids` is a flat f32 tensor (our Tensor
// is f32-only) of any rank; the kernel treats it as length N =
// totalElements(ids.shape) and produces output of shape (..., D)
// where the leading dims come from ids and D is the inner dim of
// weight.

const EMBEDDING_BLOCK_X: c_uint = 256;

/// Forward gather. `weight : (V, D)`, `ids : (...,)`. Output shape
/// is `ids.shape ++ (D,)`; i.e. one extra trailing axis.
pub fn embeddingForward(weight: Tensor, ids: Tensor) LabError!Tensor {
    try requireSameDevice(weight, ids);
    if (weight.shape.ndim() != 2) return error.ShapeMismatch;
    if (!shape_isContiguous(weight.shape, weight.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(ids.shape, ids.strides)) return error.InvalidLayout;

    const V = weight.shape.dims[0];
    const D = weight.shape.dims[1];
    const N = totalElements(ids.shape);

    // Build output shape: ids.shape ++ (D,).
    const ids_ndim = ids.shape.ndim();
    if (ids_ndim + 1 > 4) return error.ShapeMismatch; // Shape max rank = 4
    var out_dims: [4]usize = .{ 1, 1, 1, 1 };
    for (0..ids_ndim) |i| out_dims[i] = ids.shape.dims[i];
    out_dims[ids_ndim] = D;
    const out_shape = Shape{ .dims = out_dims, .rank = @intCast(ids_ndim) };

    const w_buf = try requireCudaBuffer(weight);
    const id_buf = try requireCudaBuffer(ids);
    const ctx = w_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    var out_buf = try DeviceBuffer.alloc(ctx, N * D);
    errdefer out_buf.deinit();

    const kfn = try module.getKernel(mut_ctx, "embedding", "embedding_forward");
    const N_arg: c_uint = @intCast(N);
    const D_arg: c_uint = @intCast(D);
    const V_arg: c_uint = @intCast(V);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&w_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&id_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&N_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&D_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&V_arg))),
    };
    const total = N * D;
    const grid_x: c_uint = @intCast((total + EMBEDDING_BLOCK_X - 1) / EMBEDDING_BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ EMBEDDING_BLOCK_X, 1, 1 },
        0,
        &args,
    );
    return wrapContiguousOutput(out_buf, out_shape, weight.requires_grad);
}

/// Backward scatter-add. Takes `ids : (...,)`, `grad_out : (..., D)`,
/// and vocabulary size `V`. Returns `grad_weight : (V, D)`
/// zero-initialised + scatter-added.
pub fn embeddingBackward(ids: Tensor, grad_out: Tensor, V: usize) LabError!Tensor {
    try requireSameDevice(ids, grad_out);
    if (!shape_isContiguous(ids.shape, ids.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(grad_out.shape, grad_out.strides)) return error.InvalidLayout;
    const grad_ndim = grad_out.shape.ndim();
    if (grad_ndim < 1) return error.ShapeMismatch;
    const D = grad_out.shape.dims[grad_ndim - 1];
    const N = totalElements(grad_out.shape) / D;

    const ids_n = totalElements(ids.shape);
    if (ids_n != N) return error.ShapeMismatch;

    const id_buf = try requireCudaBuffer(ids);
    const g_buf = try requireCudaBuffer(grad_out);
    const ctx = id_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    var gw_buf = try DeviceBuffer.alloc(ctx, V * D);
    errdefer gw_buf.deinit();
    try fillBits(gw_buf, 0, V * D);

    const kfn = try module.getKernel(mut_ctx, "embedding", "embedding_backward");
    const N_arg: c_uint = @intCast(N);
    const D_arg: c_uint = @intCast(D);
    const V_arg: c_uint = @intCast(V);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&id_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&g_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&gw_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&N_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&D_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&V_arg))),
    };
    const total = N * D;
    const grid_x: c_uint = @intCast((total + EMBEDDING_BLOCK_X - 1) / EMBEDDING_BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ EMBEDDING_BLOCK_X, 1, 1 },
        0,
        &args,
    );
    return wrapContiguousOutput(gw_buf, Shape.init2D(V, D), false);
}

// ---------------------------------------------------------------------------
// Unary ops (Milestone 2 — GELU, sqrt, exp, log)
// ---------------------------------------------------------------------------
//
// Same flat-index pattern as the existing unary `neg`: validate the
// input is CUDA + contiguous, allocate a fresh output, launch the
// matching kernel from the "unary" PTX module. Each dispatch caller
// is expected to have pre-loaded the module via
//   `module.loadPtxFromFile(ctx, io, "unary")`
// (same convention the other modules follow).

/// Shared launcher for a unary kernel with signature
/// `(const float*, float*, unsigned int)`. The kernel_name is the
/// symbol inside the "unary" PTX module.
fn launchUnaryKernel(
    ctx: *const CudaContext,
    kernel_name: [:0]const u8,
    a: DeviceBuffer,
    out: DeviceBuffer,
    n: usize,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try module.getKernel(mut_ctx, "unary", kernel_name);
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

/// out = 0.5 * x * (1 + erf(x / sqrt(2))). Contiguous CUDA input.
pub fn geluExact(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const n = totalElements(x.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchUnaryKernel(ctx, "unary_gelu_exact", x_buf, out_buf, n);
    return wrapContiguousOutput(out_buf, x.shape, x.requires_grad);
}

/// grad_in = grad_out * gelu'(x), where x was the original input to
/// geluExact and grad_out is the upstream gradient. All three tensors
/// must be same-shape contiguous CUDA.
pub fn geluExactBackward(x: Tensor, grad_out: Tensor) LabError!Tensor {
    try requireSameDevice(x, grad_out);
    if (!shape_equals(x.shape, grad_out.shape)) return error.ShapeMismatch;
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(grad_out.shape, grad_out.strides)) return error.InvalidLayout;

    const x_buf = try requireCudaBuffer(x);
    const g_buf = try requireCudaBuffer(grad_out);
    const ctx = x_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);
    const n = totalElements(x.shape);

    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();

    const kfn = try module.getKernel(mut_ctx, "unary", "unary_gelu_exact_backward");
    const n_arg: c_uint = @intCast(n);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&x_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&g_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&out_buf.ptr))),
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
    return wrapContiguousOutput(out_buf, x.shape, false);
}

/// out = sqrtf(x). Negative inputs produce NaN (IEEE 754).
pub fn sqrt(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const n = totalElements(x.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchUnaryKernel(ctx, "unary_sqrt", x_buf, out_buf, n);
    return wrapContiguousOutput(out_buf, x.shape, x.requires_grad);
}

/// out = expf(x).
pub fn exp(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const n = totalElements(x.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchUnaryKernel(ctx, "unary_exp", x_buf, out_buf, n);
    return wrapContiguousOutput(out_buf, x.shape, x.requires_grad);
}

/// out = logf(x). Zero / negative inputs produce -inf / NaN.
pub fn log(x: Tensor) LabError!Tensor {
    if (!shape_isContiguous(x.shape, x.strides)) return error.InvalidLayout;
    const x_buf = try requireCudaBuffer(x);
    const ctx = x_buf.ctx;
    const n = totalElements(x.shape);
    var out_buf = try DeviceBuffer.alloc(ctx, n);
    errdefer out_buf.deinit();
    try launchUnaryKernel(ctx, "unary_log", x_buf, out_buf, n);
    return wrapContiguousOutput(out_buf, x.shape, x.requires_grad);
}

// ---------------------------------------------------------------------------
// Cross-entropy (Milestone 1 — completes PR-mu)
// ---------------------------------------------------------------------------
//
// Fused forward + grad_logits for mean cross-entropy. One kernel
// launch produces both the scalar loss and the (N, C) gradient w.r.t.
// logits; backward.zig just DtoD-clones the saved grad.
//
// The kernel signature and math contract are documented in
// src/backend/cuda/kernels/ce_loss.cu. Caller-side requirements:
//   - logits  : (N, C) contiguous CUDA
//   - targets : (N,)   contiguous CUDA, f32 (rounded per row)
//   - Ordinary PTX module prerequisite: loadPtxFromFile(ctx, io, "ce_loss").

const CE_BLOCK_X: c_uint = 256;

/// One launch that writes the scalar mean loss AND the full
/// (N, C) grad_logits buffer. Both are owning CUDA tensors; caller
/// must `storage.deinit(alloc)` each when done (on the CUDA path the
/// allocator is unused, passed for signature uniformity with CPU).
///
/// Layout:
///   loss.shape        = (1,)
///   grad_logits.shape = (N, C)   (matches logits.shape)
pub fn crossEntropyFused(
    logits: Tensor,
    targets: Tensor,
) LabError!struct { loss: Tensor, grad_logits: Tensor } {
    try requireSameDevice(logits, targets);
    if (logits.shape.ndim() != 2) return error.ShapeMismatch;
    if (targets.shape.ndim() != 1) return error.ShapeMismatch;
    if (!shape_isContiguous(logits.shape, logits.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(targets.shape, targets.strides)) return error.InvalidLayout;

    const N = logits.shape.dims[0];
    const C = logits.shape.dims[1];
    if (targets.shape.dims[0] != N) return error.ShapeMismatch;

    const lg_buf = try requireCudaBuffer(logits);
    const tg_buf = try requireCudaBuffer(targets);
    const ctx = lg_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    // Pre-zero the 1-element loss buffer so the kernel's atomicAdd
    // accumulates from 0 and not from uninitialised device memory.
    // If this memset is skipped the loss is garbage + correct-delta.
    var loss_buf = try DeviceBuffer.alloc(ctx, 1);
    errdefer loss_buf.deinit();
    try fillBits(loss_buf, 0, 1);

    // grad_logits need not be pre-zeroed — the kernel writes every
    // element exactly once in pass 3.
    var grad_buf = try DeviceBuffer.alloc(ctx, N * C);
    errdefer grad_buf.deinit();

    const kfn = try module.getKernel(mut_ctx, "ce_loss", "ce_fused");
    const N_arg: c_uint = @intCast(N);
    const C_arg: c_uint = @intCast(C);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&lg_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&tg_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&loss_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&grad_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&N_arg))),
        @constCast(@as(*const anyopaque, @ptrCast(&C_arg))),
    };
    // Grid: one block per row. Kernel's block_reduce_* routines
    // require BLOCK_SIZE=256 (power of two). Softmax uses the same
    // geometry — see softmax.cu for the derivation.
    try module.launch(
        mut_ctx,
        kfn,
        .{ @intCast(N), 1, 1 },
        .{ CE_BLOCK_X, 1, 1 },
        0,
        &args,
    );

    const loss_tensor = wrapContiguousOutput(loss_buf, Shape.init1D(1), logits.requires_grad);
    // grad_logits is an intermediate passed to the tape; its
    // `requires_grad` doesn't matter (backward never traverses
    // through it) but we set it false to avoid confusing debug
    // inspection.
    const grad_tensor = wrapContiguousOutput(grad_buf, logits.shape, false);
    return .{ .loss = loss_tensor, .grad_logits = grad_tensor };
}

/// DtoD-clone a CUDA tensor into a fresh owning buffer with the same
/// shape/strides. Used by `backwardCrossEntropy` to hand the tape's
/// saved grad back as the parent gradient without leaking the
/// tape-owned copy.
pub fn cloneDevice(t: Tensor) LabError!Tensor {
    if (!shape_isContiguous(t.shape, t.strides)) return error.InvalidLayout;
    const src_buf = try requireCudaBuffer(t);
    const ctx = src_buf.ctx;
    var out_buf = try DeviceBuffer.alloc(ctx, src_buf.len);
    errdefer out_buf.deinit();
    try out_buf.copyFromDevice(src_buf);
    return wrapContiguousOutput(out_buf, t.shape, t.requires_grad);
}
//
// ---------------------------------------------------------------------------
// AdamW step (PR-mu)
// ---------------------------------------------------------------------------
//
// In-place per-parameter AdamW update. The host computes the bias-
// correction scalars `bc1 = 1 / (1 - beta1^t)` and `bc2 = 1 / (1 - beta2^t)`
// from the current step count, then launches this kernel once per
// parameter with its (param, grad, m, v) buffers and the four
// hyperparameters.

const ADAMW_BLOCK_X: c_uint = 256;

pub const AdamwConfig = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// Bias-correction scalars for the current step t. Computed on
    /// the host (simple pow loop) and passed in by value.
    bc1: f32,
    bc2: f32,
};

/// One AdamW step over `param`, `grad`, `m`, `v`. All four must be
/// contiguous, same-shape CUDA tensors. Updates `param`, `m`, `v`
/// in place. `grad` is read-only.
pub fn adamwStep(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    cfg: AdamwConfig,
) LabError!void {
    try requireSameDevice(param, grad);
    try requireSameDevice(param, m);
    try requireSameDevice(param, v);
    if (!shape_equals(param.shape, grad.shape)) return error.ShapeMismatch;
    if (!shape_equals(param.shape, m.shape)) return error.ShapeMismatch;
    if (!shape_equals(param.shape, v.shape)) return error.ShapeMismatch;
    if (!shape_isContiguous(param.shape, param.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(grad.shape, grad.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(m.shape, m.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(v.shape, v.strides)) return error.InvalidLayout;

    const p_buf = try requireCudaBuffer(param);
    const g_buf = try requireCudaBuffer(grad);
    const m_buf = try requireCudaBuffer(m);
    const v_buf = try requireCudaBuffer(v);
    const ctx = p_buf.ctx;
    const mut_ctx: *CudaContext = @constCast(ctx);

    const N = totalElements(param.shape);
    const kfn = try module.getKernel(mut_ctx, "adamw", "adamw_step");

    const lr = cfg.lr;
    const b1 = cfg.beta1;
    const b2 = cfg.beta2;
    const eps = cfg.eps;
    const wd = cfg.weight_decay;
    const bc1 = cfg.bc1;
    const bc2 = cfg.bc2;
    const N_arg: c_uint = @intCast(N);
    const args = [_]?*anyopaque{
        @constCast(@as(*const anyopaque, @ptrCast(&p_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&g_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&m_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&v_buf.ptr))),
        @constCast(@as(*const anyopaque, @ptrCast(&lr))),
        @constCast(@as(*const anyopaque, @ptrCast(&b1))),
        @constCast(@as(*const anyopaque, @ptrCast(&b2))),
        @constCast(@as(*const anyopaque, @ptrCast(&eps))),
        @constCast(@as(*const anyopaque, @ptrCast(&wd))),
        @constCast(@as(*const anyopaque, @ptrCast(&bc1))),
        @constCast(@as(*const anyopaque, @ptrCast(&bc2))),
        @constCast(@as(*const anyopaque, @ptrCast(&N_arg))),
    };
    const grid_x: c_uint = @intCast((N + ADAMW_BLOCK_X - 1) / ADAMW_BLOCK_X);
    try module.launch(
        mut_ctx,
        kfn,
        .{ grid_x, 1, 1 },
        .{ ADAMW_BLOCK_X, 1, 1 },
        0,
        &args,
    );
}
