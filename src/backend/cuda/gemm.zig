//!
//! zig-transformer-lab — CUDA GEMM via cuBLAS (Stage 7, PR-κ)
//!
//! Purpose:
//!   Row-major matmul for CUDA tensors. Wraps cublasSgemm_v2 and
//!   cublasSgemmStridedBatched using the operand-swap trick
//!   documented in docs/08_backends_cuda.md §3-§5. Every
//!   device-side matmul in the library — Linear forward, attention
//!   Q@K^T, attention A@V, lm_head — routes through this file.
//!
//! Why this file is small:
//!   All the hard work is in cuBLAS itself (NVIDIA-tuned SASS for
//!   every GPU generation). Our job is to translate between our
//!   row-major Tensor layout and cuBLAS's column-major ABI, once,
//!   correctly. The derivation lives in docs/08; reading it before
//!   modifying this file is mandatory.
//!
//! Scope (PR-κ):
//!   - `matmul(a, b)`:        2-D row-major, contiguous inputs.
//!   - `matmulBatch(a, b)`:   3-D row-major, same batch dim on
//!                             both inputs, contiguous.
//!   - No transposed-operand API; backward code materialises
//!     transposes as views.
//!   - No mixed precision.
//!   - No cuBLAS-Lt / tensor-core-opt paths.
//!
//! Layout contract:
//!   Inputs must be contiguous row-major. Non-contiguous inputs
//!   (e.g. transpose2d views) are rejected with
//!   `error.InvalidLayout`. Callers who hit this restriction
//!   should `copyTo` into a contiguous buffer before calling.
//!
//! Error contract:
//!   - error.DeviceMismatch     inputs not both CUDA, or different contexts.
//!   - error.ShapeMismatch      ranks wrong, or K dim disagrees.
//!   - error.InvalidLayout      non-contiguous input.
//!   - error.CudaError          cuBLAS returned non-success (logged
//!                              via bindings.checkCublas).
//!
//! Credits:
//!   Derivation is standard (any BLAS tutorial covers it). The
//!   specific argument mapping here matches the public PyTorch
//!   ATen CUDA backend's implementation as a cross-check. No
//!   third-party code copied.
//!

const std = @import("std");
const errors = @import("../../core/errors.zig");
const bindings = @import("bindings.zig");
const context_mod = @import("context.zig");
const mem = @import("mem.zig");

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
const shape_isContiguous = shape_mod.isContiguous;

/// Extract a CUDA DeviceBuffer from a tensor, else DeviceMismatch.
fn requireCudaBuffer(t: Tensor) LabError!DeviceBuffer {
    return switch (t.storage) {
        .cuda => |b| b,
        .cpu => error.DeviceMismatch,
    };
}

/// Wrap an owning output DeviceBuffer + shape into a Tensor.
fn wrapOut(buf: DeviceBuffer, s: Shape, requires_grad: bool) Tensor {
    const t = Tensor{
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cuda,
        .storage = .{ .cuda = buf },
        .offset = 0,
        .requires_grad = requires_grad,
        .grad = null,
        .tape_node = null,
    };
    debugCheckInvariants(t);
    return t;
}

/// Row-major 2-D matmul. `a : (M, K)`, `b : (K, N)` → `out : (M, N)`.
///
/// Implementation (see docs/08 §3.5):
///   C_cm = B_cm · A_cm → call sgemm with operand swap:
///     m = N, n = M, k = K
///     A_arg = b.ptr, lda = N
///     B_arg = a.ptr, ldb = K
///     C     = out.ptr, ldc = N
pub fn matmul(a: Tensor, b: Tensor) LabError!Tensor {
    try requireSameDevice(a, b);
    if (a.shape.ndim() != 2 or b.shape.ndim() != 2) return error.ShapeMismatch;

    const M = a.shape.dims[0];
    const K = a.shape.dims[1];
    const Kb = b.shape.dims[0];
    const N = b.shape.dims[1];
    if (K != Kb) return error.ShapeMismatch;

    if (!shape_isContiguous(a.shape, a.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(b.shape, b.strides)) return error.InvalidLayout;

    const a_buf = try requireCudaBuffer(a);
    const b_buf = try requireCudaBuffer(b);
    const ctx = a_buf.ctx;

    var out_buf = try DeviceBuffer.alloc(ctx, M * N);
    errdefer out_buf.deinit();

    const L = bindings.loader.?;
    const alpha: f32 = 1.0;
    const beta: f32 = 0.0;

    // See docs/08 §3.5: this is the operand-swap identity. Do not
    // reorder these arguments without re-reading the derivation.
    try bindings.checkCublas(L.cublasSgemm_v2(
        ctx.cublas,
        bindings.CUBLAS_OP_N, // transa — treat B_rm's bytes as col-major (no transpose)
        bindings.CUBLAS_OP_N, // transb — same for A_rm
        @intCast(N), // m  = cols of C_rm
        @intCast(M), // n  = rows of C_rm
        @intCast(K), // k  = contraction
        &alpha,
        b_buf.ptr, // A_arg = B_rm_ptr (first cuBLAS operand)
        @intCast(N), // lda  = cols of B_rm
        a_buf.ptr, // B_arg = A_rm_ptr
        @intCast(K), // ldb  = cols of A_rm
        &beta,
        out_buf.ptr,
        @intCast(N), // ldc  = cols of C_rm
    ));

    return wrapOut(out_buf, Shape.init2D(M, N), a.requires_grad or b.requires_grad);
}

/// Row-major 3-D batched matmul. `a : (B, M, K)`, `b : (B, K, N)` →
/// `out : (B, M, N)`.
///
/// Strides (see docs/08 §5):
///   strideA (cuBLAS "A") = K * N   ← elements between B_rm batches
///   strideB (cuBLAS "B") = M * K   ← elements between A_rm batches
///   strideC              = M * N
pub fn matmulBatch(a: Tensor, b: Tensor) LabError!Tensor {
    try requireSameDevice(a, b);
    if (a.shape.ndim() != 3 or b.shape.ndim() != 3) return error.ShapeMismatch;

    const Bat = a.shape.dims[0];
    if (b.shape.dims[0] != Bat) return error.ShapeMismatch;
    const M = a.shape.dims[1];
    const K = a.shape.dims[2];
    const Kb = b.shape.dims[1];
    const N = b.shape.dims[2];
    if (K != Kb) return error.ShapeMismatch;

    if (!shape_isContiguous(a.shape, a.strides)) return error.InvalidLayout;
    if (!shape_isContiguous(b.shape, b.strides)) return error.InvalidLayout;

    const a_buf = try requireCudaBuffer(a);
    const b_buf = try requireCudaBuffer(b);
    const ctx = a_buf.ctx;

    var out_buf = try DeviceBuffer.alloc(ctx, Bat * M * N);
    errdefer out_buf.deinit();

    const L = bindings.loader.?;
    const alpha: f32 = 1.0;
    const beta: f32 = 0.0;

    try bindings.checkCublas(L.cublasSgemmStridedBatched(
        ctx.cublas,
        bindings.CUBLAS_OP_N,
        bindings.CUBLAS_OP_N,
        @intCast(N),
        @intCast(M),
        @intCast(K),
        &alpha,
        b_buf.ptr, // A_arg = B_rm pointer
        @intCast(N), // lda
        @as(i64, @intCast(K * N)), // strideA = bytes between consecutive B_rm matrices
        a_buf.ptr, // B_arg = A_rm pointer
        @intCast(K), // ldb
        @as(i64, @intCast(M * K)), // strideB
        &beta,
        out_buf.ptr,
        @intCast(N),
        @as(i64, @intCast(M * N)), // strideC
        @intCast(Bat), // batchCount
    ));

    return wrapOut(out_buf, Shape.init3D(Bat, M, N), a.requires_grad or b.requires_grad);
}
