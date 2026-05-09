//!
//! zig-transformer-lab — Matrix multiplication operations
//!
//! Purpose:
//!   Provides matrix multiplication (matmul), batched matmul, and 2D
//!   transpose.  Matmul is THE fundamental operation for transformers:
//!   every linear layer, every attention score, and every output
//!   projection is a matmul.
//!
//!   This file implements the naive triple-loop algorithm.  While not as
//!   fast as cuBLAS or a tiled implementation, the naive version is
//!   pedagogically clear and correct, making it the right starting
//!   point for understanding how matrix multiplication works at the
//!   memory level.
//!
//! Shape contract:
//!   matmul:      a:(M,K) x b:(K,N) -> out:(M,N)    [rank-2 only]
//!   matmulBatch: a:(B,M,K) x b:(B,K,N) -> out:(B,M,N)  [rank-3]
//!   transpose2d: (M,N) -> (N,M)  [rank-2 only, zero-copy view]
//!
//! Math:
//!   matmul: out[i,j] = sum_k a[i,k] * b[k,j]
//!
//!   We use the ikj loop order instead of the textbook ijk:
//!     for i in 0..M:
//!       for k in 0..K:
//!         a_ik = a[i,k]        // load once, reuse for all j
//!         for j in 0..N:
//!           out[i,j] += a_ik * b[k,j]
//!
//!   Why ikj?  Both a[i,k] (with k incrementing) and b[k,j] (with j
//!   incrementing) access row-major memory sequentially.  The ijk order
//!   would stride b by its row stride in the inner loop, causing cache
//!   misses.  The ikj order hoists the a[i,k] load out of the innermost
//!   loop and walks both b[k,:] and out[i,:] contiguously — a
//!   significant cache improvement for large matrices.
//!
//! Memory ownership:
//!   matmul and matmulBatch return new owned tensors.  The caller must
//!   call deinit(allocator) on the returned tensor when done.
//!   transpose2d returns a view (owned=false) sharing the input's data.
//!
//! Errors:
//!   ShapeMismatch  — a's column count != b's row count, or batch sizes
//!   InvalidArgument — input tensors are not the expected rank
//!   OutOfMemory     — allocator could not fulfill the output buffer
//!
//! TODO:
//!   - Stage 7: replace with cuBLAS gemm for GPU tensors
//!   - Stage 8: tiled matmul for better CPU cache utilization
//!   - Consider supporting broadcast batched matmul (B,M,K) x (K,N)
//!
//! Credits:
//!   The ikj loop order insight is from "What Every Programmer Should
//!   Know About Memory" (Ulrich Drepper, 2007).  No code copied.

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Tensor = @import("../tensor.zig").Tensor;
const Shape = @import("../shape.zig").Shape;

/// Matrix multiply two rank-2 tensors.
///
/// Computes out = a x b using the cache-friendly ikj loop order.
/// Both inputs must be rank-2; the output is a new owned rank-2 tensor.
///
/// Shape: a:(M,K) x b:(K,N) -> out:(M,N)
///
/// Worked example:
///   // a = [[1,2,3],[4,5,6]]  shape (2,3)
///   // b = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]  shape (3,4)
///   // out[0,0] = 1*1+2*5+3*9 = 38
///   // out = [[38,44,50,56],[83,98,113,128]]  shape (2,4)
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn matmul(allocator: std.mem.Allocator, a: Tensor, b: Tensor) LabError!Tensor {
    // --- Rank validation ---
    // We restrict to rank-2 because general ND matmul requires
    // broadcasting rules and batch dimension handling — those belong
    // in matmulBatch.
    if (a.shape.ndim() != 2) return LabError.InvalidArgument;
    if (b.shape.ndim() != 2) return LabError.InvalidArgument;

    const M = a.shape.dims[0];
    const K = a.shape.dims[1];
    const K_b = b.shape.dims[0];
    const N = b.shape.dims[1];

    // --- Dimension compatibility ---
    // The "inner" dimension K must match: a has K columns, b has K
    // rows.  If they don't agree, the dot product is undefined.
    if (K != K_b) return LabError.ShapeMismatch;

    // --- Allocate output ---
    // Output shape is (M, N): M rows from a, N columns from b.
    // Tensor.init zero-fills, so out starts at all zeros — ready for
    // accumulation (+= in the inner loop).
    var out = try Tensor.init(allocator, Shape.init2D(M, N));
    errdefer out.deinit(allocator);

    // --- ikj triple loop ---
    // This is the core computation.  The loop order matters for
    // performance:
    //
    //   i (outer):  rows of a and out
    //   k (middle): columns of a / rows of b
    //   j (inner):  columns of b and out
    //
    // The k loop is the "middle" loop for a reason: we load a[i,k]
    // once and reuse it across all j.  This turns the inner loop
    // into a simple multiply-accumulate that walks b[k,:] and
    // out[i,:] contiguously.
    //
    // Contrast with ijk order (for i, for j, for k):
    //   b[k,j] jumps by stride[0] each k iteration (different rows
    //   of b), causing cache misses.  The ikj order avoids this.
    for (0..M) |i| {
        for (0..K) |k| {
            // Load a[i,k] once — this is the key to the ikj
            // optimization.  Without this hoist, we'd re-read the
            // same element N times in the j loop.
            const a_ik = a.data[i * a.strides.values[0] + k * a.strides.values[1]];

            for (0..N) |j| {
                // b[k,j] is contiguous in row-major order when j
                // increments.  out[i,j] is also contiguous when j
                // increments.  Both access patterns are cache-friendly.
                const b_kj = b.data[k * b.strides.values[0] + j * b.strides.values[1]];
                out.data[i * N + j] += a_ik * b_kj;
            }
        }
    }

    return out;
}

/// Batched matrix multiply two rank-3 tensors.
///
/// Applies matmul independently for each batch index.  The batch
/// dimension must match between a and b.  This is the operation used
/// in multi-head attention: each "head" is one batch element.
///
/// Shape: a:(B,M,K) x b:(B,K,N) -> out:(B,M,N)
///
/// Worked example:
///   // a shape (2,3,4), b shape (2,4,5)
///   // out[0] = a[0] x b[0], out[1] = a[1] x b[1]
///   // Each batch element is an independent 2D matmul.
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn matmulBatch(allocator: std.mem.Allocator, a: Tensor, b: Tensor) LabError!Tensor {
    // --- Rank validation ---
    // Rank-3 is the standard for batched matmul in transformers.
    // For other ranks, the caller should reshape or use matmul.
    if (a.shape.ndim() != 3) return LabError.InvalidArgument;
    if (b.shape.ndim() != 3) return LabError.InvalidArgument;

    const B = a.shape.dims[0];
    const M = a.shape.dims[1];
    const K = a.shape.dims[2];
    const B_b = b.shape.dims[0];
    const K_b = b.shape.dims[1];
    const N = b.shape.dims[2];

    // --- Batch and dimension compatibility ---
    // Both batch size and the inner K dimension must match.
    if (B != B_b) return LabError.ShapeMismatch;
    if (K != K_b) return LabError.ShapeMismatch;

    // --- Allocate output ---
    var out = try Tensor.init(allocator, Shape.init3D(B, M, N));
    errdefer out.deinit(allocator);

    // --- Loop over batch dimension ---
    // For each batch index, we compute the 2D matmul a[b] x b[b] and
    // store the result in the corresponding slice of the output.
    //
    // We access elements using the 3D strides rather than creating
    // temporary 2D view tensors.  This avoids extra allocations and
    // keeps the inner loop tight.
    for (0..B) |batch| {
        // Offset into a's flat buffer for this batch slice.
        // a[batch, :, :] starts at batch * stride[0].
        const a_off = batch * a.strides.values[0];
        // Similarly for b and out.  out is always contiguous so
        // its batch offset is batch * M * N.
        const b_off = batch * b.strides.values[0];
        const out_off = batch * M * N;

        // --- Inner ikj loop (same as matmul) ---
        for (0..M) |i| {
            for (0..K) |k| {
                // Access a[batch, i, k] using 3D strides
                const a_ik = a.data[a_off + i * a.strides.values[1] + k * a.strides.values[2]];
                for (0..N) |j| {
                    // Access b[batch, k, j] using 3D strides
                    const b_kj = b.data[b_off + k * b.strides.values[1] + j * b.strides.values[2]];
                    out.data[out_off + i * N + j] += a_ik * b_kj;
                }
            }
        }
    }

    return out;
}

/// Transpose a rank-2 tensor, returning a view.
///
/// Delegates to Tensor.transpose2d().  Provided here for API
/// completeness so that consumers can import matmul and get transpose
/// from the same namespace.
///
/// Shape: (M,N) -> (N,M)
///
/// Worked example:
///   // t shape (2,3) -> transpose2d(t) shape (3,2)
///   // The returned tensor is a view (owned=false) sharing t's data.
///
/// Memory: the returned tensor is a view — the caller must NOT deinit
/// it, and the original tensor must outlive the view.
pub fn transpose2d(tensor: Tensor) LabError!Tensor {
    // Validate rank-2 before delegating, to give a clear error from
    // this namespace rather than whatever Tensor.transpose2d raises.
    if (tensor.shape.ndim() != 2) return LabError.InvalidArgument;
    return tensor.transpose2d();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "matmul 2x3 @ 3x4 = 2x4 with hand-computed values" {
    const alloc = std.testing.allocator;

    var a = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer a.deinit(alloc);
    // a = [[1,2,3],[4,5,6]]
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;

    var b = try Tensor.init(alloc, Shape.init2D(3, 4));
    defer b.deinit(alloc);
    // b = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    for (0..12) |i| b.data[i] = @as(f32, @floatFromInt(i + 1));

    var out = try matmul(alloc, a, b);
    defer out.deinit(alloc);

    // Verify shape
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), out.shape.dims[1]);

    // Hand-computed values:
    // out[0,0] = 1*1+2*5+3*9 = 1+10+27 = 38
    // out[0,1] = 1*2+2*6+3*10 = 2+12+30 = 44
    // out[0,2] = 1*3+2*7+3*11 = 3+14+33 = 50
    // out[0,3] = 1*4+2*8+3*12 = 4+16+36 = 56
    // out[1,0] = 4*1+5*5+6*9 = 4+25+54 = 83
    // out[1,1] = 4*2+5*6+6*10 = 8+30+60 = 98
    // out[1,2] = 4*3+5*7+6*11 = 12+35+66 = 113
    // out[1,3] = 4*4+5*8+6*12 = 16+40+72 = 128
    try std.testing.expectApproxEqAbs(@as(f32, 38.0), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), out.data[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 56.0), out.data[3], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 83.0), out.data[4], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 98.0), out.data[5], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 113.0), out.data[6], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), out.data[7], 1e-4);
}

test "matmul 1x1 edge case" {
    const alloc = std.testing.allocator;

    var a = try Tensor.init(alloc, Shape.init2D(1, 1));
    defer a.deinit(alloc);
    a.data[0] = 2.0;

    var b = try Tensor.init(alloc, Shape.init2D(1, 1));
    defer b.deinit(alloc);
    b.data[0] = 3.0;

    var out = try matmul(alloc, a, b);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), out.shape.dims[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out.data[0], 1e-4);
}

test "matmul shape mismatch returns error" {
    const alloc = std.testing.allocator;

    var a = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init2D(4, 5));
    defer b.deinit(alloc);

    // a: (2,3), b: (4,5) — a.cols=3 != b.rows=4
    try std.testing.expectError(LabError.ShapeMismatch, matmul(alloc, a, b));
}

test "matmul non-rank-2 returns error" {
    const alloc = std.testing.allocator;

    var a_3d = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer a_3d.deinit(alloc);
    var b_2d = try Tensor.init(alloc, Shape.init2D(4, 5));
    defer b_2d.deinit(alloc);

    // a is rank-3, not rank-2
    try std.testing.expectError(LabError.InvalidArgument, matmul(alloc, a_3d, b_2d));
}

test "matmulBatch (2,3,4) @ (2,4,5)" {
    const alloc = std.testing.allocator;

    var a = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer a.deinit(alloc);
    // Batch 0: all 1s, Batch 1: all 2s
    for (0..12) |i| a.data[i] = 1.0;
    for (12..24) |i| a.data[i] = 2.0;

    var b = try Tensor.init(alloc, Shape.init3D(2, 4, 5));
    defer b.deinit(alloc);
    // Both batches: all 1s
    for (0..40) |i| b.data[i] = 1.0;

    var out = try matmulBatch(alloc, a, b);
    defer out.deinit(alloc);

    // Shape should be (2, 3, 5)
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 5), out.shape.dims[2]);

    // Batch 0: each element = 1*1*4 = 4.0
    for (0..15) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 4.0), out.data[i], 1e-4);
    }
    // Batch 1: each element = 2*1*4 = 8.0
    for (15..30) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 8.0), out.data[i], 1e-4);
    }
}

test "transpose2d returns transposed view" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer t.deinit(alloc);

    const tr = try transpose2d(t);

    // Shape should be (3, 2)
    try std.testing.expectEqual(@as(usize, 3), tr.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), tr.shape.dims[1]);

    // View — does not own data
    try std.testing.expect(!tr.owned);
}

test "transpose2d rejects non-rank-2" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer t.deinit(alloc);

    try std.testing.expectError(LabError.InvalidArgument, transpose2d(t));
}
