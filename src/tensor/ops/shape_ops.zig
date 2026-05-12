//!
//! zig-transformer-lab — Shape-manipulation operations with autograd support
//!
//! Purpose:
//!   Provides reshape and transpose2d as tape-tracked operations.
//!   These are needed by nn layers (Linear, Attention) which reshape
//!   between 2D and 3D, or transpose weight matrices for matmul.
//!
//!   The non-tracked versions (Tensor.reshape, Tensor.transpose2d)
//!   return views and don't record on the tape. The tracked versions
//!   here record a Node so backward can propagate gradients through
//!   the shape change correctly.
//!
//! Shape contract:
//!   reshapeTracked(t: (A,B,C), new_shape: (A*B,C)) → out: (A*B,C)
//!   transpose2dTracked(t: (M,N)) → out: (N,M)
//!
//! Math:
//!   reshape: identity function (data unchanged, shape reinterpreted)
//!   transpose2d: out[j,i] = in[i,j]
//!
//! Memory ownership:
//!   Both return owned tensors (deep copies, not views), because the
//!   backward functions assume contiguous data. The caller must deinit.
//!
//! Errors:
//!   OutOfMemory, ShapeMismatch, InvalidArgument
//!
//! Credits:
//!   Standard linear algebra operations. No code copied.

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Shape = @import("../shape.zig").Shape;
const totalElements = @import("../shape.zig").totalElements;
const shape_equals = @import("../shape.zig").equals;
const computeStrides = @import("../shape.zig").computeStrides;
const isContiguous = @import("../shape.zig").isContiguous;
const Tape = @import("../../autograd/tape.zig").Tape;
const Node = @import("../../autograd/node.zig").Node;
const OpKind = @import("../../autograd/node.zig").OpKind;
// Milestone 2 (Session 2): shape-manipulation CUDA materialisation.
// We need CUDA-aware reshape/transpose because Linear, LayerNorm and
// Attention compose these tracked ops, and the original host-side
// element-copy loops read from `tensor.data` which is the empty
// compat alias on CUDA.
const cuda_dispatch = @import("../../backend/cuda/dispatch.zig");

/// Reshape a tensor to a new shape, recording the operation on the tape.
///
/// Unlike Tensor.reshape() which returns a view, this always returns an
/// owned contiguous tensor. This is necessary for autograd: the backward
/// needs contiguous data to compute sumToShape correctly.
///
/// Shape: in:(A,B,...) -> out:(new_shape) where prod(in.shape) == prod(new_shape)
///
/// Worked example:
///   // t shape (2,3) → reshape to (6,)
///   var out = try reshapeTracked(alloc, t, Shape.init1D(6), &tape);
///   // out.shape == (6,), out.owned == true
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn reshapeTracked(allocator: std.mem.Allocator, tensor: Tensor, new_shape: Shape, tape: ?*Tape) LabError!Tensor {
    if (totalElements(tensor.shape) != totalElements(new_shape)) {
        return error.ShapeMismatch;
    }

    // Milestone 2: CUDA branch. Produce a fresh contiguous CUDA
    // buffer holding the materialised data, then relabel its shape
    // to new_shape.
    //   - Contiguous source: DtoD-clone (cuMemcpyDtoD_v2) then rewrap.
    //   - Non-contig source: broadcastTo(source, source.shape)
    //     rewrites the elements contiguously while walking via the
    //     original strides — same effect as a stride-aware gather.
    if (tensor.device == .cuda) {
        var out = blk: {
            if (isContiguous(tensor.shape, tensor.strides)) {
                break :blk try cuda_dispatch.cloneDevice(tensor);
            } else {
                break :blk try cuda_dispatch.broadcastTo(tensor, tensor.shape);
            }
        };
        // Now `out` is a contiguous CUDA tensor with tensor.shape.
        // Swap its shape/strides to the caller's new_shape. Element
        // count matches (checked above) so the underlying buffer is
        // the right size regardless of rank.
        out.shape = new_shape;
        out.strides = computeStrides(new_shape);

        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .reshape,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }

    // Always produce an owned contiguous copy. Even if the shapes match,
    // the backward expects contiguous data for sumToShape.
    var out = try Tensor.init(allocator, new_shape);
    errdefer out.deinit(allocator);

    // Copy data element-by-element using strided reads from the input.
    // This handles non-contiguous inputs (e.g., transposed views).
    const n = totalElements(new_shape);
    const ndim = tensor.shape.ndim();
    for (0..n) |flat| {
        var offset: usize = 0;
        var remaining: usize = flat;
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        offset += remaining * tensor.strides.values[axis];
        out.cpuData()[flat] = tensor.cpuData()[offset];
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .reshape,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Transpose a rank-2 tensor, recording the operation on the tape.
///
/// Unlike Tensor.transpose2d() which returns a view, this returns an
/// owned contiguous tensor. Necessary for autograd correctness.
///
/// Shape: in:(M,N) -> out:(N,M)
///
/// Worked example:
///   // t shape (2,3) → transpose2dTracked → out shape (3,2)
///   var out = try transpose2dTracked(alloc, t, &tape);
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn transpose2dTracked(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.shape.ndim() != 2) return error.InvalidArgument;

    const M = tensor.shape.dims[0];
    const N = tensor.shape.dims[1];

    // Milestone 2: CUDA branch. Take a view of the input with swapped
    // strides (tensor.transpose2d() returns a non-owning view with
    // (N,M) shape and strided reads), then use broadcastTo to
    // materialise a contiguous (N,M) buffer. This pushes the actual
    // transpose work into the existing stride-aware reduce kernel.
    if (tensor.device == .cuda) {
        const view = try tensor.transpose2d(); // non-contig (N,M) CUDA view
        var out = try cuda_dispatch.broadcastTo(view, view.shape);

        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .transpose2d,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }

    var out = try Tensor.init(allocator, Shape.init2D(N, M));
    errdefer out.deinit(allocator);

    // Write transposed: out[j,i] = in[i,j]
    for (0..M) |i| {
        for (0..N) |j| {
            const in_offset = i * tensor.strides.values[0] + j * tensor.strides.values[1];
            out.cpuData()[j * M + i] = tensor.cpuData()[in_offset];
        }
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .transpose2d,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Transpose the inner two dimensions of a rank-3 tensor, returning a view.
///
/// This is needed by batched matmul backward and by attention (K^T):
///   (B, M, K) → (B, K, M)   [swaps dims[1] and dims[2]]
///
/// Unlike transpose2d, this does NOT record on the tape — it's a pure
/// view operation used internally by ops that handle their own taping.
/// When used in attention, the QKV projections and matmulBatch calls
/// record their own tape nodes.
///
/// Shape: (B, M, K) → (B, K, M)   [rank-3 only]
///
/// Worked example:
///   // t shape (2, 3, 4) → transposeInner2d → view shape (2, 4, 3)
///   const tr = try transposeInner2d(t);
///   // tr.cpuData().ptr == t.cpuData().ptr  (shared buffer, no copy)
///
/// Memory: returns a VIEW (owned=false). Caller must NOT deinit the
///   returned tensor's data — only the original tensor owns it.
pub fn transposeInner2d(tensor: Tensor) LabError!Tensor {
    if (tensor.shape.ndim() != 3) return error.InvalidArgument;

    const B = tensor.shape.dims[0];
    const M = tensor.shape.dims[1];
    const K = tensor.shape.dims[2];

    return Tensor{
        .shape = Shape.init3D(B, K, M),
        .strides = .{
            .values = .{
                tensor.strides.values[0],
                tensor.strides.values[2],
                tensor.strides.values[1],
                0,
            },
            .rank = Shape.init3D(B, K, M).rank,
        },
        .dtype = tensor.dtype,
        .device = tensor.device,
        // Share the parent's storage but mark this as non-owning so
        // `deinit` on the view is a no-op. PR-δ seam.
        .storage = tensor_mod.nonOwningStorage(tensor.storage),
        .offset = tensor.offset,
        .requires_grad = tensor.requires_grad,
        .grad = tensor.grad,
        .tape_node = tensor.tape_node,
    };
}

/// Transpose the inner two dimensions of a rank-3 tensor, recording on the tape.
///
/// Unlike transposeInner2d (which returns a view), this returns an owned
/// contiguous copy and records a tape node so the backward can transpose
/// the gradient back to the original shape.
///
/// Shape: (B, M, K) → (B, K, M)   [rank-3 only]
///
/// Worked example:
///   // t shape (2, 3, 4) → transposeInner2dTracked → out shape (2, 4, 3)
///   var out = try transposeInner2dTracked(alloc, t, &tape);
///   // out.owned == true, out.tape_node set if t.requires_grad
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn transposeInner2dTracked(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.shape.ndim() != 3) return error.InvalidArgument;

    const B = tensor.shape.dims[0];
    const M = tensor.shape.dims[1];
    const K = tensor.shape.dims[2];

    // Milestone 2: CUDA branch. Same trick as transpose2dTracked —
    // view the tensor with inner axes swapped (transposeInner2d
    // returns a non-owning view with (B,K,M) shape and non-contig
    // strides), then materialise contiguous via broadcastTo.
    if (tensor.device == .cuda) {
        const view = try transposeInner2d(tensor); // non-contig (B,K,M) CUDA view
        var out = try cuda_dispatch.broadcastTo(view, view.shape);

        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .transpose_inner2d,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }

    var out = try Tensor.init(allocator, Shape.init3D(B, K, M));
    errdefer out.deinit(allocator);

    // Write transposed inner dims: out[b, k, m] = tensor[b, m, k]
    for (0..B) |b| {
        for (0..M) |m| {
            for (0..K) |k| {
                const in_offset = b * tensor.strides.values[0] + m * tensor.strides.values[1] + k * tensor.strides.values[2];
                const out_offset = b * K * M + k * M + m;
                out.cpuData()[out_offset] = tensor.cpuData()[in_offset];
            }
        }
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .transpose_inner2d,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Swap axes 1 and 2 of a rank-4 tensor, returning a non-owning
/// view. Used by multi-head attention to permute
/// `(B, T, n_head, d_head)` to `(B, n_head, T, d_head)` (and the
/// inverse permutation after the attention compute, since this
/// operation is its own inverse).
///
/// Shape: (B, T, H, D) → (B, H, T, D)   [rank-4 only]
///
/// Worked example:
///   // q shape (2, 4, 3, 8) → transposeAxes12_4d → view (2, 3, 4, 8)
///   const qv = try transposeAxes12_4d(q);
///   // qv.cpuData().ptr == q.cpuData().ptr (shared buffer)
///
/// Memory: returns a VIEW (owned=false). Caller must NOT deinit the
/// returned tensor's data — only the original tensor owns it.
pub fn transposeAxes12_4d(tensor: Tensor) LabError!Tensor {
    if (tensor.shape.ndim() != 4) return error.InvalidArgument;

    const d0 = tensor.shape.dims[0];
    const d1 = tensor.shape.dims[1];
    const d2 = tensor.shape.dims[2];
    const d3 = tensor.shape.dims[3];

    const new_shape = Shape.init4D(d0, d2, d1, d3);
    return Tensor{
        .shape = new_shape,
        .strides = .{
            .values = .{
                tensor.strides.values[0], // B axis unchanged
                tensor.strides.values[2], // new axis-1 reads old axis-2
                tensor.strides.values[1], // new axis-2 reads old axis-1
                tensor.strides.values[3], // d_head axis unchanged
            },
            .rank = new_shape.rank,
        },
        .dtype = tensor.dtype,
        .device = tensor.device,
        .storage = tensor_mod.nonOwningStorage(tensor.storage),
        .offset = tensor.offset,
        .requires_grad = tensor.requires_grad,
        .grad = tensor.grad,
        .tape_node = tensor.tape_node,
    };
}

/// Tracked variant of `transposeAxes12_4d`. Materialises a contiguous
/// copy so the downstream matmul + reshape pipeline does not need to
/// deal with non-unit strides, and records a `.transpose_axes12_4d`
/// tape node so backward can apply the inverse permutation to the
/// gradient.
///
/// The backward is trivially the same permutation again: applying
/// the axis-1/2 swap twice returns the identity.
///
/// Shape: (B, T, H, D) → (B, H, T, D)   [rank-4 only]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn transposeAxes12_4dTracked(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.shape.ndim() != 4) return error.InvalidArgument;

    const d0 = tensor.shape.dims[0];
    const d1 = tensor.shape.dims[1];
    const d2 = tensor.shape.dims[2];
    const d3 = tensor.shape.dims[3];

    // CUDA branch: take the non-contig view, then materialise
    // contiguous via broadcastTo of the view's shape.
    if (tensor.device == .cuda) {
        const view = try transposeAxes12_4d(tensor);
        var out = try cuda_dispatch.broadcastTo(view, view.shape);

        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .transpose_axes12_4d,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }

    // CPU branch: element-by-element copy via strided reads. Writes
    // a contiguous (d0, d2, d1, d3) output.
    const new_shape = Shape.init4D(d0, d2, d1, d3);
    var out = try Tensor.init(allocator, new_shape);
    errdefer out.deinit(allocator);

    for (0..d0) |ib| {
        for (0..d1) |it| {
            for (0..d2) |ih| {
                for (0..d3) |id| {
                    const in_offset = ib * tensor.strides.values[0] +
                        it * tensor.strides.values[1] +
                        ih * tensor.strides.values[2] +
                        id * tensor.strides.values[3];
                    // Output contiguous strides: (d2*d1*d3, d1*d3, d3, 1)
                    const out_offset = ib * (d2 * d1 * d3) +
                        ih * (d1 * d3) +
                        it * d3 +
                        id;
                    out.cpuData()[out_offset] = tensor.cpuData()[in_offset];
                }
            }
        }
    }
    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .transpose_axes12_4d,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }
    return out;
}// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "reshapeTracked (2,3) → (6,)" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer t.deinit(alloc);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    t.cpuData()[2] = 3.0;
    t.cpuData()[3] = 4.0;
    t.cpuData()[4] = 5.0;
    t.cpuData()[5] = 6.0;

    var out = try reshapeTracked(alloc, t, Shape.init1D(6), null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 6), out.shape.dims[0]);
    try std.testing.expect(out.isOwned());
    for (0..6) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), out.cpuData()[i]);
    }
}

test "transpose2dTracked (2,3) → (3,2)" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer t.deinit(alloc);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    t.cpuData()[2] = 3.0;
    t.cpuData()[3] = 4.0;
    t.cpuData()[4] = 5.0;
    t.cpuData()[5] = 6.0;

    var out = try transpose2dTracked(alloc, t, null);
    defer out.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[1]);
    try std.testing.expect(out.isOwned());
    // out[0,0] = t[0,0] = 1, out[0,1] = t[1,0] = 4
    try std.testing.expectEqual(@as(f32, 1.0), out.cpuData()[0]);
    try std.testing.expectEqual(@as(f32, 4.0), out.cpuData()[1]);
    // out[1,0] = t[0,1] = 2, out[1,1] = t[1,1] = 5
    try std.testing.expectEqual(@as(f32, 2.0), out.cpuData()[2]);
    try std.testing.expectEqual(@as(f32, 5.0), out.cpuData()[3]);
}

test "transposeInner2d (2,3,4) → (2,4,3) view" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer t.deinit(alloc);
    for (0..24) |i| t.cpuData()[i] = @floatFromInt(i);

    const tr = try transposeInner2d(t);

    // Shape should be (2, 4, 3)
    try std.testing.expectEqual(@as(usize, 2), tr.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), tr.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), tr.shape.dims[2]);

    // View — does not own data
    try std.testing.expect(!tr.isOwned());
    try std.testing.expect(tr.cpuData().ptr == t.cpuData().ptr);

    // Verify element: tr[0, 2, 1] = t[0, 1, 2]
    // t[0,1,2] = 0*12 + 1*4 + 2 = 6
    // tr[0,2,1] offset = 0*12 + 2*3 + 1 = 7 (using tr's strides)
    const t_val = t.cpuData()[0 * 12 + 1 * 4 + 2];
    const tr_offset = 0 * tr.strides.values[0] + 2 * tr.strides.values[1] + 1 * tr.strides.values[2];
    try std.testing.expectEqual(t_val, tr.cpuData()[tr_offset]);
}

test "transposeInner2d rejects non-rank-3" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(2, 3));
    defer t.deinit(std.testing.allocator);
    try std.testing.expectError(LabError.InvalidArgument, transposeInner2d(t));
}
