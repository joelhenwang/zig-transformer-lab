//!
//! autograd/grad_helpers.zig — Shared utilities for gradient computation
//!
//! Purpose:
//!   Contains helpers used by multiple backward implementations across
//!   the ops layer. Extracted from backward.zig to break circular
//!   imports: backward.zig dispatches to ops/X.zig backward functions,
//!   and those functions need these helpers. A third file breaks the
//!   cycle cleanly.
//!
//! Contents:
//!   - BackwardResult: the return type for all per-op backward functions
//!   - heapAlloc: box a Tensor value on the heap for BackwardResult
//!   - broadcastTo: expand a tensor to a target shape (CPU + CUDA)
//!   - accumulateGrad: in-place gradient accumulation (dst += src)
//!

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const totalElements = @import("../tensor/shape.zig").totalElements;
const shape_equals = @import("../tensor/shape.zig").equals;
const isContiguous = @import("../tensor/shape.zig").isContiguous;
const cuda_dispatch = @import("../tensor/device_dispatch.zig");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const max_parent_grads = 2;

/// Return type for all per-op backward functions. Each entry is either
/// a heap-allocated gradient tensor (owned by the caller) or null if
/// that parent doesn't require a gradient.
pub const BackwardResult = [max_parent_grads]?*Tensor;

// ---------------------------------------------------------------------------
// heapAlloc — box a Tensor on the heap for BackwardResult storage
// ---------------------------------------------------------------------------

/// Heap-allocate a Tensor value and return a pointer to it.
///
/// BackwardResult entries are `?*Tensor`, so every Tensor value produced
/// by an op must be boxed on the heap before being stored in the result
/// array. This helper centralizes the alloc+store pattern to avoid
/// repeating allocator.create / ptr.* = ... at every call site.
///
/// Memory ownership:
///   The returned pointer is owned by the caller, who must eventually
///   call ptr.deinit(allocator) and allocator.destroy(ptr).
pub fn heapAlloc(allocator: std.mem.Allocator, tensor: Tensor) LabError!*Tensor {
    const ptr = allocator.create(Tensor) catch return error.OutOfMemory;
    ptr.* = tensor;
    return ptr;
}

// ---------------------------------------------------------------------------
// broadcastTo — expand a tensor to a target shape
// ---------------------------------------------------------------------------

/// Expand a tensor to a target shape by replicating elements along
/// broadcast dimensions. Handles both CPU (element-wise copy with
/// stride-aware indexing) and CUDA (device_dispatch.broadcastTo).
///
/// If shapes already match, returns an owned contiguous copy (not a view)
/// because callers may deinit the source — a view would dangle.
///
/// This is used in backward to expand gradient contributions back to
/// the shape of an input that was broadcast during the forward pass.
pub fn broadcastTo(allocator: std.mem.Allocator, tensor: Tensor, target: Shape) LabError!Tensor {
    if (tensor.device == .cuda) {
        return try cuda_dispatch.broadcastTo(tensor, target);
    }
    // Same shape → return an owned copy (not a view) because callers
    // may deinit the source tensor, which would invalidate a view.
    if (shape_equals(tensor.shape, target)) {
        const out = try Tensor.init(allocator, target);
        if (isContiguous(tensor.shape, tensor.strides)) {
            @memcpy(out.cpuData(), tensor.cpuData()[0..out.cpuData().len]);
        } else {
            const n = totalElements(target);
            const ndim = tensor.shape.ndim();
            for (0..n) |flat| {
                var offset: usize = 0;
                var remaining: usize = flat;
                var axis: usize = 0;
                while (axis + 1 < ndim) : (axis += 1) {
                    var block: usize = 1;
                    var a2: usize = axis + 1;
                    while (a2 < ndim) : (a2 += 1) block *= tensor.shape.dims[a2];
                    const idx = remaining / block;
                    remaining %= block;
                    offset += idx * tensor.strides.values[axis];
                }
                offset += remaining * tensor.strides.values[axis];
                out.cpuData()[flat] = tensor.cpuData()[offset];
            }
        }
        return out;
    }

    var result = try Tensor.init(allocator, target);
    const out_n = totalElements(target);
    const ndim_out = target.ndim();
    const ndim_in = tensor.shape.ndim();

    // For each element in the output, compute the corresponding input index
    for (0..out_n) |out_i| {
        // Decompose flat output index into multi-dim coordinates
        var out_coords: [4]usize = .{ 0, 0, 0, 0 };
        var remaining = out_i;
        for (0..ndim_out) |d| {
            const rev_d = ndim_out - 1 - d;
            out_coords[rev_d] = remaining % target.dims[rev_d];
            remaining /= target.dims[rev_d];
        }

        // Map output coordinates to input coordinates (broadcast rules)
        var in_flat: usize = 0;
        const extra_dims = if (ndim_out > ndim_in) ndim_out - ndim_in else 0;
        for (0..ndim_in) |d| {
            const out_d = extra_dims + d;
            const in_dim = tensor.shape.dims[d];
            const coord = if (in_dim == 1) 0 else out_coords[out_d];
            in_flat += coord * tensor.strides.values[d];
        }

        result.cpuData()[out_i] = tensor.cpuData()[in_flat];
    }

    return result;
}

// ---------------------------------------------------------------------------
// accumulateGrad — in-place gradient accumulation
// ---------------------------------------------------------------------------

/// Accumulate src gradient into dst: dst += src.
///
/// This implements the multivariable chain rule: if a tensor is used
/// in multiple operations, its gradient is the SUM of contributions
/// from each operation.
///
/// Device routing:
///   CUDA tensors use the element-wise add kernel + DtoD copy-back.
///   CPU tensors use a simple in-place loop.
pub fn accumulateGrad(dst: *Tensor, src: *Tensor) void {
    std.debug.assert(shape_equals(dst.shape, src.shape));
    if (dst.device == .cuda) {
        // In-place accumulate via DtoD temp: tmp = dst + src, copy tmp -> dst.
        var tmp = cuda_dispatch.add(dst.*, src.*) catch |err| {
            std.log.warn("accumulateGrad CUDA failed: {}", .{err});
            return;
        };
        defer tmp.storage.deinit(undefined); // allocator unused on CUDA deinit
        const dst_buf = switch (dst.storage) {
            .cuda => |b| b,
            .cpu => unreachable,
        };
        const src_buf = switch (tmp.storage) {
            .cuda => |b| b,
            .cpu => unreachable,
        };
        dst_buf.copyFromDevice(src_buf) catch |err| {
            std.log.warn("accumulateGrad CUDA DtoD copy failed: {}", .{err});
        };
        return;
    }
    const n = dst.cpuData().len;
    for (0..n) |i| {
        dst.cpuData()[i] += src.cpuData()[i];
    }
}
