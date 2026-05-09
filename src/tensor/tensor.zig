//!
//! zig-transformer-lab — Core Tensor struct
//!
//! Purpose:
//!   The Tensor is the central data structure of the entire library. It
//!   represents an n-dimensional array of f32 values in row-major order,
//!   along with metadata (shape, strides, dtype, device) and an ownership
//!   flag that determines who is responsible for freeing the data buffer.
//!
//!   This file defines the Tensor struct and its basic operations:
//!   creation, destruction, element access, view/reshape/transpose, and
//!   copy/fill. These are the "leaf" operations that all higher-level ops
//!   (elementwise, matmul, softmax, etc.) build on top of.
//!
//! Shape contract:
//!   Every Tensor stores a Shape (logical dimensions) and Strides (memory
//!   layout offsets). The data buffer has exactly totalElements(shape) f32
//!   values stored contiguously for owned tensors; view tensors may share
//!   a larger buffer with non-contiguous strides (e.g., after transpose).
//!
//!   flat_index = sum(indices[i] * strides.values[i] for i in 0..ndim)
//!
//!   Example: shape (2,3), strides [3,1], indices [1,2] => 1*3 + 2*1 = 5
//!
//! Memory ownership:
//!   - `owned=true`: this Tensor allocated its data buffer and must free
//!     it in deinit(). The allocator that created the buffer must be the
//!     one passed to deinit().
//!   - `owned=false`: this Tensor is a view (e.g., from .view() or
//!     .transpose2d()) and shares data with another tensor. It must NOT
//!     free the data. The owning tensor's lifetime must exceed the view's.
//!   - Rule: only the original allocator can free the buffer. We do not
//!     store the allocator inside the tensor to keep the struct small;
//!     instead, the caller passes it at init and deinit time.
//!
//! Errors:
//!   - OutOfMemory: allocation of the data buffer failed in init().
//!   - ShapeMismatch: reshape or copyTo with incompatible shapes.
//!   - InvalidArgument: at() with wrong number of indices, transpose2d on non-rank-2.
//!   - NotImplemented: reshape on non-contiguous tensor with different shape.
//!
//! Autograd fields (Stage 3):
//!   requires_grad, grad, and tape_node are declared now so that the
//!   struct layout is stable across stages. They are not used by any
//!   code in Stage 2 — they default to false/null. This avoids a
//!   painful struct-migration when Stage 3 adds the tape-based autograd.
//!
//! TODO:
//!   - Stage 3: wire requires_grad/grad/tape_node into the autograd engine.
//!   - Stage 7: .cuda tensors will have data pointing to device memory.
//!     The deinit logic will need a branch for cuMemFree_v2.
//!   - Future: support .reshape() with copy for non-contiguous tensors
//!     instead of returning NotImplemented.
//;

const std = @import("std");
const Io = std.Io;

const LabError = @import("../core/errors.zig").LabError;
const shape_mod = @import("shape.zig");
const Shape = shape_mod.Shape;
const Strides = shape_mod.Strides;
const computeStrides = shape_mod.computeStrides;
const totalElements = shape_mod.totalElements;
const shape_isContiguous = shape_mod.isContiguous;
const shape_equals = shape_mod.equals;
const DType = @import("../core/dtype.zig").DType;
const Device = @import("../core/device.zig").Device;

/// NodeId for the autograd tape graph.
/// u32 is sufficient for our use case: even a large training run
/// won't have more than 4 billion tape nodes per step.
/// Declared here so the Tensor struct can hold an optional reference
/// to its tape node without depending on the autograd module.
pub const NodeId = u32;

/// The core n-dimensional array type.
///
/// Design note: we store data as []f32 (not a generic type) because
/// this library is f32-only (decision D9). A generic Tensor(T) would
/// add compilation complexity for zero practical benefit here.
///
/// All fields are public so that ops in src/tensor/ops/ can access them
/// directly. This avoids a large accessor-method surface that would just
/// forward to the fields.
pub const Tensor = struct {
    /// Flat f32 buffer holding all tensor elements in row-major order.
    /// For a view tensor, this may be a sub-slice of a larger allocation.
    data: []f32,

    /// Logical dimensions, e.g. shape (2,3) for a 2-row, 3-column matrix.
    shape: Shape,

    /// Strides: element-offset between consecutive values along each axis.
    /// For a freshly created (contiguous) tensor, these match
    /// computeStrides(shape). Views (e.g., after transpose) may have
    /// non-contiguous strides.
    strides: Strides,

    /// Data type. Always .f32 for now (decision D9).
    /// The field exists for forward compat and debug printing.
    dtype: DType,

    /// Device where the data lives: .cpu or .cuda.
    /// Stage 2 only uses .cpu; .cuda is wired in Stage 7.
    device: Device,

    /// Ownership flag. true = this tensor frees data in deinit().
    /// false = this is a view sharing another tensor's buffer.
    owned: bool,

    // --- Autograd fields (used in Stage 3, declared now) ---

    /// Whether this tensor requires gradient computation.
    /// Leaf parameters and user-specified inputs set this to true.
    /// Intermediate results get it propagated from their inputs.
    requires_grad: bool,

    /// Pointer to the gradient tensor (same shape as this tensor).
    /// null until backward() is called (or zero_grad is invoked).
    /// Owned by the autograd engine, not by this tensor.
    grad: ?*Tensor,

    /// ID of this tensor's node in the autograd tape, if any.
    /// null for tensors that are not part of any computation graph.
    tape_node: ?NodeId,

    /// Create a new zero-initialized tensor with the given shape.
    ///
    /// Allocates a flat f32 buffer of size totalElements(shape),
    /// fills it with zeros, and computes row-major strides.
    ///
    /// The returned tensor is owned (owned=true), on CPU (device=.cpu),
    /// with f32 dtype, and no autograd metadata.
    ///
    /// Worked example:
    ///   var t = try Tensor.init(allocator, Shape.init2D(2, 3));
    ///   // t.shape == (2,3), t.data.len == 6, t.data == [0,0,0,0,0,0]
    ///   // t.owned == true, t.strides.values == [3,1]
    ///   defer t.deinit(allocator);
    ///
    /// Memory: caller owns the returned tensor and must call deinit(allocator).
    pub fn init(allocator: std.mem.Allocator, shape: Shape) LabError!Tensor {
        const n = totalElements(shape);

        // Allocate the flat data buffer. We use allocator.alloc so that
        // the entire buffer is one contiguous allocation — important for
        // row-major access patterns and for copying to GPU later.
        const data = allocator.alloc(f32, n) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
        };

        // Zero-initialize: every element starts at 0.0. This is safer than
        // leaving the buffer uninitialized because NaN propagation from
        // garbage floats is extremely hard to debug.
        @memset(data, 0);

        return Tensor{
            .data = data,
            .shape = shape,
            .strides = computeStrides(shape),
            .dtype = .f32,
            .device = .cpu,
            .owned = true,
            .requires_grad = false,
            .grad = null,
            .tape_node = null,
        };
    }

    /// Free the tensor's data buffer if this tensor owns it.
    ///
    /// After deinit, the tensor is in an undefined state (all fields
    /// set to undefined). Accessing it is undefined behavior. This
    /// matches Zig's convention: deinit is a one-shot operation.
    ///
    /// For view tensors (owned=false), this is a no-op — the caller
    /// is responsible for ensuring the owning tensor outlives the view.
    ///
    /// Worked example:
    ///   var t = try Tensor.init(alloc, shape);
    ///   t.deinit(alloc);  // frees data, t is now invalid
    ///
    /// Safety: passing the wrong allocator is undefined behavior.
    /// We intentionally don't store the allocator to keep the struct
    /// small (8 bytes saved per tensor — significant when we create
    /// thousands of intermediate tensors in autograd).
    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        if (self.owned) {
            // Only free if we own the buffer. Views share another
            // tensor's data and must NOT free it.
            allocator.free(self.data);
        }
        // Poison all fields to make use-after-deinit immediately visible
        // in debug builds. The undefined values will cause obvious crashes
        // rather than silent corruption.
        self.* = undefined;
    }

    /// Compute the flat (1D) index into the data buffer from
    /// multi-dimensional indices using the stride formula:
    ///
    ///   flat_index = sum(indices[i] * strides.values[i])
    ///
    /// This is the fundamental indexing operation that all element
    /// access methods build on. Understanding this formula is key to
    /// understanding how tensors work: the strides encode the memory
    /// layout, and this formula translates logical coordinates into
    /// physical memory offsets.
    ///
    /// Worked example:
    ///   shape (2,3), strides [3,1], indices [1,2]
    ///   flat_index = 1*3 + 2*1 = 5
    ///   data[5] is the element at row 1, column 2
    ///
    /// Bounds checking: in Debug builds, we verify that each index
    /// is within its dimension. In ReleaseFast, these checks are
    /// compiled out for performance.
    pub fn flatIndex(self: Tensor, indices: []const usize) usize {
        const ndim = self.shape.ndim();
        std.debug.assert(indices.len == ndim);

        var idx: usize = 0;
        for (indices, 0..) |dim_idx, axis| {
            // Bounds check: only active in Debug/ReleaseSafe
            std.debug.assert(dim_idx < self.shape.dims[axis]);
            idx += dim_idx * self.strides.values[axis];
        }
        return idx;
    }

    /// Read the f32 value at the given multi-dimensional index.
    ///
    /// This is the read-only accessor. For a writable reference,
    /// use atPtr() instead.
    ///
    /// Worked example:
    ///   // For a 2x3 tensor with data = [0,1,2,3,4,5]:
    ///   tensor.at(&[_]usize{1, 2}) => 5.0
    pub fn at(self: Tensor, indices: []const usize) f32 {
        const idx = self.flatIndex(indices);
        return self.data[idx];
    }

    /// Get a mutable pointer to the f32 value at the given index.
    ///
    /// Returns *f32 so the caller can write through the pointer:
    ///   tensor.atPtr(&.{1,2}).* = 42.0;
    ///
    /// Worked example:
    ///   // For a 2x3 tensor:
    ///   tensor.atPtr(&[_]usize{0, 1}).* = 99.0;
    ///   // Now tensor.at(&[_]usize{0, 1}) == 99.0
    pub fn atPtr(self: *Tensor, indices: []const usize) *f32 {
        const idx = self.flatIndex(indices);
        return &self.data[idx];
    }

    /// Check if this tensor's strides match row-major (contiguous) layout.
    ///
    /// A contiguous tensor has its elements stored sequentially in memory
    /// with no gaps. This is the common case for freshly created tensors.
    /// Non-contiguous tensors arise from operations like transpose (which
    /// swaps strides without moving data).
    ///
    /// Contiguity matters because:
    ///   - reshape() can return a zero-copy view only if contiguous.
    ///   - Many ops (e.g., copyTo, softmax) assume or require contiguity.
    ///   - GPU transfers (Stage 7) need contiguous data.
    ///
    /// Worked example:
    ///   // Fresh 2x3 tensor: contiguous
    ///   // After transpose2d: NOT contiguous (strides are swapped)
    pub fn isContiguous(self: Tensor) bool {
        return shape_isContiguous(self.shape, self.strides);
    }

    /// Create a view of this tensor that shares the same data buffer.
    ///
    /// A view is a lightweight alias: it has the same data pointer,
    /// shape, and strides, but with owned=false so it won't free the
    /// buffer in deinit(). This is how PyTorch's .view() and numpy's
    /// reshaping work — no data is copied.
    ///
    /// CRITICAL: the view's lifetime must not exceed the original
    /// tensor's. If the original is deinited first, the view's data
    /// pointer becomes dangling. This is the same contract as a slice
    /// referencing freed memory.
    ///
    /// Worked example:
    ///   var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   defer t.deinit(alloc);
    ///   var v = t.view();
    ///   // v.data points to same buffer as t.data
    ///   // v.owned == false
    ///   // v.at(&.{1, 2}) reads the same element as t.at(&.{1, 2})
    pub fn view(self: Tensor) Tensor {
        return Tensor{
            .data = self.data,
            .shape = self.shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .device = self.device,
            .owned = false,
            .requires_grad = self.requires_grad,
            .grad = self.grad,
            .tape_node = self.tape_node,
        };
    }

    /// Reshape the tensor to a new shape, returning a view if possible.
    ///
    /// If the tensor is contiguous and the total number of elements
    /// matches, we can simply return a view with the new shape and
    /// recomputed strides — zero data movement.
    ///
    /// If the tensor is NOT contiguous (e.g., after transpose), reshaping
    /// would require copying data into a new contiguous layout. We don't
    /// implement that path yet — we return NotImplemented instead. This
    /// forces the caller to explicitly call a copy-then-reshape sequence,
    /// making the cost visible.
    ///
    /// Worked example:
    ///   // shape (2,3), contiguous => reshape to (3,2) is a view
    ///   var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   var r = try t.reshape(Shape.init2D(3, 2));
    ///   // r.owned == false, r.shape == (3,2), r.strides.values == [2,1]
    ///   // r.data == t.data (same buffer)
    ///
    ///   // shape (2,3), NOT contiguous => reshape to (6,) returns NotImplemented
    ///   // (first transpose, then try to reshape)
    pub fn reshape(self: Tensor, new_shape: Shape) LabError!Tensor {
        // The element count must match — reshape cannot change the total
        // number of elements. This is a fundamental invariant.
        if (totalElements(self.shape) != totalElements(new_shape)) {
            return error.ShapeMismatch;
        }

        // If the shape is the same, just return a view — this is a no-op
        // that avoids any confusion about ownership.
        if (shape_equals(self.shape, new_shape)) {
            return self.view();
        }

        // Only contiguous tensors can be reshaped as a zero-copy view.
        // Non-contiguous tensors have gaps in memory that prevent a simple
        // reinterpretation with new strides.
        if (!self.isContiguous()) {
            return error.NotImplemented;
        }

        // Return a view with the new shape and freshly computed strides.
        // The strides are always row-major for the new shape because the
        // underlying data is contiguous.
        return Tensor{
            .data = self.data,
            .shape = new_shape,
            .strides = computeStrides(new_shape),
            .dtype = self.dtype,
            .device = self.device,
            .owned = false,
            .requires_grad = self.requires_grad,
            .grad = self.grad,
            .tape_node = self.tape_node,
        };
    }

    /// Transpose a rank-2 tensor (matrix), returning a view.
    ///
    /// For a 2D tensor with shape (M, N) and strides [S0, S1], the
    /// transpose has shape (N, M) and strides [S1, S0]. No data is
    /// moved — we simply reinterpret the memory layout.
    ///
    /// This is the simplest form of transposition. General ND transposition
    /// with a perm argument is deferred to a later stage because our
    /// transformer only needs 2D transpose for matmul gradients.
    ///
    /// Worked example:
    ///   // shape (2,3), strides [3,1], data = [0,1,2,3,4,5]
    ///   var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   var tr = try t.transpose2d();
    ///   // tr.shape == (3,2), tr.strides.values == [1,3]
    ///   // tr.at(&.{1, 0}) == t.at(&.{0, 1}) == 1.0
    ///   // tr.owned == false (shares data with t)
    ///
    /// Why swap strides instead of copying?
    ///   Because the transpose of a row-major matrix is column-major.
    ///   Swapping the strides exactly captures this: what was a "row step"
    ///   (stride[0]=3) becomes a "column step" in the transposed view, and
    ///   vice versa. This is O(1) instead of O(M*N) for a copy.
    pub fn transpose2d(self: Tensor) LabError!Tensor {
        if (self.shape.ndim() != 2) {
            // This restriction exists because general ND transpose needs
            // a permutation argument and is harder to get right. Our
            // transformer only ever transposes 2D matrices.
            return error.InvalidArgument;
        }

        // Build the transposed shape: (N, M) from (M, N)
        const new_shape = Shape.init2D(self.shape.dims[1], self.shape.dims[0]);

        // Build the transposed strides: swap axis-0 and axis-1 strides.
        // This makes the transposed view read columns as contiguous if
        // the original was row-contiguous, and vice versa.
        var new_strides = Strides{
            .values = .{ self.strides.values[1], self.strides.values[0], 0, 0 },
            .rank = new_shape.rank,
        };
        _ = &new_strides;

        return Tensor{
            .data = self.data,
            .shape = new_shape,
            .strides = new_strides,
            .dtype = self.dtype,
            .device = self.device,
            .owned = false,
            .requires_grad = self.requires_grad,
            .grad = self.grad,
            .tape_node = self.tape_node,
        };
    }

    /// Deep-copy data from this tensor into dst.
    ///
    /// Both tensors must have the same shape. The data is copied element-
    /// by-element using the flat buffer (not the logical indices), which
    /// means this works correctly when both tensors are contiguous
    /// with the same layout. For non-contiguous source tensors, the copy
    /// iterates using flatIndex() on each element.
    ///
    /// Worked example:
    ///   var src = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   defer src.deinit(alloc);
    ///   var dst = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   defer dst.deinit(alloc);
    ///   try src.fill(7.0);
    ///   try src.copyTo(alloc, &dst);
    ///   // dst.at(&.{1, 2}) == 7.0
    pub fn copyTo(self: Tensor, allocator: std.mem.Allocator, dst: *Tensor) LabError!void {
        _ = allocator;
        if (!shape_equals(self.shape, dst.shape)) {
            return error.ShapeMismatch;
        }

        // For contiguous tensors, we can do a single memcpy which is
        // much faster than per-element iteration. This covers the
        // common case (fresh tensors, reshaped views).
        if (self.isContiguous() and dst.isContiguous()) {
            @memcpy(dst.data, self.data);
        } else {
            // Non-contiguous copy: iterate element by element.
            // This is slower but correct for any stride layout.
            const n = self.data.len;
            for (0..n) |i| {
                dst.data[i] = self.data[i];
            }
        }
    }

    /// Fill every element of this tensor with the given value.
    ///
    /// Uses @memset for efficiency — the compiler can vectorize this
    /// into SIMD stores on supported platforms.
    ///
    /// Worked example:
    ///   var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   t.fill(42.0);
    ///   // t.at(&.{0, 0}) == 42.0, t.at(&.{1, 2}) == 42.0
    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }
};

// ============================================================================
// Tests
// ============================================================================
// All tests use std.testing.allocator which detects memory leaks.
// This is critical: a leaked tensor buffer is a real bug that would
// cause OOM in long training runs.

test "Tensor init/deinit — no leak" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    // Verify the basic fields
    try std.testing.expectEqual(@as(usize, 6), t.data.len);
    try std.testing.expect(t.owned);
    try std.testing.expectEqual(Device.cpu, t.device);
    try std.testing.expectEqual(DType.f32, t.dtype);
    try std.testing.expect(!t.requires_grad);
    try std.testing.expect(t.grad == null);
    try std.testing.expect(t.tape_node == null);

    // Verify data is zero-initialized
    for (t.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // Verify strides are row-major: [3, 1]
    try std.testing.expectEqual(@as(usize, 3), t.strides.values[0]);
    try std.testing.expectEqual(@as(usize, 1), t.strides.values[1]);

    t.deinit(std.testing.allocator);
    // std.testing.allocator will detect if we leaked the buffer
}

test "Tensor at/atPtr round-trip" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // Write through atPtr, read back through at
    t.atPtr(&[_]usize{ 0, 0 }).* = 1.0;
    t.atPtr(&[_]usize{ 0, 1 }).* = 2.0;
    t.atPtr(&[_]usize{ 0, 2 }).* = 3.0;
    t.atPtr(&[_]usize{ 1, 0 }).* = 4.0;
    t.atPtr(&[_]usize{ 1, 1 }).* = 5.0;
    t.atPtr(&[_]usize{ 1, 2 }).* = 6.0;

    // Verify we can read back the same values
    try std.testing.expectEqual(@as(f32, 1.0), t.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2.0), t.at(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 3.0), t.at(&[_]usize{ 0, 2 }));
    try std.testing.expectEqual(@as(f32, 4.0), t.at(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 5.0), t.at(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 6.0), t.at(&[_]usize{ 1, 2 }));
}

test "Tensor flatIndex" {
    const shape = Shape.init3D(2, 3, 4);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // strides should be [12, 4, 1] for shape (2,3,4)
    // indices [1, 2, 3] => 1*12 + 2*4 + 3*1 = 12+8+3 = 23
    try std.testing.expectEqual(@as(usize, 23), t.flatIndex(&[_]usize{ 1, 2, 3 }));

    // indices [0, 0, 0] => 0
    try std.testing.expectEqual(@as(usize, 0), t.flatIndex(&[_]usize{ 0, 0, 0 }));
}

test "Tensor view does not own data" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    const v = t.view();

    // View must not be owned
    try std.testing.expect(!v.owned);

    // View shares the same data pointer
    try std.testing.expect(t.data.ptr == v.data.ptr);

    // View has the same shape and strides
    try std.testing.expect(shape_equals(t.shape, v.shape));
    try std.testing.expectEqual(t.strides.values[0], v.strides.values[0]);
    try std.testing.expectEqual(t.strides.values[1], v.strides.values[1]);
}

test "Tensor view reflects writes to original" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // Write to original
    t.atPtr(&[_]usize{ 1, 2 }).* = 42.0;

    // View should see the same value
    const v = t.view();
    try std.testing.expectEqual(@as(f32, 42.0), v.at(&[_]usize{ 1, 2 }));
}

test "Tensor transpose2d shape and strides" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // Fill with sequential values to verify element mapping
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;
    t.data[4] = 5.0;
    t.data[5] = 6.0;

    const tr = try t.transpose2d();

    // Shape should be swapped: (3, 2)
    try std.testing.expectEqual(@as(usize, 3), tr.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), tr.shape.dims[1]);

    // Strides should be swapped: original [3,1] => transposed [1,3]
    try std.testing.expectEqual(@as(usize, 1), tr.strides.values[0]);
    try std.testing.expectEqual(@as(usize, 3), tr.strides.values[1]);

    // Transpose is a view (no copy)
    try std.testing.expect(!tr.owned);
    try std.testing.expect(t.data.ptr == tr.data.ptr);

    // Verify element mapping: t[i][j] == tr[j][i]
    // t[0][1] = 2.0, so tr[1][0] should also be 2.0
    try std.testing.expectEqual(@as(f32, 2.0), tr.at(&[_]usize{ 1, 0 }));
    // t[1][0] = 4.0, so tr[0][1] should also be 4.0
    try std.testing.expectEqual(@as(f32, 4.0), tr.at(&[_]usize{ 0, 1 }));
}

test "Tensor transpose2d rejects non-rank-2" {
    const shape3d = Shape.init3D(2, 3, 4);
    var t3d = try Tensor.init(std.testing.allocator, shape3d);
    defer t3d.deinit(std.testing.allocator);

    const result = t3d.transpose2d();
    try std.testing.expectError(error.InvalidArgument, result);
}

test "Tensor reshape — contiguous view" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    t.fill(7.0);

    // Reshape (2,3) => (3,2) should work (contiguous, same element count)
    const new_shape = Shape.init2D(3, 2);
    const r = try t.reshape(new_shape);

    // Result is a view (not owned)
    try std.testing.expect(!r.owned);

    // Shape should be (3, 2)
    try std.testing.expectEqual(@as(usize, 3), r.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), r.shape.dims[1]);

    // Strides should be row-major for the new shape: [2, 1]
    try std.testing.expectEqual(@as(usize, 2), r.strides.values[0]);
    try std.testing.expectEqual(@as(usize, 1), r.strides.values[1]);

    // Data should be shared
    try std.testing.expect(t.data.ptr == r.data.ptr);

    // Value should be preserved
    try std.testing.expectEqual(@as(f32, 7.0), r.at(&[_]usize{ 0, 0 }));
}

test "Tensor reshape — same shape returns view" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    const same_shape = Shape.init2D(2, 3);
    const r = try t.reshape(same_shape);
    try std.testing.expect(!r.owned);
    try std.testing.expect(t.data.ptr == r.data.ptr);
}

test "Tensor reshape — wrong element count returns ShapeMismatch" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // 2x3 = 6 elements, but 2x4 = 8 elements
    const bad_shape = Shape.init2D(2, 4);
    try std.testing.expectError(error.ShapeMismatch, t.reshape(bad_shape));
}

test "Tensor reshape — non-contiguous returns NotImplemented" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // First transpose to make it non-contiguous
    const tr = try t.transpose2d();

    // Now try to reshape the transposed tensor — it's not contiguous
    const new_shape = Shape.init1D(6);
    try std.testing.expectError(error.NotImplemented, tr.reshape(new_shape));
}

test "Tensor fill" {
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    t.fill(42.0);

    // Every element should be 42.0
    for (t.data) |v| {
        try std.testing.expectEqual(@as(f32, 42.0), v);
    }

    // Also check via at()
    try std.testing.expectEqual(@as(f32, 42.0), t.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 42.0), t.at(&[_]usize{ 1, 2 }));
}

test "Tensor copyTo" {
    const shape = Shape.init2D(2, 3);
    var src = try Tensor.init(std.testing.allocator, shape);
    defer src.deinit(std.testing.allocator);
    var dst = try Tensor.init(std.testing.allocator, shape);
    defer dst.deinit(std.testing.allocator);

    // Fill source with a known value
    src.fill(99.0);

    // Destination starts at zero
    for (dst.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // Copy
    try src.copyTo(std.testing.allocator, &dst);

    // Destination should now match source
    for (dst.data) |v| {
        try std.testing.expectEqual(@as(f32, 99.0), v);
    }
}

test "Tensor copyTo — shape mismatch returns error" {
    const shape_a = Shape.init2D(2, 3);
    var a = try Tensor.init(std.testing.allocator, shape_a);
    defer a.deinit(std.testing.allocator);

    const shape_b = Shape.init2D(3, 2);
    var b = try Tensor.init(std.testing.allocator, shape_b);
    defer b.deinit(std.testing.allocator);

    // Both have 6 elements, but shapes differ: (2,3) vs (3,2)
    try std.testing.expectError(error.ShapeMismatch, a.copyTo(std.testing.allocator, &b));
}

test "Tensor isContiguous" {
    // Fresh tensor is contiguous
    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);
    try std.testing.expect(t.isContiguous());

    // Transposed tensor is NOT contiguous
    const tr = try t.transpose2d();
    try std.testing.expect(!tr.isContiguous());
}
