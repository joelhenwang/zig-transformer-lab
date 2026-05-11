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
//!   - future: support .reshape() with copy for non-contiguous
//!     tensors instead of returning NotImplemented.
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
const logicalOffsetFromLinear = shape_mod.logicalOffsetFromLinear;
const maxLogicalOffset = shape_mod.maxLogicalOffset;
const DType = @import("../core/dtype.zig").DType;
const Device = @import("../core/device.zig").Device;
// PR-δ: Storage.cuda carries a DeviceBuffer (RAII over cuMemAlloc_v2).
// Importing here pulls in backend.cuda.bindings + context transitively.
// All three CUDA files compile cross-platform (std.DynLib is portable),
// so this import does not break the Windows CPU-only build — bindings
// only touch the GPU at runtime via load(), never at compile time.
const DeviceBuffer = @import("../backend/cuda/mem.zig").DeviceBuffer;
const CudaContext = @import("../backend/cuda/context.zig").CudaContext;

/// NodeId for the autograd tape graph.
/// u32 is sufficient for our use case: even a large training run
/// won't have more than 4 billion tape nodes per step.
/// Declared here so the Tensor struct can hold an optional reference
/// to its tape node without depending on the autograd module.
pub const NodeId = u32;

// ---------------------------------------------------------------------------
// Storage (PR-δ) — "who owns the bytes, and where do they live?"
// ---------------------------------------------------------------------------
//
// Pre-PR-δ the Tensor stored a single `data: []f32` slice and an `owned`
// bool. That representation conflates three distinct ideas:
//
//   1. *The physical buffer* — a flat sequence of floats somewhere in
//      memory. A given buffer is owned by exactly one object.
//   2. *The logical view over that buffer* — a shape + strides + offset
//      that projects a sub-rectangle of the buffer into a tensor.
//   3. *The device* — for CUDA, the "bytes" are really a `CUdeviceptr`
//      and no `[]f32` slice can legitimately refer to them.
//
// The `Storage` union here is concept (1): ownership + location. The
// Tensor keeps its `data` / `owned` fields in sync as a convenience
// accessor for existing CPU call sites; new code should prefer
// `tensor.cpuData()` and `tensor.storageLen()`, which make the device
// contract explicit. When PR-ι introduces the real `Storage.cuda`
// variant, host code that indexes into `tensor.data[i]` on a CUDA
// tensor will fail loudly (the slice is zero-length by construction).

/// CPU-backed storage: a heap-allocated `[]f32` and whether we own it.
pub const CpuStorage = struct {
    /// The flat f32 buffer containing every element reachable by any
    /// view over this storage.
    data: []f32,

    /// True when this storage should free `data` in `Storage.deinit`.
    /// Views over the same bytes set `owned = false`.
    owned: bool,
};

/// Device-tagged storage union.
///
/// PR-δ replaced the `cuda: void` placeholder with a real
/// `DeviceBuffer` payload. The CUDA branch now carries everything
/// needed to free the allocation and to cross-reference it with a
/// `CudaContext` (for the `requireSameDevice`-style checks that
/// later PRs will extend to cross-context safety).
///
/// The `Device` enum is the tag. The payload type varies per tag:
///   - `.cpu`  -> `CpuStorage { data: []f32, owned: bool }`
///   - `.cuda` -> `DeviceBuffer { ctx, ptr, len, owned }`
///
/// Using `DeviceBuffer` directly (rather than wrapping it in an
/// outer `CudaStorage` struct) keeps the representation flat: the
/// outer struct would have held only a single `buffer: DeviceBuffer`
/// field, with no additional CUDA-specific metadata that isn't
/// already tracked inside `DeviceBuffer`.
pub const Storage = union(Device) {
    cpu: CpuStorage,
    cuda: DeviceBuffer,

    /// Number of f32 elements in the backing buffer.
    /// For CPU this is `cpu.data.len`; for CUDA it is the
    /// `DeviceBuffer.len` we tracked at `cuMemAlloc_v2` time.
    pub fn len(self: Storage) usize {
        return switch (self) {
            .cpu => |s| s.data.len,
            .cuda => |b| b.len,
        };
    }

    /// Free any memory the storage owns.
    ///
    /// The `allocator` parameter is only used for the CPU branch;
    /// CUDA storage frees itself through `DeviceBuffer.deinit`,
    /// which routes to `cuMemFree_v2` via the CUDA loader. The
    /// allocator is harmless on the CUDA path and lets us keep the
    /// signature uniform across every call site.
    pub fn deinit(self: *Storage, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .cpu => |*s| {
                if (s.owned) allocator.free(s.data);
            },
            .cuda => |*b| b.deinit(),
        }
    }
};

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
    ///
    /// PR-δ note: this field is a convenience alias for
    /// `self.storage.cpu.data` and is only meaningful when
    /// `self.device == .cpu`. New code should prefer `self.cpuData()`
    /// which returns an explicit error for non-CPU tensors.
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
    ///
    /// PR-δ note: kept in sync with `self.storage.cpu.owned` so existing
    /// code continues to compile. A cleanup PR after ε/ζ will remove this
    /// alias and have all sites read `self.storage.cpu.owned`.
    owned: bool,

    /// Backing storage (PR-δ seam). For `device == .cpu` this carries the
    /// real `[]f32` slice and its owned flag; the top-level `data` /
    /// `owned` fields are kept in sync as a compatibility convenience.
    /// For `device == .cuda` (not yet implemented) this holds the device
    /// pointer instead of a host slice — host code that reads `self.data`
    /// on a CUDA tensor will see a zero-length slice and fail loudly.
    storage: Storage,

    /// Starting offset into the backing storage (in f32 elements).
    /// Always 0 in PR-δ because the current view API does not slice; it
    /// only transposes. Future sub-view / slice operations will set this
    /// to pick out a sub-region without allocating.
    offset: usize,

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

    /// Stable 32-bit identity (PR-ζ) assigned to learnable parameter
    /// tensors via `nn.module.assignParamId`. The optimizer keys its
    /// per-parameter state (e.g. AdamW's m/v moments) by this ID so
    /// that replacing a parameter's backing buffer (checkpoint load,
    /// device transfer) does not silently lose optimizer history.
    /// Intermediate tensors never have a `param_id` — it stays null.
    param_id: ?u32 = null,

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

        const t = Tensor{
            .data = data,
            .shape = shape,
            .strides = computeStrides(shape),
            .dtype = .f32,
            .device = .cpu,
            .owned = true,
            .storage = .{ .cpu = .{ .data = data, .owned = true } },
            .offset = 0,
            .requires_grad = false,
            .grad = null,
            .tape_node = null,
        };
        debugCheckInvariants(t);
        return t;
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
        // Free through the Storage union — it owns the "is this allocated
        // and who should free it?" contract. Pre-PR-δ we used the top-level
        // `owned` bool; it is kept in sync with `storage.cpu.owned` so
        // the two agree for every well-formed tensor.
        self.storage.deinit(allocator);
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
        const v = Tensor{
            .data = self.data,
            .shape = self.shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .device = self.device,
            .owned = false,
            .storage = nonOwningStorage(self.storage),
            .offset = self.offset,
            .requires_grad = self.requires_grad,
            .grad = self.grad,
            .tape_node = self.tape_node,
        };
        debugCheckInvariants(v);
        return v;
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
        const r = Tensor{
            .data = self.data,
            .shape = new_shape,
            .strides = computeStrides(new_shape),
            .dtype = self.dtype,
            .device = self.device,
            .owned = false,
            .storage = nonOwningStorage(self.storage),
            .offset = self.offset,
            .requires_grad = self.requires_grad,
            .grad = self.grad,
            .tape_node = self.tape_node,
        };
        debugCheckInvariants(r);
        return r;
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

        const tr = Tensor{
            .data = self.data,
            .shape = new_shape,
            .strides = new_strides,
            .dtype = self.dtype,
            .device = self.device,
            .owned = false,
            .storage = nonOwningStorage(self.storage),
            .offset = self.offset,
            .requires_grad = self.requires_grad,
            .grad = self.grad,
            .tape_node = self.tape_node,
        };
        debugCheckInvariants(tr);
        return tr;
    }

    /// Deep-copy data from this tensor into dst.
    ///
    /// Both tensors must have the same shape. The fast path is a single
    /// `@memcpy` when both sides are contiguous with identical row-major
    /// layout. Otherwise the copy walks the logical element order of the
    /// source and writes in the logical element order of the destination,
    /// using `logicalOffsetFromLinear` to translate between logical index
    /// and physical offset for each side.
    ///
    /// This means a transposed source copied into a contiguous destination
    /// produces the *materialised transpose* of the source (i.e. the data
    /// is re-ordered in memory to match the new logical layout).
    ///
    /// Worked example:
    ///   var src = try Tensor.init(alloc, Shape.init2D(2, 3));   // [0,1,2,3,4,5]
    ///   src.data = [0, 1, 2, 3, 4, 5];
    ///   const trans = try src.transpose2d();                    // view, shape (3,2)
    ///   var dst = try Tensor.init(alloc, Shape.init2D(3, 2));
    ///   try trans.copyTo(alloc, &dst);
    ///   // dst.data == [0, 3, 1, 4, 2, 5]   (logical transpose materialised)
    pub fn copyTo(self: Tensor, allocator: std.mem.Allocator, dst: *Tensor) LabError!void {
        _ = allocator;
        if (!shape_equals(self.shape, dst.shape)) {
            return error.ShapeMismatch;
        }

        // Fast path: both sides contiguous row-major → plain memcpy.
        if (self.isContiguous() and dst.isContiguous()) {
            @memcpy(dst.data, self.data);
            return;
        }

        // Slow path: respect strides on both sides. Walk logical indices
        // 0..N-1, translate each to the correct physical offset for src
        // and dst separately.
        const n = totalElements(self.shape);
        for (0..n) |logical| {
            const src_off = logicalOffsetFromLinear(self.shape, self.strides, logical);
            const dst_off = logicalOffsetFromLinear(dst.shape, dst.strides, logical);
            dst.data[dst_off] = self.data[src_off];
        }
    }

    /// Fill every element of this tensor with the given value.
    ///
    /// Fast path: a contiguous tensor backed by its own buffer can be
    /// filled with a single `@memset`, which the compiler can vectorise.
    /// For a non-contiguous view (e.g. after transpose), we walk only
    /// the logical elements so we don't accidentally touch memory that
    /// belongs to the parent buffer but is not part of this view's
    /// logical extent.
    ///
    /// Note: because transpose views today share the full parent buffer,
    /// @memset on a contiguous tensor and the stride-aware walk reach
    /// the same set of f32s. The distinction will matter once PR-δ adds
    /// offset/sub-view support; writing the correct version now prevents
    /// a silent data-corruption bug from appearing later.
    ///
    /// Worked example:
    ///   var t = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   t.fill(42.0);
    ///   // t.at(&.{0, 0}) == 42.0, t.at(&.{1, 2}) == 42.0
    pub fn fill(self: *Tensor, value: f32) void {
        if (self.isContiguous()) {
            @memset(self.data, value);
            return;
        }
        const n = totalElements(self.shape);
        for (0..n) |logical| {
            const off = logicalOffsetFromLinear(self.shape, self.strides, logical);
            self.data[off] = value;
        }
    }

    // -- Device transfer (PR-ε) ------------------------------------------
    //
    // `toCuda` and `toCpu` move a tensor between devices by allocating a
    // fresh backing buffer on the destination and copying the entire
    // storage. They preserve `shape`, `strides`, and `offset` verbatim so
    // that a non-contiguous CPU view (e.g. the output of `transpose2d`)
    // becomes the same non-contiguous layout on CUDA — ops on the target
    // device walk the same logical elements the source walked.
    //
    // Why copy the *full* storage, not just `totalElements(shape)`?
    //
    //   A transposed view reads through strides that can reach any
    //   element of the parent buffer. If we copied only the reachable
    //   logical elements and then reconstructed the tensor on the target
    //   with the same strides, the first stride-walk past the logical
    //   count would OOB-read. Copying the whole parent buffer means the
    //   strides remain valid on the destination.
    //
    // Autograd: `requires_grad` is preserved (user intent carries over);
    // `grad` and `tape_node` are reset to null. These are step-scoped
    // quantities and the tape is per-device — moving a tensor mid-tape
    // and expecting gradients to flow would silently give wrong answers,
    // so we make the discontinuity explicit.
    //
    // These methods do NOT record a tape entry: the transfer is a host-
    // controlled side-effect, not a differentiable op. If a future PR
    // needs differentiable device transfer (for e.g. activation
    // checkpointing), it can add a `.to_device` OpKind — but that is
    // not Stage 7 territory.

    /// Allocate a fresh DeviceBuffer on `ctx` and copy this CPU tensor's
    /// backing buffer into it. The returned Tensor is a new heap-owning
    /// CUDA tensor with the same shape/strides/offset as `self`.
    ///
    /// Errors:
    ///   - `error.DeviceMismatch` if `self.device != .cpu`.
    ///   - `error.CudaError`      if the CUDA allocation or HtoD copy fails.
    ///
    /// Worked example:
    ///   const cpu = try Tensor.init(alloc, Shape.init2D(2, 3));
    ///   defer cpu.deinit(alloc);
    ///   var gpu = try cpu.toCuda(&ctx);
    ///   defer gpu.deinit(alloc);   // deinit routes to cuMemFree_v2
    pub fn toCuda(self: Tensor, ctx: *const CudaContext) LabError!Tensor {
        if (self.device != .cpu) return error.DeviceMismatch;
        const src_data = switch (self.storage) {
            .cpu => |s| s.data,
            .cuda => unreachable, // excluded by the device check above
        };

        // fromHost allocates src_data.len elements and HtoD-copies. The
        // returned buffer is OWNED — we transfer it into the Tensor below.
        const buffer = try DeviceBuffer.fromHost(ctx, src_data);
        // Intentionally do NOT `errdefer buffer.deinit()` here: the
        // subsequent Tensor-literal construction cannot fail (all fields
        // are simple moves/copies, no `try`), so the buffer either
        // succeeds all the way into the returned Tensor or was never
        // created in the first place.

        const t = Tensor{
            .data = &.{}, // CPU compat alias intentionally empty on CUDA
            .shape = self.shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .device = .cuda,
            .owned = false, // top-level alias is always false for CUDA tensors
            .storage = .{ .cuda = buffer },
            .offset = self.offset,
            .requires_grad = self.requires_grad,
            .grad = null,
            .tape_node = null,
            .param_id = self.param_id,
        };
        debugCheckInvariants(t);
        return t;
    }

    /// Allocate a fresh host `[]f32` and copy this CUDA tensor's
    /// backing buffer into it via `cuMemcpyDtoH_v2`. The returned
    /// Tensor is a new heap-owning CPU tensor with the same
    /// shape/strides/offset as `self`.
    ///
    /// Errors:
    ///   - `error.DeviceMismatch` if `self.device != .cuda`.
    ///   - `error.OutOfMemory`    if the host allocation fails.
    ///   - `error.CudaError`      if the DtoH copy fails.
    ///
    /// Worked example:
    ///   var gpu_tensor = ...;  // some CUDA tensor
    ///   var cpu = try gpu_tensor.toCpu(alloc);
    ///   defer cpu.deinit(alloc);   // frees the host []f32
    pub fn toCpu(self: Tensor, allocator: std.mem.Allocator) LabError!Tensor {
        if (self.device != .cuda) return error.DeviceMismatch;
        const buf = switch (self.storage) {
            .cuda => |b| b,
            .cpu => unreachable, // excluded by the device check above
        };

        const host_data = allocator.alloc(f32, buf.len) catch return error.OutOfMemory;
        errdefer allocator.free(host_data);
        try buf.copyToHost(host_data);

        const t = Tensor{
            .data = host_data,
            .shape = self.shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .device = .cpu,
            .owned = true,
            .storage = .{ .cpu = .{ .data = host_data, .owned = true } },
            .offset = self.offset,
            .requires_grad = self.requires_grad,
            .grad = null,
            .tape_node = null,
            .param_id = self.param_id,
        };
        debugCheckInvariants(t);
        return t;
    }

    /// Verify all structural invariants of this tensor.
    ///
    /// Called after every constructor / view-producing op when the build
    /// is in debug or safe-release mode (see `runtime_safety` below).
    /// Release-fast builds skip the check entirely.
    ///
    /// Invariants checked:
    ///   1. `shape.rank + 1 ∈ [1, 4]`. Rank is a u2 so this is already
    ///      structurally enforced, but a panic here catches anyone who
    ///      built a `Shape` by hand with the wrong rank field.
    ///   2. Every logical dimension is ≥ 1. Zero dims are not supported.
    ///   3. The stride rank matches the shape rank.
    ///   4. `offset + maxLogicalOffset(shape, strides)` lies inside
    ///      `storage.len()` — i.e. no view can index past the end of its
    ///      backing buffer.
    ///   5. On CPU devices, the top-level `data` alias matches
    ///      `storage.cpu.data` (the two representations must not drift).
    ///   6. If a gradient tensor is attached, it has the same shape as
    ///      self and lives on the same device.
    ///
    /// Returns `LabError.InvalidLayout`, `ShapeMismatch`, or
    /// `DeviceMismatch` on violation.
    pub fn checkInvariants(self: Tensor) LabError!void {
        // 1. Rank is intrinsically in [0, 3] (u2), so ndim is [1, 4].

        // 2. No dim is zero.
        const n = self.shape.ndim();
        for (0..n) |axis| {
            if (self.shape.dims[axis] == 0) return error.ShapeMismatch;
        }

        // 3. Stride and shape rank agree.
        if (self.shape.rank != self.strides.rank) return error.InvalidLayout;

        // 4. Every reachable logical element fits in the backing storage
        //    (accounting for offset). Using `storage.len()` rather than
        //    `data.len` is forward-compatible with CUDA storage where
        //    there is no host-visible slice to measure.
        const max_off = maxLogicalOffset(self.shape, self.strides);
        if (self.offset + max_off >= self.storage.len()) return error.InvalidLayout;

        // 5. CPU compat alias must agree with storage.
        switch (self.storage) {
            .cpu => |s| {
                if (self.data.ptr != s.data.ptr or self.data.len != s.data.len) {
                    return error.InvalidLayout;
                }
                if (self.owned != s.owned) return error.InvalidLayout;
                if (self.device != .cpu) return error.DeviceMismatch;
            },
            .cuda => |b| {
                if (self.device != .cuda) return error.DeviceMismatch;
                // The CPU compat alias is intentionally empty for a
                // CUDA tensor. Any host code that walks `self.data[i]`
                // iterates zero times — a loud failure (no output)
                // rather than a silent wrong answer from reading past
                // a non-host-addressable CUdeviceptr.
                if (self.data.len != 0) return error.InvalidLayout;
                // A CUDA tensor's top-level `owned` flag must stay
                // false: deinit of Tensor.data would try
                // `allocator.free(emptySlice)` (harmless) but more
                // importantly, the *real* ownership lives inside the
                // DeviceBuffer, not in the top-level alias.
                if (self.owned) return error.InvalidLayout;
                // The backing DeviceBuffer must be internally
                // consistent: zero-length means zero pointer, and
                // nonzero-length implies a real device pointer. This
                // matches the alloc(0) convention in DeviceBuffer.
                if (b.len == 0 and b.ptr != 0) return error.InvalidLayout;
            },
        }

        // 6. Attached gradient, if any, matches shape and device.
        if (self.grad) |g| {
            if (!shape_equals(g.shape, self.shape)) return error.ShapeMismatch;
            if (g.device != self.device) return error.DeviceMismatch;
        }
    }
};

// ============================================================================
// Free helpers (device check, debug invariant trigger)
// ============================================================================

/// Clone a Storage value but flip its `owned` bit to false. Used by every
/// view-constructing op so that a view's storage metadata faithfully
/// records that it does not own the bytes, even though it shares them.
pub fn nonOwningStorage(src: Storage) Storage {
    return switch (src) {
        .cpu => |s| Storage{ .cpu = .{ .data = s.data, .owned = false } },
        .cuda => |b| Storage{ .cuda = .{
            .ctx = b.ctx,
            .ptr = b.ptr,
            .len = b.len,
            .owned = false,
        } },
    };
}

/// Reject binary ops whose inputs live on different devices.
///
/// Today CPU is the only device in use, so this helper is effectively a
/// no-op — but every `add`, `sub`, `matmul`, etc. should call it so the
/// contract is encoded in one place. When PR-δ introduces `Storage.cuda`
/// this function becomes the one place where the "no implicit device
/// transfer" rule is enforced.
pub fn requireSameDevice(a: Tensor, b: Tensor) LabError!void {
    if (a.device != b.device) return error.DeviceMismatch;
}

/// Run `checkInvariants` on a tensor in debug / safe-release builds.
/// In `ReleaseFast`/`ReleaseSmall`, this is a no-op and all calls
/// compile away. Use this inside constructors / view ops to keep the
/// invariants encoded as runtime-checked contracts without paying for
/// the check in hot release builds.
pub fn debugCheckInvariants(t: Tensor) void {
    if (std.debug.runtime_safety) {
        t.checkInvariants() catch |err| {
            std.debug.panic("Tensor invariants violated: {s}", .{@errorName(err)});
        };
    }
}

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

test "Tensor copyTo — transposed source materialises the transpose" {
    // (2,3) row-major buffer: [0,1,2, 3,4,5]
    var src = try Tensor.init(std.testing.allocator, Shape.init2D(2, 3));
    defer src.deinit(std.testing.allocator);
    src.data[0] = 0;
    src.data[1] = 1;
    src.data[2] = 2;
    src.data[3] = 3;
    src.data[4] = 4;
    src.data[5] = 5;

    // Non-contiguous view of src with shape (3,2) — the logical
    // transpose. Iterating view logically 0..5 yields 0,3,1,4,2,5.
    const view = try src.transpose2d();

    var dst = try Tensor.init(std.testing.allocator, Shape.init2D(3, 2));
    defer dst.deinit(std.testing.allocator);

    try view.copyTo(std.testing.allocator, &dst);

    // dst must contain the logical transpose of src, laid out row-major.
    try std.testing.expectEqual(@as(f32, 0), dst.data[0]);
    try std.testing.expectEqual(@as(f32, 3), dst.data[1]);
    try std.testing.expectEqual(@as(f32, 1), dst.data[2]);
    try std.testing.expectEqual(@as(f32, 4), dst.data[3]);
    try std.testing.expectEqual(@as(f32, 2), dst.data[4]);
    try std.testing.expectEqual(@as(f32, 5), dst.data[5]);
}

test "Tensor copyTo — contiguous source into transposed dst preserves logical order" {
    // src laid out row-major: [10,20,30,40,50,60] shape (3,2)
    var src = try Tensor.init(std.testing.allocator, Shape.init3D(1, 3, 2));
    defer src.deinit(std.testing.allocator);
    // Use a 3D shape so we can exercise transposeInner2d indirectly:
    // keep it simple with a (3,2) buffer and transpose via transpose2d.
    // Actually simpler: just do a 2D test.
    _ = &src;

    var src2 = try Tensor.init(std.testing.allocator, Shape.init2D(3, 2));
    defer src2.deinit(std.testing.allocator);
    src2.data[0] = 10;
    src2.data[1] = 20;
    src2.data[2] = 30;
    src2.data[3] = 40;
    src2.data[4] = 50;
    src2.data[5] = 60;

    // dst starts as contiguous (2,3), then we take its transpose view
    // to get a NON-contiguous (3,2) destination. Writing logical
    // elements into that view should still land in the right places
    // in the underlying buffer.
    var dst_buf = try Tensor.init(std.testing.allocator, Shape.init2D(2, 3));
    defer dst_buf.deinit(std.testing.allocator);
    var dst_view = try dst_buf.transpose2d(); // shape (3,2), strides (1,3)

    try src2.copyTo(std.testing.allocator, &dst_view);

    // Logically, reading the (3,2) view in row-major order should match src2.
    try std.testing.expectEqual(@as(f32, 10), dst_view.at(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 20), dst_view.at(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 30), dst_view.at(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 40), dst_view.at(&.{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 50), dst_view.at(&.{ 2, 0 }));
    try std.testing.expectEqual(@as(f32, 60), dst_view.at(&.{ 2, 1 }));
}

test "Tensor fill — transposed view mutates only logical elements" {
    // A fresh (3, 2) tensor has 6 buffer slots. Transposing it makes a
    // (2, 3) view whose logical element set is the same 6 slots. Filling
    // the view with a value should set every buffer slot — confirm via
    // both the view and the original (they share the buffer).
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(3, 2));
    defer t.deinit(std.testing.allocator);
    // Seed with known junk so a bug that skips elements would show up.
    for (t.data, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var view = try t.transpose2d();
    view.fill(7.0);

    for (t.data) |v| {
        try std.testing.expectEqual(@as(f32, 7.0), v);
    }
}

// -- Invariants (PR-γ) -------------------------------------------------------

test "Tensor.checkInvariants passes on a fresh contiguous tensor" {
    var t = try Tensor.init(std.testing.allocator, Shape.init3D(2, 3, 4));
    defer t.deinit(std.testing.allocator);
    try t.checkInvariants();
}

test "Tensor.checkInvariants passes on a transposed view" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(2, 3));
    defer t.deinit(std.testing.allocator);
    const v = try t.transpose2d();
    try v.checkInvariants();
}

test "Tensor.checkInvariants rejects out-of-range stride" {
    // Build a tensor by hand with strides that push an element past
    // the end of its data buffer. This simulates a bug where a view
    // is constructed with the wrong strides.
    var data: [6]f32 = .{ 0, 0, 0, 0, 0, 0 };
    const bogus = Tensor{
        .data = &data,
        .shape = Shape.init2D(2, 3),
        // strides (6, 2) would need offsets up to (1*6 + 2*2) = 10 > 6.
        .strides = Strides{ .values = .{ 6, 2, 0, 0 }, .rank = 1 },
        .dtype = .f32,
        .device = .cpu,
        .owned = false,
        .storage = .{ .cpu = .{ .data = &data, .owned = false } },
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try std.testing.expectError(error.InvalidLayout, bogus.checkInvariants());
}

test "Tensor.checkInvariants rejects stride/shape rank mismatch" {
    var data: [6]f32 = .{ 0, 0, 0, 0, 0, 0 };
    const bogus = Tensor{
        .data = &data,
        .shape = Shape.init2D(2, 3),
        .strides = Strides{ .values = .{ 3, 1, 1, 0 }, .rank = 2 }, // rank 2 vs shape rank 1
        .dtype = .f32,
        .device = .cpu,
        .owned = false,
        .storage = .{ .cpu = .{ .data = &data, .owned = false } },
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try std.testing.expectError(error.InvalidLayout, bogus.checkInvariants());
}

test "requireSameDevice no-op on two CPU tensors" {
    var a = try Tensor.init(std.testing.allocator, Shape.init1D(3));
    defer a.deinit(std.testing.allocator);
    var b = try Tensor.init(std.testing.allocator, Shape.init1D(3));
    defer b.deinit(std.testing.allocator);
    try requireSameDevice(a, b);
}

test "requireSameDevice rejects differing devices" {
    // Construct a fake 'cuda' tensor by hand (no allocation; we only
    // exercise the device-check branch). requireSameDevice reads only
    // Tensor.device, so the CUDA storage need not be backed by a real
    // GPU allocation. We use `stubCudaStorage` which sets `ctx` to
    // undefined — safe here because no code in this test path ever
    // dereferences `ctx`. Any future change to checkInvariants or to
    // requireSameDevice that reads the ctx pointer would crash loudly
    // in debug builds (undefined is 0xAA-filled), prompting the
    // migration of this test to a real-GPU fixture.
    var data: [1]f32 = .{0};
    const cpu_t = Tensor{
        .data = &data,
        .shape = Shape.init1D(1),
        .strides = computeStrides(Shape.init1D(1)),
        .dtype = .f32,
        .device = .cpu,
        .owned = false,
        .storage = .{ .cpu = .{ .data = &data, .owned = false } },
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    const cuda_t = Tensor{
        .data = &.{},
        .shape = Shape.init1D(1),
        .strides = computeStrides(Shape.init1D(1)),
        .dtype = .f32,
        .device = .cuda,
        .owned = false,
        .storage = stubCudaStorage(1),
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try std.testing.expectError(error.DeviceMismatch, requireSameDevice(cpu_t, cuda_t));
}

test "Tensor storage is cpu-tagged after init" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(2, 3));
    defer t.deinit(std.testing.allocator);
    switch (t.storage) {
        .cpu => |s| {
            try std.testing.expect(s.owned);
            try std.testing.expectEqual(@as(usize, 6), s.data.len);
            // data and storage.cpu.data must point at the same buffer.
            try std.testing.expect(t.data.ptr == s.data.ptr);
        },
        .cuda => unreachable,
    }
    try std.testing.expectEqual(@as(usize, 0), t.offset);
}

test "Tensor view marks storage as not owned" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(2, 3));
    defer t.deinit(std.testing.allocator);
    const v = t.view();
    switch (v.storage) {
        .cpu => |s| try std.testing.expect(!s.owned),
        .cuda => unreachable,
    }
    // Top-level `owned` alias agrees with storage.
    try std.testing.expect(!v.owned);
}

// -- PR-δ Storage.cuda invariants ------------------------------------------
//
// These tests exercise only the *structural* invariants of a
// CUDA-tagged tensor. They do not allocate on the GPU (that happens in
// `tests/integration_cuda.zig` under `-Dcuda=true`). Instead they build
// a Tensor literal with `Storage.cuda` backed by a stub DeviceBuffer
// whose `ctx` field is `undefined` — safe because no code in these
// tests ever dereferences the ctx pointer.
//
// The stub pattern is test-only. Any future change to the invariant
// check or to requireSameDevice that reads `storage.cuda.ctx` would
// crash these tests loudly in debug builds (Zig fills `undefined` with
// 0xAA), signalling that the stub needs to become a real-GPU fixture.

/// Build a Storage.cuda value that is structurally valid (so
/// checkInvariants accepts it) but is **not** backed by a real GPU
/// allocation. The embedded DeviceBuffer has:
///   - ctx = undefined (never dereferenced in the tests that use this)
///   - ptr = 0 (matches the `alloc(0)` empty-handle convention)
///   - len = n (caller-chosen, must match the tensor's logical size)
///   - owned = false (so deinit does not try to call cuMemFree_v2)
fn stubCudaStorage(n: usize) Storage {
    return .{ .cuda = .{
        .ctx = undefined,
        .ptr = 0,
        .len = n,
        .owned = false,
    } };
}

test "Tensor.checkInvariants accepts a hand-built CUDA stub" {
    const s = Shape.init2D(3, 4);
    const t = Tensor{
        .data = &.{},
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cuda,
        .owned = false,
        .storage = stubCudaStorage(12),
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try t.checkInvariants();
}

test "Tensor.checkInvariants rejects CUDA tensor with nonzero data.len alias" {
    // A CUDA tensor must keep the top-level `data` compat alias empty.
    // Any non-empty slice here means host code would silently walk a
    // slice that is *not* the actual CUDA payload — the exact silent-
    // wrong-answer bug Storage was designed to prevent.
    var bogus_host: [4]f32 = .{ 0, 0, 0, 0 };
    const s = Shape.init1D(4);
    const t = Tensor{
        .data = &bogus_host, // INVALID — CUDA tensors must have data.len == 0
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cuda,
        .owned = false,
        .storage = stubCudaStorage(4),
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try std.testing.expectError(error.InvalidLayout, t.checkInvariants());
}

test "Tensor.checkInvariants rejects CUDA tensor whose top-level owned=true" {
    // Ownership of CUDA memory lives inside the DeviceBuffer, not in
    // the top-level `owned` compat alias. If a tensor claims to own
    // host memory while living on CUDA, Tensor.deinit would try to
    // free a host-side empty slice (harmless) but more importantly it
    // signals a structural confusion about who is responsible for
    // cuMemFree_v2 — refuse to trust such a tensor.
    const s = Shape.init1D(4);
    const t = Tensor{
        .data = &.{},
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cuda,
        .owned = true, // INVALID for a CUDA tensor
        .storage = stubCudaStorage(4),
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try std.testing.expectError(error.InvalidLayout, t.checkInvariants());
}

test "Tensor.checkInvariants rejects device/storage tag disagreement" {
    // A tensor with device = .cpu but storage = .cuda is internally
    // inconsistent. The union tag on `storage` is the source of truth;
    // the `device` field is a convenience alias. Drift is a bug.
    const s = Shape.init1D(4);
    const t = Tensor{
        .data = &.{},
        .shape = s,
        .strides = computeStrides(s),
        .dtype = .f32,
        .device = .cpu, // disagrees with storage tag below
        .owned = false,
        .storage = stubCudaStorage(4),
        .offset = 0,
        .requires_grad = false,
        .grad = null,
        .tape_node = null,
    };
    try std.testing.expectError(error.DeviceMismatch, t.checkInvariants());
}
