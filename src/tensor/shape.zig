//!
//! zig-transformer-lab — Tensor shape, strides, and broadcasting
//!
//! Purpose:
//!   Defines `Shape` and `Strides` — the two small structs that describe how a
//!   tensor's logical dimensions map onto its flat data buffer.  All tensor
//!   operations start by inspecting or transforming a Shape, so this file is
//!   the foundation of the entire tensor layer.
//!
//! Shape contract for entry points:
//!   computeStrides(shape)   → Strides   (row-major, C-contiguous)
//!   totalElements(shape)    → usize     (product of dims)
//!   isContiguous(s, strides) → bool     (strides match row-major?)
//!   equals(a, b)            → bool      (same rank & dims?)
//!   toString(shape, buf)    → []const u8 (human-readable, e.g. "(2, 3)")
//!   broadcastShapes(a, b)   → !Shape    (NumPy-style broadcast, or error)
//!   squeeze(shape, axis)    → Shape     (remove size-1 dim(s))
//!   logicalOffsetFromLinear(shape, strides, linear) → usize
//!                                       (row-major logical index → physical offset)
//!   maxLogicalOffset(shape, strides) → usize
//!                                       (largest physical offset any element hits)
//!
//! Math:
//!   Row-major strides (C order):
//!     stride[ndim-1] = 1
//!     stride[i]      = stride[i+1] * dims[i+1]     for i = ndim-2 .. 0
//!
//!   Total elements:
//!     N = Π_{i=0}^{ndim-1} dims[i]
//!
//!   NumPy broadcasting (right-aligned comparison):
//!     Two dims d_a and d_b are compatible iff d_a == d_b  OR  d_a == 1  OR  d_b == 1.
//!     The output dim is max(d_a, d_b).  If any pair is incompatible, error.
//!
//! Memory ownership:
//!   Shape and Strides are fixed-size (4 × usize + 1 × u2 each = 34 bytes
//!   on 64-bit).  No allocation anywhere.  Functions return by value.
//!
//! Error conditions:
//!   broadcastShapes  → LabError.ShapeMismatch  when dims are incompatible.
//!
//! TODO:
//!   - future: `unsqueeze(shape, axis)` - the inverse of squeeze,
//!     useful for aligning rank between broadcasting partners.
//!
//! Credits:
//!   Broadcasting rules are identical to NumPy (numpy/doc/broadcasting.rst).
//!   Row-major stride derivation appears in every tensor library; no code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of dimensions a Shape can represent.
///
/// Why 4?  A transformer needs at most (batch, seq_len, heads, head_dim) —
/// four axes.  Keeping this small lets us store dims in a fixed-size array
/// (no allocator) and use u2 for the rank field.  If you need 5D+ tensors,
/// change this constant and the rank type together.
pub const max_rank: usize = 4;

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Describes the logical dimensions of a tensor.
///
/// Invariants:
///   - `rank` stores (ndim - 1), so rank 0 → 1D, rank 1 → 2D, …, rank 3 → 4D.
///     This lets us fit the rank in a u2 (0..3) while supporting 1..4 dims.
///   - `dims[i]` for i < ndim are the actual dimension sizes (all ≥ 1).
///   - `dims[i]` for i ≥ ndim are always 1.  This makes totalElements() a
///     simple product of all four entries and lets broadcasting logic treat
///     unused dims as broadcastable size-1 dimensions.
///
/// Worked example:
///   Shape for a 2×3 matrix:
///     .dims  = [2, 3, 1, 1]
///     .rank  = 1          // ndim = rank + 1 = 2
pub const Shape = struct {
    dims: [max_rank]usize,
    rank: u2,

    /// Returns the number of dimensions (1..4).
    ///
    /// Stored as rank+1 because u2 can only hold 0..3 and we need 1..4.
    ///
    /// Worked example:
    ///   shape{ .dims = [2, 3, 1, 1], .rank = 1 }.ndim()  ==  2
    pub fn ndim(self: Shape) usize {
        return @as(usize, self.rank) + 1;
    }

    /// Construct a 1D shape. Debug-asserts that the dimension is non-zero;
    /// zero-sized tensors are not supported in this project (PR-γ).
    pub fn init1D(d0: usize) Shape {
        std.debug.assert(d0 > 0);
        return .{ .dims = .{ d0, 1, 1, 1 }, .rank = 0 };
    }

    /// Construct a 2D shape. Both dimensions must be non-zero.
    pub fn init2D(d0: usize, d1: usize) Shape {
        std.debug.assert(d0 > 0 and d1 > 0);
        return .{ .dims = .{ d0, d1, 1, 1 }, .rank = 1 };
    }

    /// Construct a 3D shape. All three dimensions must be non-zero.
    pub fn init3D(d0: usize, d1: usize, d2: usize) Shape {
        std.debug.assert(d0 > 0 and d1 > 0 and d2 > 0);
        return .{ .dims = .{ d0, d1, d2, 1 }, .rank = 2 };
    }

    /// Construct a 4D shape. All four dimensions must be non-zero.
    pub fn init4D(d0: usize, d1: usize, d2: usize, d3: usize) Shape {
        std.debug.assert(d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0);
        return .{ .dims = .{ d0, d1, d2, d3 }, .rank = 3 };
    }
};

// ---------------------------------------------------------------------------
// Strides
// ---------------------------------------------------------------------------

/// Describes how to step through a flat data buffer to reach the next element
/// along each dimension.
///
/// In row-major (C-contiguous) order, the last dimension is contiguous
/// (stride 1) and earlier dimensions have larger strides.
///
/// Invariants:
///   - `rank` matches the Shape's rank (stores ndim - 1).
///   - `values[i]` for i < ndim are the byte-offset steps (in elements, not
///     bytes — we multiply by @sizeOf(f32) only when computing byte offsets).
///   - `values[i]` for i ≥ ndim are 0 (unused).
///
/// Worked example:
///   Strides for a 2×3 matrix (row-major):
///     .values = [3, 1, 0, 0]
///     .rank   = 1
pub const Strides = struct {
    values: [max_rank]usize,
    rank: u2,

    /// Returns the number of dimensions this strides object describes.
    pub fn ndim(self: Strides) usize {
        return @as(usize, self.rank) + 1;
    }
};

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Compute row-major (C-contiguous) strides for the given shape.
///
/// Formula:
///   stride[ndim-1] = 1
///   stride[i] = stride[i+1] * dims[i+1]    for i = ndim-2 … 0
///
/// Worked example:
///   computeStrides(Shape.init3D(2, 3, 4))
///     → Strides{ .values = [12, 4, 1, 0], .rank = 2 }
///   // element [i][j][k] lives at offset i*12 + j*4 + k*1
///
/// Memory: no allocation.
pub fn computeStrides(shape: Shape) Strides {
    var strides = Strides{
        .values = [_]usize{0} ** max_rank,
        .rank = shape.rank,
    };

    const n = shape.ndim();
    if (n == 0) return strides;

    // The rightmost dimension is contiguous — adjacent elements are next to
    // each other in memory, so stepping by 1 f32 gets you the next element.
    strides.values[n - 1] = 1;

    // Walk left: each stride is the product of all dims to its right.
    // This is the defining property of row-major layout.
    if (n >= 2) {
        var i: usize = n - 1;
        while (i > 0) : (i -= 1) {
            strides.values[i - 1] = strides.values[i] * shape.dims[i];
        }
    }

    return strides;
}

/// Translate a row-major logical index into a physical data-buffer offset,
/// honouring arbitrary strides.
///
/// A "logical index" is what a user-facing loop `for (0..totalElements(shape))`
/// sees: 0, 1, 2, ... reading the tensor in row-major order (the last dim
/// varies fastest). The physical offset is where that logical element
/// actually lives in the flat `[]f32` buffer, which depends on the tensor's
/// strides — and the strides of a transposed or otherwise strided view
/// are NOT row-major.
///
/// Algorithm (unravel then dot):
///   1. Walk dimensions right-to-left.
///   2. At each dim, split the linear index into (coord along this dim,
///      remaining prefix) via `coord = remaining % dim_size;
///      remaining = remaining / dim_size;`.
///   3. Accumulate `coord * strides[dim]` into the physical offset.
///
/// This is the inverse of the "row-major index formula" in `Tensor.flatIndex`:
///   flatIndex(indices) uses strides to go from multi-dim indices to offset;
///   logicalOffsetFromLinear does the same job starting from a linear
///   logical counter instead of explicit indices.
///
/// Worked example:
///   shape = (2, 3), strides = (1, 2)   (a transposed view of a (3,2) buffer)
///   logical 0 → coord_1=0, coord_0=0 → 0*2 + 0*1 = 0
///   logical 1 → coord_1=1, coord_0=0 → 1*2 + 0*1 = 2
///   logical 2 → coord_1=2, coord_0=0 → 2*2 + 0*1 = 4
///   logical 3 → coord_1=0, coord_0=1 → 0*2 + 1*1 = 1
///
/// Memory: no allocation.
pub fn logicalOffsetFromLinear(shape: Shape, strides: Strides, linear: usize) usize {
    const n = shape.ndim();
    var remaining = linear;
    var offset: usize = 0;

    // Walk right-to-left: the last dim is the fastest-varying (row-major
    // logical order). `i` is 1-based here so we can stop before underflow.
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        const dim_size = shape.dims[i];
        const coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset += coord * strides.values[i];
    }
    return offset;
}

/// Compute the largest physical offset reachable by any logical element of
/// a tensor with this shape and strides. Used by `Tensor.checkInvariants`
/// to verify that every logical element of a view lies inside its backing
/// storage.
///
/// Formula:
///   max_offset = Σ_i (dims[i] - 1) * strides[i]
///
/// Rationale: the physical offset of element (c0, c1, ..., c_{n-1}) is
/// `Σ c_i * strides[i]`. Each coordinate `c_i` independently ranges over
/// `[0, dims[i] - 1]`. Strides are non-negative (we never produce
/// negative-stride views), so each term is maximised by picking the
/// largest coordinate, and the sum of the per-axis maxima equals the
/// overall maximum offset.
///
/// Precondition: every dim is ≥ 1 (enforced by `Shape.initND` asserts).
/// If a dim were 0, `dims[i] - 1` would wrap; we rely on zero-dim
/// rejection to keep this function total.
///
/// Worked example:
///   shape (3, 2), strides (1, 3)   (a transposed view of a (2,3) buffer)
///   max = (3-1)*1 + (2-1)*3 = 2 + 3 = 5 → fits in buffer of length 6 ✓
pub fn maxLogicalOffset(shape: Shape, strides: Strides) usize {
    const n = shape.ndim();
    var max_off: usize = 0;
    for (0..n) |i| {
        std.debug.assert(shape.dims[i] >= 1);
        max_off += (shape.dims[i] - 1) * strides.values[i];
    }
    return max_off;
}


///
/// Formula:
///   N = Π_{i=0}^{ndim-1} dims[i]
///
/// Because unused dims are always 1, we can safely multiply all four entries
/// without worrying about which ones are "real".
///
/// Worked example:
///   totalElements(Shape.init2D(2, 3))  ==  6
///   totalElements(Shape.init3D(2, 3, 4))  ==  24
///
/// Memory: no allocation.
pub fn totalElements(shape: Shape) usize {
    // Multiply all four entries; unused dims are 1 so they don't affect
    // the product.  This is simpler and less error-prone than only iterating
    // up to ndim.
    var product: usize = 1;
    for (shape.dims) |d| {
        product *= d;
    }
    return product;
}

/// Check whether the given strides correspond to row-major (C-contiguous)
/// layout for the given shape.
///
/// A tensor is contiguous if iterating through the flat buffer in order
/// visits elements in row-major order.  This is true when the strides match
/// what `computeStrides` would produce.
///
/// Worked example:
///   isContiguous(Shape.init2D(2, 3), Strides{ .values = [3, 1, 0, 0], .rank = 1 })
///     → true
///   isContiguous(Shape.init2D(2, 3), Strides{ .values = [1, 3, 0, 0], .rank = 1 })
///     → false  (column-major strides, not row-major)
///
/// Memory: no allocation.
pub fn isContiguous(shape: Shape, strides: Strides) bool {
    // Quick check: ranks must match.
    if (shape.rank != strides.rank) return false;

    const expected = computeStrides(shape);
    const n = shape.ndim();

    // Compare only the meaningful stride values (0..ndim-1).
    // The unused entries (ndim..max_rank-1) are irrelevant.
    for (0..n) |i| {
        if (strides.values[i] != expected.values[i]) return false;
    }
    return true;
}

/// Check whether two shapes have the same rank and identical dimensions.
///
/// Worked example:
///   equals(Shape.init2D(2, 3), Shape.init2D(2, 3))  ==  true
///   equals(Shape.init2D(2, 3), Shape.init2D(3, 2))  ==  false
///
/// Memory: no allocation.
pub fn equals(a: Shape, b: Shape) bool {
    if (a.rank != b.rank) return false;
    const n = a.ndim();
    for (0..n) |i| {
        if (a.dims[i] != b.dims[i]) return false;
    }
    return true;
}

/// Format a Shape as a human-readable string like "(2, 3)".
///
/// The caller provides the buffer; the returned slice is a view into it.
/// If the buffer is too small, returns an empty slice.
///
/// Worked example:
///   var buf: [64]u8 = undefined;
///   const s = toString(Shape.init2D(2, 3), &buf);
///   // s == "(2, 3)"
///
/// Memory: no allocation (writes into caller-supplied buffer).
pub fn toString(shape: Shape, buf: []u8) []const u8 {
    // In Zig 0.16.0, std.Io.Writer provides a fixed-buffer mode via
    // Writer.fixed(buf) that writes into the caller-supplied slice.
    // This is the idiomatic way to build a string without allocating.
    var w: std.Io.Writer = .fixed(buf);

    w.writeByte('(') catch return buf[0..0];

    const n = shape.ndim();
    for (0..n) |i| {
        if (i > 0) {
            // Separate dims with ", " — the standard mathematical notation.
            w.writeAll(", ") catch return buf[0..0];
        }
        // Format the dimension as a decimal integer.
        w.printInt(shape.dims[i], 10, .lower, .{}) catch return buf[0..0];
    }

    w.writeByte(')') catch return buf[0..0];

    return w.buffered();
}

/// Compute the broadcast shape of two inputs using NumPy-style rules.
///
/// Broadcasting aligns shapes from the right and compares each pair of dims:
///   - Equal dims → keep that dim.
///   - One dim is 1 → take the other (the larger) dim.
///   - Neither is 1 and they differ → ShapeMismatch error.
///
/// Worked example:
///   broadcastShapes(Shape.init2D(2, 1), Shape.init2D(1, 3))
///     → Shape.init2D(2, 3)
///   broadcastShapes(Shape.init1D(3), Shape.init2D(2, 3))
///     → Shape.init2D(2, 3)
///   broadcastShapes(Shape.init1D(3), Shape.init1D(4))
///     → error.ShapeMismatch
///
/// Memory: no allocation.
pub fn broadcastShapes(a: Shape, b: Shape) !Shape {
    const ndim_a = a.ndim();
    const ndim_b = b.ndim();
    const ndim_out = @max(ndim_a, ndim_b);

    // Safety check: our shapes can represent at most 4D, so if broadcasting
    // somehow needs more dimensions, we error out.
    if (ndim_out > max_rank) return LabError.ShapeMismatch;

    var result = Shape{
        .dims = [_]usize{1} ** max_rank,
        .rank = @intCast(ndim_out - 1),
    };

    // Walk from the rightmost dimension toward the left.  Right-alignment is
    // the key insight of NumPy broadcasting: (3,) and (2, 3) broadcast to
    // (2, 3) because the 3 in the 1D shape aligns with the 3 in the 2D shape.
    var i: usize = 0;
    while (i < ndim_out) : (i += 1) {
        const da = if (i < ndim_a) a.dims[ndim_a - 1 - i] else 1;
        const db = if (i < ndim_b) b.dims[ndim_b - 1 - i] else 1;

        // Two dims are compatible if they are equal, or one of them is 1.
        if (da == db) {
            result.dims[ndim_out - 1 - i] = da;
        } else if (da == 1) {
            // The size-1 dim "stretches" to match the other.
            result.dims[ndim_out - 1 - i] = db;
        } else if (db == 1) {
            result.dims[ndim_out - 1 - i] = da;
        } else {
            // Neither is 1 and they differ — broadcasting is impossible.
            return LabError.ShapeMismatch;
        }
    }

    return result;
}

/// Remove size-1 dimensions from a shape.
///
/// If `axis` is non-null, only the dimension at that axis is removed (if its
/// size is 1; otherwise the shape is returned unchanged).  If `axis` is null,
/// *all* size-1 dimensions are removed.
///
/// The result's rank is never reduced below 0 (meaning ndim never goes below
/// 1).  If squeezing would eliminate all dimensions, the result is a 1D shape
/// of size 1 — our system does not represent 0D scalar shapes.
///
/// Worked example:
///   squeeze(Shape.init2D(1, 3), 0)  →  Shape.init1D(3)
///   squeeze(Shape.init4D(1, 3, 1, 4), null)  →  Shape.init2D(3, 4)
///   squeeze(Shape.init2D(2, 3), 0)  →  Shape.init2D(2, 3)  (dim not 1, no change)
///
/// Memory: no allocation.
pub fn squeeze(shape: Shape, axis: ?u2) Shape {
    if (axis) |ax| {
        // If the requested axis is not size 1, return the shape unchanged.
        // This mirrors NumPy's legacy behaviour (newer NumPy raises, but
        // our signature doesn't allow error returns).
        if (shape.dims[ax] != 1) return shape;

        // If this is the only dimension, don't squeeze it (would produce 0D).
        if (shape.rank == 0) return shape;

        // Build the new shape by copying all dims except the one at `ax`.
        var result = Shape{
            .dims = [_]usize{1} ** max_rank,
            .rank = 0,
        };
        const n = shape.ndim();
        var out_idx: usize = 0;
        for (0..n) |i| {
            if (i == ax) continue;
            result.dims[out_idx] = shape.dims[i];
            out_idx += 1;
        }
        // New ndim is out_idx, so rank = out_idx - 1 (out_idx >= 1 since we
        // started with ndim >= 2 — we returned early for rank == 0).
        result.rank = @intCast(out_idx - 1);
        return result;
    } else {
        // Remove ALL size-1 dimensions.
        var result = Shape{
            .dims = [_]usize{1} ** max_rank,
            .rank = 0,
        };
        const n = shape.ndim();
        var out_idx: usize = 0;
        for (0..n) |i| {
            if (shape.dims[i] == 1) continue;
            result.dims[out_idx] = shape.dims[i];
            out_idx += 1;
        }

        if (out_idx == 0) {
            // All dims were 1 — result would be 0D (scalar).  Our system
            // requires at least 1D, so we return a 1D shape of size 1.
            // This is the same as Shape.init1D(1).
            result.dims[0] = 1;
            result.rank = 0;
        } else {
            result.rank = @intCast(out_idx - 1);
        }
        return result;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "computeStrides 1D" {
    const shape = Shape.init1D(5);
    const s = computeStrides(shape);
    try std.testing.expectEqual(@as(usize, 1), s.values[0]);
    try std.testing.expectEqual(@as(u2, 0), s.rank);
}

test "computeStrides 2D" {
    const shape = Shape.init2D(2, 3);
    const s = computeStrides(shape);
    try std.testing.expectEqual(@as(usize, 3), s.values[0]);
    try std.testing.expectEqual(@as(usize, 1), s.values[1]);
}

test "computeStrides 3D" {
    const shape = Shape.init3D(2, 3, 4);
    const s = computeStrides(shape);
    try std.testing.expectEqual(@as(usize, 12), s.values[0]);
    try std.testing.expectEqual(@as(usize, 4), s.values[1]);
    try std.testing.expectEqual(@as(usize, 1), s.values[2]);
}

test "computeStrides 4D" {
    const shape = Shape.init4D(2, 3, 4, 5);
    const s = computeStrides(shape);
    try std.testing.expectEqual(@as(usize, 60), s.values[0]);
    try std.testing.expectEqual(@as(usize, 20), s.values[1]);
    try std.testing.expectEqual(@as(usize, 5), s.values[2]);
    try std.testing.expectEqual(@as(usize, 1), s.values[3]);
}

test "totalElements" {
    try std.testing.expectEqual(@as(usize, 5), totalElements(Shape.init1D(5)));
    try std.testing.expectEqual(@as(usize, 6), totalElements(Shape.init2D(2, 3)));
    try std.testing.expectEqual(@as(usize, 24), totalElements(Shape.init3D(2, 3, 4)));
    try std.testing.expectEqual(@as(usize, 120), totalElements(Shape.init4D(2, 3, 4, 5)));
}

test "isContiguous true for row-major strides" {
    const shape = Shape.init2D(2, 3);
    const strides = Strides{ .values = .{ 3, 1, 0, 0 }, .rank = 1 };
    try std.testing.expect(isContiguous(shape, strides));
}

test "isContiguous false for column-major strides" {
    const shape = Shape.init2D(2, 3);
    const strides = Strides{ .values = .{ 1, 2, 0, 0 }, .rank = 1 };
    try std.testing.expect(!isContiguous(shape, strides));
}

test "isContiguous true for 3D row-major" {
    const shape = Shape.init3D(2, 3, 4);
    const strides = Strides{ .values = .{ 12, 4, 1, 0 }, .rank = 2 };
    try std.testing.expect(isContiguous(shape, strides));
}

test "isContiguous mismatched rank" {
    const shape = Shape.init2D(2, 3);
    const strides = Strides{ .values = .{ 3, 1, 1, 0 }, .rank = 2 };
    try std.testing.expect(!isContiguous(shape, strides));
}

test "equals same shapes" {
    try std.testing.expect(equals(Shape.init2D(2, 3), Shape.init2D(2, 3)));
}

test "equals different dims" {
    try std.testing.expect(!equals(Shape.init2D(2, 3), Shape.init2D(3, 2)));
}

test "equals different rank" {
    try std.testing.expect(!equals(Shape.init1D(6), Shape.init2D(2, 3)));
}

test "toString 1D" {
    var buf: [64]u8 = undefined;
    const s = toString(Shape.init1D(5), &buf);
    try std.testing.expectEqualStrings("(5)", s);
}

test "toString 2D" {
    var buf: [64]u8 = undefined;
    const s = toString(Shape.init2D(2, 3), &buf);
    try std.testing.expectEqualStrings("(2, 3)", s);
}

test "toString 3D" {
    var buf: [64]u8 = undefined;
    const s = toString(Shape.init3D(2, 3, 4), &buf);
    try std.testing.expectEqualStrings("(2, 3, 4)", s);
}

test "toString 4D" {
    var buf: [64]u8 = undefined;
    const s = toString(Shape.init4D(2, 3, 4, 5), &buf);
    try std.testing.expectEqualStrings("(2, 3, 4, 5)", s);
}

test "broadcastShapes same shape" {
    const result = try broadcastShapes(Shape.init2D(2, 3), Shape.init2D(2, 3));
    try std.testing.expect(equals(Shape.init2D(2, 3), result));
}

test "broadcastShapes (2,1) x (1,3) → (2,3)" {
    const result = try broadcastShapes(Shape.init2D(2, 1), Shape.init2D(1, 3));
    try std.testing.expect(equals(Shape.init2D(2, 3), result));
}

test "broadcastShapes (3,) x (2,3) → (2,3)" {
    const result = try broadcastShapes(Shape.init1D(3), Shape.init2D(2, 3));
    try std.testing.expect(equals(Shape.init2D(2, 3), result));
}

test "broadcastShapes (1,) x (3,) → (3,)" {
    const result = try broadcastShapes(Shape.init1D(1), Shape.init1D(3));
    try std.testing.expect(equals(Shape.init1D(3), result));
}

test "broadcastShapes incompatible → ShapeMismatch" {
    const result = broadcastShapes(Shape.init1D(3), Shape.init1D(4));
    try std.testing.expectError(LabError.ShapeMismatch, result);
}

test "broadcastShapes incompatible 2D → ShapeMismatch" {
    const result = broadcastShapes(Shape.init2D(2, 3), Shape.init2D(3, 2));
    try std.testing.expectError(LabError.ShapeMismatch, result);
}

test "broadcastShapes (8,1,6) x (7,1,5) → ShapeMismatch" {
    const a = Shape.init3D(8, 1, 6);
    const b = Shape.init3D(7, 1, 5);
    const result = broadcastShapes(a, b);
    try std.testing.expectError(LabError.ShapeMismatch, result);
}

test "broadcastShapes (8,1,6) x (1,1,6) → (8,1,6)" {
    const a = Shape.init3D(8, 1, 6);
    const b = Shape.init3D(1, 1, 6);
    const result = try broadcastShapes(a, b);
    try std.testing.expect(equals(Shape.init3D(8, 1, 6), result));
}

test "squeeze removes specific axis" {
    const shape = Shape.init2D(1, 3);
    const result = squeeze(shape, 0);
    try std.testing.expect(equals(Shape.init1D(3), result));
}

test "squeeze no-op when dim is not 1" {
    const shape = Shape.init2D(2, 3);
    const result = squeeze(shape, 0);
    try std.testing.expect(equals(Shape.init2D(2, 3), result));
}

test "squeeze null removes all size-1 dims" {
    const shape = Shape.init4D(1, 3, 1, 4);
    const result = squeeze(shape, null);
    try std.testing.expect(equals(Shape.init2D(3, 4), result));
}

test "squeeze null on all-1s shape returns 1D of size 1" {
    const shape = Shape.init3D(1, 1, 1);
    const result = squeeze(shape, null);
    try std.testing.expect(equals(Shape.init1D(1), result));
}

test "squeeze on 1D shape with axis 0 returns unchanged" {
    const shape = Shape.init1D(5);
    const result = squeeze(shape, 0);
    try std.testing.expect(equals(Shape.init1D(5), result));
}

test "squeeze 3D to 2D" {
    const shape = Shape.init3D(1, 4, 5);
    const result = squeeze(shape, 0);
    try std.testing.expect(equals(Shape.init2D(4, 5), result));
}

test "Shape.ndim returns correct values" {
    try std.testing.expectEqual(@as(usize, 1), Shape.init1D(5).ndim());
    try std.testing.expectEqual(@as(usize, 2), Shape.init2D(2, 3).ndim());
    try std.testing.expectEqual(@as(usize, 3), Shape.init3D(2, 3, 4).ndim());
    try std.testing.expectEqual(@as(usize, 4), Shape.init4D(2, 3, 4, 5).ndim());
}

test "logicalOffsetFromLinear: contiguous row-major matches identity" {
    // A freshly created tensor has row-major strides, so the logical
    // index equals the physical offset exactly.
    const shape = Shape.init3D(2, 3, 4);
    const strides = computeStrides(shape);
    for (0..24) |i| {
        try std.testing.expectEqual(i, logicalOffsetFromLinear(shape, strides, i));
    }
}

test "logicalOffsetFromLinear: transposed 2D view" {
    // Original (3, 2) row-major buffer: indices 0..5 in row-major order.
    // After transpose, the view has shape (2, 3) and strides (1, 2)
    // (axis strides swapped). Iterating logically 0..5 over the view
    // should hit the original buffer indices in this pattern:
    //   view[0,0] → orig[0,0] = 0
    //   view[0,1] → orig[1,0] = 2
    //   view[0,2] → orig[2,0] = 4
    //   view[1,0] → orig[0,1] = 1
    //   view[1,1] → orig[1,1] = 3
    //   view[1,2] → orig[2,1] = 5
    const view_shape = Shape.init2D(2, 3);
    const view_strides = Strides{ .values = .{ 1, 2, 0, 0 }, .rank = 1 };
    const expected = [_]usize{ 0, 2, 4, 1, 3, 5 };
    for (expected, 0..) |want, i| {
        try std.testing.expectEqual(want, logicalOffsetFromLinear(view_shape, view_strides, i));
    }
}

test "logicalOffsetFromLinear: 3D inner transpose" {
    // (2, 3, 4) contiguous → strides (12, 4, 1).
    // Inner transpose to (2, 4, 3) swaps strides[1] and strides[2] → (12, 1, 4).
    const view_shape = Shape.init3D(2, 4, 3);
    const view_strides = Strides{ .values = .{ 12, 1, 4, 0 }, .rank = 2 };
    // Logical index 0 → (0, 0, 0) → 0
    try std.testing.expectEqual(@as(usize, 0), logicalOffsetFromLinear(view_shape, view_strides, 0));
    // Logical (0, 1, 0) = linear 3 → 0*12 + 1*1 + 0*4 = 1
    try std.testing.expectEqual(@as(usize, 1), logicalOffsetFromLinear(view_shape, view_strides, 3));
    // Logical (0, 0, 1) = linear 1 → 0*12 + 0*1 + 1*4 = 4
    try std.testing.expectEqual(@as(usize, 4), logicalOffsetFromLinear(view_shape, view_strides, 1));
    // Logical (1, 3, 2) = linear 1*12 + 3*3 + 2 = 23 → 1*12 + 3*1 + 2*4 = 23
    try std.testing.expectEqual(@as(usize, 23), logicalOffsetFromLinear(view_shape, view_strides, 23));
}
