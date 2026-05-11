//!
//! zig-transformer-lab — Debug helpers: device-aware tensor comparison
//!
//! Purpose:
//!   Compute element-wise absolute and relative diffs between two
//!   tensors, with either one allowed to live on CUDA. Used for
//!   CPU-vs-GPU parity debugging outside the formal oracle harness:
//!   when an op's output drifts unexpectedly, `compare` reports the
//!   worst offender so a human can localise the drift.
//!
//!   Complements `src/testing/oracle.zig` which computes the same
//!   diffs but is private to the test-harness layer and assumes
//!   both tensors are CPU-resident. `compare` is the production-safe
//!   variant: it handles CUDA inputs by DtoH-copying, and returns a
//!   structured `CompareReport` rather than raising on mismatch.
//!
//! Shape contract:
//!   compare(alloc, a, b, opts) → CompareReport
//!
//! Memory ownership:
//!   Scratch allocations for DtoH copies are released before the
//!   function returns. The returned `CompareReport` is a plain value
//!   struct — no allocator-owned memory inside.
//!
//! Errors:
//!   ShapeMismatch — a and b differ in shape
//!   CudaError     — DtoH copy failure (CUDA inputs)
//!   OutOfMemory   — scratch allocation failure
//!
//! Device:
//!   - CPU + CPU: direct scan of both tensors' data.
//!   - CPU + CUDA (either order): DtoH-copy the CUDA one, then scan.
//!   - CUDA + CUDA: DtoH-copy both.
//!
//!   `sameDevice` and `requireSameDevice` are deliberately NOT used:
//!   the whole point of `compare` is to cross device boundaries.
//!

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const shape_equals = @import("../tensor/shape.zig").equals;
const totalElements = @import("../tensor/shape.zig").totalElements;

/// Structured result of a comparison. Returned by `compare`, never
/// thrown. Callers decide what to do with the numbers (assert,
/// log, threshold).
///
/// Fields:
///   max_abs_diff   — largest |a_i - b_i| across all elements
///   max_rel_err    — largest |a_i - b_i| / max(|b_i|, denom_floor)
///   worst_idx      — flat index of the element that produced max_abs_diff
///                    (if the max ties across many elements, the first
///                    encountered in row-major order wins)
///   n_elements     — total element count; useful for confirming the
///                    compare looked at what the caller expected
pub const CompareReport = struct {
    max_abs_diff: f32,
    max_rel_err: f32,
    worst_idx: usize,
    n_elements: usize,

    /// Convenience predicate matching the oracle-style OR tolerance:
    /// pass if max_rel_err < rel_tol OR max_abs_diff < abs_tol.
    pub fn withinTolerance(self: CompareReport, rel_tol: f32, abs_tol: f32) bool {
        return self.max_rel_err < rel_tol or self.max_abs_diff < abs_tol;
    }
};

pub const CompareOpts = struct {
    /// Floor applied to the reference value in the relative-error
    /// denominator. Matches the convention in `src/testing/oracle.zig`:
    /// near-zero references would otherwise spike rel_err arbitrarily.
    /// Default `1e-8` preserves tiny honest relative errors; use a
    /// larger floor (e.g. `1e-2`) for comparisons near zero.
    denom_floor: f32 = 1e-8,
};

fn tensorToHostAlloc(allocator: std.mem.Allocator, t: Tensor) LabError![]f32 {
    return switch (t.storage) {
        .cpu => blk: {
            // Return a fresh copy so the caller can free uniformly on
            // both paths. The cost is one memcpy for CPU tensors —
            // acceptable in a debug helper.
            const out = allocator.alloc(f32, t.data.len) catch return error.OutOfMemory;
            @memcpy(out, t.data);
            break :blk out;
        },
        .cuda => |buf| blk: {
            const host = allocator.alloc(f32, buf.len) catch return error.OutOfMemory;
            errdefer allocator.free(host);
            try buf.copyToHost(host);
            break :blk host;
        },
    };
}

/// Element-wise diff between two tensors of identical shape.
///
/// Returns a `CompareReport` describing the worst absolute and
/// relative deviation. Does NOT raise on tolerance — the caller
/// decides via `report.withinTolerance(...)` or explicit thresholds.
///
/// Worked example:
///   const r = try debug.compare(alloc, cuda_out, cpu_out, .{});
///   std.debug.print("worst idx={d} abs={e} rel={e}\n", .{
///       r.worst_idx, r.max_abs_diff, r.max_rel_err,
///   });
///   try std.testing.expect(r.withinTolerance(1e-3, 1e-4));
pub fn compare(
    allocator: std.mem.Allocator,
    a: Tensor,
    b: Tensor,
    opts: CompareOpts,
) LabError!CompareReport {
    if (!shape_equals(a.shape, b.shape)) return error.ShapeMismatch;
    const n = totalElements(a.shape);

    const a_host = try tensorToHostAlloc(allocator, a);
    defer allocator.free(a_host);
    const b_host = try tensorToHostAlloc(allocator, b);
    defer allocator.free(b_host);

    // The underlying buffers may be larger than the logical element
    // count (e.g. CUDA tensors whose DeviceBuffer.len > product of
    // shape.dims). We only care about the first `n` elements in
    // row-major order — for the freshly-allocated CUDA outputs our
    // codebase produces this always matches, but guard regardless.
    const n_a = @min(n, a_host.len);
    const n_b = @min(n, b_host.len);
    const n_cmp = @min(n_a, n_b);

    var worst_abs: f32 = 0.0;
    var worst_rel: f32 = 0.0;
    var worst_idx: usize = 0;
    for (0..n_cmp) |i| {
        const d = @abs(a_host[i] - b_host[i]);
        const denom = @max(@abs(b_host[i]), opts.denom_floor);
        const r = d / denom;
        if (d > worst_abs) {
            worst_abs = d;
            worst_idx = i;
        }
        if (r > worst_rel) worst_rel = r;
    }

    return .{
        .max_abs_diff = worst_abs,
        .max_rel_err = worst_rel,
        .worst_idx = worst_idx,
        .n_elements = n_cmp,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const Shape = @import("../tensor/shape.zig").Shape;

test "compare: identical tensors report zero diff" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(4));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(4));
    defer b.deinit(alloc);
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    @memcpy(b.data, a.data);

    const r = try compare(alloc, a, b, .{});
    try std.testing.expectEqual(@as(f32, 0.0), r.max_abs_diff);
    try std.testing.expectEqual(@as(f32, 0.0), r.max_rel_err);
    try std.testing.expectEqual(@as(usize, 4), r.n_elements);
}

test "compare: small abs diff in the 2nd element" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(3));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(3));
    defer b.deinit(alloc);
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    b.data[0] = 1.0;
    b.data[1] = 2.001;
    b.data[2] = 3.0;

    const r = try compare(alloc, a, b, .{});
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), r.max_abs_diff, 1e-6);
    try std.testing.expectEqual(@as(usize, 1), r.worst_idx);
}

test "compare: shape mismatch raises ShapeMismatch" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(4));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(3));
    defer b.deinit(alloc);
    try std.testing.expectError(error.ShapeMismatch, compare(alloc, a, b, .{}));
}

test "compare: near-zero reference uses denom floor" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(1));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(1));
    defer b.deinit(alloc);
    a.data[0] = 1e-6;
    b.data[0] = 0.0;
    // With default denom_floor=1e-8, rel = 1e-6 / 1e-8 = 100 -> huge.
    const r_tight = try compare(alloc, a, b, .{});
    try std.testing.expect(r_tight.max_rel_err > 10.0);

    // With denom_floor=1e-2, rel = 1e-6 / 1e-2 = 1e-4 -> small.
    const r_loose = try compare(alloc, a, b, .{ .denom_floor = 1e-2 });
    try std.testing.expect(r_loose.max_rel_err < 1e-3);
}

test "withinTolerance: OR composition" {
    const r = CompareReport{
        .max_abs_diff = 1e-6,
        .max_rel_err = 100.0, // near-zero ref blew this up
        .worst_idx = 0,
        .n_elements = 1,
    };
    // Fails rel alone, passes abs alone, so OR passes.
    try std.testing.expect(r.withinTolerance(1e-4, 1e-5));
    // Fails both.
    try std.testing.expect(!r.withinTolerance(1e-4, 1e-10));
}

test "compare: 2D tensors scan row-major order" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer b.deinit(alloc);
    for (0..6) |i| {
        a.data[i] = @floatFromInt(i);
        b.data[i] = @floatFromInt(i);
    }
    b.data[4] = 100.0; // (row 1, col 1)

    const r = try compare(alloc, a, b, .{});
    try std.testing.expectEqual(@as(usize, 4), r.worst_idx);
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), r.max_abs_diff, 1e-6);
}
