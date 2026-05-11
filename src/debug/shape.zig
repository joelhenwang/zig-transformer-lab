//!
//! zig-transformer-lab — Debug helpers: shape assertions
//!
//! Purpose:
//!   Opt-in runtime shape assertions for development-time sanity checks.
//!   These helpers produce detailed, immediately-actionable error messages
//!   when a tensor does not have the shape the calling code expects.
//!
//!   The canonical Stage 2-7 style is "compute the shape by hand, trust
//!   the op." This is fine for correct code; but when a transpose or a
//!   reshape introduces a subtle off-by-one, the error surfaces as a
//!   `ShapeMismatch` three calls downstream, often from a line that
//!   doesn't obviously reference the wrong shape. `assertShape` lets
//!   a debug session pin the wrong-shape location on its first
//!   occurrence.
//!
//!   Failures print a one-line diagnostic via `std.debug.print` (matching
//!   the project's existing `backward.zig` debug pattern) and return
//!   `error.ShapeMismatch`. The diagnostic includes both the expected
//!   and actual shapes so the reader doesn't need to re-run with
//!   additional logging to understand what went wrong.
//!
//!   These helpers are intentionally NOT used inside `src/` production
//!   code. They are opt-in tools for example scripts, ad-hoc debugging,
//!   and new-op development. The existing ops keep their terse
//!   `return error.ShapeMismatch;` style.
//!
//! Shape contract:
//!   assertShape(t, expected)    → void or error.ShapeMismatch
//!   assertRank(t, expected)     → void or error.ShapeMismatch
//!   assertDim(t, axis, expected) → void or error.ShapeMismatch or error.InvalidArgument
//!
//! Memory ownership:
//!   None. All helpers are pure validators over stack-allocated Shape
//!   values plus a const ref to the tensor.
//!
//! Errors:
//!   ShapeMismatch — tensor shape does not match expectation
//!   InvalidArgument — axis out of range for assertDim
//!
//! Device:
//!   Device-agnostic. Shape checks inspect `tensor.shape` only, which
//!   is a value type on the Tensor struct regardless of CPU or CUDA
//!   backing storage.
//!

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const shape_equals = @import("../tensor/shape.zig").equals;
const shape_toString = @import("../tensor/shape.zig").toString;

/// Format a Shape into a stack buffer and return a slice view.
///
/// Factored out so the error-path prints don't need to handle the
/// buffer management. All shape-string representations fit comfortably
/// in 64 bytes (max rank 4, each dim <= 20 digits).
fn fmtShape(s: Shape) []const u8 {
    // Use a thread-local buffer so concurrent failures from multiple
    // tests don't interleave. One buffer per device/helper is fine
    // because each assertion prints and returns before the next runs.
    const T = struct {
        var buf: [64]u8 = undefined;
    };
    // shape_toString writes into the buffer and returns a slice into
    // it. Copies are intentional: the slice lives until the caller's
    // print returns.
    return shape_toString(s, &T.buf);
}

/// Assert that `t.shape` equals `expected`. Emits a one-line
/// diagnostic on mismatch.
///
/// Worked example:
///   // Inside new-op development:
///   try debug.assertShape(logits, Shape.init3D(B, T, V));
///   // If logits is (B, T+1, V), prints:
///   //   assertShape FAIL: expected (2, 4, 16), got (2, 5, 16)
///
/// Usage in a test:
///   try debug.assertShape(out, Shape.init2D(3, 5));
///
/// Returns `error.ShapeMismatch` on failure; `void` on success.
pub fn assertShape(t: Tensor, expected: Shape) LabError!void {
    if (shape_equals(t.shape, expected)) return;
    std.debug.print(
        "  assertShape FAIL: expected {s}, got {s}\n",
        .{ fmtShape(expected), fmtShape(t.shape) },
    );
    return error.ShapeMismatch;
}

/// Assert that `t.shape.ndim() == expected_rank`. Use when the
/// specific dimensions are not yet known (e.g. early in a function
/// that will branch on rank).
///
/// Worked example:
///   // Input must be 3D, but the specific dims come from config:
///   try debug.assertRank(x, 3);
///   const B = x.shape.dims[0];
///   const T = x.shape.dims[1];
///   const D = x.shape.dims[2];
///
/// Returns `error.ShapeMismatch` on failure.
pub fn assertRank(t: Tensor, expected_rank: usize) LabError!void {
    if (t.shape.ndim() == expected_rank) return;
    std.debug.print(
        "  assertRank FAIL: expected rank {d}, got rank {d} (shape {s})\n",
        .{ expected_rank, t.shape.ndim(), fmtShape(t.shape) },
    );
    return error.ShapeMismatch;
}

/// Assert that `t.shape.dims[axis] == expected_size`. Use for
/// per-axis checks where the other axes are irrelevant (e.g. "the
/// last axis must equal vocab_size" regardless of batch shape).
///
/// Errors:
///   InvalidArgument — axis >= t.shape.ndim()
///   ShapeMismatch   — t.shape.dims[axis] != expected_size
///
/// Worked example:
///   // Regardless of input batch / time shape, the last axis must
///   // be the vocabulary.
///   try debug.assertDim(logits, logits.shape.ndim() - 1, cfg.vocab_size);
pub fn assertDim(t: Tensor, axis: usize, expected_size: usize) LabError!void {
    const ndim = t.shape.ndim();
    if (axis >= ndim) {
        std.debug.print(
            "  assertDim FAIL: axis {d} out of range for rank-{d} shape {s}\n",
            .{ axis, ndim, fmtShape(t.shape) },
        );
        return error.InvalidArgument;
    }
    const actual = t.shape.dims[axis];
    if (actual == expected_size) return;
    std.debug.print(
        "  assertDim FAIL: expected axis {d} = {d}, got {d} (shape {s})\n",
        .{ axis, expected_size, actual, fmtShape(t.shape) },
    );
    return error.ShapeMismatch;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "assertShape: passes on matching shape" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(3, 5));
    defer t.deinit(std.testing.allocator);
    try assertShape(t, Shape.init2D(3, 5));
}

test "assertShape: fails with ShapeMismatch on mismatch" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(3, 5));
    defer t.deinit(std.testing.allocator);
    try std.testing.expectError(error.ShapeMismatch, assertShape(t, Shape.init2D(3, 4)));
}

test "assertShape: rank mismatch also reports ShapeMismatch" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(3, 5));
    defer t.deinit(std.testing.allocator);
    try std.testing.expectError(error.ShapeMismatch, assertShape(t, Shape.init3D(3, 5, 1)));
}

test "assertRank: passes on matching rank" {
    var t = try Tensor.init(std.testing.allocator, Shape.init3D(2, 3, 4));
    defer t.deinit(std.testing.allocator);
    try assertRank(t, 3);
}

test "assertRank: fails on wrong rank" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(3, 5));
    defer t.deinit(std.testing.allocator);
    try std.testing.expectError(error.ShapeMismatch, assertRank(t, 3));
}

test "assertDim: passes on matching size" {
    var t = try Tensor.init(std.testing.allocator, Shape.init3D(2, 3, 4));
    defer t.deinit(std.testing.allocator);
    try assertDim(t, 1, 3);
    try assertDim(t, 2, 4);
}

test "assertDim: fails on wrong size" {
    var t = try Tensor.init(std.testing.allocator, Shape.init3D(2, 3, 4));
    defer t.deinit(std.testing.allocator);
    try std.testing.expectError(error.ShapeMismatch, assertDim(t, 0, 5));
}

test "assertDim: returns InvalidArgument for out-of-range axis" {
    var t = try Tensor.init(std.testing.allocator, Shape.init2D(3, 5));
    defer t.deinit(std.testing.allocator);
    try std.testing.expectError(error.InvalidArgument, assertDim(t, 4, 1));
}

test "assertShape: works on 1D, 3D, 4D shapes" {
    var t1 = try Tensor.init(std.testing.allocator, Shape.init1D(7));
    defer t1.deinit(std.testing.allocator);
    try assertShape(t1, Shape.init1D(7));

    var t3 = try Tensor.init(std.testing.allocator, Shape.init3D(2, 3, 4));
    defer t3.deinit(std.testing.allocator);
    try assertShape(t3, Shape.init3D(2, 3, 4));

    var t4 = try Tensor.init(std.testing.allocator, Shape.init4D(1, 2, 3, 4));
    defer t4.deinit(std.testing.allocator);
    try assertShape(t4, Shape.init4D(1, 2, 3, 4));
}
