//!
//! src/testing/oracle.zig — Load PyTorch-generated `.ztlt` fixtures
//!
//! Purpose:
//!   The Python oracle (`tools/oracle.py`) writes a directory of `.ztlt`
//!   binary files per test case, plus a `meta.json`. This module loads
//!   them into `Tensor` values so Zig tests can compare our forward /
//!   backward implementations against PyTorch.
//!
//!   The format is documented in `tools/oracle.py` and `docs/07d_
//!   checkpoint_format.md` (the ZTLC format is the model-level cousin).
//!   Briefly, each file is:
//!
//!     offset  size   field
//!         0    4     "ZTLT" magic
//!         4    4     u32 version = 1
//!         8    1     u8 rank
//!         9    3     pad
//!        12   16     u32[4] dims
//!        28    4     u32 n_elements
//!        32  n*4     f32[n_elements]
//!
//! Design note:
//!   This module is *test-only*. It is imported by
//!   `tests/integration_oracle.zig` and never from production code.
//!   It allocates `Tensor` values via the caller's allocator so the
//!   existing ownership rules apply.
//!
//!   Tolerance comparison is split into `maxAbsDiff`, `maxRelErr`, and
//!   an assertion helper `expectClose` that matches the pattern used by
//!   `src/autograd/gradcheck.zig`: pass if max_rel_err < rel_tol OR
//!   max_abs_diff < abs_tol. Relative-only fails spuriously when the
//!   reference value is near zero; absolute-only fails spuriously when
//!   the reference value is huge. The OR composition is the pragmatic
//!   choice for f32 numerical work.
//!
//! Memory ownership:
//!   `loadTensor` allocates a fresh `Tensor` that the caller owns and
//!   must `deinit`. The file is read in full via `readFileAlloc` and
//!   the intermediate buffer is freed before returning.
//!

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const shape_mod = @import("../tensor/shape.zig");
const Shape = shape_mod.Shape;
const totalElements = shape_mod.totalElements;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Magic bytes at the start of every `.ztlt` file. Mismatch means the
/// file was not produced by our oracle (wrong format, corrupted, or
/// someone pointed the loader at the wrong path).
pub const ZTLT_MAGIC: [4]u8 = .{ 'Z', 'T', 'L', 'T' };
pub const ZTLT_VERSION: u32 = 1;
const ZTLT_MAX_RANK: usize = 4;
const ZTLT_HEADER_SIZE: usize = 32;

/// Read a whole `.ztlt` file into a heap-allocated Tensor.
///
/// Caller receives an owned contiguous CPU tensor. The returned
/// tensor's shape is exactly what the oracle wrote — it is the
/// caller's responsibility to check that the shape matches what the
/// op under test expects (or to reshape deliberately).
///
/// Errors:
///   error.IoError         — magic or version mismatch, truncated file,
///                           inconsistent header/payload size
///   error.ShapeMismatch   — rank out of range, or dims contradict
///                           n_elements
///   error.OutOfMemory     — allocator failure
pub fn loadTensor(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) LabError!Tensor {
    const cwd = std.Io.Dir.cwd();

    // Read the whole file. Tests are small (under 1 MiB each); no need
    // to stream. `readFileAlloc` returns []u8 owned by the caller.
    const bytes = cwd.readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024)) catch {
        return error.IoError;
    };
    defer allocator.free(bytes);

    if (bytes.len < ZTLT_HEADER_SIZE) return error.IoError;

    // Parse header.
    if (!std.mem.eql(u8, bytes[0..4], &ZTLT_MAGIC)) return error.IoError;
    const version = std.mem.readInt(u32, bytes[4..8], .little);
    if (version != ZTLT_VERSION) return error.IoError;

    const rank: usize = bytes[8];
    if (rank < 1 or rank > ZTLT_MAX_RANK) return error.ShapeMismatch;
    // bytes[9..12] are pad, we don't validate them (writer zeroes them).

    var dims: [ZTLT_MAX_RANK]usize = undefined;
    for (0..ZTLT_MAX_RANK) |i| {
        const off = 12 + i * 4;
        dims[i] = std.mem.readInt(u32, bytes[off..][0..4], .little);
    }
    const n_elements = std.mem.readInt(u32, bytes[28..32], .little);

    // Sanity: product of the first `rank` dims must equal n_elements,
    // and dims beyond `rank` must be zero.
    var expected_n: usize = 1;
    for (0..rank) |i| {
        if (dims[i] == 0) return error.ShapeMismatch;
        expected_n *= dims[i];
    }
    for (rank..ZTLT_MAX_RANK) |i| {
        if (dims[i] != 0) return error.ShapeMismatch;
    }
    if (expected_n != @as(usize, n_elements)) return error.ShapeMismatch;

    // Validate payload size.
    const expected_bytes = ZTLT_HEADER_SIZE + @as(usize, n_elements) * @sizeOf(f32);
    if (bytes.len < expected_bytes) return error.IoError;
    // Trailing bytes (if any) are ignored — future versions may append
    // metadata here. We could reject extras for strictness, but the
    // manifest.json is the authoritative list of files; a tensor file
    // with extras is harmless.

    // Build a Shape of the right rank.
    const shape = switch (rank) {
        1 => Shape.init1D(dims[0]),
        2 => Shape.init2D(dims[0], dims[1]),
        3 => Shape.init3D(dims[0], dims[1], dims[2]),
        4 => Shape.init4D(dims[0], dims[1], dims[2], dims[3]),
        else => unreachable,
    };

    // Allocate the destination tensor and copy the payload.
    var t = try Tensor.init(allocator, shape);
    errdefer t.deinit(allocator);

    const payload = bytes[ZTLT_HEADER_SIZE..expected_bytes];
    // On every target we support the payload layout is identical to
    // f32 row-major little-endian, so we can memcpy directly.
    @memcpy(std.mem.sliceAsBytes(t.data), payload);

    return t;
}

/// Tolerance-based comparison used throughout oracle tests.
/// Passes when `max_rel_err < rel_tol` OR `max_abs_diff < abs_tol`.
pub const CloseOptions = struct {
    rel_tol: f32,
    abs_tol: f32,
};

/// Elementwise max(|a - b|) between two tensors with identical shape.
/// Requires contiguous layouts (our oracle always produces contiguous
/// tensors, and all our op outputs are contiguous).
pub fn maxAbsDiff(a: Tensor, b: Tensor) LabError!f32 {
    if (totalElements(a.shape) != totalElements(b.shape)) return error.ShapeMismatch;
    const n = a.data.len;
    if (b.data.len != n) return error.ShapeMismatch;
    var worst: f32 = 0.0;
    for (0..n) |i| {
        const d = @abs(a.data[i] - b.data[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/// Max relative error with a small floor on the denominator to avoid
/// dividing by near-zero reference values.
///
/// Matches the convention in `src/autograd/gradcheck.zig`:
///
///     rel_err_i = |a_i - b_i| / max(|b_i|, denom_floor)
///
/// Using `|b_i|` as the reference scale (b is the oracle answer) biases
/// toward "how big is the error relative to the correct answer", which
/// is the intuitive notion.
pub fn maxRelErr(a: Tensor, b: Tensor, denom_floor: f32) LabError!f32 {
    if (totalElements(a.shape) != totalElements(b.shape)) return error.ShapeMismatch;
    const n = a.data.len;
    if (b.data.len != n) return error.ShapeMismatch;
    var worst: f32 = 0.0;
    for (0..n) |i| {
        const d = @abs(a.data[i] - b.data[i]);
        const denom = @max(@abs(b.data[i]), denom_floor);
        const r = d / denom;
        if (r > worst) worst = r;
    }
    return worst;
}

/// Assertion helper: passes if |a - b| is close per the tolerance
/// policy. Returns `error.NumericalError` on failure, which the test
/// framework converts into a clear test failure.
pub fn expectClose(
    a: Tensor,
    b: Tensor,
    opts: CloseOptions,
) LabError!void {
    const abs_diff = try maxAbsDiff(a, b);
    const rel_err = try maxRelErr(a, b, 1e-8);
    const pass = rel_err < opts.rel_tol or abs_diff < opts.abs_tol;
    if (!pass) {
        std.debug.print(
            "  oracle compare FAIL: max_abs_diff={d:.6}  max_rel_err={d:.6}  (abs_tol={d:.6}, rel_tol={d:.6})\n",
            .{ abs_diff, rel_err, opts.abs_tol, opts.rel_tol },
        );
        return error.NumericalError;
    }
}

// --------------------------------------------------------------------------
// Tests for the loader itself
// --------------------------------------------------------------------------

fn testIo() !std.Io {
    const T = struct {
        var instance: ?std.Io.Threaded = null;
    };
    if (T.instance == null) {
        T.instance = std.Io.Threaded.init(std.testing.allocator, .{});
    }
    return T.instance.?.io();
}

test "loadTensor: ZTLT magic and version validation" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    // Write a file with the wrong magic.
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/ztlt-bad-magic.bin";
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, path, .{});
    defer cwd.deleteFile(io, path) catch {};
    defer file.close(io);
    var buf: [64]u8 = undefined;
    var w = file.writer(io, &buf);
    try w.interface.writeAll("WRONG....");
    try w.flush();

    try std.testing.expectError(error.IoError, loadTensor(alloc, io, path));
}

test "maxAbsDiff and maxRelErr on identical tensors are zero" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(3));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(3));
    defer b.deinit(alloc);
    a.data[0] = 1.0; a.data[1] = 2.0; a.data[2] = 3.0;
    b.data[0] = 1.0; b.data[1] = 2.0; b.data[2] = 3.0;
    try std.testing.expectEqual(@as(f32, 0.0), try maxAbsDiff(a, b));
    try std.testing.expectEqual(@as(f32, 0.0), try maxRelErr(a, b, 1e-8));
}

test "expectClose passes on small differences within tolerance" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(2));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(2));
    defer b.deinit(alloc);
    a.data[0] = 1.00001; a.data[1] = 2.00001;
    b.data[0] = 1.00000; b.data[1] = 2.00000;
    try expectClose(a, b, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

test "expectClose fails on large differences" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, Shape.init1D(1));
    defer a.deinit(alloc);
    var b = try Tensor.init(alloc, Shape.init1D(1));
    defer b.deinit(alloc);
    a.data[0] = 1.0;
    b.data[0] = 2.0;
    try std.testing.expectError(
        error.NumericalError,
        expectClose(a, b, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 }),
    );
}
