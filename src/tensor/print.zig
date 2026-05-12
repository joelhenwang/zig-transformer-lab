//!
//! zig-transformer-lab — Debug printing for tensors
//!
//! Purpose:
//!   Provides two debug utilities for inspecting tensors during development:
//!
//!   1. debugSummary() — prints shape, strides, dtype, device, ownership,
//!      and computed statistics (min, mean, max, NaN count, Inf count).
//!      This is the first thing to call when something looks wrong: the
//!      stats tell you at a glance if the tensor is all zeros, has NaNs,
//!      or has blown up to Inf.
//!
//!   2. printValues() — prints up to N f32 values from the flat data buffer
//!      as a concise list like [1.000, 2.000, 3.000, ...].
//!      Useful for spot-checking small tensors or the beginning of a
//!      large tensor.
//!
//! Why single-pass stats?
//!   debugSummary() computes min/mean/max/NaN/Inf in one loop over the
//!   data buffer. This avoids three separate passes (one for min/max,
//!   one for mean, one for NaN/Inf) which would each touch all elements.
//!   For a large tensor (say 1M elements = 4 MB), three passes means 12 MB
//!   of memory reads; one pass means 4 MB. The data stays in L2 cache.
//!
//! Shape contract:
//!   Both functions read tensor.cpuData()[0..totalElements()]. They do NOT
//!   use flatIndex() because stats are computed over the flat buffer
//!   regardless of stride layout. For a non-contiguous tensor, the
//!   "flat" stats may not match the logical element order, but they
//!   still tell you if the buffer contains NaN/Inf.
//!
//!   future: add a per-axis variant that respects strides so that
//!   non-contiguous tensors print correctly. Today's fast path only
//!   works for contiguous tensors.
//!
//! Memory ownership:
//!   Neither function allocates memory. They write directly to the
//!   provided writer.
//!
//! Errors:
//!   - IoError: if writing to the writer fails (e.g., broken pipe).
//!
//! TODO:
//!   - future: support non-contiguous tensors by iterating with
//!     flatIndex() (today's loop assumes contiguous storage).
//!   - future: a printMatrix() variant that formats 2D tensors as
//!     grids with aligned column widths.
//;

const std = @import("std");
const Io = std.Io;

const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("tensor.zig").Tensor;
const totalElements = @import("shape.zig").totalElements;

/// Print a one-line summary of the tensor's metadata and data statistics.
///
/// Output format:
///   Tensor shape=(2,3) strides=[3,1] dtype=f32 device=cpu owned=true
///     min=0.000 mean=3.500 max=6.000 nan=0 inf=0
///
/// The statistics (min, mean, max, NaN count, Inf count) are computed in
/// a single pass over the flat data buffer for cache efficiency.
///
/// Worked example:
///   // For a 2x3 tensor with values [1,2,3,4,5,6]:
///   debugSummary(tensor, writer)
///   // Output includes: min=1.000 mean=3.500 max=6.000 nan=0 inf=0
pub fn debugSummary(tensor: Tensor, writer: *std.Io.Writer) !void {
    // --- Print metadata ---
    // We format shape using the (d,d,...) notation and strides with [s0,s1,...].
    try writer.print("Tensor shape=(", .{});
    const ndim = tensor.shape.ndim();
    for (0..ndim) |i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print("{d}", .{tensor.shape.dims[i]});
    }
    try writer.print(") strides=[", .{});
    for (0..ndim) |i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print("{d}", .{tensor.strides.values[i]});
    }
    try writer.print("] dtype={s} device={s} owned={}\n", .{
        tensor.dtype.label(),
        tensor.device.label(),
        tensor.isOwned(),
    });

    // --- Compute stats in a single pass ---
    // We use the flat data buffer, not flatIndex(), because:
    //   1. All elements are in the buffer regardless of strides.
    //   2. A single linear scan is cache-friendly.
    //   3. For contiguous tensors, this is exactly the logical order.
    const n = tensor.cpuData().len;

    var min_val: f32 = std.math.inf(f32); // Start at +inf so any real value is smaller
    var max_val: f32 = -std.math.inf(f32); // Start at -inf so any real value is larger
    var sum: f64 = 0.0; // Accumulate in f64 to avoid precision loss on large sums
    var nan_count: usize = 0;
    var inf_count: usize = 0;

    for (tensor.cpuData()) |v| {
        // Check for NaN first — NaN comparisons are always false, so
        // we must detect it before the min/max comparisons. A NaN would
        // never update min_val or max_val because (NaN < x) is false
        // and (NaN > x) is false, silently corrupting the stats.
        if (std.math.isNan(v)) {
            nan_count += 1;
            continue;
        }
        // Inf is valid for min/max (e.g., a mask tensor might use -Inf),
        // but we still count it separately so the user knows.
        if (std.math.isInf(v)) {
            inf_count += 1;
        }
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        // Accumulate into f64 to avoid precision loss when summing many
        // f32 values. A 1M-element tensor of f32s can accumulate ~1e-6
        // relative error if summed in f32; f64 reduces this to ~1e-15.
        sum += @as(f64, @floatCast(v));
    }

    // Compute mean: total sum / number of elements.
    // If the tensor is empty (0 elements), mean is NaN by convention.
    const mean_val: f32 = if (n > 0)
        @floatCast(sum / @as(f64, @floatFromInt(n)))
    else
        std.math.nan(f32);

    // If all values were NaN, min and max are still at their initial
    // +/- Inf values. We report them as NaN to indicate "no valid values".
    if (nan_count == n) {
        min_val = std.math.nan(f32);
        max_val = std.math.nan(f32);
    }

    try writer.print("  min={d:.3} mean={d:.3} max={d:.3} nan={} inf={}\n", .{
        min_val,
        mean_val,
        max_val,
        nan_count,
        inf_count,
    });
}

/// Print up to `max_items` f32 values from the tensor's flat data buffer.
///
/// Output format: [1.000, 2.000, 3.000, ...]
/// If the tensor has more elements than max_items, the output is
/// truncated with "..." after the last printed value.
///
/// This is useful for quick sanity checks: "does my tensor contain
/// reasonable-looking numbers?" For detailed inspection, use debugSummary().
///
/// Worked example:
///   // For a tensor with data [1, 2, 3, 4, 5, 6] and max_items=3:
///   printValues(tensor, writer, 3)
///   // Output: [1.000, 2.000, 3.000, ...]
pub fn printValues(tensor: Tensor, writer: *std.Io.Writer, max_items: usize) !void {
    try writer.print("[", .{});
    const n = tensor.cpuData().len;
    const limit = @min(n, max_items);

    for (0..limit) |i| {
        if (i > 0) try writer.print(", ", .{});
        try writer.print("{d:.3}", .{tensor.cpuData()[i]});
    }

    // If we truncated, show "..." to indicate there are more values
    if (n > max_items) {
        try writer.print(", ...", .{});
    }

    try writer.print("]\n", .{});
}

// ============================================================================
// Tests
// ============================================================================

test "debugSummary on a small known tensor" {
    const Shape = @import("shape.zig").Shape;

    const shape = Shape.init2D(2, 3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // Fill with known values: [1, 2, 3, 4, 5, 6]
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    t.cpuData()[2] = 3.0;
    t.cpuData()[3] = 4.0;
    t.cpuData()[4] = 5.0;
    t.cpuData()[5] = 6.0;

    // Write to a fixed buffer so we can inspect the output.
    // In Zig 0.16.0, std.Io.Writer.fixed(buf) creates a writer that
    // writes into a fixed buffer. writer.buffered() returns the slice
    // of bytes that have been written so far.
    var buf: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    try debugSummary(t, &writer);

    const output = writer.buffered();

    // Verify key metadata appears in the output.
    // We check for substrings rather than exact match because
    // formatting may vary slightly across Zig versions.
    try std.testing.expect(std.mem.indexOf(u8, output, "shape=(2,3)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "strides=[3,1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "device=cpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "owned=true") != null);

    // Verify stats: min=1.000, max=6.000, mean=3.500, nan=0, inf=0
    try std.testing.expect(std.mem.indexOf(u8, output, "min=1.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "max=6.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "mean=3.500") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "nan=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "inf=0") != null);
}

test "debugSummary detects NaN and Inf" {
    const Shape = @import("shape.zig").Shape;

    const shape = Shape.init1D(4);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // Put in a NaN, an Inf, a -Inf, and a normal value
    t.cpuData()[0] = std.math.nan(f32);
    t.cpuData()[1] = std.math.inf(f32);
    t.cpuData()[2] = -std.math.inf(f32);
    t.cpuData()[3] = 1.0;

    var buf: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    try debugSummary(t, &writer);

    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "nan=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "inf=2") != null);
}

test "printValues with truncation" {
    const Shape = @import("shape.zig").Shape;

    const shape = Shape.init1D(6);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    // Fill with [1, 2, 3, 4, 5, 6]
    for (0..6) |i| {
        t.cpuData()[i] = @floatFromInt(i + 1);
    }

    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    // Print only 3 items
    try printValues(t, &writer, 3);

    const output = writer.buffered();
    // Should contain first 3 values and "..."
    try std.testing.expect(std.mem.indexOf(u8, output, "1.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "2.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "3.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "...") != null);
    // Should NOT contain values 4-6
    try std.testing.expect(std.mem.indexOf(u8, output, "4.000") == null);
}

test "printValues without truncation" {
    const Shape = @import("shape.zig").Shape;

    const shape = Shape.init1D(3);
    var t = try Tensor.init(std.testing.allocator, shape);
    defer t.deinit(std.testing.allocator);

    t.cpuData()[0] = 10.0;
    t.cpuData()[1] = 20.0;
    t.cpuData()[2] = 30.0;

    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    // Print all items (max_items >= len)
    try printValues(t, &writer, 10);

    const output = writer.buffered();
    // Should contain all 3 values and no "..."
    try std.testing.expect(std.mem.indexOf(u8, output, "10.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "20.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "30.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "...") == null);
}
