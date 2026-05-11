//!
//! zig-transformer-lab — Debug helpers: tensor dump / load (.ztlt format)
//!
//! Purpose:
//!   Persist a single tensor to disk in the ZTLT binary format (same
//!   format emitted by `tools/oracle.py:write_tensor` and consumed by
//!   `src/testing/oracle.zig:loadTensor`). Lets a debug session
//!   compare a Zig-produced tensor against a PyTorch oracle, or
//!   vice-versa, by sharing `.ztlt` files on disk.
//!
//!   Symmetric `load` re-reads the dump. Implemented as a pass-through
//!   to the existing oracle loader so the format is consistent across
//!   production code and tests.
//!
//! Shape contract:
//!   dump(io, path, t) → void
//!   load(alloc, io, path) → Tensor  (owning CPU tensor)
//!
//! File format (32-byte header + payload):
//!   offset  size   field
//!       0    4     "ZTLT" magic
//!       4    4     u32 version = 1
//!       8    1     u8 rank (1..4)
//!       9    3     pad (zeros)
//!      12   16     u32[4] dims (entries beyond rank are zero)
//!      28    4     u32 n_elements
//!      32  n*4     f32[n_elements] little-endian
//!
//!   Full derivation and rationale in `docs/oracle.md`.
//!
//! Memory ownership:
//!   - `dump` allocates NOTHING on the CPU path (writes through a
//!     stack buffer + direct byte slice of `t.data`).
//!   - `dump` on a CUDA tensor allocates a temporary scratch buffer
//!     via the caller's allocator for the DtoH copy; released before
//!     return.
//!   - `load` returns an owning CPU Tensor allocated via the caller's
//!     allocator. Caller must `deinit`.
//!
//! Errors:
//!   IoError          — file create/write/read failure
//!   ShapeMismatch    — rank out of [1, 4] or dims contradict n_elements
//!                      (from oracle loader on load; dump rejects
//!                      rank > 4 before writing)
//!   OutOfMemory      — scratch / output buffer allocation failure
//!   CudaError        — DtoH copy failure on CUDA dump
//!
//! Device:
//!   - CPU dump: write `t.data` directly as little-endian bytes.
//!   - CUDA dump: DtoH-copy buf.len elements, then write those bytes.
//!     Note: CUDA tensors may have DeviceBuffer.len > shape-implied
//!     element count if the storage was sized for a larger view.
//!     We write only the first `totalElements(t.shape)` floats — the
//!     caller's logical tensor size, not the underlying buffer size.
//!

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const totalElements = @import("../tensor/shape.zig").totalElements;
const oracle = @import("../testing/oracle.zig");

/// Magic bytes at the start of every .ztlt file. Shared with the
/// Python oracle writer and the Zig oracle loader.
const ZTLT_MAGIC = "ZTLT";
const ZTLT_VERSION: u32 = 1;
const ZTLT_MAX_RANK: usize = 4;
const ZTLT_HEADER_SIZE: usize = 32;

/// Copy the logical elements of a tensor into a freshly-allocated
/// host buffer, regardless of device. Caller owns the returned slice
/// and must `allocator.free` it.
///
/// For CPU tensors this is a memcpy; we could skip the copy by
/// writing `t.data` directly, but keeping a single code path
/// simplifies `dump` at negligible cost (a debug helper is not a hot
/// path). For CUDA tensors this is the DtoH scratch buffer.
///
/// Returns exactly `totalElements(t.shape)` floats — the CUDA
/// DeviceBuffer may be larger than the logical shape; we write only
/// the logical region.
fn tensorToHost(allocator: std.mem.Allocator, t: Tensor) LabError![]f32 {
    const n = totalElements(t.shape);
    return switch (t.storage) {
        .cpu => blk: {
            if (t.data.len < n) return error.ShapeMismatch;
            const out = allocator.alloc(f32, n) catch return error.OutOfMemory;
            @memcpy(out, t.data[0..n]);
            break :blk out;
        },
        .cuda => |buf| blk: {
            // DeviceBuffer.copyToHost wants the destination length to
            // match buf.len. Allocate the full device buffer size,
            // copy, then slice down to the logical n. Under normal
            // usage `buf.len == n` (fresh op output); only toCuda()
            // of a shape-subset source would make buf.len > n.
            const scratch = allocator.alloc(f32, buf.len) catch return error.OutOfMemory;
            errdefer allocator.free(scratch);
            try buf.copyToHost(scratch);
            if (buf.len == n) break :blk scratch;
            // Shrink: allocate the exact logical size, copy, free the
            // scratch. An allocator.resize would be nicer but not
            // guaranteed for every allocator.
            const out = allocator.alloc(f32, n) catch return error.OutOfMemory;
            @memcpy(out, scratch[0..n]);
            allocator.free(scratch);
            break :blk out;
        },
    };
}

/// Write a tensor to `path` as a .ztlt file.
///
/// Worked example:
///   try debug.dump.dump(io, "/tmp/logits.ztlt", logits);
///   // Then in Python:
///   //   from tools.oracle import read_tensor_ztlt
///   //   t = read_tensor_ztlt("/tmp/logits.ztlt")
///   //   print(t.shape, t[:5])
pub fn dump(allocator: std.mem.Allocator, io: std.Io, path: []const u8, t: Tensor) LabError!void {
    const rank = t.shape.ndim();
    if (rank < 1 or rank > ZTLT_MAX_RANK) return error.ShapeMismatch;

    const host = try tensorToHost(allocator, t);
    defer allocator.free(host);

    const cwd = std.Io.Dir.cwd();
    const file = cwd.createFile(io, path, .{}) catch return error.IoError;
    defer file.close(io);

    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);
    const w = &writer.interface;

    // Header — mirror tools/oracle.py:write_tensor exactly.
    w.writeAll(ZTLT_MAGIC) catch return error.IoError;
    w.writeInt(u32, ZTLT_VERSION, .little) catch return error.IoError;
    w.writeInt(u8, @intCast(rank), .little) catch return error.IoError;
    w.writeAll(&.{ 0, 0, 0 }) catch return error.IoError; // 3-byte pad
    for (0..ZTLT_MAX_RANK) |i| {
        const dim: u32 = if (i < rank) @intCast(t.shape.dims[i]) else 0;
        w.writeInt(u32, dim, .little) catch return error.IoError;
    }
    const n_elements: u32 = @intCast(host.len);
    w.writeInt(u32, n_elements, .little) catch return error.IoError;

    // Payload — raw f32 bytes little-endian. Our supported targets
    // are all LE; a big-endian port would byte-swap here.
    const bytes = std.mem.sliceAsBytes(host);
    w.writeAll(bytes) catch return error.IoError;

    writer.flush() catch return error.IoError;
}

/// Re-read a .ztlt file into a fresh owning CPU tensor. Delegates to
/// the oracle loader (which validates magic/version/rank and produces
/// an owned tensor).
pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8) LabError!Tensor {
    return oracle.loadTensor(allocator, io, path);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn testIo() !std.Io {
    const T = struct {
        var instance: ?std.Io.Threaded = null;
    };
    if (T.instance == null) {
        T.instance = std.Io.Threaded.init(std.testing.allocator, .{});
    }
    return T.instance.?.io();
}

test "dump/load round-trip preserves 1D tensor values" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/debug_dump_1d.ztlt";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var a = try Tensor.init(alloc, Shape.init1D(5));
    defer a.deinit(alloc);
    a.data[0] = 1.5;
    a.data[1] = -2.75;
    a.data[2] = 0.0;
    a.data[3] = 3.14159;
    a.data[4] = 100.0;

    try dump(alloc, io, path, a);

    var b = try load(alloc, io, path);
    defer b.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 5), b.shape.dims[0]);
    try std.testing.expectEqual(@as(u2, 0), b.shape.rank); // rank field encodes ndim-1
    for (0..5) |i| {
        try std.testing.expectEqual(a.data[i], b.data[i]);
    }
}

test "dump/load round-trip preserves 3D tensor shape and values" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/debug_dump_3d.ztlt";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var a = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
    defer a.deinit(alloc);
    for (0..24) |i| a.data[i] = @floatFromInt(i);

    try dump(alloc, io, path, a);

    var b = try load(alloc, io, path);
    defer b.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), b.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), b.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), b.shape.dims[2]);
    for (0..24) |i| {
        try std.testing.expectEqual(a.data[i], b.data[i]);
    }
}

test "dump/load round-trip preserves 4D tensor" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/debug_dump_4d.ztlt";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var a = try Tensor.init(alloc, Shape.init4D(1, 2, 3, 4));
    defer a.deinit(alloc);
    for (0..24) |i| a.data[i] = @as(f32, @floatFromInt(i)) * 0.5;

    try dump(alloc, io, path, a);

    var b = try load(alloc, io, path);
    defer b.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), b.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), b.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), b.shape.dims[2]);
    try std.testing.expectEqual(@as(usize, 4), b.shape.dims[3]);
    for (0..24) |i| try std.testing.expectEqual(a.data[i], b.data[i]);
}
