//! target: Zig 0.16.0 — illustrative; run with:
//!   zig build-exe file-read-io-0-16.zig && .\file-read-io-0-16.exe
//!
//! Read a whole file into an arena-allocated buffer and print its length.
//!
//! Uses the 0.16 API:
//!   - `std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(n))`
//!   - Error rename: `FileTooBig` → `StreamTooLong`

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena_a = init.arena.allocator();

    const path = "file-read-io-0-16.zig"; // read ourselves
    const contents = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        path,
        arena_a,
        .limited(1 << 16), // 64 KiB cap
    );

    var buf: [128]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("read {d} bytes from {s}\n", .{ contents.len, path });
    try fw.flush();
}
