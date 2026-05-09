//! target: Zig 0.16.0 — illustrative; run with:
//!   zig build-exe hello-world-io-0-16.zig && .\hello-world-io-0-16.exe
//!
//! Demonstrates the modern I/O path: Juicy Main + std.Io.File.stdout().

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    // One-shot write: no buffer needed.
    try std.Io.File.stdout().writeStreamingAll(init.io, "hello via std.Io\n");

    // Multiple writes: use a buffered writer.
    var buf: [256]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("pid-ish value = {d}\n", .{@as(u32, 0xCAFE)});
    try fw.interface.print("second line\n", .{});
    try fw.flush(); // REQUIRED
}
