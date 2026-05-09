//! target: Zig 0.16.0 — illustrative, verify with `zig build run` on 0.16.0.
//!
//! Minimal Juicy-Main program: takes `init: std.process.Init`, writes one
//! line to stdout with an explicit `std.Io` buffered writer, and flushes.
//!
//! Key 0.16.0 idioms demonstrated:
//!   - `pub fn main(init: std.process.Init) !void`
//!   - `init.io` for I/O
//!   - `std.Io.File.stdout().writer(io, &buf)` for buffered stdout
//!   - `try fw.flush()` — required or output vanishes
//!
//! TODO(learner): swap `writer(io, &buf)` for `writer(io, &.{})` to get an
//! unbuffered writer. Observe the behavioral difference.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var buf: [1024]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);

    try fw.interface.print("hello from modern Zig 0.16.0\n", .{});

    try fw.flush();
}
