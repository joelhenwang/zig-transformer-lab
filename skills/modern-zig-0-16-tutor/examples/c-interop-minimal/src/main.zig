//! target: Zig 0.16.0 — illustrative.
//!
//! Imports the translated-C module and calls a C function we compile
//! alongside the Zig code.

const std = @import("std");
const c = @import("c");

pub fn main(init: std.process.Init) !void {
    const sum = c.example_add(3, 4);

    var buf: [128]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("example_add(3, 4) = {d}\n", .{sum});
    try fw.flush();
}
