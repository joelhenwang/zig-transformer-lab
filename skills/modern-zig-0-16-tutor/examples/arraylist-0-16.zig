//! target: Zig 0.16.0 — illustrative; run with:
//!   zig build-exe arraylist-0-16.zig && .\arraylist-0-16.exe
//!   zig test    arraylist-0-16.zig
//!
//! `std.ArrayList` in the 0.16 unmanaged idiom.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;

    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(a);

    try list.append(a, 10);
    try list.append(a, 20);
    try list.appendSlice(a, &.{ 30, 40, 50 });

    var buf: [128]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    for (list.items, 0..) |v, i| {
        try fw.interface.print("[{d}] = {d}\n", .{ i, v });
    }
    try fw.flush();
}

test "list integrity" {
    const a = std.testing.allocator;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);
    try list.appendSlice(a, &.{ 1, 2, 3, 4 });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, list.items);
}
