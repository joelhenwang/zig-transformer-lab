//! target: Zig 0.16.0 — illustrative.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var buf: [256]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    try fw.interface.print("argc = {d}\n", .{args.len});
    for (args, 0..) |arg, i| {
        try fw.interface.print("  argv[{d}] = {s}\n", .{ i, arg });
    }

    try fw.flush();
}

test "1 + 1 == 2" {
    try std.testing.expectEqual(@as(i32, 2), 1 + 1);
}
