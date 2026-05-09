# Recipes: containers

## Recipe 5 — `std.ArrayList` with the unmanaged idiom

**Problem.** Grow a list of numbers, iterate, and free it correctly.

**Mentor explanation.** In 0.16, `std.ArrayList(T)` is default-initialized
with `.empty` and does not store its allocator. Every mutating method takes
the allocator as the first argument. This lets you move containers across
allocators without carrying baggage.

**Minimal snippet.**
```zig
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;

    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);

    for (0..5) |i| try list.append(a, @intCast(i * i));

    var buf: [128]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    for (list.items, 0..) |v, i| {
        try fw.interface.print("[{d}] = {d}\n", .{ i, v });
    }
    try fw.flush();
}
```

**Test to write.**
```zig
test "list grows and frees" {
    const a = std.testing.allocator;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);
    try list.append(a, 10);
    try list.append(a, 20);
    try list.appendSlice(a, &.{ 30, 40 });
    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 30, 40 }, list.items);
}

test "ensureTotalCapacity reduces reallocs" {
    const a = std.testing.allocator;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);
    try list.ensureTotalCapacity(a, 1024);
    for (0..1024) |i| list.appendAssumeCapacity(@intCast(i));
    try std.testing.expectEqual(@as(usize, 1024), list.items.len);
}
```

**Stale pattern to avoid.**
```zig
// WRONG — stores allocator, uses old method signatures
var list = std.ArrayList(u32).init(a);
defer list.deinit();
try list.append(42);
```

**Exercise.** Replace the `ArrayList` with a `std.AutoHashMap(u32, []const u8)`
(unmanaged idiom). Add, look up, and remove keys. What extra step is needed
to iterate the entries?

<!-- ~0.8k tokens -->
