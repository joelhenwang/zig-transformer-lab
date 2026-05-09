# Recipes: allocators

## Recipe 3 — Allocating and freeing a slice

**Problem.** Allocate a slice of `u8`, fill it, and free it, without leaks.

**Mentor explanation.** Zig has no default allocator. You pass one in. The
allocating function owns the error; the caller owns the memory.

**Minimal snippet.**
```zig
const std = @import("std");

pub fn makeBuf(a: std.mem.Allocator, n: usize) ![]u8 {
    const buf = try a.alloc(u8, n);
    errdefer a.free(buf);        // clean up if any later step fails

    @memset(buf, 'x');
    return buf;                   // ownership transferred to caller
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const buf = try makeBuf(a, 16);
    defer a.free(buf);            // caller frees

    try std.Io.File.stdout().writeStreamingAll(init.io, buf);
    try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
}
```

**Test to write.**
```zig
test "makeBuf has no leak" {
    const a = std.testing.allocator;
    const buf = try makeBuf(a, 8);
    defer a.free(buf);
    try std.testing.expectEqual(@as(usize, 8), buf.len);
    for (buf) |b| try std.testing.expectEqual(@as(u8, 'x'), b);
}
```

**Stale pattern to avoid.**
```zig
// WRONG — defer before the alloc may succeed, and no allocator param
fn makeBad() ![]u8 {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const buf = try gpa.allocator().alloc(u8, 16);
    defer gpa.allocator().free(buf);   // runs at scope end — too early
    return buf;
}
```

**Exercise.** Modify `makeBuf` to take a comptime element type `T`; return
`[]T`. What changes in the signature, and why is that a good design choice
for a numeric kernel?

---

## Recipe 4 — Using `std.testing.allocator`

**Problem.** Detect leaks in a test automatically.

**Mentor explanation.** `std.testing.allocator` is a leak-detecting
allocator. At the end of the test it reports every address that wasn't
freed, with a stack trace. Every allocating test should use it.

**Minimal snippet.**
```zig
const std = @import("std");

test "no leak in a small ArrayList" {
    const a = std.testing.allocator;

    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);

    try list.append(a, 1);
    try list.append(a, 2);
    try list.append(a, 3);

    try std.testing.expectEqual(@as(usize, 3), list.items.len);
}
```

**Stale pattern to avoid.**
```zig
// WRONG — managed style, and stores allocator inside list
var list = std.ArrayList(u32).init(std.testing.allocator);
defer list.deinit();
try list.append(1);
```

**Exercise.** Intentionally introduce a leak (comment out `defer
list.deinit(a);`). Run `zig test`. What does the error message show?

<!-- ~1.0k tokens -->
