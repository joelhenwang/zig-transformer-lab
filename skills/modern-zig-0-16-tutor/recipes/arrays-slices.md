# Recipes: arrays & slices

## Recipe 9 — Slice bounds, ownership, and the slice-by-length trick

**Problem.** Given a larger buffer, create a fixed-length view at a
runtime-determined offset, suitable for `@memcpy` / SIMD.

**Mentor explanation.** If **both** bounds of a slice expression are
comptime-known, the result is a **pointer to array** (`*[N]T`), which
carries length in the type. That lets the compiler generate better code
and lets SIMD intrinsics accept the result without a runtime bounds check.

The trick: `buf[offset..][0..length]` — the first slice has a runtime
start, the second has both bounds comptime-known.

**Minimal snippet.**
```zig
const std = @import("std");

pub fn copyWindow(dst: *[4]u32, src: []const u32, start: usize) !void {
    if (start + 4 > src.len) return error.OutOfRange;
    const window: *const [4]u32 = src[start..][0..4];
    @memcpy(dst, window);
}
```

**Test to write.**
```zig
test "copyWindow copies a fixed block" {
    var dst: [4]u32 = .{ 0, 0, 0, 0 };
    const src: []const u32 = &.{ 10, 20, 30, 40, 50, 60 };
    try copyWindow(&dst, src, 2);
    try std.testing.expectEqualSlices(u32, &.{ 30, 40, 50, 60 }, &dst);
}

test "copyWindow rejects out-of-range" {
    var dst: [4]u32 = undefined;
    const src: []const u32 = &.{ 1, 2, 3 };
    try std.testing.expectError(error.OutOfRange, copyWindow(&dst, src, 1));
}
```

**Stale pattern to avoid.**
```zig
// WRONG — slice of runtime length is []u32, cannot @memcpy into *[4]u32
const win = src[start .. start + 4];
@memcpy(dst, win);  // compile error: length mismatch
```

**Exercise.** Generalize to `comptime N: usize`. What parts of the function
can Zig eliminate at compile time?

## Ownership and slices — the cheat sheet

| Situation | Who frees? |
|---|---|
| `fn foo(s: []const u8)` | Caller. The slice is borrowed. |
| `fn foo(a: Allocator, s: []const u8) !void` — copies `s` | The copy is owned by `a`; original caller still owns `s`. |
| `fn bar(a: Allocator) ![]u8` | Caller; `bar` transfers ownership. |
| `fn baz(list: *std.ArrayList(u8), v: u8) !void` | `list` owns all bytes; caller's allocator is unaffected. |

Document ownership in doc comments. Never make the reader guess.

<!-- ~1.0k tokens -->
