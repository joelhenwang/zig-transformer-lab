# Recipes: error handling

## Recipe 6 — `defer` and `errdefer`

**Problem.** Build a struct that allocates two buffers; guarantee both are
freed if any step fails, and both are freed on normal destruction.

**Mentor explanation.** `defer` runs at scope end in reverse order.
`errdefer` runs only on error exit. Place each `errdefer` **immediately
after** the matching allocation. That way if a later `try` fires, the right
buffers unwind.

**Minimal snippet.**
```zig
const std = @import("std");

const Two = struct {
    a_buf: []u8,
    b_buf: []u32,

    pub fn init(a: std.mem.Allocator) !Two {
        const ab = try a.alloc(u8, 64);
        errdefer a.free(ab);

        const bb = try a.alloc(u32, 16);
        errdefer a.free(bb);

        if (ab.len + bb.len < 1) return error.Impossible; // force a fail path
        return .{ .a_buf = ab, .b_buf = bb };
    }

    pub fn deinit(self: *Two, a: std.mem.Allocator) void {
        a.free(self.b_buf);
        a.free(self.a_buf);
        self.* = undefined;
    }
};
```

**Test to write.**
```zig
test "Two.init leaks nothing on success" {
    const a = std.testing.allocator;
    var t = try Two.init(a);
    defer t.deinit(a);
    try std.testing.expectEqual(@as(usize, 64), t.a_buf.len);
    try std.testing.expectEqual(@as(usize, 16), t.b_buf.len);
}
```

**Stale pattern to avoid.**
```zig
// WRONG — defer on success path runs too early, and no errdefer for ab
const ab = try a.alloc(u8, 64);
defer a.free(ab);                 // frees immediately, even on success return
const bb = try a.alloc(u32, 16);  // ab now dangles if this succeeds
```

**Exercise.** Force `init` to fail after the second allocation. Use
`std.testing.FailingAllocator` with `.fail_index = 1`. Verify both buffers
are freed by the leak detector.

---

## Recipe 7 — Error union handling

**Problem.** Call a fallible function and handle each error case explicitly.

**Mentor explanation.** Three patterns: `try` (propagate), `catch default`
(provide a value), `if (...) |v| ... else |err| switch (err) { ... }`
(exhaustive handling). Pick the narrowest.

**Minimal snippet.**
```zig
const ParseErr = error{ InvalidChar, Overflow };

fn parseByte(s: []const u8) ParseErr!u8 {
    var n: u16 = 0;
    for (s) |c| {
        if (c < '0' or c > '9') return error.InvalidChar;
        n = n * 10 + (c - '0');
        if (n > 255) return error.Overflow;
    }
    return @intCast(n);
}

pub fn main(init: std.process.Init) !void {
    _ = init;

    const a = parseByte("12") catch 0;
    _ = a;

    const b = try parseByte("255");
    _ = b;

    if (parseByte("abc")) |v| {
        _ = v;
    } else |err| switch (err) {
        error.InvalidChar => std.debug.print("bad\n", .{}),
        error.Overflow => std.debug.print("big\n", .{}),
    }
}
```

**Stale pattern to avoid.**
```zig
// WRONG — swallows errors silently
const v = parseByte("abc") catch 0;
```

**Exercise.** Change `parseByte` to return an **inferred** error set (`!u8`).
What's the downside for callers?

---

## Recipe 8 — Optional handling

**Problem.** Read an environment variable that may not exist.

**Mentor explanation.** `?T` is the "maybe T" type. Unwrap with `orelse`
(provide default), `if (maybe) |v| {}`, or `.?` (panic if null — last
resort).

**Minimal snippet.**
```zig
pub fn main(init: std.process.Init) !void {
    const home: ?[]const u8 = init.environ_map.get("HOME");
    const h = home orelse return error.HomeMissing;

    var buf: [128]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("HOME = {s}\n", .{h});
    try fw.flush();
}
```

**Stale pattern to avoid.**
```zig
// WRONG — silent panic on missing env var
const h = init.environ_map.get("HOME").?;
```

**Exercise.** Replace `return error.HomeMissing` with a default (`"/home/me"`)
using `orelse`. When would you choose each?

<!-- ~1.3k tokens -->
