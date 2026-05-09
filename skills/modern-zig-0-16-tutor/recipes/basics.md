# Recipes: basics

## Recipe 1 — Hello world with `std.process.Init` and `std.Io`

**Problem.** Write a Zig 0.16.0 program that prints a greeting to stdout.

**Mentor explanation.** `main` now takes an `init: std.process.Init`. `init.io`
is the I/O implementation for the target, and `std.Io.File.stdout()` returns
the stdout file handle. For a one-shot string, `writeStreamingAll` is simpler
than buffering.

**Minimal snippet.**
```zig
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    try std.Io.File.stdout().writeStreamingAll(init.io, "Hello, World!\n");
}
```

**Test to write.**
```zig
test "writeStreamingAll on fixed buffer" {
    var out: [32]u8 = undefined;
    var w = std.Io.Writer.fixed(&out);
    try w.writeAll("Hi\n");
    try std.testing.expectEqualStrings("Hi\n", out[0..3]);
}
```

**Stale pattern to avoid.**
```zig
// WRONG
const stdout = std.io.getStdOut().writer();
try stdout.print("Hello, World!\n", .{});
```

**Exercise.** Rewrite the program to use a buffered writer with a 1024-byte
user buffer, print three separate lines, and verify the program prints
nothing if you forget `try fw.flush()`.

---

## Recipe 2 — When `std.debug.print` is acceptable

**Problem.** You want a quick debug print without plumbing `io` through every
function.

**Mentor explanation.** `std.debug.print` writes to **stderr** and does not
require `io`. Use it for: ad-hoc debugging, fatal messages, programs that
don't take `init: std.process.Init`, and tests. Never use it in library code
that takes `io`.

**Minimal snippet.**
```zig
const std = @import("std");

pub fn main() void {
    const n = 42;
    std.debug.print("debug: n = {d}\n", .{n});
}
```

**Test to write.** You don't test `std.debug.print` output directly; trust
the stdlib.

**Stale pattern to avoid.**
```zig
// WRONG — over-engineered debug path
const stderr = std.io.getStdErr().writer();
try stderr.print("debug: n = {d}\n", .{42});
```

**Exercise.** Add a `std.log` scoped logger to the same file, and compare
the ergonomics of `log.debug(...)` vs `std.debug.print(...)`. Which one
would you keep in committed code?

<!-- ~0.9k tokens -->
