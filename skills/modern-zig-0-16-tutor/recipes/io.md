# Recipes: I/O

## Stdout via a buffered writer

**Problem.** Print several lines to stdout with acceptable throughput,
without forgetting `flush()`.

**Mentor explanation.** A buffered `File.Writer` holds a user-owned buffer.
You call `fw.interface.print(...)` or `.writeAll(...)` to queue bytes; the
bytes reach the file only after `fw.flush()` or when the buffer fills and
is spilled implicitly by a writer method. **Forgetting `flush()` before
scope exit is the most common "nothing prints" bug** in 0.16.

**Minimal snippet.**
```zig
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var buf: [4096]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);

    for (0..5) |i| {
        try fw.interface.print("line {d}\n", .{i});
    }

    try fw.flush();   // REQUIRED
}
```

**Test to write.** Use a `std.Io.Writer.fixed(&sink)` in the test, print
into it, and assert the bytes.

```zig
test "fixed writer captures formatted output" {
    var sink: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&sink);
    try w.print("x = {d}\n", .{42});
    try std.testing.expectEqualStrings("x = 42\n", w.buffered());
    // `buffered()` returns the written prefix of the user buffer (illustrative,
    // verify method name against Zig 0.16.0 stdlib).
}
```

> VERIFY WITH ZIG 0.16.0 LOCALLY: the exact method to read the written
> prefix of a fixed `std.Io.Writer`. If `buffered()` is not the spelling,
> compare `sink[0..w.writeIndex]` or the documented equivalent.

**Stale pattern to avoid.**
```zig
// WRONG
const stdout = std.io.getStdOut().writer();
var bw = std.io.bufferedWriter(stdout);
const w = bw.writer();
try w.print("line\n", .{});
try bw.flush();
```

**Exercise.** Reduce `buf` to 8 bytes. Before the final `flush()`, print
at least 40 characters. Does the program still print everything, or do you
see partial output? Why? (Hint: the writer's internal flush threshold when
the user buffer fills.)

## File read — whole file into memory

```zig
pub fn readWholeFile(init: std.process.Init, a: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 20));
}
```

Error renamed: `FileTooBig` → `StreamTooLong` when the limit is exceeded.

## File write — fresh file

```zig
pub fn writeHello(init: std.process.Init, path: []const u8) !void {
    const f = try std.Io.Dir.cwd().createFile(init.io, path, .{});
    defer f.close(init.io);

    var buf: [64]u8 = undefined;
    var fw = f.writer(init.io, &buf);
    try fw.interface.writeAll("hello file\n");
    try fw.flush();
}
```

<!-- ~1.0k tokens -->
