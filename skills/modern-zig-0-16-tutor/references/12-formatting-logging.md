# 12 — Formatting and logging (0.16.0)

Formatting in 0.16 is interface-driven: format methods take a
`*std.Io.Writer`, not a generic writer. Several identifiers were renamed.
The `{D}` duration specifier is gone, replaced by `std.Io.Duration`.

## Printing via a writer

```zig
pub fn main(init: std.process.Init) !void {
    var buf: [4096]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);

    try fw.interface.print("int = {d}, str = {s}, hex = {x}\n",
        .{ 42, "hi", 0xdead });

    try fw.flush();
}
```

The format string mini-language is unchanged:

| Specifier | Meaning |
|---|---|
| `{d}` | Decimal (integer or float) |
| `{x}` / `{X}` | Hex, lower / upper |
| `{o}` | Octal |
| `{b}` | Binary |
| `{c}` | Char (one byte) |
| `{s}` | String |
| `{any}` | Debug-style, any type |
| `{e}` / `{E}` | Float scientific |
| `{f}` | Calls `.format(writer)` on the value |
| `{?}` / `{!}` | Optional / error-union wrappers |

Width / precision / fill / align all still work:
`{d:>8}`, `{d:0>4}`, `{e:.3}`, `{s:<10}`.

## Custom `format` methods (0.16 signature)

```zig
const MyPoint = struct {
    x: f32, y: f32,

    pub fn format(self: MyPoint, writer: *std.Io.Writer) !void {
        try writer.print("({d:.2}, {d:.2})", .{ self.x, self.y });
    }
};

// Usage:
// try fw.interface.print("{f}\n", .{ MyPoint{ .x = 1.0, .y = 2.0 } });
```

The `{f}` specifier invokes `format(writer)`. The pre-0.14 four-argument
signature `fn format(self, comptime fmt: []const u8, opts: FormatOptions,
writer: anytype) !void` is **stale**.

## Durations with `std.Io.Duration`

```zig
// WRONG
try writer.print("took {D}\n", .{elapsed_ns});

// CORRECT
try writer.print("took {f}\n", .{std.Io.Duration{ .nanoseconds = elapsed_ns }});
```

`{D}` has been removed. `Io.Duration` owns its own `format` method.

## Renamed identifiers (0.16)

| Stale | Modern 0.16 |
|---|---|
| `std.fmt.Formatter` | `std.fmt.Alt` |
| `std.fmt.FormatOptions` | `std.fmt.Options` |
| `std.fmt.bufPrintZ` | `std.fmt.bufPrintSentinel` |
| `std.fmt.format(writer, ...)` | `std.Io.Writer.print(...)` via the writer interface |

## `std.log`

```zig
const log = std.log.scoped(.myapp);

pub fn main(init: std.process.Init) !void {
    _ = init;
    log.info("starting", .{});
    log.warn("suspicious config: {s}", .{"foo"});
    log.err("failed: {s}", .{"bar"});
}
```

Logging is configurable per-root-module:

```zig
pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = myLogFn,
};

fn myLogFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime fmt: []const u8,
    args: anytype,
) void {
    // route to your own sink
    _ = level; _ = scope;
    std.debug.print(fmt ++ "\n", args);
}
```

## `allocPrint` / `bufPrint` / `comptimePrint`

```zig
const msg = try std.fmt.allocPrint(a, "x = {d}", .{42});
defer a.free(msg);

var buf: [64]u8 = undefined;
const slice = try std.fmt.bufPrint(&buf, "y = {d}", .{17});
_ = slice;

const compiled: []const u8 = std.fmt.comptimePrint("build = {s}", .{"0.1"});
```

For null-terminated output, use `std.fmt.bufPrintSentinel` (new name).

## Common review findings

- `std.io.getStdOut().writer().print(...)` — stale.
- Custom `format(self, comptime fmt, opts, writer) ...` — stale signature.
- `std.fmt.Formatter` / `FormatOptions` / `bufPrintZ` — stale names.
- `{D}` for durations — gone.
- Using `std.debug.print` instead of `io` in a library function — SUGGEST
  refactor.

## Common mentor diagnostic questions

- "What's the max length of this output? Can `bufPrint` handle it?"
- "Is this log statement on a hot path? If so, use `.debug` scoped and keep
  it out of release builds."
- "What does the user see if the `flush()` is missing?"

<!-- ~2.0k tokens · verified against Zig 0.16.0 release notes -->
