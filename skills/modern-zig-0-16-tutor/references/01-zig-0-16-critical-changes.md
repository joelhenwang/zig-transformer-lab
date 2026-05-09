# 01 — Zig 0.16.0 critical changes (anchor)

This is the **first file to load** when the user's question or error involves
version-sensitive code. It covers the deltas that a pre-0.16 LLM will get
wrong. Every section: **WRONG / CORRECT** diff + mentor note + deeper link.

## I/O as an Interface (`std.Io`)

`std.Io` is a vtable + userdata interface, modelled after `Allocator`. The
application picks an implementation once (in `main`), and every library takes
`io: std.Io` as a parameter. Writers/readers are non-generic; buffers are
explicit; flushing is explicit.

```zig
// WRONG (0.13-era)
const stdout = std.io.getStdOut().writer();
var bw = std.io.bufferedWriter(stdout);
const w = bw.writer();
try w.print("x = {d}\n", .{x});
try bw.flush();

// CORRECT (0.16)
var buf: [4096]u8 = undefined;
var fw = std.Io.File.stdout().writer(io, &buf);
try fw.interface.print("x = {d}\n", .{x});
try fw.flush();
```

**Mentor note.** Buffered writers do not implicitly flush on close; forgetting
`try fw.flush()` is the #1 source of "nothing was printed" bugs. Also deleted:
`GenericReader`, `AnyReader`, `FixedBufferStream`. See `references/07-io-0-16.md`.

## Juicy Main (`pub fn main(init: std.process.Init) !void`)

`Init` bundles: `io`, `gpa`, `arena`, `environ_map`, `minimal.args`,
`minimal.environ`, `preopens`.

```zig
// WRONG
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();
    const args = try std.process.argsAlloc(a);
    defer std.process.argsFree(a, args);
    try std.io.getStdOut().writer().print("argc={d}\n", .{args.len});
}

// CORRECT
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var buf: [64]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("argc={d}\n", .{args.len});
    try fw.flush();
}
```

**Mentor note.** Only `main` constructs the `Init`. Library functions accept
`allocator: std.mem.Allocator`, `io: std.Io`, and/or a `*const std.process.Environ.Map`
as explicit parameters — no globals. See `references/07-io-0-16.md`.

## Environment variables + process arguments are non-global

```zig
// WRONG
const home = try std.process.getEnvVarOwned(a, "HOME");
const args = try std.process.argsAlloc(a);

// CORRECT
const home = init.environ_map.get("HOME"); // ?[]const u8 borrowed
const args = try init.minimal.args.toSlice(init.arena.allocator());
```

**Mentor note.** The old globals (`std.os.environ`, `std.process.argsAlloc`)
could not be populated from non-libc Zig and were racy under threaded
`setenv`. VERIFY WITH ZIG 0.16.0 LOCALLY: exact members of `Environ.Map`
(`get` vs `getPtr`).

## `@cImport` moved to the build system

```zig
// WRONG (0.13-era) — in source
pub const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("GLFW/glfw3.h");
});
// and in build.zig:
exe.linkSystemLibrary("glfw");
exe.linkLibC();

// CORRECT (0.16) — create a c.h header file
// c/c_api.h:
// #include <stdio.h>
// #include <GLFW/glfw3.h>

// build.zig:
const translate_c = b.addTranslateC(.{
    .root_source_file = b.path("c/c_api.h"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});
translate_c.linkSystemLibrary("glfw", .{});

const exe = b.addExecutable(.{
    .name = "app",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = translate_c.createModule() },
        },
    }),
});

// src/main.zig:
// const c = @import("c");
```

**Mentor note.** `@cImport` still parses but is deprecated. The new form
lets IDE tooling (clangd etc.) see your `.h` file, and `linkSystemLibrary`
moves onto the `TranslateC` step. See `references/09-c-interop-0-16.md`.

## `@Type` replaced with 8 individual builtins

```zig
// WRONG
const U10 = @Type(.{ .int = .{ .signedness = .unsigned, .bits = 10 } });
const S   = @Type(.{ .@"struct" = .{ ... } });

// CORRECT
const U10 = @Int(.unsigned, 10);
const T   = @Tuple(&.{ u32, [2]f64 });
// @Struct, @Union, @Enum, @Fn, @Pointer, @EnumLiteral also exist
```

**Mentor note.** No `@Float` (use `std.meta.Float`), no `@Array` (use
`[len]Elem` / `[len:s]Elem`), no `@Opaque` (write `opaque {}`), no
`@ErrorSet` (declare `error{...}`). See `references/11-comptime-metaprogramming.md`.

## Containers migrated to the unmanaged idiom

```zig
// WRONG
var list = std.ArrayList(u32).init(a);
defer list.deinit();
try list.append(1);

var q = std.PriorityQueue(u32, void, lessThan).init(a, {});
try q.add(3);
_ = q.remove();

// CORRECT
var list: std.ArrayList(u32) = .empty;
defer list.deinit(a);
try list.append(a, 1);

var q: std.PriorityQueue(u32, void, lessThan) = .empty;
defer q.deinit(a);
try q.push(a, 3);
_ = q.pop();
```

**Mentor note.** `PriorityQueue` renames: `add→push`, `addSlice→pushSlice`,
`remove→pop`, `removeOrNull→pop`, `removeMin→popMin`, `removeMax→popMax`,
`removeIndex→popIndex`. VERIFY WITH ZIG 0.16.0 LOCALLY whether the public
name is still `ArrayListUnmanaged` or just `ArrayList`. See `references/06-containers-0-16.md`.

## Build system: `root_module` discipline

```zig
// WRONG
const exe = b.addExecutable(.{
    .name = "app",
    .root_source_file = b.path("src/main.zig"),
    .target = target,
    .optimize = optimize,
});
exe.addModule("util", util_mod);

// CORRECT
const exe = b.addExecutable(.{
    .name = "app",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "util", .module = util_mod },
        },
    }),
});
```

Linking still happens on the artifact (`exe.linkLibrary(lib)`), but imports
and `addCSourceFile` / `linkSystemLibrary` live on **`exe.root_module`** now.
See `references/08-build-system-0-16.md`.

## Thread pool removed; concurrency via `Io.Group`

```zig
// WRONG
var pool: std.Thread.Pool = undefined;
try pool.init(.{ .allocator = a });
defer pool.deinit();
var wg: std.Thread.WaitGroup = .{};
pool.spawnWg(&wg, work, .{ item });
wg.wait();

// CORRECT
var g: std.Io.Group = .init;
errdefer g.cancel(io);
g.async(io, work, .{ io, item });
try g.await(io);
```

`std.Thread.{Mutex,Condition,Futex,Semaphore,RwLock}` moved to
`std.Io.{Mutex,Condition,Futex,Semaphore,RwLock}`. The async/await keywords
are **gone**; `io.async(fn, args)` is a method call, `future.await(io)` is too.

## `DebugAllocator` replaces `GeneralPurposeAllocator`; `ArenaAllocator` is lock-free

```zig
// WRONG
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
var safe = std.heap.ThreadSafeAllocator{ .child_allocator = arena.allocator() };

// CORRECT
var gpa_state: std.heap.DebugAllocator(.{}) = .init;
defer _ = gpa_state.deinit();
const gpa = gpa_state.allocator();

// ArenaAllocator is already threadsafe; no wrapper needed
var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
defer arena.deinit();
```

See `references/05-memory-allocators.md`.

## `mem` renames + cuts

```zig
// WRONG
if (std.mem.indexOf(u8, s, "=")) |i| { ... }

// CORRECT (preferred)
if (std.mem.cut(u8, s, "=")) |pair| {
    const k = pair.prefix;
    const v = pair.suffix;
}
// or at least
if (std.mem.find(u8, s, "=")) |i| { ... }
```

Concepts: `find` = return index; `pos` = with start index; `last` = from end;
`scalar` = single-element needle. VERIFY WITH ZIG 0.16.0 LOCALLY exact rename
table.

## Format + duration + logging

- `fmt.Formatter` → `fmt.Alt`
- `fmt.FormatOptions` → `fmt.Options`
- `fmt.bufPrintZ` → `fmt.bufPrintSentinel`
- Top-level `std.fmt.format` → `std.Io.Writer.print`
- `{D}` duration specifier gone → `std.Io.Duration{ .nanoseconds = ns }` + `{f}`

## Removed (confirmed)

- `std.io.getStdOut()` / `getStdIn()` / `getStdErr()` → `std.Io.File.stdout()` etc.
- `GenericReader` / `AnyReader` / `FixedBufferStream` — use `std.Io.Reader`
- `std.heap.GeneralPurposeAllocator` → `DebugAllocator`
- `std.heap.ThreadSafeAllocator`
- `std.Thread.Pool`, `std.once`, `std.Thread.Mutex.Recursive`
- `std.SegmentedList`, `std.meta.declList`
- `fs.getAppDataDir` (use `known-folders`)
- `usingnamespace` (since 0.15)
- `async`/`await`/`suspend`/`resume` keywords (since ~0.10)

## Smaller language changes (flag if you see them)

- `callconv(.C)` → `callconv(.c)` (lowercase)
- `@intToPtr`/`@ptrToInt`/`@enumToInt`/`@intToEnum`/`@floatToInt`/`@intToFloat`/
  `@boolToInt`/`@errSetCast` → `@ptrFromInt`/`@intFromPtr`/`@intFromEnum`/
  `@enumFromInt`/`@intFromFloat`/`@floatFromInt`/`@intFromBool`/`@errorCast`
- `@typeInfo(T).Struct.fields` → `.@"struct".fields`; `.ErrorUnion` → `.error_union`; `.Pointer` → `.pointer`
- `return &local_var;` now a compile error
- Small-int → float implicit coercion allowed (if bits ≤ significand)
- `@floor`/`@ceil`/`@round`/`@trunc` forward integer result types;
  `@intFromFloat` now effectively deprecated
- Pointers forbidden inside `packed struct` / `packed union`
- Explicit backing int required for packed types in extern contexts

<!-- ~4.0k tokens · verified against Zig 0.16.0 release notes -->
