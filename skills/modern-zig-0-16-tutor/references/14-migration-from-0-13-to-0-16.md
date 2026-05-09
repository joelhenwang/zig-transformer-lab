# 14 — Migration from Zig 0.13 → 0.16

This is the file to load when the user pastes 0.13-era code and asks for help.
Every section: **WRONG / CORRECT**, a one-line mentor note, and a compiler
error fragment you can match against.

## Migration discipline (mentor rules)

1. **Do not rewrite.** Produce a focused diff.
2. If there are many changes, group them. Start with the change that unblocks
   compilation, not the stylistic one.
3. Verify each change against the reference file it touches (see the links
   throughout).
4. When a change alters ownership or semantics, call it out in prose.

## Table of canonical migrations

| # | Stale (≤0.13) | Modern 0.16 | Compiler-error hint |
|---|---|---|---|
| 1 | `pub fn main() !void` + hand-rolled `GPA` | `pub fn main(init: std.process.Init) !void` | "no member 'GeneralPurposeAllocator'" |
| 2 | `std.io.getStdOut().writer()` | `std.Io.File.stdout().writeStreamingAll(io, "...")` | "no member 'getStdOut'" |
| 3 | `std.os.environ` / `getEnvVarOwned` | `init.environ_map.get("...")` | "no member 'getEnvVarOwned'" |
| 4 | `std.process.argsAlloc(a)` | `init.minimal.args.toSlice(arena)` | "no member 'argsAlloc'" |
| 5 | `@cImport({...})` + `linkSystemLibrary` on exe | `b.addTranslateC(...).createModule()` + `addImport("c", m)` | warning `@cImport` deprecated |
| 6 | `@Type(.{...})` | `@Int`, `@Struct`, `@Enum`, `@Union`, `@Tuple`, `@Pointer`, `@Fn`, `@EnumLiteral` | "use of undeclared identifier '@Type'" |
| 7 | `ArrayList(T).init(a)` + `append(v)` + `deinit()` | `.empty` + `append(a, v)` + `deinit(a)` | "expected 2 arguments, found 1" |
| 8 | `b.addExecutable(.{ .root_source_file, .target, .optimize })` | `.root_module = b.createModule({...})` | "no field 'root_source_file'" |
| 9 | `b.addStaticLibrary(...)` / `b.addSharedLibrary(...)` | `b.addLibrary(.{ .linkage = ... })` | "no member 'addStaticLibrary'" |
| 10 | `exe.addModule(name, m)` | `exe.root_module.addImport(name, m)` | "no member 'addModule'" |
| 11 | `std.heap.GeneralPurposeAllocator(.{}){}` | `std.heap.DebugAllocator(.{}) = .init` | "no member 'GeneralPurposeAllocator'" |
| 12 | `std.heap.ThreadSafeAllocator` | `ArenaAllocator` is already threadsafe | "no member 'ThreadSafeAllocator'" |
| 13 | `std.Thread.Pool` + `spawnWg` | `std.Io.Group` + `group.async(io, ...)` / `await(io)` | "no member 'Pool' in 'std.Thread'" |
| 14 | `async` / `await` / `suspend` / `resume` keywords | `io.async(fn, args)` / `future.await(io)` | "unexpected token 'async'" |
| 15 | `usingnamespace mixin;` | explicit re-exports | "unexpected token 'usingnamespace'" |
| 16 | `@intToPtr` / `@ptrToInt` / `@enumToInt` / `@intToEnum` / `@floatToInt` / `@intToFloat` / `@boolToInt` / `@errSetCast` | `@ptrFromInt` / `@intFromPtr` / `@intFromEnum` / `@enumFromInt` / `@intFromFloat` / `@floatFromInt` / `@intFromBool` / `@errorCast` | "no member '@intToPtr'" |
| 17 | `@typeInfo(T).Struct` / `.Pointer` / `.ErrorUnion` | `.@"struct"` / `.pointer` / `.error_union` | "no member 'Struct' in 'std.builtin.Type'" |
| 18 | `callconv(.C)` | `callconv(.c)` | "invalid enum value '.C'" |
| 19 | `FileSource` / `std.zig.CrossTarget` | `std.Build.LazyPath` / `std.Target.Query` | "no member 'CrossTarget'" |
| 20 | `std.build.Builder` | `std.Build` | "no member 'Builder'" |
| 21 | `std.mem.indexOf(...)` / `indexOfScalar` / `lastIndexOf` | `std.mem.find(...)` / `findScalar` / `findLast`; often `cut*` | "no member 'indexOf'" |
| 22 | `{D}` format specifier for nanoseconds | `std.Io.Duration{ .nanoseconds = ns }` + `{f}` | fmt error "invalid specifier 'D'" |

Each row above is expanded in the linked reference. The table below is a
curated set of the **worked before/after blocks** with mentor commentary.

---

## Worked migration 1 — the hello-world

```zig
// WRONG (0.13)
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    const args = try std.process.argsAlloc(a);
    defer std.process.argsFree(a, args);

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();
    try stdout.print("argc = {d}\n", .{args.len});
    try bw.flush();
}
```

```zig
// CORRECT (0.16)
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var buf: [128]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("argc = {d}\n", .{args.len});
    try fw.flush();
}
```

**Mentor note.** Three changes collapsed into one: entry point signature,
arg access, and I/O. See refs 01, 02, 07.

## Worked migration 2 — `ArrayList` / allocator

```zig
// WRONG
var list = std.ArrayList(i32).init(a);
defer list.deinit();
try list.append(1);
try list.append(2);
try list.appendSlice(&.{ 3, 4 });

// CORRECT
var list: std.ArrayList(i32) = .empty;
defer list.deinit(a);
try list.append(a, 1);
try list.append(a, 2);
try list.appendSlice(a, &.{ 3, 4 });
```

See ref 06.

## Worked migration 3 — build.zig

```zig
// WRONG (0.13-era)
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "hello",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = mode,
    });
    exe.linkSystemLibrary("z");
    exe.linkLibC();
    exe.install();
}
```

```zig
// CORRECT (0.16)
pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "hello",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.linkSystemLibrary("z", .{});
    b.installArtifact(exe);
}
```

See ref 08.

## Worked migration 4 — `@cImport` → translate-c

```zig
// WRONG
const c = @cImport({
    @cInclude("stdio.h");
});
pub fn main() void {
    _ = c.puts("hello");
}
// build.zig: exe.linkLibC();
```

Create `c/api.h`:

```c
#include <stdio.h>
```

```zig
// build.zig — CORRECT
const translate_c = b.addTranslateC(.{
    .root_source_file = b.path("c/api.h"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});

const exe = b.addExecutable(.{
    .name = "hi",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = translate_c.createModule() },
        },
    }),
});
```

```zig
// src/main.zig — CORRECT
const c = @import("c");

pub fn main(init: std.process.Init) !void {
    _ = init;
    _ = c.puts("hello");
}
```

See ref 09.

## Worked migration 5 — `@Type` → new builtins

```zig
// WRONG
const Small = @Type(.{ .int = .{ .signedness = .unsigned, .bits = 7 } });
const info  = @typeInfo(Small).Int;  // also stale casing

// CORRECT
const Small = @Int(.unsigned, 7);
const info  = @typeInfo(Small).int;  // snake_case
```

See ref 11.

## Worked migration 6 — concurrency

```zig
// WRONG
var pool: std.Thread.Pool = undefined;
try pool.init(.{ .allocator = a });
defer pool.deinit();

var wg: std.Thread.WaitGroup = .{};
pool.spawnWg(&wg, work, .{ &pool, &wg, 1 });
pool.spawnWg(&wg, work, .{ &pool, &wg, 2 });
wg.wait();

// CORRECT
var g: std.Io.Group = .init;
errdefer g.cancel(io);
g.async(io, work, .{ io, 1 });
g.async(io, work, .{ io, 2 });
try g.await(io);
```

See ref 01.

## Worked migration 7 — builtins rename

```zig
// WRONG
const i: usize = @ptrToInt(ptr);
const p: *u32  = @intToPtr(*u32, i);
const e: E     = @intToEnum(E, 1);
const x: i32   = @enumToInt(e);
const f: f32   = @intToFloat(f32, 7);
const n: i32   = @floatToInt(i32, 3.14);
const b: u1    = @boolToInt(true);

// CORRECT
const i: usize = @intFromPtr(ptr);
const p: *u32  = @ptrFromInt(i);
const e: E     = @enumFromInt(1);
const x: i32   = @intFromEnum(e);
const f: f32   = @floatFromInt(7);
const n: i32   = @intFromFloat(3.14);   // or simpler: @trunc
const b: u1    = @intFromBool(true);
```

See ref 03.

## If in doubt

Offer the user a diff. Don't rewrite the file.

<!-- ~3.4k tokens · verified against Zig 0.16.0 release notes -->
