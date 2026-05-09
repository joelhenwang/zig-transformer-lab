# 02 — Language basics (0.16.0)

Rules a learner must internalize before anything else. Keep this file short;
deeper topics live in refs 03–13.

## Source files are structs

Every `.zig` file **is a struct**. Top-level decls are that struct's members.
Files that contain fields are conventionally **TitleCased**; files that
contain only `pub fn`/`pub const` are conventionally **snake_case**.

```zig
// src/util.zig
pub const version = "0.1.0";
pub fn add(a: i32, b: i32) i32 { return a + b; }
```

## `const` vs `var`

- Prefer `const` when mutation is not needed.
- Shadowing an outer identifier is a compile error.
- `var` at container scope = global data; its initializer runs at comptime.
- `const` with a comptime-known value → read-only data section.

## `undefined`

`undefined` coerces to any type. In Debug / ReleaseSafe it is written as the
byte pattern `0xAA` so wild reads are easy to spot. Reading an `undefined`
value is unchecked undefined behavior in ReleaseFast / ReleaseSmall.

## Destructuring

```zig
var a: u32 = 0;
var b: u32 = 0;
a, b = .{ 3, 4 };     // inside a block
const x, const y = .{ 1, 2 };
```

## String literals

- `"hello"` has type `*const [5:0]u8` — a pointer to a null-terminated array.
- Coerces to `[]const u8`, `[*:0]const u8`, `[*]const u8`.
- When passing to C variadics: `[*:0]const u8` is the portable choice.

## Entry points

| Shape | When to use |
|---|---|
| `pub fn main(init: std.process.Init) !void` | Default. You need argv, env, io, gpa, arena. |
| `pub fn main(init: std.process.Init.Minimal) !void` | You want raw argv/env but not the default allocators. |
| `pub fn main() !void` | Simplest; no argv/env/io provided. |
| `pub fn main() u8` / `void` / `E!u8` | All supported as return types. |
| `export fn main(argc: c_int, argv: [*]const [*:0]const u8) c_int` | When linking libc and you want C's `main`. |
| `pub const _start = {};` | Disable `std.start`. Freestanding. |

## Root-module knobs

Declare these at the top of the file chosen as `root_source_file`:

```zig
pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = myLogFn,
};

pub const panic = std.debug.FullPanic(myPanic);

fn myPanic(msg: []const u8, first_trace_addr: ?usize) noreturn {
    // ... print + exit
    _ = msg; _ = first_trace_addr;
    @trap();
}
```

## Naming conventions (`zig fmt` + style guide)

- Files with only decls: `snake_case.zig`.
- Files exporting a single public struct type: `TitleCase.zig`.
- Types (incl. functions returning `type`): `TitleCase`.
- Functions, variables, fields: `camelCase` (functions) / `snake_case` (fields).
- Error sets: `PascalCase` members (`error.InvalidChar`).

## Common builtin renames a 0.13-era LLM will get wrong

| Stale | Modern 0.16 |
|---|---|
| `@intToPtr(T, addr)` | `@ptrFromInt(addr)` with result-type context |
| `@ptrToInt(p)` | `@intFromPtr(p)` |
| `@enumToInt(e)` | `@intFromEnum(e)` |
| `@intToEnum(E, i)` | `@enumFromInt(i)` |
| `@floatToInt(T, f)` | `@intFromFloat(f)` (often unnecessary; see `@trunc`) |
| `@intToFloat(T, i)` | `@floatFromInt(i)` |
| `@boolToInt(b)` | `@intFromBool(b)` |
| `@errSetCast(T, e)` | `@errorCast(e)` |
| `@typeInfo(T).Struct` | `@typeInfo(T).@"struct"` |
| `@typeInfo(T).Pointer` | `@typeInfo(T).pointer` |
| `@typeInfo(T).ErrorUnion` | `@typeInfo(T).error_union` |

## Forbidden (compile errors in 0.16)

- `return &local_var;` — returning address of an expired local.
- `usingnamespace` — removed from the grammar; re-export explicitly.
- `async` / `await` / `suspend` / `resume` keywords — removed (method calls
  `io.async(...)` / `future.await(io)` are not keywords).
- `@Type(.{ ... })` — replaced by 8 builtins, see ref 11.

<!-- ~2.0k tokens · verified against Zig 0.16.0 release notes -->
