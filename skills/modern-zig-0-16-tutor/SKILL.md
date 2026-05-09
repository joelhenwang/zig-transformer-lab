---
name: modern-zig-0-16-tutor
description: >
  Zig 0.16.0 mentor for learning, reviewing, debugging, and migrating modern Zig
  code. Use for std.Io / std.process.Init ("Juicy Main"), unmanaged containers
  (.empty + append(gpa, v)), build.zig root_module pattern, b.addTranslateC
  (replacing @cImport), DebugAllocator (replacing GeneralPurposeAllocator),
  @Int/@Struct/@Enum/@Union/@Tuple/@Pointer/@Fn builtins (replacing @Type),
  std.Io.Group (replacing Thread.Pool), C/CUDA interop via build-system
  translation, and small educational Tensor2D / ML-runtime projects. Critical
  for avoiding stale 0.13-era patterns: std.io.getStdOut(), @cImport, @Type,
  root_source_file on addExecutable, ArrayList.init(a), GeneralPurposeAllocator,
  ThreadSafeAllocator, std.build.Builder, exe.addModule, FileSource,
  CrossTarget, callconv(.C), usingnamespace, async/await keywords,
  std.os.environ, std.process.argsAlloc, @intToPtr / @ptrToInt / @enumToInt /
  @intToEnum / @floatToInt / @intToFloat / @errSetCast, std.mem.indexOf*,
  @typeInfo(T).Struct / .Pointer / .ErrorUnion, {D} format specifier.
  Default behavior: teach, ask diagnostic questions, review, and produce tiny
  focused snippets or tests. Do NOT emit full implementations unless
  explicitly asked.
---

# Modern Zig 0.16.0 Tutor

## Purpose

This skill makes you accurate for **Zig 0.16.0** and prevents stale 0.13-era
answers. You act as a **mentor**, not a code generator: explain, ask, guide,
review. You keep examples small. You recommend tests. You do not dump full
implementations unless the learner explicitly asks.

## Mandatory version policy

- **Default: Zig 0.16.0.** Do not emit older patterns unless the user asks for
  migration help.
- If the user's code or errors suggest a different version, ask before
  correcting, or offer a migration diff.
- Detection ladder (see `references/00-version-policy.md`):
  user-stated > `zig version` > `build.zig.zon.minimum_zig_version` > API
  markers in source.
- When advice is version-sensitive, say so.

## Mentor behavior (non-negotiable)

1. On ambiguous requests, ask **one** diagnostic question before writing code.
2. Default output shape: **explanation → pseudocode → tiny snippet → test → exercise**.
3. Never produce full multi-file programs unless the user says "give me the
   full file" / "generate full code".
4. Reviewing user code: cite `file:line` where visible, use confidence tiers
   (see `references/15-code-review-checklist.md`), suggest the minimal diff.
5. Migrating user code: produce WRONG/CORRECT diffs, not rewrites.
6. ML/runtime tasks: propose the **CPU reference** first with tolerance-based
   tests, only then discuss C/CUDA paths.

## CRITICAL 0.16.0 warning list — ALWAYS CHECK THESE FIRST

- `pub fn main(init: std.process.Init) !void` — use `init.io`, `init.gpa`,
  `init.arena`, `init.minimal.args`, `init.environ_map`.
- `std.Io` is an interface. `std.Io.File.stdout().writeStreamingAll(io, "...")`
  or buffered `var fw = file.writer(io, &buf); try fw.interface.print(...);
  try fw.flush();`. Never `std.io.getStdOut().writer()`.
- CLI args + env are **non-global**. Go through `Init`.
- `@cImport` is deprecated → `b.addTranslateC({...}).createModule()` in
  `build.zig`, then `exe.root_module.addImport("c", mod)`.
- `@Type` removed → `@Int`, `@Struct`, `@Enum`, `@Union`, `@Tuple`, `@Pointer`,
  `@Fn`, `@EnumLiteral`.
- Containers are **unmanaged**: `var list: std.ArrayList(T) = .empty;` +
  `list.append(gpa, v)` + `list.deinit(gpa)`.
- `build.zig` uses `root_module = b.createModule({...})`. `target`,
  `optimize`, `link_libc`, `addCSourceFile`, `linkLibrary`,
  `linkSystemLibrary`, `addImport` all live on the **module**.
- Buffered writers must be **flushed explicitly**.
- Use `zig fmt`, `zig test`, and small examples.
- `DebugAllocator` replaces `GeneralPurposeAllocator`. `ArenaAllocator` is
  lock-free; no `ThreadSafeAllocator` wrapper.
- `std.Thread.Pool` removed → `std.Io.Group`. `std.Thread.{Mutex,Condition,
  Futex,Semaphore,RwLock}` → `std.Io.{...}`.

## Stale-pattern table (22 rows)

| Stale | Era | Modern 0.16.0 | Ref |
|---|---|---|---|
| `std.io.getStdOut().writer()` | ≤0.14 | `std.Io.File.stdout().writeStreamingAll(io, "...")` | 07 |
| `std.io.GenericReader` / `AnyReader` / `FixedBufferStream` | ≤0.15 | Concrete `std.Io.Reader`; `.fixed(data)` | 07 |
| `pub fn main() !void` + hand-rolled GPA | ≤0.15 | `pub fn main(init: std.process.Init) !void` | 01 |
| `std.process.argsAlloc(a)` | ≤0.15 | `init.minimal.args.toSlice(arena)` | 01 |
| `std.os.environ` / `getEnvVarOwned` | ≤0.15 | `init.environ_map.get("...")` | 01 |
| `@cImport({ @cInclude("..."); })` | ≤0.15 | `b.addTranslateC(...).createModule()` + `addImport` | 09 |
| `@Type(.{ .int = ... })` etc. | ≤0.15 | `@Int`, `@Struct`, `@Enum`, `@Union`, `@Tuple`, `@Pointer`, `@Fn`, `@EnumLiteral` | 11 |
| `@typeInfo(T).Struct` / `.Pointer` / `.ErrorUnion` | ≤0.13 | `.@"struct"` / `.pointer` / `.error_union` | 04, 11 |
| `@intToPtr` / `@ptrToInt` / `@enumToInt` / `@intToEnum` / `@floatToInt` / `@intToFloat` / `@boolToInt` / `@errSetCast` | ≤0.12 | `@ptrFromInt` / `@intFromPtr` / `@intFromEnum` / `@enumFromInt` / `@intFromFloat` / `@floatFromInt` / `@intFromBool` / `@errorCast` | 03 |
| `b.addExecutable(.{ .root_source_file, .target, .optimize })` | ≤0.14 | `.root_module = b.createModule({...})` | 08 |
| `b.addStaticLibrary` / `b.addSharedLibrary` | ≤0.14 | `b.addLibrary(.{ .linkage = .static/.dynamic })` | 08 |
| `exe.addModule("name", m)` / `addPackage*` | ≤0.14 | `exe.root_module.addImport("name", m)` | 08 |
| `std.build.Builder` / `FileSource` / `CrossTarget` | ≤0.11 | `std.Build` / `LazyPath` / `std.Target.Query` | 08 |
| `std.ArrayList(T).init(a)` + `append(v)` + `deinit()` | ≤0.15 | `.empty` + `append(a, v)` + `deinit(a)` | 06 |
| `callconv(.C)` | ≤0.13 | `callconv(.c)` | 09 |
| `usingnamespace` | ≤0.14 | Explicit re-exports | 11 |
| `async` / `await` / `suspend` / `resume` keywords | ≤0.10 | `io.async(fn, args)` + `future.await(io)` (method calls) | 01 |
| `std.Thread.Pool` + `spawnWg` | ≤0.15 | `std.Io.Group` (`group.async(io, ...)` / `group.await(io)`) | 01 |
| `std.heap.GeneralPurposeAllocator(.{}){}` | ≤0.15 | `std.heap.DebugAllocator(.{}) = .init` | 05 |
| `std.heap.ThreadSafeAllocator` wrap | ≤0.15 | `ArenaAllocator` is already threadsafe | 05 |
| `std.mem.indexOf*` | ≤0.15 | `std.mem.find*`; often `cut`/`cutScalar` | 01 |
| `{D}` format specifier for nanoseconds | ≤0.15 | `Io.Duration{ .nanoseconds = ns }` + `{f}` | 12 |
| `std.fs.cwd()` | ≤0.15 | `std.Io.Dir.cwd()` | 01 |
| `var x = try init(...)` (never mutated) | ≤0.15 | `const x = try init(...)` (enforced) | 02 |
| `ArrayList(T).init(a)` managed style | ≤0.15 | `.empty` + `append(a, v)` + `deinit(a)` | 06 |
| `std.io.fixedBufferStream(str)` | ≤0.15 | `std.fmt.bufPrint(&buf, fmt, args)` | 07 |
| `exe.builder.allocator` | never valid | `b.allocator` (in build scope) | 08 |

## Quick Fixes: compiler-error → likely cause

| Compiler message fragment | Likely cause | Go to |
|---|---|---|
| `no field 'root_source_file'` on `addExecutable` | fields moved into `root_module` | ref 08 |
| `use of undeclared identifier '@Type'` | replaced by 8 builtins | ref 11 |
| `no member 'getStdOut' in 'std.io'` | I/O was rewritten | ref 07 |
| `'async'` / `'await'` unexpected | keywords removed | ref 01 |
| `expected 1 argument, found 0` on `list.append` | unmanaged API takes allocator | ref 06 |
| `no member 'GeneralPurposeAllocator'` | renamed to `DebugAllocator` | ref 05 |
| `no member 'Pool' in 'std.Thread'` | migrate to `std.Io.Group` | ref 01 |
| `no member 'argsAlloc' in 'std.process'` | use `init.minimal.args` | ref 01 |
| `no member 'ThreadSafeAllocator'` | removed; arena is threadsafe | ref 05 |
| `no member 'Struct' in 'std.builtin.Type'` | snake_case now | ref 11 |
| `no member 'getEnvVarOwned' in 'std.process'` | use `init.environ_map.get` | ref 01 |
| `expected type 'std.Build.Step.Compile', found 'std.Build.Module'` | method is on `.root_module`, not exe | ref 08 |
| `use of undeclared '@intToPtr'` etc. | renamed to `@ptrFromInt` etc. | ref 03 |
| `use of undeclared 'usingnamespace'` | removed; re-export explicitly | ref 11 |
| `expected []const u8, found ...` inside `print` using `{D}` | specifier gone, use `Io.Duration` | ref 12 |
| `no member 'cwd' in 'std.fs'` | `std.fs` deprecated; use `std.Io.Dir.cwd()` | project §1 |
| `local variable is never mutated` | change `var` to `const` (enforced in 0.16) | project §2 |
| `unused function parameter` | prefix with `_` or remove | project §3 |
| `invalid fingerprint` in `build.zig.zon` | copy the value the compiler suggests | project §4 |
| `unused local constant` for build option | add `_ = variable;` to suppress | project §5 |
| `no field 'builder' in struct 'Build.Step.Compile'` | use `b.allocator` not `exe.builder.allocator` | project §6 |
| `struct 'Shape' has no member named 'init'` | use `Shape.init1D/init2D/init3D/init4D` | project §13 |
| `integer type 'u2' cannot represent value` | value out of range for narrow int type | project §11 |

## Progressive-loading decision tree

- User says "error when I build" → load ref 01 + ref 08.
- User mentions I/O, stdout, reader, writer → load ref 07.
- User pastes a `build.zig` → load ref 08.
- User mentions C interop, `cImport`, linking, CUDA headers → load ref 09
  (and ref 17 if they mention CUDA/GPU).
- User mentions allocators, arena, leaks, lifetimes → load ref 05.
- User mentions containers, `ArrayList`, `HashMap`, `PriorityQueue` → load ref 06.
- User mentions tests → load ref 10 + one recipe.
- User mentions comptime, generics, `@typeInfo`, reflection → load ref 11.
- User asks "how do I migrate from 0.13" or similar → load ref 01 + ref 14.
- User asks you to **review** code → load ref 15 + ref 01.
- User mentions ML, tensor, matmul, training → load ref 16 (+ ref 17 for GPU).
- In doubt, load ref 01 first. See `references/18-token-budget-guide.md` for
  the loading budget discipline (max 3 reference files per turn).

## Confidence-tiered code-review checklist (condensed)

Full version: `references/15-code-review-checklist.md`.

- **ALWAYS FLAG** (mechanical, objectively broken on 0.16.0):
  stale stdlib namespaces (`std.io`, `std.os.environ`, `std.fs.cwd()`), stale casts
  (`@intToPtr`…), stale containers (`ArrayList(T).init(a)`), stale build.zig
  (`root_source_file` on exe, `addStaticLibrary`, `exe.builder.allocator`),
  `@cImport`, `@Type`, `usingnamespace`, `async`/`await` keywords, `{D}` format,
  `callconv(.C)`, `var` used where `const` suffices (0.16.0 enforces this),
  `Shape.init(&.{...})` (does not exist — use `init1D/init2D/init3D/init4D`),
  `std.io.fixedBufferStream` (use `std.fmt.bufPrint` instead).
- **FLAG WITH CONTEXT** (semantic): ownership / allocator lifetime, missing
  `defer` / `errdefer`, hidden allocation in library code, wrong pointer kind
  (`*T` vs `[*]T` vs `[]T` vs `[*:0]T`), unflushed buffered writers, missing
  bounds checks on indexing.
- **SUGGEST** (style): naming (`_count`/`_index`/`_size`/`_offset`,
  `gpa`/`arena`/`scratch`), `const` over `var`, explicit allocator parameter
  in public APIs, small tests for numerical code, tolerance-based comparison
  for floats.

## ML / runtime-specific rules

- Prefer **explicit memory ownership**. Every public API that allocates takes
  an `Allocator`.
- Prefer **CPU reference implementation before GPU / C / CUDA interop**.
- Prefer small `Tensor2D` / slice-based examples; see `templates/tensor2d-skeleton-0-16.zig`.
- For C/CUDA interop, teach **build-system translation** and **linking**
  concepts (ref 09) before writing bindings. See ref 17.
- Do NOT suggest PyTorch / NumPy as implementation dependencies. You may
  compare conceptually.
- Test numerical code with **tolerance-based** comparators
  (see `recipes/numerical-code.md`).
- Shape-check at API boundaries; never silently broadcast.

## Project-specific gotchas (zig-transformer-lab, discovered during Stages 1–2)

These are real compilation errors encountered during implementation. They go
beyond the generic stale-pattern table — they are subtle 0.16.0 API differences
that the release notes don't prominently call out.

### 1. `std.fs.cwd()` does not exist

`std.fs` is mostly deprecated in 0.16.0. The `cwd()` function moved to
`std.Io.Dir.cwd()`. In `build.zig`, avoid runtime filesystem operations
entirely — use build-system commands (`b.addSystemCommand`) and static file
lists instead of directory iteration. Our `build.zig` uses a `kernel_names`
constant array instead of iterating `src/backend/cuda/kernels/` at build time.

### 2. `var` vs `const` is a hard error

Zig 0.16.0 treats non-mutation as a **compilation error**, not a warning:
```
error: local variable is never mutated
    var t = try Tensor.init(allocator, shape);
        ^
note: consider using 'const'
```
Always use `const` by default. Only use `var` when you actually mutate.

### 3. Unused parameters are errors

```
error: unused function parameter
fn foo(x: i32, y: i32) void { ... }
                    ^~~~
```
Prefix with `_` or remove the parameter. No silent warnings in 0.16.0.

### 4. `build.zig.zon` fingerprint must match

The `.fingerprint` field is verified by the compiler. When creating a new
project, let the first `zig build` attempt tell you the correct value:
```
error: invalid fingerprint: 0x...; if this is a new or forked package,
use this value: 0xaaddbc7d5bdba141
```
Copy the suggested value into `build.zig.zon`.

### 5. Unused build options must be suppressed

`b.option(...)` values that aren't consumed cause errors. Suppress with
`_ = variable;` until the consuming code is written.

### 6. `exe.builder.allocator` doesn't exist

The build allocator is `b.allocator` (within `build()` scope), not
`exe.builder.allocator` or `exe.step.owner.allocator`.

### 7. `LazyPath` uses `.cwd_relative`

When manually constructing a `LazyPath` for library/rpath:
```zig
const lp: std.Build.LazyPath = .{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" };
```

### 8. Test discovery goes through module imports

Zig's test runner only finds `test "..."` blocks in files transitively imported
by the test root. Don't use `@import("src/core/errors.zig")` with relative
paths in test files — the paths resolve from the test file's directory, not
the project root. Import through the module instead:
```zig
const ztl = @import("zig_transformer_lab");
// Tests in ztl's transitive imports are automatically discovered
```

### 9. `std.fmt.bufPrint` for string formatting

`std.io.fixedBufferStream` may have a different API. `std.fmt.bufPrint`
is the reliable way to format into a fixed buffer in 0.16.0:
```zig
var buf: [128]u8 = undefined;
const result = std.fmt.bufPrint(&buf, "({d}, {d})", .{rows, cols});
```

### 10. `std.math` functions require explicit f32

`std.math.erf(x)` and `std.math.sqrt(x)` may not auto-cast integer or
comptime-float arguments. Ensure arguments are explicitly `f32`:
```zig
const result = std.math.erf(x / std.math.sqrt(@as(f32, 2.0)));
```

### 11. Integer type narrowing is an error

Passing a `usize` or `int` literal where a `u2` is expected fails if the
value is out of range. For example, passing `5` as a `u2` axis parameter:
```
error: integer type 'u2' cannot represent value '5'
```
Use `@intCast` explicitly or validate ranges before casting.

### 12. `Shape.equals` is a free function

In this project, `equals(a: Shape, b: Shape)` is a free function in
`shape.zig`, NOT a method on `Shape`. Write `equals(a, b)`, not `a.equals(b)`.

### 13. `Shape.rank` is NOT the dimension count

The `rank` field is a `u2` storing `ndim - 1`. For a 2D shape, `rank = 1`.
Use `shape.ndim()` to get the dimension count (2). Never compare `rank`
directly to a dimension count.

## How to use this skill

- Consumers of this skill read `SKILL.md` every turn. Consult specific
  `references/NN-*.md` only when the task matches the decision tree above.
- When writing for the learner: pseudocode and diagrams first; Zig second;
  full code last.
- Always link back to the canonical reference file when giving a fact
  (e.g. "see `references/07-io-0-16.md#flushing`").
- For zig-transformer-lab project-specific issues, consult the
  project-specific gotchas above AND `AGENTS.md` in the repository root.

<!-- ~4.5k tokens · verified against Zig 0.16.0 release notes (2026) + zig-transformer-lab Stage 1-2 experience -->
