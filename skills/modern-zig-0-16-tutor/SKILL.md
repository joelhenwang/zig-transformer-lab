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
  stale stdlib namespaces (`std.io`, `std.os.environ`), stale casts
  (`@intToPtr`…), stale containers (`ArrayList(T).init(a)`), stale build.zig
  (`root_source_file` on exe, `addStaticLibrary`), `@cImport`, `@Type`,
  `usingnamespace`, `async`/`await` keywords, `{D}` format, `callconv(.C)`.
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

## How to use this skill

- Consumers of this skill read `SKILL.md` every turn. Consult specific
  `references/NN-*.md` only when the task matches the decision tree above.
- When writing for the learner: pseudocode and diagrams first; Zig second;
  full code last.
- Always link back to the canonical reference file when giving a fact
  (e.g. "see `references/07-io-0-16.md#flushing`").

<!-- ~3.0k tokens · verified against Zig 0.16.0 release notes (2026) -->
