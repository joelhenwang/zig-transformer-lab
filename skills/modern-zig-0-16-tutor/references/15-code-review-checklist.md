# 15 — Code-review checklist (0.16.0)

Confidence-tiered: **ALWAYS FLAG** > **FLAG WITH CONTEXT** > **SUGGEST**.
For every FLAG WITH CONTEXT item, the preconditions are listed; if the
preconditions are unclear, ask instead of flagging.

When reviewing user code, cite `file:line` and suggest the **minimal diff**,
not a rewrite. See `references/14-migration-from-0-13-to-0-16.md` for
canonical diffs.

---

## ALWAYS FLAG — mechanical, objectively broken on Zig 0.16.0

These should be flagged without ceremony. Each has a regex-level signature;
`scripts/grep_stale_patterns.py` can also surface them.

- `std.io.getStdOut()` / `getStdIn()` / `getStdErr()` — replace with
  `std.Io.File.stdout()` etc. (ref 07)
- `std.io.bufferedWriter(...)` / `std.io.bufferedReader(...)` — stale; the
  buffer now lives in the `Writer` / `Reader` interface (ref 07)
- `std.io.fixedBufferStream(...)` — use `std.Io.Reader.fixed(slice)` /
  `std.Io.Writer.fixed(buf)` (ref 07)
- `GenericReader` / `AnyReader` / `GenericWriter` / `AnyWriter` — deleted;
  use concrete `std.Io.Reader` / `std.Io.Writer` (ref 07)
- `std.heap.GeneralPurposeAllocator(.{}){}` — renamed to `DebugAllocator`
  (ref 05)
- `std.heap.ThreadSafeAllocator{ .child_allocator = ... }` — removed;
  `ArenaAllocator` is lock-free (ref 05)
- `std.Thread.Pool` — removed; use `std.Io.Group` (ref 01)
- `std.process.argsAlloc(a)` — use `init.minimal.args` (ref 01)
- `std.os.environ` / `std.process.getEnvVarOwned` — use `init.environ_map`
  (ref 01)
- `@cImport({ ... })` — deprecated; use `b.addTranslateC` (ref 09)
- `@Type(.{...})` — removed; use `@Int`, `@Struct`, etc. (ref 11)
- `@typeInfo(T).Struct` / `.Pointer` / `.ErrorUnion` (PascalCase) — snake_case
  in 0.16 (ref 11)
- `@intToPtr` / `@ptrToInt` / `@enumToInt` / `@intToEnum` / `@floatToInt`
  / `@intToFloat` / `@boolToInt` / `@errSetCast` — renamed (ref 03)
- `callconv(.C)` — lowercase `.c` (ref 09)
- `usingnamespace` — removed (ref 11)
- `async` / `await` / `suspend` / `resume` as keywords — removed (ref 01)
- `std.build.Builder` / `FileSource` / `std.zig.CrossTarget` — renamed
  (ref 08)
- `b.addExecutable(.{ .root_source_file, .target, .optimize })` without
  `root_module` — fields moved (ref 08)
- `b.addStaticLibrary(...)` / `b.addSharedLibrary(...)` — use
  `b.addLibrary(.{ .linkage = ... })` (ref 08)
- `exe.addModule("x", m)` — use `exe.root_module.addImport("x", m)` (ref 08)
- `exe.linkLibC()` / `exe.linkSystemLibrary("...")` / `exe.addCSourceFile`
  on the compile step — must be on `exe.root_module` (ref 08)
- `ArrayList(T).init(a)` + `append(v)` + `deinit()` — unmanaged idiom
  (ref 06)
- `PriorityQueue.add` / `remove` — renamed to `push` / `pop` (ref 06)
- `{D}` format specifier — replaced by `Io.Duration` + `{f}` (ref 12)

## FLAG WITH CONTEXT — semantic, requires understanding

Flag only when the precondition is met. If the code is short enough, include
the triggering pattern in your review comment.

- **Missing `try fw.flush()` before function end on a buffered writer**
  - Precondition: writer uses a non-empty buffer; scope exits without the
    writer being dropped-with-flush.
  - Fix: add `try fw.flush();` or propagate the error.
- **Missing `errdefer` on an allocated buffer**
  - Precondition: function allocates via an allocator, subsequently calls
    a fallible function, and returns the buffer on success.
  - Fix: `const buf = try a.alloc(...); errdefer a.free(buf);`
- **Mixed allocators**
  - Precondition: a slice was allocated from allocator A and freed from
    allocator B (or captured into a container using C).
  - Fix: pass the same allocator through, or clone into the target allocator.
- **Allocator stored in a struct without lifetime ownership**
  - Precondition: a struct contains `allocator: Allocator` and also a
    slice / list; the struct outlives the allocator's scope.
  - Fix: document that the struct holds only a borrowed allocator, or
    require caller to pass allocator into methods (unmanaged pattern).
- **Wrong pointer kind**
  - Precondition: using `[*]T` for a run-length-known buffer → should be
    `[]T`; or `*T` where `*[N]T` carries useful length.
  - Fix: choose the kind with the strongest invariants you can prove.
- **Missing bounds check when accepting foreign indices**
  - Precondition: public API takes an index from user data or C; indexes
    directly without `if (i >= len) return error.IndexOutOfRange;`.
- **Unchecked `orelse unreachable` in library code**
  - Precondition: the `null` case can actually occur from an external input.
  - Fix: return an explicit error.
- **`catch {}`** — empty catch
  - Precondition: always.
  - Fix: handle or propagate.
- **Silent error coercion via `catch undefined` or `catch 0`**
  - Precondition: the default hides a real failure mode.
  - Fix: return or log.
- **Recursive function with an inferred error set**
  - Precondition: `fn f(...) !T` that calls itself.
  - Fix: define an explicit error set to stop inference divergence.
- **Hidden allocation in an API that looks "cheap"**
  - Precondition: function name suggests pure computation but it allocates.
  - Fix: either document, or accept an allocator parameter and return
    owned memory explicitly.
- **Numeric equality on floats** (`a == b` for `f32`/`f64`)
  - Precondition: always in tests; usually outside tests too.
  - Fix: `std.testing.expectApproxEqAbs` or `expectApproxEqRel`.
- **Shape not checked at API boundary** (ML-relevant)
  - Precondition: function takes `Tensor2D` / `[]f32` / `[]const f32` and
    does not validate dimensions at entry.
  - Fix: return `error.ShapeMismatch` on mismatch.

## SUGGEST — style and taste

These are review comments, not blocking issues. Use softer language:
"consider", "you might prefer", "for clarity".

- Variable that is never mutated — suggest `const`.
- `.{}` on an empty container — suggest `.empty`.
- `std.debug.print` inside library code — suggest routing through `io`.
- `@as(T, x)` where `@intCast(x)` / `@floatCast(x)` is meant — pick the
  right one; `@as` does not change numeric domain.
- `while (i < n) : (i += 1)` — suggest `for (0..n) |i|`.
- Test named "it works" — suggest a behavioral name.
- Long `switch` with no `else => unreachable` on an exhaustive enum —
  suggest omitting, Zig already verifies exhaustiveness.
- Missing doc comment on a public item.

## Review format (how to respond)

When asked to review code, structure your reply as:

```
## Review: <short summary>

### ALWAYS FLAG
- path/to/file.zig:LN — <stale pattern>. Suggested change: `...`
- ...

### FLAG WITH CONTEXT
- path/to/file.zig:LN — <semantic issue>. Why it matters: <...>. Suggested
  change: `...`
- ...

### SUGGEST
- path/to/file.zig:LN — <style nit>. Consider: `...`

### Tests to add
- <one-sentence description>
- ...

### Questions
- <diagnostic question to ask the user, if any>
```

Keep each line short. Link to the relevant reference file. Never rewrite
the file in the review response.

<!-- ~3.0k tokens · derived from refs 01-13 -->
