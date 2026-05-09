# Recipes: debugging

## Recipe 17 — Detect stale Zig patterns in a codebase

**Problem.** You received a Zig project that doesn't compile on 0.16.0. You
want a fast list of probably-stale lines before you dig in.

**Mentor explanation.** Use `scripts/grep_stale_patterns.py` (shipped with
this skill). It scans `.zig`, `build.zig`, and `.md` files for a curated
list of known-stale patterns. The tool doesn't fix anything — it points
you at the lines.

**Run it.**
```pwsh
python scripts\grep_stale_patterns.py C:\path\to\project
```

Sample output:
```
src\main.zig:4:1 — std.io.getStdOut()         → see references/07-io-0-16.md
src\main.zig:7:5 — GeneralPurposeAllocator    → see references/05-memory-allocators.md
build.zig:12:20 — root_source_file on exe     → see references/08-build-system-0-16.md
```

**Mentor workflow.**
1. Run the script first, before reading the code.
2. Fix **build.zig** issues first — those block everything.
3. Fix the top of `main.zig` (main signature + stdio) next.
4. Work down to container / allocator renames.
5. Run `zig build` between each cluster of fixes to narrow errors.

## Reading compiler errors in 0.16

Common message → likely cause:

| Error fragment | Likely cause |
|---|---|
| `no field 'root_source_file'` | moved into `root_module` (ref 08) |
| `no member 'getStdOut'` | use `std.Io.File.stdout()` (ref 07) |
| `no member 'GeneralPurposeAllocator'` | renamed `DebugAllocator` (ref 05) |
| `expected 2 arguments, found 1` on `append` | unmanaged API; pass allocator (ref 06) |
| `use of undeclared identifier '@Type'` | split into 8 builtins (ref 11) |
| `use of undeclared '@intToPtr'` | `@ptrFromInt` (ref 03) |
| `invalid enum value '.C'` | use `.c` (ref 09) |
| `unexpected token 'async'` | keywords removed (ref 01) |

## `std.debug.dumpStackTrace` + error return trace

```zig
pub fn main(init: std.process.Init) !void {
    _ = init;
    doWork() catch |err| {
        if (@errorReturnTrace()) |trace| std.debug.dumpStackTrace(trace.*);
        return err;
    };
}
```

In Debug builds, `@errorReturnTrace()` returns a rich call-site trace from
every place the error propagated — often more useful than the stack trace.

**Exercise.** Introduce an intentional stale pattern (`@cImport`) in a test
file. Run the grepper. Does the line number match? Does the suggested
reference point where you'd need to go?

<!-- ~1.0k tokens -->
