# Agent Brief — zig-transformer-lab

## One-sentence mission

Build a pedagogical Zig 0.16.0 library that trains a tiny 1-block 1-head word-level
transformer on CPU, then on CUDA, with extensive documentation and heavily commented
code that teaches how PyTorch-like systems work internally.

## Hard rules

- Do not violate Locked Decisions D1 through D14 or policies P1 through P4.
  Ask the user before changing any decision. See `docs/00_overview.md` for the full table.
- No third-party Zig dependencies.
- Zig version pinned at 0.16.0 (exact).
- f32 only. No mixed precision.
- GPU kernels are CUDA C only. No pure-Zig GPU kernels.
- Kernels are compiled offline with nvcc to .ptx and loaded via cuModuleLoadData.
- Never copy code from LGPL sources. Architectural ideas only, attributed in file headers.

## Workflow

- Implement stages 1 through 9 in order. Do not interleave stages.
- Commit after each stage with `stage(N): <summary>`; paste acceptance outputs
  into the commit body.
- Docs chapter for a stage ships in the same commit as the stage's code.

## Style

- File header block: purpose, shape contract, math formulas, ownership, errors,
  TODOs, credits.
- Explicit allocators. No hidden globals. No `@panic` outside tests.
- `zig fmt` clean. Tests co-located via `test "..."` blocks.
- Every public function has a worked shape example in its doc comment.
- Pedagogical comments: explain *why*, not just *what*. ~30-40% of non-blank
  lines should be comments.

## Zig 0.16.0 patterns

Use modern 0.16 API exclusively:
- `pub fn main(init: std.process.Init) !void`
- `b.createModule(...)`, `root_module.addImport(...)`
- `var list: ArrayList(T) = .empty; list.append(gpa, v)`
- `@ptrFromInt`, `@intFromPtr`, `@intFromEnum` (not the old cast builtins)
- `callconv(.c)` not `callconv(.C)`
- Explicit re-exports, never `usingnamespace`
- `std.Io.Dir.cwd()` not `std.fs.cwd()`
- `const` for non-mutated locals (Zig 0.16.0 enforces this as an error, not a warning)

## This project's Shape API (MUST follow — tripped up 3 subagents)

The `Shape` struct in `src/tensor/shape.zig` uses rank-specific constructors:

```zig
const s1 = Shape.init1D(10);          // 1D: (10,)
const s2 = Shape.init2D(2, 3);       // 2D: (2, 3)
const s3 = Shape.init3D(2, 3, 4);    // 3D: (2, 3, 4)
const s4 = Shape.init4D(2, 3, 4, 5); // 4D: (2, 3, 4, 5)
```

**CRITICAL:** There is NO `Shape.init(&.{2, 3})` constructor. It does not exist.
If you write `Shape.init(...)` with a slice/array argument, compilation will fail.

The `rank` field is a `u2` storing `ndim - 1` (0-3 for 1D-4D). Use `shape.ndim()`
to get the dimension count (returns `rank + 1`). Never compare `rank` directly
to a dimension count without adding 1.

`Shape.equals` is a **free function**, not a method: `equals(a, b)` not `a.equals(b)`.

## Zig 0.16.0 gotchas encountered during Stage 2

These are real compilation errors we hit. Learn from them:

1. **`std.fs.cwd()` does not exist in 0.16.0.** The `std.fs` module is mostly
   deprecated. Use `std.Io.Dir.cwd()` for filesystem access. In `build.zig`,
   avoid runtime filesystem operations entirely — use build-system commands
   and static file lists instead of directory iteration.

2. **`var` vs `const` is enforced as an error.** If a local variable is never
   mutated after initialization, Zig 0.16.0 emits a hard error (not a warning).
   Use `const` by default. Only use `var` when you actually mutate the variable.

3. **Unused function parameters are errors.** If a function parameter isn't
   used, either prefix with `_` or remove it. No silent warnings.

4. **`build.zig.zon` fingerprint must match.** The `fingerprint` field is
   verified by the compiler. If you create a new project, let the compiler
   tell you the correct value on first build — it prints the expected value.

5. **Unused build options must be suppressed.** If you declare `b.option(...)`
   but don't use the value, add `_ = variable;` to suppress the error.

6. **`addSystemCommand` takes `[]const []const u8`.** All arguments must be
   available at build-configuration time. `b.fmt()` returns `[]const u8` which
   works, but you can't do runtime string construction inside the arg array.

7. **`LazyPath` uses `.cwd_relative = "path"`.** When adding library/rpath
   entries, use `std.Build.LazyPath{ .cwd_relative = "/path" }`.

8. **`exe.builder.allocator` doesn't exist.** In 0.16.0, access the build
   allocator via `b.allocator` (inside `build` function scope), not through
   the compile step.

9. **Test discovery goes through module imports.** Zig's test runner only finds
   `test "..."` blocks in files transitively imported by the test root. Our
   `src/root.zig` re-exports everything and its own `test { }` block references
   each sub-module. Don't use relative file paths in test files — import
   through the module.

10. **`std.fmt.bufPrint` for string formatting.** `std.io.fixedBufferStream`
    may not exist or have a different API. `std.fmt.bufPrint(buf, fmt, args)`
    is the reliable way to format into a buffer.

## CUDA sacred spots

- Row-major to column-major wrapping in `src/backend/cuda/gemm.zig`. Dedicated tests.
- Bounds checks in every kernel.
- Offline .ptx only (no NVRTC).
- Bindings module opens libraries with dlopen at runtime.
- Link `libc` + `dl`, never `libcuda` directly.
- In `build.zig`, use a static `kernel_names` list for nvcc compilation —
  don't iterate the filesystem at build time (avoids `std.fs.cwd()` issues).

## Zig 0.16.0 skill reference

The `skills/modern-zig-0-16-tutor/` directory contains a comprehensive Zig 0.16.0
reference library. **Read `skills/modern-zig-0-16-tutor/SKILL.md` every session** — it
has the canonical stale-pattern table (28 rows), compiler-error quick lookup (22 entries),
and 13 project-specific gotchas discovered during Stages 1-2.

When you hit a Zig compilation error or API question, load reference files by topic
(max 3 per turn to manage token budget):

| Task | Load |
|------|------|
| Build system errors | `references/08-build-system-0-16.md` |
| I/O (stdout, writer, formatting) | `references/07-io-0-16.md` |
| Allocators, arena, leaks | `references/05-memory-allocators.md` |
| ArrayList / HashMap | `references/06-containers-0-16.md` |
| C/CUDA interop | `references/09-c-interop-0-16.md` + `references/17-zig-cuda-interop-notes.md` |
| Testing, debugging | `references/10-testing-debugging.md` |
| Code review | `references/15-code-review-checklist.md` |
| ML/tensor design | `references/16-zig-for-ml-runtime-projects.md` |
| Tensor2D skeleton | `templates/tensor2d-skeleton-0-16.zig` |
| Numerical code testing | `recipes/numerical-code.md` |
| Stale pattern scanning | `scripts/grep_stale_patterns.py <dir>` |

Also available: 19 reference files, 12 recipes, 7 templates, 4 validation scripts.
See `skills/modern-zig-0-16-tutor/README.md` for the full directory map.

## When stuck

- Ask the user with a crisp options-style question (max 4 options).
- Never guess at hardware, environment, or decisions.
- Check the skill's compiler-error quick lookup table in `SKILL.md`.
- See `SESSION_GUIDE.md` for full project state and continuation instructions.

## Full plan

See `plan.md` at the repository root and `docs/00_overview.md`.
