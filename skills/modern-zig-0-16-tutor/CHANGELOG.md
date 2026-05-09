# Changelog

All notable changes to `modern-zig-0-16-tutor`.

## [0.1.0] — Initial release

**Target Zig version:** 0.16.0

### Added

- `SKILL.md` with YAML frontmatter, aggressive anti-stale `description`,
  mentor-behavior contract, 22-row stale-pattern table, 15-row compiler
  error→cause Quick Fixes table, progressive-loading decision tree,
  condensed confidence-tiered review checklist, ML/runtime rules.
- 19 topic references (`references/00` through `references/18`):
  - `00` version policy / detection ladder
  - `01` Zig 0.16.0 critical changes (anchor file)
  - `02` language basics
  - `03` types, pointers, slices
  - `04` errors, optionals, defer
  - `05` memory + allocators (DebugAllocator, Arena, FBA, smp)
  - `06` containers (unmanaged idiom, `.empty`)
  - `07` I/O as an Interface (std.Io)
  - `08` build system (root_module, createModule)
  - `09` C interop (addTranslateC, extern/export, callconv(.c))
  - `10` testing + debugging
  - `11` comptime + metaprogramming (@Int, @Struct, ... replacing @Type)
  - `12` formatting + logging (fmt.Alt, Io.Duration)
  - `13` style guide
  - `14` 0.13 → 0.16 migration guide
  - `15` confidence-tiered code-review checklist
  - `16` Zig for ML/runtime projects (Tensor2D, @Vector)
  - `17` Zig-side CUDA interop notes (host-side only; recommended reading)
  - `18` token-budget discipline for consumers of this skill
- 12 recipe files containing 17 numbered recipes with mentor format.
- 7 templates with `// target: Zig 0.16.0 — illustrative` headers, including a
  `Tensor2D` skeleton with naive scalar matmul and 3 tests.
- 6 example targets: 4 single-file illustrations + 2 project folders
  (`build-basic-project`, `c-interop-minimal`).
- 4 Python helper scripts (`detect_zig_version.py`, `validate_examples.py`,
  `grep_stale_patterns.py`, `build_skill_package.py`) — Python stdlib only.
- `tests/expected_stale_patterns.txt` and `tests/expected_modern_patterns.txt`
  fixtures for the grepper smoke test.
- `QUALITY_REPORT.md` generated at packaging time.

### Known uncertainties (carried into `QUALITY_REPORT.md`)

These items are flagged inline in the relevant reference files with
`VERIFY WITH ZIG 0.16.0 LOCALLY` callouts:

1. Exact member names on `std.process.Init.Environ.Map` (`get` vs `getPtr`).
2. Return shape of `std.mem.cut` (struct with `.prefix`/`.suffix` vs tuple).
3. Full `mem.indexOf*` → `find*` rename table.
4. Whether `async`/`await`/`suspend`/`resume` remain reserved words or are
   fully deleted from the grammar.
5. Whether bare `pub fn main() !void` still compiles (release notes suggest
   yes, without argv/env access).
6. Whether `std.ArrayList(T)` in 0.16 is now the unmanaged variant by rename,
   or whether `ArrayListUnmanaged` is still the spelling.
7. Exact namespace of the `.limited(n)` helper used by `readFileAlloc` and
   `allocRemaining`.
8. Whether `translate_c.linkSystemLibrary(...)` makes `exe.linkLibC()`
   redundant.

### Not tested

- Examples and templates are marked **illustrative** and have not been compiled
  against Zig 0.16.0 locally (no Zig toolchain on authoring host).
- `scripts/validate_examples.py` is runnable by the user once Zig is installed.

## [Unreleased]

- Placeholder for future 0.16.1 / 0.17.0 updates.
