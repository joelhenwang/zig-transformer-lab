# Quality Report — `modern-zig-0-16-tutor` v0.1.0

**Generated:** 2026-05-09
**Target Zig version:** 0.16.0
**Authoring host:** Windows 11, Python 3.12, Zig *not* installed.

## What was created

| Category | Count | Detail |
|---|---|---|
| Top-level | 3 | `SKILL.md`, `README.md`, `CHANGELOG.md` |
| References | 19 | `references/00` through `references/18` |
| Recipes | 12 | `recipes/*.md`, 17 numbered recipes |
| Templates | 7 | `templates/*.zig` with `0.16` annotations |
| Examples | 10 | 4 single-file + 2 project folders (incl. `c/` sources) |
| Scripts | 4 | Python stdlib only (`detect_zig_version.py`, `grep_stale_patterns.py`, `validate_examples.py`, `build_skill_package.py`) |
| Tests | 2 | `tests/expected_stale_patterns.txt`, `tests/expected_modern_patterns.txt` |

## Validation performed

### ✅ Grepper smoke tests

```
scripts/grep_stale_patterns.py tests/expected_stale_patterns.txt  → 37 hits
scripts/grep_stale_patterns.py tests/expected_modern_patterns.txt →  0 hits
scripts/grep_stale_patterns.py templates/ examples/               →  0 hits
scripts/grep_stale_patterns.py references/                        → 43 hits
```

- 37/37 canonical stale patterns flagged in the fixture.
- 0 false positives in the modern fixture.
- Templates and examples are clean (no stale patterns shipped).
- Reference MD file hits are all legitimate **prose discussion** of stale
  identifiers (not code blocks or tables). These are the point of the
  reference files and are not a defect.

### ✅ Version detector

```
scripts/detect_zig_version.py
→ {"version": null, "is_0_16": false, "warning": "zig is not on PATH"}
→ exit code 3
```

Handled correctly when Zig is absent.

### ❌ Not performed (no Zig toolchain on authoring host)

These steps must be run by the user after installing Zig 0.16.0:

- `zig fmt --check` over `templates/` and `examples/`
- `zig test` on `templates/basic-test-0-16.zig`,
  `templates/allocator-owned-buffer-0-16.zig`,
  `templates/tensor2d-skeleton-0-16.zig`
- `zig build` in `examples/build-basic-project/`
- `zig build` in `examples/c-interop-minimal/` (requires a C compiler too)

All are wired up in `scripts/validate_examples.py`; run it after installing
Zig 0.16.0 to sanity-check.

### ⚠️ Known uncertainties carried forward

All are flagged inline in the referenced files with
**`VERIFY WITH ZIG 0.16.0 LOCALLY`** callouts.

1. Exact member names on `std.process.Init.Environ.Map` (`get` vs
   `getPtr`). — `references/01`.
2. Return shape of `std.mem.cut` (struct with `.prefix`/`.suffix` vs
   tuple). — `references/01`.
3. Full `mem.indexOf*` → `find*` rename table. — `references/01`.
4. Whether `async`/`await`/`suspend`/`resume` remain reserved words or
   are fully deleted from the grammar. — `references/01`.
5. Whether bare `pub fn main() !void` still compiles (release notes
   suggest yes, with no argv/env access). — `references/02`.
6. Whether `std.ArrayList(T)` in 0.16 is now the unmanaged variant by
   rename, or whether `ArrayListUnmanaged` is still the spelling. —
   `references/06`.
7. Exact namespace of `.limited(n)` used by `readFileAlloc` and
   `allocRemaining`. — `references/07`, `recipes/io.md`.
8. Whether `translate_c.linkSystemLibrary(...)` makes
   `exe.root_module.link_libc = true` redundant. — `references/08`,
   `references/09`.

## Size budget check

Estimated on-disk totals:

| Group | Lines | Tokens (est.) |
|---|---|---|
| SKILL.md | ~235 | ~3.0 k |
| README + CHANGELOG | ~200 | ~2.5 k |
| References (19 files) | ~3 800 | ~48 k |
| Recipes (12 files) | ~900 | ~12 k |
| Templates (7 files) | ~500 | ~6 k |
| Examples (10 files) | ~200 | ~2.5 k |
| Scripts (4 files) | ~600 | ~6 k |
| Fixtures | ~130 | ~1.5 k |
| **Total** | **~6 500** | **~81 k** |

Typical per-turn load (see `references/18-token-budget-guide.md`):

| Scenario | Files | Tokens |
|---|---|---|
| Quick syntax question | SKILL.md + 1 recipe | ~4 k |
| I/O question | SKILL.md + ref 07 + hello-world example | ~7 k |
| Build setup | SKILL.md + ref 08 + exe template | ~8 k |
| C interop | SKILL.md + refs 08, 09 + interop template | ~13 k |
| Migration audit | SKILL.md + refs 01, 14 + 3 k user code | ~14 k |
| Code review (10 k user code) | SKILL.md + refs 15, 01 + code | ~21 k |
| CUDA + ML | SKILL.md + refs 16, 17, 09 | ~14 k |

All under the 30–40 k ceiling.

## Install

See `README.md`. Short form:

```pwsh
Copy-Item -Recurse -Force `
  "C:\Users\z00517bz\Documents\dev\modern-zig-0-16-tutor" `
  "C:\Users\z00517bz\.claude\skills\modern-zig-0-16-tutor"
```

## How to update for Zig 0.16.1 / 0.17.0

1. Read the new release notes.
2. Patch change sections in `references/01-zig-0-16-critical-changes.md`.
3. Update `SKILL.md` stale-pattern table + `scripts/grep_stale_patterns.py`
   regex list if any new stale patterns appeared.
4. Update `CHANGELOG.md`.
5. Re-run `scripts/grep_stale_patterns.py tests/` to ensure fixtures pass.
6. Re-run `scripts/build_skill_package.py` to produce a new zip.

## Zip

Produced by `scripts/build_skill_package.py` at
`C:\Users\z00517bz\Documents\dev\modern-zig-0-16-tutor.zip`.
