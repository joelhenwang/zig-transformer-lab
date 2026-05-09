# modern-zig-0-16-tutor

An AI mentor skill for **Zig 0.16.0**. Helps a learner write, review, debug,
and migrate modern Zig — and eventually build a tiny educational deep-learning
runtime with CUDA interop.

Designed to prevent stale 0.13-era answers. See `SKILL.md` for the activation
contract.

## What this skill does

- Pins answers to **Zig 0.16.0** language and stdlib APIs.
- Enforces **mentor behavior**: explain → ask → pseudocode → tiny snippet →
  test → exercise. Never dump full code unless asked.
- Flags **~22 canonical stale patterns** automatically (see `SKILL.md` stale
  table and `scripts/grep_stale_patterns.py`).
- Provides a **confidence-tiered code-review checklist**
  (`references/15-code-review-checklist.md`).
- Covers **build.zig** (`root_module` + `createModule`), **C interop** via
  `b.addTranslateC`, and **Zig-side CUDA interop** concepts.
- Includes a runnable **Tensor2D** skeleton for CPU ML reference work
  (`templates/tensor2d-skeleton-0-16.zig`).

## Install

### Option A — copy to the Claude skills directory (recommended)

```pwsh
Copy-Item -Recurse -Force `
  "C:\Users\z00517bz\Documents\dev\modern-zig-0-16-tutor" `
  "C:\Users\z00517bz\.claude\skills\modern-zig-0-16-tutor"
```

Restart your client if needed. The skill will then be listed under
available skills.

### Option B — use the packaged zip

```pwsh
cd C:\Users\z00517bz\Documents\dev
Expand-Archive -Path .\modern-zig-0-16-tutor.zip `
  -DestinationPath "$HOME\.claude\skills\" -Force
```

## Use

- In an AI client with skill loading, trigger the skill by asking a
  Zig 0.16.0 question. Or explicitly: "load modern-zig-0-16-tutor".
- The consumer model will read `SKILL.md` automatically and pull topic
  references on demand (see the progressive-loading decision tree in
  `SKILL.md`).

## Validate the examples (after you install Zig 0.16.0)

Download Zig 0.16.0 from <https://ziglang.org/download/>, add to `PATH`,
then:

```pwsh
python .\scripts\detect_zig_version.py
python .\scripts\validate_examples.py
python .\scripts\grep_stale_patterns.py .
```

- `detect_zig_version.py` prints JSON: `{version, is_0_16, warning}`.
- `validate_examples.py` runs `zig fmt --check`, `zig test`, and `zig build`
  on the examples/templates. Degrades cleanly if `zig` is absent.
- `grep_stale_patterns.py` scans `.zig` / `build.zig` / `*.md` for stale
  0.13-era patterns and suggests modern replacements.

## Update to a newer Zig (e.g. 0.16.1 or 0.17.0)

1. Fetch the new release notes.
2. Walk `references/01-zig-0-16-critical-changes.md` and patch each change
   section.
3. If any stale pattern list grew, update `SKILL.md` stale-pattern table
   and `scripts/grep_stale_patterns.py` regex list.
4. Update `CHANGELOG.md`.
5. Re-run `scripts/build_skill_package.py` to produce a fresh zip.
6. Test against representative code with `scripts/grep_stale_patterns.py`
   and `scripts/validate_examples.py`.

## Directory map

```
SKILL.md                        activation contract, stale table, review checklist
README.md                       this file
CHANGELOG.md                    version history
references/00-18                topic-scoped deep dives
recipes/                        12 files, 17 numbered recipes
templates/                      7 starter .zig files
examples/                       runnable snippets + 2 project folders
scripts/                        version detector, stale grepper, validator, packager
tests/                          expected stale + modern patterns for grepper smoke tests
QUALITY_REPORT.md               emitted by packaging step
```

## License / provenance

All code examples target Zig 0.16.0 and are derived from the official release
notes, language reference, and standard library docs. Examples are marked
`illustrative, verify with Zig 0.16.0` where not personally compiled; see
`QUALITY_REPORT.md`.

<!-- ~1.4k tokens -->
