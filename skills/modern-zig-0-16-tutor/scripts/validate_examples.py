#!/usr/bin/env python3
"""validate_examples.py — smoke-test the shipped Zig examples and templates.

Runs (when `zig` is on PATH):
  1. `zig fmt --check` over every .zig file in templates/ and examples/.
  2. `zig test <file>` on single-file templates/examples that look like tests.
  3. `zig build` in examples/build-basic-project/ and examples/c-interop-minimal/.

If `zig` is not installed, exits 0 and prints a clear JSON record saying so;
nothing is silently skipped.

Usage:
    python scripts/validate_examples.py
    python scripts/validate_examples.py --json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Files known to contain only test blocks (so `zig test` is the right runner).
TEST_FILES = [
    ROOT / "templates" / "basic-test-0-16.zig",
    ROOT / "templates" / "allocator-owned-buffer-0-16.zig",
    ROOT / "templates" / "tensor2d-skeleton-0-16.zig",
]

# Single-file programs that should at least `zig fmt --check` and be
# syntactically sound. They are not executed here.
FMT_CHECK_DIRS = [
    ROOT / "templates",
    ROOT / "examples",
]

BUILD_PROJECTS = [
    ROOT / "examples" / "build-basic-project",
    ROOT / "examples" / "c-interop-minimal",
]


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {
        "zig_available": False,
        "fmt_check": [],
        "zig_test": [],
        "zig_build": [],
    }

    if shutil.which("zig") is None:
        report["note"] = "zig is not on PATH; nothing executed"
        out = json.dumps(report, indent=2)
        if args.json:
            print(out)
        else:
            print(out)
            print("\nInstall Zig 0.16.0 and re-run to actually validate examples.")
        return 0

    report["zig_available"] = True

    # 1. zig fmt --check across template + example dirs (non-fatal on per-dir
    # failures; we collect all).
    for d in FMT_CHECK_DIRS:
        if not d.exists():
            continue
        r = run(["zig", "fmt", "--check", str(d)])
        report["fmt_check"].append(r)  # type: ignore[attr-defined]

    # 2. zig test on test-y files.
    for t in TEST_FILES:
        if not t.exists():
            continue
        r = run(["zig", "test", str(t)])
        report["zig_test"].append(r)  # type: ignore[attr-defined]

    # 3. zig build in project folders.
    for pd in BUILD_PROJECTS:
        if not (pd / "build.zig").exists():
            continue
        r = run(["zig", "build"], cwd=pd)
        report["zig_build"].append(r)  # type: ignore[attr-defined]

    # Aggregate pass/fail.
    def any_fail(group: list[dict[str, object]]) -> bool:
        return any(rec.get("returncode") != 0 for rec in group)

    report["any_failure"] = any(
        any_fail(report[k])  # type: ignore[index]
        for k in ("fmt_check", "zig_test", "zig_build")
    )

    out = json.dumps(report, indent=2)
    if args.json:
        print(out)
    else:
        print(out)

    return 1 if report["any_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
