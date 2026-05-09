#!/usr/bin/env python3
"""build_skill_package.py — create a distributable ZIP of the skill directory.

Validates required files exist, excludes build / cache artifacts, and writes
the archive next to the skill directory.

Usage:
    python scripts/build_skill_package.py
    python scripts/build_skill_package.py --out C:\\path\\to\\out.zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "SKILL.md",
    "README.md",
    "CHANGELOG.md",
    "references/00-version-policy.md",
    "references/01-zig-0-16-critical-changes.md",
    "references/14-migration-from-0-13-to-0-16.md",
    "references/15-code-review-checklist.md",
    "references/18-token-budget-guide.md",
]

REQUIRED_DIRS = [
    "references",
    "recipes",
    "templates",
    "examples",
    "scripts",
    "tests",
]

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "zig-cache",
    "zig-out",
    ".zig-cache",
    "node_modules",
}
EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo", ".zip"}


def verify() -> list[str]:
    issues: list[str] = []
    for f in REQUIRED_FILES:
        if not (ROOT / f).exists():
            issues.append(f"missing required file: {f}")
    for d in REQUIRED_DIRS:
        if not (ROOT / d).is_dir():
            issues.append(f"missing required dir: {d}")
    return issues


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts[:-1])
        if parts & EXCLUDE_DIR_NAMES:
            continue
        if p.suffix in EXCLUDE_FILE_SUFFIXES:
            continue
        yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT.parent / f"{ROOT.name}.zip",
        help="output zip path (default: <parent>/<skill-name>.zip)",
    )
    args = parser.parse_args()

    issues = verify()
    if issues:
        for i in issues:
            sys.stderr.write(f"ERROR: {i}\n")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in iter_files(ROOT):
            rel = Path(ROOT.name) / p.relative_to(ROOT)
            zf.write(p, arcname=rel.as_posix())

    print(f"wrote {args.out}")
    print(f"size: {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
