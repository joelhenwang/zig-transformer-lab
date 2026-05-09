#!/usr/bin/env python3
"""grep_stale_patterns.py — flag stale Zig patterns in .zig / build.zig / .md.

Scans one or more paths (files or directories) for the canonical stale
patterns from Zig 0.13-era code. Suggests the modern replacement and the
reference file to consult.

Usage:
    python scripts/grep_stale_patterns.py PATH [PATH ...] [--json] [--ignore-md]

Exit codes:
    0 - no hits
    1 - hits found
    2 - usage error

Output (text mode):
    path:line:col  —  <pattern description>  (→ references/NN-*.md)

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr if possible (Windows cp1252 hates arrows).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# Each pattern: (human_name, regex, reference_filename)
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("std.io.getStdOut", re.compile(r"std\.io\.getStd(?:Out|In|Err)\s*\("),
     "references/07-io-0-16.md"),
    ("std.io.bufferedWriter", re.compile(r"std\.io\.bufferedWriter\s*\("),
     "references/07-io-0-16.md"),
    ("std.io.fixedBufferStream", re.compile(r"std\.io\.fixedBufferStream\s*\("),
     "references/07-io-0-16.md"),
    ("GenericReader/AnyReader",
     re.compile(r"std\.io\.(?:GenericReader|AnyReader|GenericWriter|AnyWriter)\b"),
     "references/07-io-0-16.md"),
    ("@cImport", re.compile(r"@cImport\s*\("),
     "references/09-c-interop-0-16.md"),
    ("@Type", re.compile(r"@Type\s*\("),
     "references/11-comptime-metaprogramming.md"),
    ("@typeInfo PascalCase",
     re.compile(r"@typeInfo\s*\([^)]+\)\s*\.(?:Struct|Pointer|ErrorUnion|Array|Enum|Fn|Union|Int|Float|Optional)\b"),
     "references/11-comptime-metaprogramming.md"),
    ("std.heap.GeneralPurposeAllocator",
     re.compile(r"std\.heap\.GeneralPurposeAllocator\s*\("),
     "references/05-memory-allocators.md"),
    ("std.heap.ThreadSafeAllocator",
     re.compile(r"std\.heap\.ThreadSafeAllocator\b"),
     "references/05-memory-allocators.md"),
    ("std.Thread.Pool", re.compile(r"std\.Thread\.Pool\b"),
     "references/01-zig-0-16-critical-changes.md"),
    ("std.process.argsAlloc", re.compile(r"std\.process\.argsAlloc\s*\("),
     "references/01-zig-0-16-critical-changes.md"),
    ("std.os.environ", re.compile(r"std\.os\.environ\b"),
     "references/01-zig-0-16-critical-changes.md"),
    ("std.process.getEnvVarOwned",
     re.compile(r"std\.process\.getEnvVarOwned\s*\("),
     "references/01-zig-0-16-critical-changes.md"),
    ("std.build.Builder", re.compile(r"std\.build\.Builder\b"),
     "references/08-build-system-0-16.md"),
    ("b.addStaticLibrary", re.compile(r"\baddStaticLibrary\s*\("),
     "references/08-build-system-0-16.md"),
    ("b.addSharedLibrary", re.compile(r"\baddSharedLibrary\s*\("),
     "references/08-build-system-0-16.md"),
    ("exe.addModule", re.compile(r"\bexe\.addModule\s*\("),
     "references/08-build-system-0-16.md"),
    ("std.zig.CrossTarget", re.compile(r"std\.zig\.CrossTarget\b"),
     "references/08-build-system-0-16.md"),
    ("FileSource type", re.compile(r"\bFileSource\b"),
     "references/08-build-system-0-16.md"),
    ("root_source_file on addExecutable/addTest",
     re.compile(r"\baddExecutable\s*\(\s*\.\s*\{[^}]*\.root_source_file"),
     "references/08-build-system-0-16.md"),
    ("ArrayList.init(allocator)",
     re.compile(r"ArrayList\s*\([^)]*\)\s*\.\s*init\s*\("),
     "references/06-containers-0-16.md"),
    ("callconv(.C)", re.compile(r"callconv\s*\(\s*\.C\s*\)"),
     "references/09-c-interop-0-16.md"),
    ("usingnamespace", re.compile(r"\busingnamespace\b"),
     "references/11-comptime-metaprogramming.md"),
    ("async/await/suspend/resume keyword",
     re.compile(r"\b(?:async|await|suspend|resume|nosuspend)\b"),
     "references/01-zig-0-16-critical-changes.md"),
    ("@intToPtr", re.compile(r"@intToPtr\s*\("),
     "references/03-types-pointers-slices.md"),
    ("@ptrToInt", re.compile(r"@ptrToInt\s*\("),
     "references/03-types-pointers-slices.md"),
    ("@enumToInt", re.compile(r"@enumToInt\s*\("),
     "references/02-language-basics.md"),
    ("@intToEnum", re.compile(r"@intToEnum\s*\("),
     "references/02-language-basics.md"),
    ("@floatToInt", re.compile(r"@floatToInt\s*\("),
     "references/02-language-basics.md"),
    ("@intToFloat", re.compile(r"@intToFloat\s*\("),
     "references/02-language-basics.md"),
    ("@boolToInt", re.compile(r"@boolToInt\s*\("),
     "references/02-language-basics.md"),
    ("@errSetCast", re.compile(r"@errSetCast\s*\("),
     "references/02-language-basics.md"),
    ("std.mem.indexOf*",
     re.compile(r"std\.mem\.(?:indexOf|indexOfScalar|indexOfAny|indexOfPos|lastIndexOf)\b"),
     "references/01-zig-0-16-critical-changes.md"),
    ("{D} format specifier", re.compile(r'"[^"]*\{D\}[^"]*"'),
     "references/12-formatting-logging.md"),
]

# Markdown files are allowed to *describe* stale patterns. Suppress hits on:
#   - Table rows (lines whose first non-space char is `|`), because the
#     reference tables list stale names in the left column by design.
#   - Lines inside a fenced code block where the preceding block marker or
#     the block's first comment contains "WRONG" / "stale" / "DEPRECATED".
#   - Lines that obviously framing the pattern as an anti-example
#     (contain "WRONG", "Stale", "stale", "DEPRECATED", "deprecated").
MD_FRAMING_RE = re.compile(
    r"(WRONG|Stale|stale|DEPRECATED|deprecated|≤\s*0\.)",
)

SCAN_EXTENSIONS = {".zig", ".md"}
SCAN_FILENAMES = {"build.zig", "build.zig.zon"}


def iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in SCAN_FILENAMES or p.suffix in SCAN_EXTENSIONS:
            out.append(p)
    return out


def _md_suppress_line(
    line: str,
    in_code_block: bool,
    block_is_stale: bool,
) -> bool:
    """Return True if this markdown line should be skipped by the grepper."""
    stripped = line.lstrip()
    # Table rows: documentation, never real code.
    if stripped.startswith("|"):
        return True
    # Anti-example framing on the same line.
    if MD_FRAMING_RE.search(line):
        return True
    # Inside a fenced code block whose header / first comment says WRONG/stale.
    if in_code_block and block_is_stale:
        return True
    return False


def scan_file(path: Path, ignore_md: bool) -> list[dict[str, object]]:
    is_md = path.suffix == ".md"
    if is_md and ignore_md:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    hits: list[dict[str, object]] = []

    # Track markdown fenced-code context.
    in_code_block = False
    block_is_stale = False
    # Capture up to 4 lines of context inside a fresh code block to decide
    # whether it is a "WRONG" block.
    context_lines_left = 0

    for lineno, line in enumerate(text.splitlines(), start=1):
        if is_md:
            # Fence toggling.
            if line.lstrip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    block_is_stale = False
                    context_lines_left = 4
                    # Check header line for WRONG marker.
                    if MD_FRAMING_RE.search(line):
                        block_is_stale = True
                else:
                    in_code_block = False
                    block_is_stale = False
                    context_lines_left = 0
                continue

            # Inside a fresh code block, sniff the first few lines for
            # a WRONG-comment that marks the block as anti-example.
            if in_code_block and context_lines_left > 0:
                if MD_FRAMING_RE.search(line):
                    block_is_stale = True
                context_lines_left -= 1

            if _md_suppress_line(line, in_code_block, block_is_stale):
                continue

        for name, regex, ref in PATTERNS:
            m = regex.search(line)
            if m:
                hits.append({
                    "file": str(path),
                    "line": lineno,
                    "col": m.start() + 1,
                    "pattern": name,
                    "suggest_ref": ref,
                    "snippet": line.strip()[:200],
                })
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag stale Zig 0.13-era patterns.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of human-readable text")
    parser.add_argument("--ignore-md", action="store_true",
                        help="skip .md files (useful when scanning only code)")
    args = parser.parse_args()

    all_hits: list[dict[str, object]] = []
    for p in args.paths:
        if not p.exists():
            sys.stderr.write(f"WARN: {p} does not exist\n")
            continue
        for f in iter_files(p):
            all_hits.extend(scan_file(f, args.ignore_md))

    if args.json:
        json.dump(all_hits, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for h in all_hits:
            print(f"{h['file']}:{h['line']}:{h['col']}  -  {h['pattern']}  "
                  f"(-> {h['suggest_ref']})")
            print(f"    {h['snippet']}")
        if not all_hits:
            print("no stale patterns found", file=sys.stderr)

    return 1 if all_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
