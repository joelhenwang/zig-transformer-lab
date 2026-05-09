#!/usr/bin/env python3
"""detect_zig_version.py — detect the installed Zig toolchain version.

Runs `zig version` and emits JSON to stdout:

    {"version": "0.16.0", "is_0_16": true, "warning": null}

Exits 0 if Zig is 0.16.x.
Exits 2 if Zig is reachable but the version is not 0.16.x.
Exits 3 if Zig is not on PATH.

Usage:
    python scripts/detect_zig_version.py
    python scripts/detect_zig_version.py --quiet   # JSON only, no stderr

Standard library only; no third-party dependencies.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

EXPECTED_MAJOR_MINOR = (0, 16)
VERSION_RE = re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)(?:[-+][A-Za-z0-9.]+)?\s*$")


def detect() -> dict[str, object]:
    if shutil.which("zig") is None:
        return {
            "version": None,
            "is_0_16": False,
            "warning": "zig is not on PATH",
            "exit_code": 3,
        }
    try:
        proc = subprocess.run(
            ["zig", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as ex:  # noqa: BLE001
        return {
            "version": None,
            "is_0_16": False,
            "warning": f"failed to run zig version: {ex}",
            "exit_code": 3,
        }
    raw = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    match = VERSION_RE.match(raw)
    if match is None:
        return {
            "version": raw or None,
            "is_0_16": False,
            "warning": f"could not parse zig version output: {raw!r}",
            "exit_code": 2,
        }
    major, minor, _patch = map(int, match.groups())
    is_ok = (major, minor) == EXPECTED_MAJOR_MINOR
    return {
        "version": raw,
        "is_0_16": is_ok,
        "warning": None if is_ok else (
            f"expected 0.16.x, got {raw}. This skill targets Zig 0.16.0."
        ),
        "exit_code": 0 if is_ok else 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect Zig toolchain version.")
    parser.add_argument("--quiet", action="store_true",
                        help="only emit JSON on stdout; no stderr messages")
    args = parser.parse_args()

    info = detect()
    exit_code = int(info.pop("exit_code"))  # type: ignore[arg-type]
    json.dump(info, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if not args.quiet and info.get("warning"):
        sys.stderr.write(f"WARN: {info['warning']}\n")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
