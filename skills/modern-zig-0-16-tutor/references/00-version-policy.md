# 00 — Version policy

**Default target: Zig 0.16.0.** Do not emit older patterns unless the user
explicitly asks for migration help.

## Detection ladder

Ordered most-authoritative to least:

1. **User-stated version.** If the user said "I am on 0.16.0" or "I'm still on
   0.13", use that.
2. **`zig version` output.** If the user pastes it or you can run it, trust it.
3. **`build.zig.zon`** — check for `minimum_zig_version`. In 0.16 this field
   is required alongside a `fingerprint`:
   ```zig
   .{
       .name = .my_project,
       .version = "0.1.0",
       .minimum_zig_version = "0.16.0",
       .fingerprint = 0x...,
       .dependencies = .{},
       .paths = .{""},
   }
   ```
4. **API markers in source code.** See the table below.

| Marker you see | Likely Zig era |
|---|---|
| `pub fn main(init: std.process.Init) !void` | 0.16+ |
| `std.Io.File.stdout()` | 0.16+ |
| `b.addTranslateC(...)` | 0.16+ |
| `std.heap.DebugAllocator` | 0.16+ |
| `@Int(.unsigned, 32)` / `@Struct(...)` | 0.16+ |
| `.empty` on a container + `.append(a, v)` | 0.15–0.16 |
| `std.io.getStdOut().writer()` | 0.10–0.14 |
| `@cImport({ @cInclude("..."); })` | 0.10–0.15 (works in 0.15, deprecated in 0.16) |
| `@Type(.{ .int = ... })` | 0.11–0.15 |
| `std.ArrayList(T).init(a)` + `append(v)` | 0.10–0.14 |
| `std.heap.GeneralPurposeAllocator(.{}){}` | 0.10–0.15 |
| `@intToPtr` / `@enumToInt` etc. | 0.10–0.11 |
| `usingnamespace` | 0.10–0.14 |
| `async` / `await` keywords | 0.9–0.10 |
| `std.build.Builder` | 0.9 and earlier |

## What to do when the user's code appears older

- **Do not silently rewrite.** Ask: "Are you targeting Zig 0.16.0? Some of
  this code looks like 0.13-era patterns. I can show a 0.16 migration diff,
  or we can keep 0.13 for now."
- If they pick 0.16: produce WRONG/CORRECT diffs, not a full rewrite.
  See `references/14-migration-from-0-13-to-0-16.md`.
- If they pick 0.13 or earlier: warn them the 0.16 stdlib is substantially
  different, this skill is tuned for 0.16, and defer to the official release
  notes of their target version.

## Minimum Zig version enforcement

When starting a new project, include `minimum_zig_version = "0.16.0"` in
`build.zig.zon`. `zig build` will refuse with a clear message if the user's
compiler is older. This is the cheapest guardrail against version drift.

## Running `zig version`

The user can verify their toolchain with:
```pwsh
zig version
```

Expected for this skill: `0.16.0` (or a 0.16.x patch release).

If the user reports `0.15.x` or `0.14.x`, offer to either help them upgrade
or explicitly switch to migration mode (ref 14).

<!-- ~1.2k tokens · verified against Zig 0.16.0 release notes -->
