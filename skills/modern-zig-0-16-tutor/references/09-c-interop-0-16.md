# 09 — C interop (0.16.0)

Zig's C interop is first-class: ABI-compatible layout, `extern` declarations,
`export` to produce a Zig library callable from C, and **build-system-driven
translation** of C headers. `@cImport` still parses but is **deprecated** in
0.16 — new code should use `b.addTranslateC` in `build.zig`.

## C primitive types

| Zig name | C meaning |
|---|---|
| `c_char` | `char` |
| `c_short` / `c_ushort` | `(unsigned) short` |
| `c_int` / `c_uint` | `(unsigned) int` |
| `c_long` / `c_ulong` | `(unsigned) long` |
| `c_longlong` / `c_ulonglong` | `(unsigned) long long` |
| `c_longdouble` | `long double` |
| `anyopaque` | `void` (for type-erased pointers) |
| `[*:0]const u8` | `const char *` when it is a C-string |
| `[*c]T` | C pointer (accept from translate-c; do not construct) |

## Declaring external functions

```zig
pub extern "c" fn puts(s: [*:0]const u8) c_int;
pub extern "c" fn printf(format: [*:0]const u8, ...) c_int;

// With a non-"c" library symbol namespace:
pub extern "libfoo" fn foo_do(x: c_int) c_int;
```

**`callconv(.c)`** — lowercase. `callconv(.C)` was removed.

```zig
export fn my_add(a: c_int, b: c_int) c_int {   // implicitly callconv(.c)
    return a + b;
}

fn fast_path(x: c_int) callconv(.c) c_int {
    return x * 2;
}
```

## `@cImport` — deprecated but still works

```zig
// DEPRECATED in 0.16; prefer build-system translation (see below)
const c = @cImport({
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("stdio.h");
});

pub fn main(init: std.process.Init) !void {
    _ = init;
    _ = c.puts("hello");
}
```

If you use `@cImport`, you still need `link_libc = true` on the module.

## Modern 0.16: build-system translation

In `src/main.zig`:

```zig
const c = @import("c");   // a translated-C module

pub fn main(init: std.process.Init) !void {
    _ = init;
    _ = c.puts("hello via translate-c");
}
```

In `c/api.h`:

```c
#include <stdio.h>
// Add any other headers your program needs.
```

In `build.zig`:

```zig
const translate_c = b.addTranslateC(.{
    .root_source_file = b.path("c/api.h"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});

const exe = b.addExecutable(.{
    .name = "app",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = translate_c.createModule() },
        },
    }),
});

b.installArtifact(exe);
```

**Mentor note.** Advantages of the build-system form:
- IDE tooling (clangd, ZLS) can see the real `.h` file.
- `linkSystemLibrary` and include paths attach to the `TranslateC` step.
- Translated output is identical to `@cImport`.

## Linking system libraries

```zig
exe.root_module.linkSystemLibrary("z", .{});       // -lz
exe.root_module.linkSystemLibrary("cuda", .{});    // libcuda (NVIDIA driver)
exe.root_module.linkSystemLibrary("cudart", .{});  // CUDA runtime
```

Add include paths for headers not in the default search set:

```zig
exe.root_module.addIncludePath(b.path("third_party/cuda/include"));
// or a system path at configure time:
exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
```

## Linking a C source file

```zig
exe.root_module.link_libc = true;
exe.root_module.addCSourceFile(.{
    .file = b.path("c/glue.c"),
    .flags = &.{ "-std=c99", "-Wall", "-Wextra" },
});
exe.root_module.addIncludePath(b.path("c/include"));
```

## Exporting Zig for C callers

```zig
// mathtest.zig
export fn mt_add(a: c_int, b: c_int) c_int {
    return a + b;
}
```

```zig
// build.zig
const lib = b.addLibrary(.{
    .linkage = .dynamic,
    .name = "mathtest",
    .root_module = b.createModule(.{
        .root_source_file = b.path("mathtest.zig"),
        .target = target,
        .optimize = optimize,
    }),
    .version = .{ .major = 1, .minor = 0, .patch = 0 },
});
b.installArtifact(lib);
```

The resulting shared library exposes `mt_add` with the standard C ABI.

## Calling through `[*c]T` (C pointers)

- Only consume; do not synthesize. Translate-c produces them.
- Dereference a C pointer to a struct as `p.*.field` (analogous to `->`).
- Prefer converting to `*T` or `[*]T` early if you can prove non-null /
  lengths.

## CUDA-relevant notes (zig-side; full story in ref 17)

- You will typically point `translate_c` at a small `cuda_api.h` that
  `#include`s `<cuda.h>` (Driver API) or `<cuda_runtime.h>` (Runtime API).
- `linkSystemLibrary("cuda", .{})` for the driver stub;
  `linkSystemLibrary("cudart", .{})` for the runtime.
- Include path: NVIDIA's `include/` under `CUDA_PATH`.
- Kernels (`.cu`) are compiled **outside** `zig build` (with `nvcc`) and
  linked as a `.lib`/`.a`/`.so` via `linkLibrary` / `linkSystemLibrary`.
  Zig 0.16 has no native NVPTX pipeline for general kernels.

## Common review findings

- `@cImport(...)` in new code — SUGGEST moving to `b.addTranslateC`.
- `callconv(.C)` (uppercase) — FLAG.
- `exe.linkSystemLibrary("...")` on the artifact — FLAG; move to
  `exe.root_module.linkSystemLibrary("...", .{})`.
- Missing `link_libc = true` when using libc headers — FLAG.
- Manually constructed `[*c]T` — FLAG hard.
- `@as([*c]u8, ...)` — FLAG; usually the code wants `[*]u8` or `[]u8`.

## Common mentor diagnostic questions

- "Is this header from a system package or vendored? Is the include path set?"
- "Will you ship a single `.h` that `#include`s everything, or several?"
- "Is this ABI-stable? Which C types cross the boundary?"
- "Who owns the memory the C function returns? Does it say in the doc?"

<!-- ~2.8k tokens · verified against Zig 0.16.0 release notes + langref -->
