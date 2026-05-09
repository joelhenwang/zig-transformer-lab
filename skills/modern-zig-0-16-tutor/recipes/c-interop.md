# Recipes: C interop

## Recipe 12 — Translate / import C headers through the build system

**Problem.** Call `puts` from a C library in your Zig program, using the
modern 0.16 build-system translation instead of deprecated `@cImport`.

**Mentor explanation.** Put your C includes in a real `.h` file. Let
`b.addTranslateC` turn it into a Zig module. Add the module as an import on
your executable's root module. Reference it in Zig as `@import("c")`.

**Minimal snippet — project layout.**
```
my-app/
├── build.zig
├── c/
│   └── api.h
└── src/
    └── main.zig
```

**`c/api.h`.**
```c
#include <stdio.h>
```

**`build.zig`.**
```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("c/api.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "c-hello",
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
}
```

**`src/main.zig`.**
```zig
const std = @import("std");
const c = @import("c");

pub fn main(init: std.process.Init) !void {
    _ = init;
    _ = c.puts("hello from C");
}
```

**Stale pattern to avoid.**
```zig
// WRONG
const c = @cImport({ @cInclude("stdio.h"); });
// build.zig: exe.linkLibC();
```

**Exercise.** Add `zlib`. Extend `c/api.h` with `#include <zlib.h>`; add
`translate_c.linkSystemLibrary("z", .{});`. Call `c.zlibVersion()` and
print it.

---

## Recipe 13 — Link a C source file

**Problem.** Write a small C helper (`c/add.c`) and call it from Zig.

**Mentor explanation.** `.c` sources attach to the **module** of the
compile step, not the compile step itself. `link_libc = true` is required
for almost any non-trivial C code.

**Minimal snippet — `c/add.h` and `c/add.c`.**
```c
// c/add.h
int my_add(int a, int b);
```
```c
// c/add.c
#include "add.h"
int my_add(int a, int b) { return a + b; }
```

**`build.zig`.**
```zig
const exe = b.addExecutable(.{
    .name = "adder",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }),
});

exe.root_module.addCSourceFile(.{
    .file = b.path("c/add.c"),
    .flags = &.{ "-std=c99", "-Wall" },
});
exe.root_module.addIncludePath(b.path("c"));

b.installArtifact(exe);
```

**`src/main.zig`.**
```zig
const std = @import("std");

pub extern "c" fn my_add(a: c_int, b: c_int) c_int;

pub fn main(init: std.process.Init) !void {
    var buf: [64]u8 = undefined;
    var fw = std.Io.File.stdout().writer(init.io, &buf);
    try fw.interface.print("2 + 3 = {d}\n", .{my_add(2, 3)});
    try fw.flush();
}
```

**Stale pattern to avoid.**
```zig
// WRONG — method on the artifact, not on the module
exe.addCSourceFile("c/add.c", &.{"-std=c99"});
exe.linkLibC();
```

**Exercise.** Add a second C file that depends on `add.h`. Confirm both
compile. What does `zig build --verbose` show under the hood?

<!-- ~1.3k tokens -->
