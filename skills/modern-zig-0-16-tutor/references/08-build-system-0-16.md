# 08 — Build system (0.16.0)

The canonical shape in 0.16 is **`root_module = b.createModule(...)`** on
every compile step. Target, optimize, `link_libc`, `addImport`,
`addCSourceFile`, `linkSystemLibrary` all live on the **module**, not on the
compile step. `linkLibrary` and `installArtifact` still live on the artifact.

## Canonical `build.zig`

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // Run step
    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the app").dependOn(&run.step);

    // Test step
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run unit tests").dependOn(&run_tests.step);
}
```

## Imports (Zig-to-Zig modules)

```zig
const util_mod = b.createModule(.{
    .root_source_file = b.path("src/util.zig"),
});

exe.root_module.addImport("util", util_mod);
// then in src/main.zig:  const util = @import("util");
```

`exe.addModule(...)` is **stale**.

## Linking

```zig
// Another Zig-built library:
exe.root_module.linkLibrary(lib_compile_step);

// System library (pkg-config / search path):
exe.root_module.linkSystemLibrary("z", .{});

// libc:
// Option A — set on the module:
const mod = b.createModule(.{
    .root_source_file = b.path("src/main.zig"),
    .target = target, .optimize = optimize,
    .link_libc = true,
});
// Option B — on the module method:
exe.root_module.link_libc = true;
```

## C source files

```zig
exe.root_module.addCSourceFile(.{
    .file = b.path("c/glue.c"),
    .flags = &.{"-std=c99", "-Wall"},
});
exe.root_module.addIncludePath(b.path("include"));
```

## Libraries

```zig
const lib = b.addLibrary(.{
    .name = "mylib",
    .linkage = .static,   // or .dynamic
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    }),
    .version = .{ .major = 0, .minor = 1, .patch = 0 },
});
b.installArtifact(lib);
```

`b.addStaticLibrary` and `b.addSharedLibrary` are **stale** — both are now
`b.addLibrary(.{ .linkage = ... })`.

## C translation (replaces `@cImport`)

```zig
const translate_c = b.addTranslateC(.{
    .root_source_file = b.path("c/c_api.h"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});
translate_c.linkSystemLibrary("z", .{});

exe.root_module.addImport("c", translate_c.createModule());
// Then in Zig: const c = @import("c");
```

> VERIFY WITH ZIG 0.16.0 LOCALLY whether `translate_c.linkSystemLibrary(...)`
> makes `exe.root_module.link_libc = true` redundant. In the example above
> we set both; redundancy is harmless.

## Build options (conditional compilation)

```zig
const version = b.option([]const u8, "version", "semver") orelse "0.0.0";
const opts = b.addOptions();
opts.addOption([]const u8, "version", version);
opts.addOption(bool, "use_cuda", false);

exe.root_module.addOptions("config", opts);
// In Zig:  const config = @import("config");
```

## Cross-target / release matrix

```zig
const targets: []const std.Target.Query = &.{
    .{ .cpu_arch = .x86_64,  .os_tag = .linux, .abi = .musl },
    .{ .cpu_arch = .aarch64, .os_tag = .macos },
    .{ .cpu_arch = .x86_64,  .os_tag = .windows },
};

for (targets) |q| {
    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.resolveTargetQuery(q),
            .optimize = .ReleaseSafe,
        }),
    });
    const inst = b.addInstallArtifact(exe, .{
        .dest_dir = .{ .override = .{ .custom = try q.zigTriple(b.allocator) } },
    });
    b.getInstallStep().dependOn(&inst.step);
}
```

## `build.zig.zon` (0.16 requires `fingerprint`)

```zig
.{
    .name = .my_project,           // enum literal in 0.16
    .version = "0.1.0",
    .minimum_zig_version = "0.16.0",
    .fingerprint = 0x1122334455667788,
    .dependencies = .{},
    .paths = .{""},
}
```

## Dependencies

Packages are fetched into a **project-local** `zig-pkg/` in 0.16, not the
global cache. Use `zig fetch --save=<name> <url>` to pin and download.

## Unit test timeouts (0.16)

```
zig build test --test-timeout 500ms
```

Kills individual tests that exceed the budget. Handy in CI.

## Stale-pattern table (flag in user build.zig)

| Stale | Modern 0.16 |
|---|---|
| `std.build.Builder` | `std.Build` |
| `FileSource` | `std.Build.LazyPath` via `b.path(...)` |
| `std.zig.CrossTarget` | `std.Target.Query` via `b.resolveTargetQuery(...)` |
| `b.addExecutable(.{ .root_source_file = ..., .target = ..., .optimize = ... })` | `.root_module = b.createModule({...})` |
| `b.addStaticLibrary(...)` / `b.addSharedLibrary(...)` | `b.addLibrary(.{ .linkage = .static/.dynamic })` |
| `exe.addModule("x", m)` / `exe.addPackage(...)` | `exe.root_module.addImport("x", m)` |
| `exe.linkSystemLibrary("z")` | `exe.root_module.linkSystemLibrary("z", .{})` |
| `exe.linkLibC()` | `exe.root_module.link_libc = true` |
| `exe.addCSourceFile(...)` | `exe.root_module.addCSourceFile(.{ .file, .flags })` |
| `exe.install()` | `b.installArtifact(exe)` |
| `exe.run()` | `b.addRunArtifact(exe)` |
| `@cImport` in source | `b.addTranslateC({...}).createModule()` + `addImport("c", m)` |

## Common mentor diagnostic questions

- "Is this option on the module or on the artifact? `root_module` is the
  module, `exe` is the artifact."
- "Where is the `-lc` / `-lm` flag coming from? Is `link_libc` set on the
  right level?"
- "Is this a `LazyPath` (runtime-built file) or a source file on disk?"
- "Have you set `minimum_zig_version` in `build.zig.zon`?"

<!-- ~2.8k tokens · verified against Zig 0.16.0 release notes + build guide -->
