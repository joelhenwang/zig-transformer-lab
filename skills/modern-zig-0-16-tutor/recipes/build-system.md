# Recipes: build system

## Recipe 10 — Simple executable with modern `build.zig`

**Problem.** Start a Zig 0.16 project: one source file, one executable,
`zig build` just works, `zig build run` runs it.

**Mentor explanation.** The 0.16 shape is `root_module = b.createModule(...)`
on every compile step. Target and optimize live on the module, not on the
executable.

**Minimal snippet (`build.zig`).**
```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "hello",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the app").dependOn(&run.step);
}
```

**Minimal `src/main.zig`.**
```zig
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    try std.Io.File.stdout().writeStreamingAll(init.io, "Hello, build system!\n");
}
```

**Stale pattern to avoid.**
```zig
// WRONG
const exe = b.addExecutable(.{
    .name = "hello",
    .root_source_file = .{ .path = "src/main.zig" },
    .target = target,
    .optimize = optimize,
});
exe.install();
```

**Exercise.** Run `zig build -Doptimize=ReleaseSafe`. Which files change
under `zig-out/`? Why is `Debug` the default for `zig build`?

---

## Recipe 11 — Add a test step

**Problem.** Add `zig build test` to run your unit tests.

**Mentor explanation.** Create a `Test` compile step that shares the same
source file, wrap it in `b.addRunArtifact`, and attach to a `"test"` step.

**Minimal snippet.** Append to the `build.zig` above:
```zig
const tests = b.addTest(.{
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    }),
});
const run_tests = b.addRunArtifact(tests);
b.step("test", "Run unit tests").dependOn(&run_tests.step);
```

**Stale pattern to avoid.**
```zig
// WRONG
const tests = b.addTest("src/main.zig");
tests.install();
```

**Exercise.** Add `--test-timeout 500ms` to your CI command. Force a test
to exceed the timeout. What does the reporter say?

<!-- ~1.0k tokens -->
