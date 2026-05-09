//! target: Zig 0.16.0 — illustrative, verify with `zig build` on 0.16.0.
//!
//! Canonical library build.zig in the 0.16 shape:
//!   - `b.addLibrary(.{ .linkage = .static | .dynamic })`
//!     (NOT `b.addStaticLibrary` / `b.addSharedLibrary`)
//!   - `.root_module = b.createModule({...})`
//!   - A consumer executable that links the library via
//!     `exe.root_module.linkLibrary(lib)`
//!
//! Layout:
//!   .
//!   ├── build.zig
//!   ├── src/
//!   │   ├── main.zig       (consumer exe)
//!   │   └── mylib.zig      (library root)

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- library ---
    const lib = b.addLibrary(.{
        .name = "mylib",
        .linkage = .static,   // swap to .dynamic if you want a .so / .dll
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mylib.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });
    b.installArtifact(lib);

    // --- consumer exe that links the lib ---
    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.linkLibrary(lib);
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    b.step("run", "Run the consumer exe").dependOn(&run.step);

    // --- library tests ---
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mylib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    b.step("test", "Run library tests").dependOn(&run_lib_tests.step);
}

// TODO(learner): expose the library's Zig API as a module consumers can
// import:
//   const mylib_mod = b.createModule(.{
//       .root_source_file = b.path("src/mylib.zig"),
//       .target = target, .optimize = optimize,
//   });
//   exe.root_module.addImport("mylib", mylib_mod);
// Then `const mylib = @import("mylib");` inside src/main.zig.
