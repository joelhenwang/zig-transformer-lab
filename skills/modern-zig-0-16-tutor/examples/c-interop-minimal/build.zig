//! target: Zig 0.16.0 — illustrative.
//!
//! Minimal C-interop project that translates a C header via the build
//! system (replaces @cImport) and calls a C function.
//!
//! Layout:
//!   .
//!   ├── build.zig
//!   ├── src/main.zig
//!   ├── c/
//!   │   ├── example.h
//!   │   └── example.c

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Translate the header into a Zig module.
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("c/example.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Consumer exe.
    const exe = b.addExecutable(.{
        .name = "c-interop-minimal",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "c", .module = translate_c.createModule() },
            },
        }),
    });

    // Compile the C source file into the exe.
    exe.root_module.addCSourceFile(.{
        .file = b.path("c/example.c"),
        .flags = &.{ "-std=c99", "-Wall", "-Wextra" },
    });
    exe.root_module.addIncludePath(b.path("c"));

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    b.step("run", "Run the demo").dependOn(&run.step);
}
