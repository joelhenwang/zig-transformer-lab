//! target: Zig 0.16.0 — illustrative, verify with `zig build run` and
//! `zig build test` on 0.16.0.
//!
//! Canonical executable build.zig. Demonstrates the modern 0.16.0 shape:
//!   - `root_module = b.createModule({...})`
//!   - `target`, `optimize`, `link_libc` live on the module
//!   - `b.installArtifact(exe)` (not `exe.install()`)
//!   - `b.addRunArtifact(exe)` for the "run" step
//!   - `b.addTest(...)` + `b.addRunArtifact(tests)` for the "test" step
//!
//! Project layout this build.zig expects:
//!   .
//!   ├── build.zig            (this file)
//!   ├── build.zig.zon        (see TODO below)
//!   └── src/
//!       └── main.zig

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- executable ---
    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // --- run step ---
    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the app").dependOn(&run.step);

    // --- test step ---
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

// TODO(learner): add a sibling `build.zig.zon`:
//   .{
//       .name = .app,
//       .version = "0.1.0",
//       .minimum_zig_version = "0.16.0",
//       .fingerprint = 0x1122334455667788,
//       .dependencies = .{},
//       .paths = .{""},
//   }
