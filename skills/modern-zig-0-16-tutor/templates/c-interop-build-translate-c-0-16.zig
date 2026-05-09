//! target: Zig 0.16.0 — illustrative, verify with `zig build run` on 0.16.0.
//!
//! C interop via modern build-system translation (REPLACES @cImport).
//!
//!   - `b.addTranslateC({...}).createModule()` turns a C header into a Zig
//!     module.
//!   - `exe.root_module.addImport("c", ...)` exposes it as `@import("c")`.
//!   - `linkSystemLibrary` calls move onto the TranslateC step, not the exe.
//!   - `link_libc = true` is set on the module (either on the TranslateC
//!     step or on the consumer exe's root module, usually both is harmless).
//!
//! Layout:
//!   .
//!   ├── build.zig
//!   ├── c/
//!   │   └── c_api.h            (your single C surface; #includes other headers)
//!   └── src/
//!       └── main.zig

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Translate the consolidated C header into a Zig module.
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("c/c_api.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Attach any system libraries to the TranslateC step (NOT the exe):
    // translate_c.linkSystemLibrary("z", .{});
    // translate_c.linkSystemLibrary("glfw", .{});

    // Consumer executable imports the translated C as @import("c").
    const exe = b.addExecutable(.{
        .name = "c-interop",
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
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the C-interop exe").dependOn(&run.step);
}

// TODO(learner):
//   1. Create c/c_api.h with a single #include <stdio.h>.
//   2. In src/main.zig:
//        const std = @import("std");
//        const c = @import("c");
//        pub fn main(init: std.process.Init) !void {
//            _ = init;
//            _ = c.puts("hello from translate-c");
//        }
//   3. `zig build run`. Verify it prints.
//   4. Add #include <math.h> and call c.sqrt(2.0).
//   5. Compare to the DEPRECATED @cImport approach — read
//      references/09-c-interop-0-16.md for the side-by-side.
