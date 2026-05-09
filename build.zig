const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda = b.option(bool, "cuda", "Enable CUDA backend compilation") orelse false;
    const cuda_arch = b.option([]const u8, "cuda_arch", "nvcc -arch flag (default: sm_89)") orelse "sm_89";
    const cuda_home = b.option([]const u8, "cuda_home", "Path to CUDA toolkit root") orelse "";
    const example_name = b.option([]const u8, "example", "Name of example to run (without .zig extension)") orelse "";
    const seed = b.option(u64, "seed", "Random seed for examples") orelse 1337;

    // Silence unused-variable warning until examples use the seed
    _ = seed;

    // --- Library module ---
    // The single module that all examples and tests import.
    // As files are added in later stages, src/root.zig re-exports them.
    const lib_mod = b.addModule("zig_transformer_lab", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // --- Test step ---
    // Runs all unit tests. We use src/root.zig as the test root because
    // Zig's test runner only discovers `test` blocks in files that belong
    // to the SAME module as the test root. Since root.zig already imports
    // every sub-module, its `test { _ = ...; }` block pulls in all tests.
    const test_step = b.step("test", "Run all tests");

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    // CUDA integration tests: only compiled and run when -Dcuda=true
    if (cuda) {
        const cuda_test_mod = b.createModule(.{
            .root_source_file = b.path("tests/integration_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig_transformer_lab", .module = lib_mod },
            },
        });
        const cuda_tests = b.addTest(.{
            .root_module = cuda_test_mod,
        });
        linkCudaRuntime(b, cuda_tests, cuda_home);
        const run_cuda_tests = b.addRunArtifact(cuda_tests);
        test_step.dependOn(&run_cuda_tests.step);
    }

    // --- Kernel compilation step ---
    // Compiles each .cu file under src/backend/cuda/kernels/ to zig-out/ptx/<name>.ptx
    // using nvcc. Only runs when -Dcuda=true. When no .cu files exist yet, this is
    // effectively a no-op because the kernel list is empty.
    const kernel_step = b.step("kernels", "Compile CUDA .cu kernels to PTX (only when -Dcuda=true)");
    if (cuda) {
        buildKernels(b, kernel_step, cuda_arch, cuda_home);
    }

    // --- Example runner ---
    // Usage: zig build run-example -Dexample=01_tensor_playground
    // Resolves to examples/<name>.zig, compiles it, and runs it.
    if (example_name.len > 0) {
        const example_exe = b.addExecutable(.{
            .name = example_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("examples/{s}.zig", .{example_name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zig_transformer_lab", .module = lib_mod },
                },
            }),
        });
        if (cuda) linkCudaRuntime(b, example_exe, cuda_home);
        b.installArtifact(example_exe);

        const run_step = b.step("run-example", "Run the named example");
        const run_cmd = b.addRunArtifact(example_exe);
        run_step.dependOn(&run_cmd.step);
    }

    // --- Docs check step ---
    // Prints line counts for each docs/*.md chapter. Used to verify
    // each chapter meets the 500-line minimum.
    const docs_step = b.step("docs", "Print docs chapter line counts");
    addDocsCheck(b, docs_step);
}

/// Links libc and libdl (for dlopen), and adds the CUDA toolkit library path
/// so that libcublas.so can be found at runtime via dlopen.
/// Does NOT link libcuda directly — that library is opened at runtime via dlopen.
fn linkCudaRuntime(b: *std.Build, exe: *std.Build.Step.Compile, cuda_home: []const u8) void {
    exe.root_module.linkSystemLibrary("c", .{});
    exe.root_module.linkSystemLibrary("dl", .{});

    // Add CUDA toolkit lib path so the runtime linker can find libcublas.so
    const lib_dir_path = if (cuda_home.len > 0)
        b.fmt("{s}/targets/x86_64-linux/lib", .{cuda_home})
    else
        "/usr/local/cuda/targets/x86_64-linux/lib";

    const lib_dir: std.Build.LazyPath = .{ .cwd_relative = lib_dir_path };
    exe.root_module.addLibraryPath(lib_dir);
    exe.root_module.addRPath(lib_dir);
}

/// List of CUDA kernel .cu files to compile.
/// As new kernels are added in Stage 7, add their names here.
/// Each entry is the filename stem (without .cu extension).
const kernel_names = [_][]const u8{
    // Stage 7 will add:
    // "elementwise",
    // "softmax",
    // "layernorm",
    // "gelu",
    // "embedding",
    // "causal_mask",
    // "ce_loss",
    // "adamw",
};

/// Creates nvcc compile commands for each kernel in kernel_names,
/// producing zig-out/ptx/<name>.ptx for each one.
fn buildKernels(b: *std.Build, kernel_step: *std.Build.Step, cuda_arch: []const u8, cuda_home: []const u8) void {
    const nvcc_path = if (cuda_home.len > 0)
        b.fmt("{s}/bin/nvcc", .{cuda_home})
    else
        "nvcc";

    // Create the PTX output directory via a mkdir command
    const mkdir_cmd = b.addSystemCommand(&.{
        "mkdir", "-p", "zig-out/ptx",
    });
    kernel_step.dependOn(&mkdir_cmd.step);

    for (kernel_names) |name| {
        const cu_src = b.fmt("src/backend/cuda/kernels/{s}.cu", .{name});
        const ptx_out = b.fmt("zig-out/ptx/{s}.ptx", .{name});
        const arch_flag = b.fmt("-arch={s}", .{cuda_arch});

        const cmd = b.addSystemCommand(&.{
            nvcc_path,
            "-O3",
            arch_flag,
            "-ptx",
            "--use_fast_math",
            "-Xcompiler",
            "-fPIC",
            "-o",
            ptx_out,
            cu_src,
        });
        cmd.setName(b.fmt("nvcc compile {s}.cu", .{name}));
        // Each kernel compile depends on the mkdir, and the kernel step
        // depends on each compile.
        cmd.step.dependOn(&mkdir_cmd.step);
        kernel_step.dependOn(&cmd.step);
    }
}

/// Adds a step that prints line counts for each docs/0X_*.md chapter.
/// Used to verify documentation completeness at the end of each stage.
fn addDocsCheck(b: *std.Build, docs_step: *std.Build.Step) void {
    const check = b.addSystemCommand(&.{
        "sh", "-c",
        \\echo "=== docs/ line counts ===" && \
        \\for f in docs/0*.md; do [ -f "$f" ] && wc -l "$f"; done || true
    });
    docs_step.dependOn(&check.step);
}
