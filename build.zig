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

    // --- Kernel compilation step ---
    // Compiles each .cu file under src/backend/cuda/kernels/ to zig-out/ptx/<name>.ptx
    // using nvcc. Only runs when -Dcuda=true AND the build target is Linux
    // (nvcc is not installed on our Windows dev host; CUDA work happens
    // on the remote RTX box per docs/stage7_plan.md). PR-zeta onward,
    // the CUDA test step depends on this so that `zig build test -Dcuda=true`
    // produces PTX automatically — the PTX loader in
    // src/backend/cuda/module.zig reads zig-out/ptx/<stem>.ptx at
    // runtime and would error.IoError without these files present.
    //
    // Declared BEFORE the CUDA test block so that block can attach a
    // dependency on `kernel_step`.
    const kernel_step = b.step("kernels", "Compile CUDA .cu kernels to PTX (only when -Dcuda=true)");
    const kernels_enabled = cuda and target.result.os.tag == .linux;
    if (kernels_enabled) {
        buildKernels(b, kernel_step, cuda_arch, cuda_home);
    }

    // CUDA integration tests: only compiled and run when -Dcuda=true.
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
        linkCudaRuntime(b, cuda_tests, target, cuda_home);
        const run_cuda_tests = b.addRunArtifact(cuda_tests);
        // Run from the repo root so the PTX loader's relative path
        // "zig-out/ptx/<stem>.ptx" resolves regardless of where the
        // user invoked `zig build`.
        run_cuda_tests.setCwd(b.path("."));
        // PR-zeta: tests depend on kernels. This makes the test
        // command self-sufficient: `zig build test -Dcuda=true`
        // compiles .cu -> .ptx first, then runs the integration tests
        // that load those .ptx files. Only wired on Linux — Windows
        // doesn't have nvcc and the CUDA tests SkipZigTest there.
        if (kernels_enabled) {
            run_cuda_tests.step.dependOn(kernel_step);
        }
        test_step.dependOn(&run_cuda_tests.step);
    }

    // --- Oracle parity tests ---
    // Compares our CPU ops against PyTorch-generated fixtures under
    // tests/fixtures/. The fixtures must be generated first with:
    //
    //   python tools/oracle.py generate
    //
    // The test step is separate (`zig build test-oracle`) rather than
    // folded into the main test step so that a fresh clone without
    // fixtures does not fail the default test run. Once fixtures are
    // present, `zig build test-oracle` exercises them.
    const oracle_test_step = b.step("test-oracle", "Run PyTorch oracle parity tests (requires tests/fixtures/)");
    const oracle_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration_oracle.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zig_transformer_lab", .module = lib_mod },
        },
    });
    const oracle_tests = b.addTest(.{
        .root_module = oracle_test_mod,
    });
    const run_oracle_tests = b.addRunArtifact(oracle_tests);
    // Run from the repo root so relative paths like "tests/fixtures/..."
    // resolve correctly regardless of where the user invoked the build.
    run_oracle_tests.setCwd(b.path("."));
    oracle_test_step.dependOn(&run_oracle_tests.step);

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
        if (cuda) linkCudaRuntime(b, example_exe, target, cuda_home);
        b.installArtifact(example_exe);

        const run_step = b.step("run-example", "Run the named example");
        const run_cmd = b.addRunArtifact(example_exe);
        run_step.dependOn(&run_cmd.step);
    }

    // --- Docs check step ---
    // Prints line counts for each docs/*.md chapter. Used to verify
    // each chapter meets the 500-line minimum. Cross-platform: walks the
    // filesystem via a tiny Zig helper executable rather than `sh -c wc -l`,
    // so it works on Windows, Linux, and macOS.
    const docs_step = b.step("docs", "Print docs chapter line counts");
    addDocsCheck(b, docs_step, target);
}

/// Links libc (and libdl on Linux for dlopen), and adds the CUDA toolkit library
/// path so that libcublas can be found at runtime via dlopen. Does NOT link the
/// driver library directly — that is opened at runtime via dlopen / LoadLibrary.
///
/// Platform notes:
///   - Linux: links libc + libdl, adds `<cuda_home>/targets/x86_64-linux/lib`.
///   - Windows: links libc, adds `<cuda_home>/lib/x64`. (Not yet functional —
///     Stage 7 is developed on Linux first. Any Windows CUDA attempt will
///     fail at runtime when it tries to dlopen `.so.1`, which is the
///     correct failure mode for now.)
///   - Other OSes: no changes; CUDA is not supported there.
fn linkCudaRuntime(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    cuda_home: []const u8,
) void {
    const os_tag = target.result.os.tag;

    exe.root_module.linkSystemLibrary("c", .{});
    if (os_tag == .linux) {
        exe.root_module.linkSystemLibrary("dl", .{});
    }

    // Pick a default toolkit path per OS. `cuda_home` overrides if provided.
    const lib_dir_path = if (cuda_home.len > 0) blk: {
        break :blk switch (os_tag) {
            .linux => b.fmt("{s}/targets/x86_64-linux/lib", .{cuda_home}),
            .windows => b.fmt("{s}/lib/x64", .{cuda_home}),
            else => b.fmt("{s}/lib", .{cuda_home}),
        };
    } else switch (os_tag) {
        .linux => "/usr/local/cuda/targets/x86_64-linux/lib",
        .windows => "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64",
        else => "/usr/local/cuda/lib",
    };

    const lib_dir: std.Build.LazyPath = .{ .cwd_relative = lib_dir_path };
    exe.root_module.addLibraryPath(lib_dir);
    // rpath is meaningful on Linux / macOS; harmless on Windows.
    exe.root_module.addRPath(lib_dir);
}

/// List of CUDA kernel .cu files to compile.
/// As new kernels are added in Stage 7, add their names here.
/// Each entry is the filename stem (without .cu extension).
const kernel_names = [_][]const u8{
    // Stage 7 additions (PR-zeta onward):
    "vector_add",
    "elementwise",
    "reduce",
    // Future PRs will add:
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
///
/// Linux-only (gated by the caller). Uses `mkdir -p` to ensure the
/// output directory exists before nvcc runs; nvcc does not create
/// parent directories on its own and would fail with "cannot write
/// output" otherwise.
///
/// NOTE: `--use_fast_math` is intentionally NOT passed. Stage 7 correctness
/// mode compares CUDA results against CPU element-wise within tight
/// tolerances; fast math would relax those. A future performance mode may
/// opt in via a separate `-Dcuda_fast_math=true` option.
fn buildKernels(b: *std.Build, kernel_step: *std.Build.Step, cuda_arch: []const u8, cuda_home: []const u8) void {
    if (kernel_names.len == 0) return;

    const nvcc_path = if (cuda_home.len > 0)
        b.fmt("{s}/bin/nvcc", .{cuda_home})
    else
        "nvcc";

    // Ensure the PTX output directory exists before any nvcc
    // invocation. `mkdir -p` is idempotent and every nvcc command
    // below depends on this step so it runs exactly once.
    const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", "zig-out/ptx" });
    mkdir_cmd.setName("mkdir -p zig-out/ptx");

    for (kernel_names) |name| {
        const cu_src = b.fmt("src/backend/cuda/kernels/{s}.cu", .{name});
        const ptx_out = b.fmt("zig-out/ptx/{s}.ptx", .{name});
        const arch_flag = b.fmt("-arch={s}", .{cuda_arch});

        const cmd = b.addSystemCommand(&.{
            nvcc_path,
            "-O3",
            arch_flag,
            "-ptx",
            "-Xcompiler",
            "-fPIC",
            "-o",
            ptx_out,
            cu_src,
        });
        cmd.setName(b.fmt("nvcc compile {s}.cu", .{name}));
        cmd.step.dependOn(&mkdir_cmd.step);
        kernel_step.dependOn(&cmd.step);
    }
}

/// Adds a step that prints line counts for each docs/0X_*.md chapter,
/// cross-platform. Uses a small Zig helper (`tools/docs_linecount.zig`)
/// compiled and run as part of the step. This replaces the previous
/// `sh -c ... wc -l ...` invocation that only worked on POSIX systems.
fn addDocsCheck(b: *std.Build, docs_step: *std.Build.Step, target: std.Build.ResolvedTarget) void {
    const exe = b.addExecutable(.{
        .name = "docs_linecount",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/docs_linecount.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    const run = b.addRunArtifact(exe);
    // Run from the project root so `docs/` is resolvable as a relative path.
    run.setCwd(b.path("."));
    docs_step.dependOn(&run.step);
}
