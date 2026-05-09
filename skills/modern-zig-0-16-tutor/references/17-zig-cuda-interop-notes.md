# 17 — Zig-side CUDA interop notes (0.16.0)

This file covers the **Zig side** of a Zig-plus-CUDA project. It is **not**
CUDA documentation. For CUDA itself (kernels, streams, occupancy, shared
memory, PTX), consult the recommended reading list at the bottom.

## Guiding principle

- **Zig owns host / runtime code.** Allocation, orchestration, graph
  building, I/O, configuration.
- **CUDA C (`.cu`) owns kernels**, at least initially. Zig 0.16 does not
  have a native general-purpose NVPTX pipeline for kernels.
- Link the two via the standard C ABI.

## Suggested project layout

```
my-runtime/
├── build.zig
├── build.zig.zon
├── include/
│   └── cuda_api.h                  # your curated C API surface
├── src/
│   ├── main.zig
│   ├── runtime.zig                 # Zig host-side runtime
│   └── cuda.zig                    # Zig wrapper around translate-c
├── kernels/
│   ├── matmul.cu                   # compiled with nvcc
│   ├── softmax.cu
│   └── CMakeLists.txt              # or a simple shell script
└── tests/
    ├── matmul_test.zig
    └── softmax_test.zig
```

Kernels compile out-of-band:

```
nvcc -O3 -arch=sm_80 -Xcompiler -fPIC \
     -c kernels/matmul.cu -o build/matmul.o
nvcc -shared -o build/libmlkernels.so build/matmul.o build/softmax.o
```

## Linking CUDA in `build.zig`

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zml",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    // CUDA headers
    exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });

    // Link driver + runtime libs
    exe.root_module.linkSystemLibrary("cuda", .{});     // libcuda (driver stub)
    exe.root_module.linkSystemLibrary("cudart", .{});   // libcudart (runtime)

    // Link your nvcc-built kernel library
    exe.root_module.addLibraryPath(.{ .cwd_relative = "build" });
    exe.root_module.linkSystemLibrary("mlkernels", .{});

    b.installArtifact(exe);
}
```

On Windows: use `CUDA_PATH` (`C:\Program Files\NVIDIA GPU Computing
Toolkit\CUDA\v...\include` / `\lib\x64`). Library names: `cuda.lib`,
`cudart.lib`.

## Translating CUDA headers

Create a small surface:

```c
// include/cuda_api.h
#include <cuda.h>          // Driver API
#include <cuda_runtime.h>  // Runtime API (optional; some projects pick one)
```

```zig
// build.zig — translate once, import everywhere
const tc = b.addTranslateC(.{
    .root_source_file = b.path("include/cuda_api.h"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});
tc.addSystemIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
exe.root_module.addImport("cuda", tc.createModule());
```

```zig
// src/cuda.zig
const cuda = @import("cuda");
```

## Driver API concepts (no kernel code shown)

The Driver API is thin and stable. Conceptually, a Zig wrapper wants to
expose these primitives:

- **Context**: `cuInit(0)`, `cuDeviceGet`, `cuCtxCreate`, `cuCtxDestroy`.
- **Module**: `cuModuleLoad` / `cuModuleLoadData` (load a `.ptx` or `.cubin`),
  `cuModuleGetFunction` (resolve a kernel by name).
- **Launch**: `cuLaunchKernel(function, gridDim, blockDim, sharedMem,
  stream, args, extra)`.
- **Memory**: `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH`,
  `cuMemcpyAsync`.
- **Stream**: `cuStreamCreate`, `cuStreamSynchronize`, `cuStreamDestroy`.
- **Event**: `cuEventCreate`, `cuEventRecord`, `cuEventElapsedTime`.

Zig-side wrapper pattern (skeleton — deliberately incomplete):

```zig
const cuda = @import("cuda");

pub const CudaError = error{
    CudaInvalidValue,
    CudaOutOfMemory,
    CudaLaunchFailed,
    CudaUnknown,
    // ... expand as needed
};

pub inline fn check(code: cuda.CUresult) CudaError!void {
    if (code == cuda.CUDA_SUCCESS) return;
    return switch (code) {
        cuda.CUDA_ERROR_INVALID_VALUE => error.CudaInvalidValue,
        cuda.CUDA_ERROR_OUT_OF_MEMORY => error.CudaOutOfMemory,
        cuda.CUDA_ERROR_LAUNCH_FAILED => error.CudaLaunchFailed,
        else => error.CudaUnknown,
    };
}
```

**Mentor note.** Wrap every call site in `try check(...)`. Do not let raw
CUDA return codes leak into business logic.

## Memory model reminders (applies to Zig wrappers)

- **Host allocations** with `std.mem.Allocator` are NOT directly accessible
  from the device.
- **Pinned (page-locked) host memory** via `cudaHostAlloc` / `cuMemAllocHost`
  accelerates H2D / D2H transfers but costs OS pages — allocate sparingly.
- **Device allocations** are opaque `CUdeviceptr` integers; treat them as
  typed handles in Zig (wrap in a `DeviceSlice(f32)` struct).
- **Unified memory** (`cuMemAllocManaged`) is a convenience, not a free
  lunch; it causes page faults under pressure.

## Streams and async

- Everything non-trivial goes through a stream.
- Work submitted to different streams may overlap on modern GPUs.
- `cuStreamSynchronize(stream)` is a **hard barrier** — you want to avoid
  it on the hot path.
- CUDA events measure elapsed time without a full synchronize.

## What not to do (in 0.16.0)

- **Do not try to emit PTX from Zig directly** for production kernels. The
  `nvptx64-cuda` target is listed under "Additional Platforms"; it is not a
  complete kernel-authoring pipeline in 0.16.
- **Do not call into cuBLAS / cuDNN via `@cImport`.** Use build-system
  translation of `cublas_v2.h` etc., same pattern as above.
- **Do not store a raw `CUdeviceptr` as a Zig pointer type.** It is a
  device address, not a host address; dereferencing it on the host is UB.

## Testing with CUDA

- Keep a **CPU reference** for every kernel. Test the CUDA path against it
  with `expectApproxEqAbs`.
- Gate GPU tests behind a build option:
  ```zig
  const cuda_enabled = b.option(bool, "cuda", "enable CUDA tests") orelse false;
  ```
- Skip tests with `return error.SkipZigTest;` when CUDA is not available.

## Recommended reading (external; not part of this skill)

- NVIDIA CUDA C Programming Guide — the canonical reference for the CUDA
  programming model.
- NVIDIA CUDA Runtime API Reference — for `cudart` functions.
- NVIDIA CUDA Driver API Reference — for `libcuda`.
- PTX ISA Reference — when you eventually inspect generated assembly.
- cuBLAS / cuDNN / cuFFT / NCCL reference manuals — for off-the-shelf BLAS
  / DNN / FFT / collective kernels.
- CUTLASS (github.com/NVIDIA/cutlass) — modern CUDA template patterns.
- NVIDIA HPC SDK documentation — for newer NVHPC-tooling workflows.
- **Not** TensorRT / PyTorch — those are not implementation dependencies
  for this project.

## Common mentor diagnostic questions

- "Do we have a CPU reference for this kernel? Have we tested against it?"
- "Which stream is this on? Can it overlap with something else?"
- "Is that slice device memory or host memory? Can you show the
  `DeviceSlice(T)` wrapper?"
- "Are errors being checked from every CUDA call?"
- "Is the kernel launch argument pack correct? (This is the #1 source of
  silent wrong-answer bugs.)"

<!-- ~2.8k tokens · Zig 0.16.0 side only; CUDA concepts not normative -->
