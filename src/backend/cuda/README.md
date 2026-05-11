# `src/backend/cuda/`

NVIDIA CUDA backend for the zig-transformer-lab. Compiles offline with
nvcc to `.ptx`, loaded at runtime via `cuModuleLoadData`, invoked
through dlopen'd `libcuda.so.1` and `libcublas.so.13`. No link-time
dependency on the CUDA toolkit — the binary loads on any machine and
probes for CUDA at runtime.

## Files

- `bindings.zig` — dlopen-based loader for the ~30 CUDA Driver API
  and cuBLAS entry points we use. Resolved once on first context
  creation; subsequent calls are indirect-call fast.
- `context.zig` — `CudaContext`: device, context, stream, cuBLAS
  handle, PTX module cache. One per training run, heap-allocated so
  the `*const CudaContext` pointer stored in every `DeviceBuffer`
  remains valid.
- `mem.zig` — `DeviceBuffer`: RAII over `cuMemAlloc_v2`. `alloc`,
  `fromHost` (alloc + HtoD), `copyFromHost`, `copyToHost`,
  `copyFromDevice`, `deinit`.
- `module.zig` — PTX file loader. Reads `zig-out/ptx/<stem>.ptx` off
  disk, null-terminates it, hands it to `cuModuleLoadData`, caches
  the resulting `CUmodule` by stem. Kernel resolution via
  `cuModuleGetFunction`.
- `dispatch.zig` — host-side dispatch functions. Every CUDA op in
  this library ultimately calls a function here. Wraps kernel
  launches with argument packing, grid/block math, and error-checked
  `cuLaunchKernel` calls.
- `gemm.zig` — cuBLAS GEMM wrappers. Implements the row-major →
  column-major operand-swap trick so callers can pass row-major
  tensors and get row-major results.
- `kernels/` — source `.cu` files compiled offline to
  `zig-out/ptx/*.ptx` via nvcc. Listed below.

## Kernels

- `vector_add.cu` — smoke-test kernel (PR-ζ).
- `elementwise.cu` — same-shape and broadcast elementwise ops:
  add/sub/mul/div/neg/addScalar/mulScalar plus rank-4 stride-aware
  broadcast variants.
- `reduce.cu` — `reduce_sum_all` (atomicAdd), `reduce_sum_axis`,
  `bcast_copy` (rank-4 stride-aware gather for broadcastTo).
- `softmax.cu` — `softmax_last` and `logsoftmax_last` (last-axis
  reductions with numerical stabilisation).
- `embedding.cu` — `embedding_forward` (gather) and
  `embedding_backward` (scatter-add with atomic).
- `unary.cu` — `unary_gelu_exact` + backward, `unary_sqrt`,
  `unary_exp`, `unary_log`.
- `ce_loss.cu` — `ce_fused`, the fused forward + backward-logits
  kernel that produces both loss and `grad_logits` in one launch.
- `adamw.cu` — `adamw_step`, the per-parameter AdamW update.

## If you're new here

Read `docs/08_backends_cuda.md` end to end. The row-major /
column-major discussion in §3-§4 is the single most important idea in
this folder. Then the bridge chapter `docs/08b_from_cuda_to_training.md`
for what it means to train on GPU.

## Cross-references

- Debug workflow: `docs/09_debugging.md` §§6-8
- CUDA-specific compilation gotchas: `AGENTS.md` §"CUDA sacred spots"
- Stage 7 plan (as executed): `docs/stage7_plan.md`
