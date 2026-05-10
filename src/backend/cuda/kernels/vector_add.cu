//
// zig-transformer-lab — vector_add CUDA kernel (Stage 7, PR-zeta)
//
// Purpose:
//   Minimal smoke-test kernel. Adds two f32 vectors elementwise into a
//   third. This is deliberately the smallest possible kernel that
//   exercises the end-to-end build + load + launch pipeline introduced
//   in PR-zeta.
//
//   Later PRs add real op kernels (elementwise, softmax, layernorm,
//   matmul, embedding, cross-entropy, adamw). Those kernels follow
//   the same structure this file establishes:
//     - `extern "C"` to keep the symbol name unmangled so
//       cuModuleGetFunction can find it by its literal name.
//     - `__global__ void ...` — CUDA device kernel entry point.
//     - Explicit bounds check at the top (`if (i >= n) return;`) per
//       the Stage 7 policy in AGENTS.md / docs/stage7_plan.md.
//
// Build:
//   Compiled by build.zig's `kernels` step via:
//     nvcc -O3 -arch=sm_89 -ptx -Xcompiler -fPIC -o
//       zig-out/ptx/vector_add.ptx src/backend/cuda/kernels/vector_add.cu
//   On sm_89 (RTX 4060 Ti, Ada Lovelace) f32 add is a plain FADD; no
//   tensor cores or FMA fusion, so results are bit-identical to the
//   CPU reference for non-denormal inputs.
//
// Invocation:
//   Launched from src/backend/cuda/module.zig via cuLaunchKernel with
//   four arguments packed into kernelParams:
//     a, b: const float*
//     c:    float*
//     n:    unsigned int
//   Grid: ((n + 255) / 256, 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: 0
//

extern "C" __global__
void vector_add(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}
