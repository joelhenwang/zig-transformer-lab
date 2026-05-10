//
// zig-transformer-lab — elementwise CUDA kernels (Stage 7, PR-eta)
//
// Purpose:
//   Same-shape elementwise binary and unary kernels covering the ops
//   that pr-eta wires into the Zig dispatch layer: add / sub / mul /
//   div / neg / add_scalar / mul_scalar. All kernels operate over a
//   flat contiguous f32 buffer of length `n`.
//
// Layout contract:
//   Inputs and outputs are contiguous, row-major, same shape (no
//   broadcasting). Broadcasting lands in a later PR (PR-theta).
//
// Kernel ABI:
//   All kernels are `extern "C"` so cuModuleGetFunction resolves them
//   by their literal name. Every kernel starts with the standard
//   bounds check:
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n) return;
//   This is the Stage 7 policy (AGENTS.md hard rules) and the reason
//   the dispatch layer can safely round grid size up to the next
//   multiple of block size.
//
// Scalar ops:
//   add_scalar / mul_scalar take the scalar by value (as f32), not by
//   pointer. cuLaunchKernel's kernelParams array reads sizeof(f32)
//   bytes for the scalar slot, matching the kernel's f32 parameter.
//
// Why no fused kernel for e.g. (a + b) * c?
//   PR-eta mirrors the CPU op set one-for-one to keep parity tests
//   tractable. Fusion is a Stage 9 performance optimisation.
//

extern "C" __global__
void elw_add(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

extern "C" __global__
void elw_sub(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] - b[i];
}

extern "C" __global__
void elw_mul(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] * b[i];
}

extern "C" __global__
void elw_div(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] / b[i];
}

extern "C" __global__
void elw_neg(const float* a, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = -a[i];
}

extern "C" __global__
void elw_add_scalar(const float* a, float s, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + s;
}

extern "C" __global__
void elw_mul_scalar(const float* a, float s, float* c, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] * s;
}
