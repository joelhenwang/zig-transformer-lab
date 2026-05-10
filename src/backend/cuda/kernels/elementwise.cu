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

//
// Broadcast kernels (PR-theta)
// ---------------------------------------------------------------------------
//
// General-purpose stride-aware rank-4 elementwise kernels. The host
// prepares effective strides per input by right-aligning with the
// output shape and setting stride=0 for broadcasted (size-1 or
// missing) axes. The kernel decomposes the flat output index i into
// (i0, i1, i2, i3) against the output shape, then computes each
// input's physical offset via a dot product with that input's
// effective strides.
//
// Rank-4 is the universal form for shapes up to ndim=4 (the library
// limit, enforced by Shape.rank being u2 plus 1). Shapes with fewer
// dims are padded to 4D with size=1 and stride=0 on the left
// (broadcasting convention). This keeps the kernel launch uniform
// across all rank combinations without a template per rank.
//
// Why int (signed) strides?
//   Future slice ops may introduce negative strides (reversed views).
//   Today all strides are non-negative, so the signedness is a
//   no-cost safety margin.
//

extern "C" __global__
void elw_broadcast_add(
    const float* a, const float* b, float* c,
    unsigned int n,
    unsigned int d0, unsigned int d1, unsigned int d2, unsigned int d3,
    int a_s0, int a_s1, int a_s2, int a_s3,
    int b_s0, int b_s1, int b_s2, int b_s3)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int tmp = i;
    unsigned int i3 = tmp % d3; tmp /= d3;
    unsigned int i2 = tmp % d2; tmp /= d2;
    unsigned int i1 = tmp % d1; tmp /= d1;
    unsigned int i0 = tmp;

    int a_off = (int)i0 * a_s0 + (int)i1 * a_s1 + (int)i2 * a_s2 + (int)i3 * a_s3;
    int b_off = (int)i0 * b_s0 + (int)i1 * b_s1 + (int)i2 * b_s2 + (int)i3 * b_s3;

    c[i] = a[a_off] + b[b_off];
}

extern "C" __global__
void elw_broadcast_sub(
    const float* a, const float* b, float* c,
    unsigned int n,
    unsigned int d0, unsigned int d1, unsigned int d2, unsigned int d3,
    int a_s0, int a_s1, int a_s2, int a_s3,
    int b_s0, int b_s1, int b_s2, int b_s3)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tmp = i;
    unsigned int i3 = tmp % d3; tmp /= d3;
    unsigned int i2 = tmp % d2; tmp /= d2;
    unsigned int i1 = tmp % d1; tmp /= d1;
    unsigned int i0 = tmp;
    int a_off = (int)i0 * a_s0 + (int)i1 * a_s1 + (int)i2 * a_s2 + (int)i3 * a_s3;
    int b_off = (int)i0 * b_s0 + (int)i1 * b_s1 + (int)i2 * b_s2 + (int)i3 * b_s3;
    c[i] = a[a_off] - b[b_off];
}

extern "C" __global__
void elw_broadcast_mul(
    const float* a, const float* b, float* c,
    unsigned int n,
    unsigned int d0, unsigned int d1, unsigned int d2, unsigned int d3,
    int a_s0, int a_s1, int a_s2, int a_s3,
    int b_s0, int b_s1, int b_s2, int b_s3)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tmp = i;
    unsigned int i3 = tmp % d3; tmp /= d3;
    unsigned int i2 = tmp % d2; tmp /= d2;
    unsigned int i1 = tmp % d1; tmp /= d1;
    unsigned int i0 = tmp;
    int a_off = (int)i0 * a_s0 + (int)i1 * a_s1 + (int)i2 * a_s2 + (int)i3 * a_s3;
    int b_off = (int)i0 * b_s0 + (int)i1 * b_s1 + (int)i2 * b_s2 + (int)i3 * b_s3;
    c[i] = a[a_off] * b[b_off];
}

extern "C" __global__
void elw_broadcast_div(
    const float* a, const float* b, float* c,
    unsigned int n,
    unsigned int d0, unsigned int d1, unsigned int d2, unsigned int d3,
    int a_s0, int a_s1, int a_s2, int a_s3,
    int b_s0, int b_s1, int b_s2, int b_s3)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tmp = i;
    unsigned int i3 = tmp % d3; tmp /= d3;
    unsigned int i2 = tmp % d2; tmp /= d2;
    unsigned int i1 = tmp % d1; tmp /= d1;
    unsigned int i0 = tmp;
    int a_off = (int)i0 * a_s0 + (int)i1 * a_s1 + (int)i2 * a_s2 + (int)i3 * a_s3;
    int b_off = (int)i0 * b_s0 + (int)i1 * b_s1 + (int)i2 * b_s2 + (int)i3 * b_s3;
    c[i] = a[a_off] / b[b_off];
}
