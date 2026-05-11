//
// zig-transformer-lab — unary elementwise CUDA kernels
//                        (Stage 7, Milestone 2)
//
// Purpose:
//   Unary elementwise ops that are reachable from the transformer
//   forward / backward path: geluExact, sqrt, exp, log. Each kernel
//   is a straight flat-index map; backward is implemented either
//   via composition of existing ops (sqrt backward = mul by
//   0.5/sqrt(x) via elementwise), or with a dedicated fused
//   kernel here when the composed path would be more launches
//   than necessary.
//
// Kernel signatures (all extern "C"):
//
//   unary_gelu_exact(const float* x, float* out, unsigned int n)
//       out[i] = 0.5 * x[i] * (1 + erf(x[i] * 1/sqrt(2)))
//
//   unary_gelu_exact_backward(
//       const float* x, const float* grad_out, float* grad_in,
//       unsigned int n)
//       grad_in[i] = grad_out[i] * gelu_prime(x[i])
//
//     where gelu_prime(x) = 0.5 * (1 + erf(x/sqrt(2))) +
//                           x/sqrt(2*pi) * exp(-x^2/2)
//
//   unary_sqrt(const float* x, float* out, unsigned int n)
//       out[i] = sqrtf(x[i])
//
//   unary_exp(const float* x, float* out, unsigned int n)
//       out[i] = expf(x[i])
//
//   unary_log(const float* x, float* out, unsigned int n)
//       out[i] = logf(x[i])
//
// Precision note (GELU):
//   CPU `geluExact` uses a 5-term Abramowitz & Stegun polynomial
//   approximation of erf in f64, then casts back to f32. CUDA's
//   `erff` intrinsic is compiled into ptxas-produced code that
//   typically achieves <= 2 ULP error. Expect ~1e-6 drift per
//   element on random inputs in [-3, 3]. Unit tests use abs_tol
//   1e-5 to comfortably absorb this.
//
// GELU backward derivation:
//   f(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
//   f'(x) = 0.5 * (1 + erf(x/sqrt(2)))     [d/dx of the 0.5*x factor]
//         + 0.5 * x * d/dx erf(x/sqrt(2))   [d/dx of erf factor]
//   d/dx erf(x/sqrt(2)) = 2/sqrt(pi) * exp(-x^2/2) * (1/sqrt(2))
//                       = sqrt(2/pi) * exp(-x^2/2)
//   Substituting:
//     f'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * sqrt(1/(2*pi)) * exp(-x^2/2)
//   This matches the closed form used in PyTorch's
//   torch.nn.functional.gelu backward and in our CPU
//   backwardGelu implementation (see src/autograd/backward.zig).
//

#include <math_constants.h>

#define BLOCK_SIZE 256

// 1.0f / sqrtf(2.0f) as a literal. Avoids a divide every thread.
__device__ __constant__ float kInvSqrt2 = 0.7071067811865475f;
// 1.0f / sqrtf(2.0f * pi) — used in the GELU backward.
__device__ __constant__ float kInvSqrt2Pi = 0.3989422804014327f;

extern "C" __global__
void unary_gelu_exact(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    // erff is an sm_89 intrinsic: ~2 ULP, saturating at +/-inf.
    out[idx] = 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

extern "C" __global__
void unary_gelu_exact_backward(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    // phi = 0.5 * (1 + erf(v / sqrt(2)))  [the cumulative Gaussian]
    float phi = 0.5f * (1.0f + erff(v * kInvSqrt2));
    // pdf = 1/sqrt(2pi) * exp(-v^2 / 2)
    float pdf = kInvSqrt2Pi * expf(-0.5f * v * v);
    // gelu'(v) = phi + v * pdf
    grad_in[idx] = grad_out[idx] * (phi + v * pdf);
}

extern "C" __global__
void unary_sqrt(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;
    out[idx] = sqrtf(x[idx]);
}

extern "C" __global__
void unary_exp(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;
    out[idx] = expf(x[idx]);
}

extern "C" __global__
void unary_log(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;
    out[idx] = logf(x[idx]);
}
