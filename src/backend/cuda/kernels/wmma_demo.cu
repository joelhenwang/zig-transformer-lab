// wmma_demo.cu — Pedagogical tensor-core WMMA kernel
//
// Demonstrates FP16 x FP16 -> FP32 matrix multiply-accumulate using
// Nvidia's WMMA (Warp Matrix Multiply-Accumulate) API. This is a
// MINIMAL example: one warp computes one 16x16x16 tile.
//
// NOT a production GEMM — just teaches the WMMA fragment model.
// See docs/cuda_depth.md section 5 for the full walkthrough.
//
// Launch: grid = (1, 1, 1), block = (32, 1, 1) [one warp]
// Inputs: A (16x16 half), B (16x16 half)
// Output: C (16x16 float)
//
// Requires: compute capability >= 7.0 (Volta+)
// RTX 4060 Ti is SM 8.9 (Ada Lovelace) — fully supported.

#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_matmul_16x16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C)
{
    // Declare fragments — these are distributed across the 32 threads
    // in the warp. Each thread holds a few elements of each fragment.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Zero the accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Load 16x16 tiles from global memory into fragments
    // Leading dimension = 16 (row stride for row-major layout)
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    // THE KEY INSTRUCTION: tensor-core matrix multiply-accumulate
    // D = A * B + C  (all in one hardware instruction)
    // A: 16x16 FP16, B: 16x16 FP16, C/D: 16x16 FP32
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result to global memory
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
