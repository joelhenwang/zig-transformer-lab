// tiled_gemm.cu — Pedagogical shared-memory tiled GEMM kernel
//
// This kernel demonstrates the core GPU optimization technique: loading
// data cooperatively into shared memory to reduce global memory traffic.
// It is NOT a replacement for cuBLAS — it exists to TEACH tiling.
//
// See docs/cuda_depth.md section 3 for the full walkthrough.
//
// Launch: grid = (N/TILE, M/TILE), block = (TILE, TILE)
// where TILE = 16 and M, N, K are matrix dimensions.

#define TILE 16

extern "C" __global__ void tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Cooperative load: each thread loads one element of each tile
        if (row < M && a_col < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Naive GEMM baseline for comparison (one thread per output element)
extern "C" __global__ void naive_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
