//
// zig-transformer-lab — softmax / log-softmax CUDA kernels
//                        (Stage 7, PR-lambda)
//
// Purpose:
//   Row-wise softmax and log-softmax over the last axis of a
//   contiguous tensor of shape (..., C). The host reshapes the
//   logical (N_pre, C) view by flattening all leading dims into
//   num_rows = prod(leading dims) before launching -- the kernel
//   then treats the input as a contiguous (num_rows, C) matrix
//   and processes one row per block.
//
// Numerical stability:
//   Both kernels use the max-subtraction trick:
//     softmax_i     = exp(x_i - max) / sum_k exp(x_k - max)
//     log_softmax_i = (x_i - max) - log(sum_k exp(x_k - max))
//   Without the max subtract, exp(100) = Inf and the sum becomes
//   Inf, giving NaN after division.
//
// Block geometry:
//   One block per row, block size = BLOCK_SIZE threads. Threads
//   stride-loop across C (which may be larger or smaller than
//   BLOCK_SIZE). Shared memory holds BLOCK_SIZE f32s for the two
//   reductions (max, then sum). A single __shared__ array is
//   reused across the reductions -- we __syncthreads() between
//   phases so the second reduction cannot see stale values.
//
// C size limits:
//   The kernel handles arbitrarily large C via the stride loop in
//   each phase. Shared memory sizing is fixed (BLOCK_SIZE, not C)
//   so no launch-time bounds on C. Our current transformer shapes
//   use C = 2000 (vocab for cross-entropy) at most; a block of
//   256 threads and three passes over 2000 elements is ~8
//   iterations per thread -- well within warp-scheduling budget.
//
// Block size choice:
//   256 threads = 8 warps, matches our elementwise default. A
//   power of two is required for the O(log2) reduction loop
//   below. If we ever want to experiment with 128 or 512,
//   BLOCK_SIZE can be templated, but 256 is a sweet spot.
//

#define BLOCK_SIZE 256

// Block-wide max reduction over `sdata[0..BLOCK_SIZE]`, result in sdata[0].
__device__ inline void block_reduce_max(float* sdata, unsigned int tid) {
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid];
            float b = sdata[tid + s];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
}

// Block-wide sum reduction.
__device__ inline void block_reduce_sum(float* sdata, unsigned int tid) {
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

extern "C" __global__
void softmax_last(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int num_rows,
    unsigned int C)
{
    unsigned int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* row_in = x + row * C;
    float* row_out = out + row * C;

    unsigned int tid = threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];

    // Pass 1: find the row's max (for numerical stability).
    float local_max = -INFINITY;
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    block_reduce_max(sdata, tid);
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2: write exp(x - max) and accumulate the sum.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    block_reduce_sum(sdata, tid);
    float row_sum = sdata[0];
    __syncthreads();

    // Pass 3: normalise. After this, each row of out sums to 1.0.
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        row_out[i] = row_out[i] / row_sum;
    }
}

extern "C" __global__
void log_softmax_last(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int num_rows,
    unsigned int C)
{
    unsigned int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* row_in = x + row * C;
    float* row_out = out + row * C;

    unsigned int tid = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    // Pass 1: row max.
    float local_max = -INFINITY;
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    block_reduce_max(sdata, tid);
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2: compute sum of exp(x - max). We don't persist the
    // exp values; they're cheap to recompute and log_softmax's
    // final formula needs only `x - max - log(sum)`.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        local_sum += expf(row_in[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    block_reduce_sum(sdata, tid);
    float log_sum = logf(sdata[0]);
    __syncthreads();

    // Pass 3: write x - max - log_sum.
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        row_out[i] = (row_in[i] - row_max) - log_sum;
    }
}
