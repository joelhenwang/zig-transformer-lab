//
// zig-transformer-lab — cross-entropy loss CUDA kernel
//                        (Stage 7, Milestone 1 / PR-mu completion)
//
// Purpose:
//   Fused forward + backward for mean cross-entropy classification
//   loss. One kernel invocation produces both the scalar `loss` and
//   the (N, C) `grad_logits` in a single launch, avoiding the
//   CPU-roundtrip tax the training loop would pay every step.
//
// Shape contract:
//   logits       : (N, C) contiguous
//   targets      : (N,)   contiguous f32 (rounded to int per row)
//   loss_out     : (1,)   pre-zeroed on host; kernel atomicAdds into [0]
//   grad_logits  : (N, C) contiguous; written elementwise
//
// Math:
//   For each row i, with target t_i = round(targets[i]) in [0, C):
//     m_i          = max_j logits[i, j]
//     logsumexp_i  = m_i + log(sum_j exp(logits[i,j] - m_i))
//     row_loss_i   = logsumexp_i - logits[i, t_i]
//     loss         = (1/N) * sum_i row_loss_i
//     dL/dlogits[i, j] = (softmax(logits)[i, j] - 1_{j == t_i}) / N
//
//   This matches PyTorch's torch.nn.functional.cross_entropy with
//   reduction='mean'. The oracle fixture cross_entropy_3d uses this
//   exact formula.
//
// Block geometry:
//   One block per row, BLOCK_SIZE=256 threads, three passes over C:
//     1. find row max (shared-memory reduction)
//     2. sum exp(x - max) (shared-memory reduction); take log
//     3. write (softmax - one_hot)/N to grad_logits elementwise;
//        thread 0 atomicAdds the row's mean-scaled loss contribution
//        into loss_out.
//
//   The kernel shape mirrors softmax.cu so bit-level accumulation
//   order of the sum-of-exps matches the log-softmax path — helpful
//   for cross-op consistency.
//
// Numerical stability:
//   Max-subtraction before every expf, same trick as softmax.cu.
//   Without it, exp(large_logit) -> Inf and the whole row becomes
//   NaN after normalisation.
//
// Indeterminism note:
//   atomicAdd on loss_out is non-deterministic by a few ULPs across
//   runs because thread-block order is scheduler-dependent. This is
//   acceptable for training (the gradient scale is unaffected) and
//   comfortably within the oracle cross_entropy_3d tolerance of
//   rel=1e-4 / abs=1e-4. A deterministic variant (tree reduction
//   across rows) can be added later if a Stage 9 perf pass calls
//   for bit-exact reproduction.
//
// Note on re-computing expf in pass 3:
//   We could persist the exp values from pass 2 into grad_logits
//   (write e = exp(x - max) there, then divide by row_sum in pass 3).
//   That saves one expf per element but requires a second sync and
//   complicates the later normalise+subtract step. The current
//   three-pass layout is the simplest correct implementation and
//   measured fine on our B*T <= 100-ish rows; optimise only if
//   profiling says to.
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
void ce_fused(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ loss_out,      // (1,), pre-zeroed by host
    float* __restrict__ grad_logits,   // (N, C)
    unsigned int N,
    unsigned int C)
{
    unsigned int row = blockIdx.x;
    if (row >= N) return;
    unsigned int tid = threadIdx.x;

    const float* row_in = logits + row * C;
    float* row_out_grad = grad_logits + row * C;

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

    // Pass 2: sum of exp(x - max). We don't persist the exp values;
    // pass 3 recomputes them. See header note.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        local_sum += expf(row_in[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    block_reduce_sum(sdata, tid);
    float row_sum = sdata[0];
    float log_sum = logf(row_sum);
    __syncthreads();

    // Target index for this row. rintf rounds-half-to-even; our f32
    // targets come from integer casts on the Python side so the
    // rounding path is exact. Host-side validation is the caller's
    // responsibility (loss.zig does it for the CPU path; CUDA path
    // trusts the training loop to produce in-range indices).
    int target_idx = (int)rintf(targets[row]);

    // The mean-over-batch factor. Bake it into every gradient write
    // and the loss accumulator so the backward pass needs no further
    // scaling.
    float inv_N = 1.0f / (float)N;

    // Pass 3: write grad_logits = (softmax - one_hot) / N.
    for (unsigned int i = tid; i < C; i += BLOCK_SIZE) {
        float p = expf(row_in[i] - row_max) / row_sum;
        float one_hot = ((int)i == target_idx) ? 1.0f : 0.0f;
        row_out_grad[i] = (p - one_hot) * inv_N;
    }

    // Loss contribution for this row:
    //   row_loss = -log(softmax[row, target])
    //            = -(logits[row, target] - row_max - log_sum)
    //            =  log_sum + row_max - logits[row, target]
    // Only thread 0 emits to avoid over-counting.
    if (tid == 0) {
        float target_logit = row_in[target_idx];
        float row_loss = (log_sum + row_max - target_logit) * inv_N;
        atomicAdd(loss_out, row_loss);
    }
}
