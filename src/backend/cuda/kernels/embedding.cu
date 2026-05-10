//
// zig-transformer-lab — embedding CUDA kernels (Stage 7, PR-mu)
//
// Purpose:
//   Embedding table: forward gathers rows from weight[ids[i]],
//   backward scatter-adds grad_output[i] back into grad_weight[ids[i]]
//   via atomicAdd. The scatter step is the reason atomicAdd exists —
//   two rows of the output may reference the same weight row, and
//   their gradient contributions must BOTH accumulate into that
//   shared slot.
//
// Layout contract:
//   weight:      (V, D)   contiguous
//   ids:         (N,)     contiguous f32 (stored as floats because our
//                         Tensor is f32-only; rounded to int via rintf)
//   grad_output: (N, D)   contiguous  (caller may flatten higher-rank
//                         grad tensors into this shape on the host)
//   grad_weight: (V, D)   contiguous  (caller pre-zeros via
//                         cuMemsetD32_v2 before launch)
//
// Bounds on ids:
//   Each id must satisfy 0 <= id < V. The kernel silently skips
//   OOB ids (returns without writing) — higher layers should
//   validate on the host before launch. Silent skip avoids
//   corrupting a valid grad_weight slot next to the OOB one.
//

extern "C" __global__
void embedding_forward(
    const float* weight,
    const float* ids,
    float* out,
    unsigned int N,
    unsigned int D,
    unsigned int V)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * D) return;
    unsigned int i = t / D;
    unsigned int j = t % D;
    int idx = (int)rintf(ids[i]);
    if (idx < 0 || idx >= (int)V) return;
    out[i * D + j] = weight[idx * D + j];
}

extern "C" __global__
void embedding_backward(
    const float* ids,
    const float* grad_out,
    float* grad_weight,
    unsigned int N,
    unsigned int D,
    unsigned int V)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * D) return;
    unsigned int i = t / D;
    unsigned int j = t % D;
    int idx = (int)rintf(ids[i]);
    if (idx < 0 || idx >= (int)V) return;
    atomicAdd(&grad_weight[idx * D + j], grad_out[i * D + j]);
}
