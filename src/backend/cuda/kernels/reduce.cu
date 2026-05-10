//
// zig-transformer-lab — reduction + broadcast-copy CUDA kernels
//                        (Stage 7, PR-iota)
//
// Purpose:
//   Reduction primitives that backward passes use constantly:
//     reduce_sum_all     -- whole-tensor sum to a scalar (atomicAdd
//                           single-pass; the host zero-inits the
//                           output buffer via cuMemsetD32_v2 first).
//     reduce_sum_axis    -- sum along one axis of a CONTIGUOUS
//                           row-major input. Output shape matches
//                           input with the reduced axis set to 1.
//     bcast_copy         -- rank-4 stride-aware gather; used for
//                           broadcastTo (scalar -> tensor, or
//                           shape-match identity copy).
//
// Determinism note:
//   reduce_sum_all uses atomicAdd on f32. Order of accumulation is
//   not deterministic across runs, so the result can vary by at most
//   a few ULPs. Oracle tolerances (rel_tol=1e-4) absorb this. A
//   deterministic tree reduction is a Stage 9 perf/teaching concern.
//
// Layout assumption:
//   reduce_sum_axis assumes the input is CONTIGUOUS row-major. Non-
//   contiguous reductions ship in a later PR if a use case needs
//   them -- our backward path only reduces freshly-contiguous
//   gradient tensors.
//

extern "C" __global__
void reduce_sum_all(const float* x, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(out, x[i]);
}

extern "C" __global__
void reduce_sum_axis(
    const float* x, float* out,
    unsigned int out_n,
    unsigned int axis_size,
    unsigned int inner_count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_n) return;
    unsigned int outer = i / inner_count;
    unsigned int inner = i % inner_count;
    float acc = 0.0f;
    unsigned int base = outer * axis_size * inner_count + inner;
    for (unsigned int k = 0; k < axis_size; k++) {
        acc += x[base + k * inner_count];
    }
    out[i] = acc;
}

extern "C" __global__
void bcast_copy(
    const float* x, float* out,
    unsigned int n,
    unsigned int d0, unsigned int d1, unsigned int d2, unsigned int d3,
    int s0, int s1, int s2, int s3)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tmp = i;
    unsigned int i3 = tmp % d3; tmp /= d3;
    unsigned int i2 = tmp % d2; tmp /= d2;
    unsigned int i1 = tmp % d1; tmp /= d1;
    unsigned int i0 = tmp;
    int off = (int)i0 * s0 + (int)i1 * s1 + (int)i2 * s2 + (int)i3 * s3;
    out[i] = x[off];
}
