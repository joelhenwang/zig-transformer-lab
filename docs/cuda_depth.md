# CUDA Depth — From Basic Kernels to Tensor Cores

A bridge from the basic CUDA backend in `docs/08_backends_cuda.md`
to the techniques that make real LLM kernels 10-100x faster. This
chapter goes deeper: shared memory, tiling, warp primitives, tensor
cores, kernel fusion, streams, and profiling.

**Prerequisites:** read `docs/08_backends_cuda.md` and
`docs/08b_from_cuda_to_training.md` first. You should already
understand: thread/block/grid, kernel launches, HtoD/DtoH transfers,
DeviceBuffer, PTX loading, and the cuBLAS operand-swap trick.

**Hardware context:** all examples target the RTX 4060 Ti (Ada
Lovelace, SM 8.9, 16 GB GDDR6, 288 GB/s bandwidth, 165 TFLOPS FP16
tensor core, 22 TFLOPS FP32 CUDA core).

**Runnable demos:** this chapter is paired with two pedagogical
kernels you can build and run:
- `examples/11_cuda_tiled_gemm_demo.zig` — naive vs tiled vs cuBLAS
- `examples/12_cuda_wmma_demo.zig` — WMMA correctness check

---

## 1. GPU Memory Hierarchy

Understanding GPU performance starts with understanding where your
data lives and how expensive it is to access.

### 1.1 The hierarchy

```
+---------------------------------------------------+
| Register file (per-thread)                        |
|   Size: ~256 KB per SM (64K 32-bit registers)    |
|   Latency: 0 cycles (free)                       |
|   Bandwidth: effectively infinite                 |
+---------------------------------------------------+
| Shared memory / L1 cache (per-block)             |
|   Size: configurable, up to 100 KB per SM        |
|   Latency: ~20-30 cycles                         |
|   Bandwidth: ~19 TB/s (across all SMs)           |
+---------------------------------------------------+
| L2 cache (chip-wide)                             |
|   Size: 32 MB (RTX 4060 Ti)                      |
|   Latency: ~200 cycles                           |
|   Bandwidth: ~3 TB/s                             |
+---------------------------------------------------+
| Global memory (GDDR6)                            |
|   Size: 16 GB                                     |
|   Latency: ~400-600 cycles                       |
|   Bandwidth: 288 GB/s                            |
+---------------------------------------------------+
| Host memory (CPU DRAM, via PCIe)                  |
|   Size: 32+ GB                                    |
|   Latency: ~10,000+ cycles                       |
|   Bandwidth: 25 GB/s (PCIe 4.0 x16)             |
+---------------------------------------------------+
```

### 1.2 The key insight

Global memory is 700x slower than registers and 10x slower than
shared memory. Any optimization that moves data closer to the compute
units — from global to shared to registers — wins proportionally.

For matrix multiplication:
- Naive: each thread reads N values from global memory per output element
- Tiled: each thread reads N/TILE_SIZE values from global, the rest from shared
- Ratio: TILE_SIZE fewer global memory accesses (16x for TILE=16)

This is why tiled GEMM is 10-50x faster than naive GEMM for large
matrices, even though both compute exactly the same arithmetic.

### 1.3 Arithmetic intensity

**Arithmetic intensity** = FLOPs / bytes loaded from memory.

For matrix multiply C = A @ B where all matrices are NxN:
- FLOPs: 2*N^3 (N^2 output elements, each needs N multiply-adds)
- Bytes loaded (naive): 2*N^3 * 4 bytes = 8*N^3 bytes (each thread loads a full row + col)
- Arithmetic intensity (naive): 2*N^3 / 8*N^3 = 0.25 FLOP/byte

With tiling (tile size T):
- Bytes loaded: 2*N^3/T * 4 bytes
- Arithmetic intensity: T/4 FLOP/byte
- For T=16: 4 FLOP/byte (16x better!)

The GPU's peak compute is 22 TFLOPS FP32. Its peak bandwidth is
288 GB/s. The crossover ("roofline") is at 22e12 / 288e9 = 76
FLOP/byte. Any kernel below this ratio is memory-bound; above is
compute-bound. Tiling moves matmul from deeply memory-bound to
approaching compute-bound.

---

## 2. Naive GEMM — The Baseline

Before optimizing, establish what "naive" looks like.

### 2.1 One thread per output element

```cuda
// Naive GEMM: C[row][col] = sum_k A[row][k] * B[k][col]
__global__ void naive_gemm(
    const float* A, const float* B, float* C,
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
```

### 2.2 Why it's slow

Each thread computes one output element. For C[row][col], it loads:
- K values from row `row` of A (stride K in memory)
- K values from column `col` of B (stride N in memory — NOT coalesced!)

For N=K=1024: each thread loads 2*1024 = 2048 floats = 8 KB from
global memory. With 1M output elements, that's 8 TB of global loads.
At 288 GB/s bandwidth, that's ~28 seconds — but the GPU has millions
of threads sharing the bandwidth, so the actual time is limited by
memory bandwidth, not arithmetic.

Measured on RTX 4060 Ti at 1024x1024: ~15 ms. cuBLAS: ~0.3 ms.
That's a 50x gap to close.

---

## 3. Tiled Shared-Memory GEMM (The Centerpiece)

This is the most important optimization in GPU programming. Every
high-performance kernel — GEMM, attention, convolution — uses some
form of tiling.

### 3.1 The idea

Instead of each thread loading from global memory independently:
1. A **block** of threads cooperatively loads a **tile** of A and B
   into shared memory
2. Each thread computes its output using data from shared memory
3. Repeat for the next tile until all K values are consumed

This means each value loaded from global memory is reused by
TILE_SIZE threads — a TILE_SIZE x reduction in global bandwidth.

### 3.2 The algorithm

```
For each output tile (block of C):
  Initialize accumulator to 0
  For t = 0, 1, ..., K/TILE - 1:
    1. COOPERATIVELY load tile of A into shared_A  (blockDim.y x TILE)
    2. COOPERATIVELY load tile of B into shared_B  (TILE x blockDim.x)
    3. __syncthreads()  (ensure all threads finished loading)
    4. Each thread: accumulate dot product from shared_A row * shared_B col
    5. __syncthreads()  (ensure all threads finished reading before next load)
  Write accumulator to C[row][col]
```

### 3.3 Full kernel (annotated)

This kernel is in `src/backend/cuda/kernels/tiled_gemm.cu`:

```cuda
#define TILE 16

__global__ void tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory tiles — visible to all threads in this block
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    // This thread's output position
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles along the K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // --- Phase 1: Cooperative load from global to shared ---
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Load A[row][a_col] into shared_A[threadIdx.y][threadIdx.x]
        if (row < M && a_col < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B[b_row][col] into shared_B[threadIdx.y][threadIdx.x]
        if (b_row < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        // --- Phase 2: Wait for all threads to finish loading ---
        __syncthreads();

        // --- Phase 3: Compute partial dot product from shared memory ---
        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // --- Phase 4: Wait before next tile overwrites shared memory ---
        __syncthreads();
    }

    // Write final result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 3.4 Why this is faster

Per output element:
- **Naive**: 2K global memory loads
- **Tiled**: 2K/TILE global loads + 2K shared memory loads

Shared memory is ~10x faster than global. For K=1024, TILE=16:
- Naive: 2048 global loads
- Tiled: 128 global loads + 2048 shared loads

That's a 16x reduction in the expensive (global) loads, replaced
with cheap (shared) loads. In practice the speedup is 10-50x
depending on matrix size and GPU occupancy.

### 3.5 Bank conflicts

Shared memory is divided into 32 "banks." If two threads in a warp
access the same bank simultaneously, the accesses serialize ("bank
conflict"). The pattern `sB[k][threadIdx.x]` in our kernel is
conflict-free because consecutive threads access consecutive columns
(consecutive banks). The pattern `sA[threadIdx.y][k]` could conflict
if k iterates such that threads hit the same bank — but since each
thread uses a different `threadIdx.y` and all read the same `k`,
this is a broadcast (one read shared by all), not a conflict.

For more advanced kernels (128x128 tiles with register blocking),
bank conflicts become a real optimization target. At our TILE=16
teaching scale, they don't dominate.

### 3.6 How to run the demo

```bash
# On the remote RTX 4060 Ti:
zig build run-example -Dexample=11_cuda_tiled_gemm_demo -Dcuda=true -Doptimize=ReleaseFast
```

Expected output (approximate):
```
Matrix size: 1024 x 1024
Naive GEMM:  ~15.2 ms
Tiled GEMM:  ~0.9 ms   (16.8x speedup)
cuBLAS GEMM: ~0.3 ms   (50.6x vs naive)
Max error (tiled vs cuBLAS): < 1e-4
```

The gap between tiled (~0.9 ms) and cuBLAS (~0.3 ms) comes from
further optimizations cuBLAS applies: 128x128 tiles, register
blocking, double-buffering, vectorized loads, and tensor cores.
Our 16x16 kernel teaches the concept; cuBLAS ships the production
performance.

---

## 4. Warp-Level Primitives

A **warp** is 32 threads that execute in lockstep on the same SM.
Warp-level primitives let threads within a warp communicate without
going through shared memory — they use register shuffle instructions.

### 4.1 __shfl_down_sync — warp reduction

The most common use case: reducing 32 values to 1 (e.g., for sumAll).

```cuda
// Reduce 32 values across a warp to thread 0
__device__ float warp_reduce_sum(float val) {
    // Each step: thread i gets the value from thread i+offset
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // Only thread 0 in the warp has the final sum
}
```

Why this is fast: no shared memory needed, no `__syncthreads()`,
completes in 5 instructions (log2(32) = 5 shuffle steps).

### 4.2 __ballot_sync — predicate aggregation

Returns a 32-bit mask where bit i is set if thread i's predicate
is true. Useful for counting, early termination, compaction.

```cuda
// Count how many threads in the warp have x > threshold
unsigned int mask = __ballot_sync(0xFFFFFFFF, x > threshold);
int count = __popc(mask);  // population count (number of set bits)
```

### 4.3 When to use warp primitives

- **Small reductions** (≤ 32 elements): warp shuffle is optimal
- **Block reductions** (> 32 elements): warp-reduce each warp, then
  combine warp results in shared memory
- **Any thread-to-thread communication within a warp**: shuffle avoids
  the shared memory round-trip

Our existing `reduce_sum.cu` kernel uses a block-level reduction with
shared memory. For the final 32-element reduction within each warp,
a shuffle-based approach would be ~2x faster (but at our scale the
launch overhead dominates, so the improvement is unmeasurable).

---

## 5. Tensor Cores via WMMA

### 5.1 What tensor cores are

Tensor cores are fixed-function hardware units that compute small
matrix multiply-accumulate operations in a single clock cycle:

```
D = A * B + C
where A is M×K, B is K×N, C and D are M×N
Supported sizes: 16×16×16, 32×8×16, 8×32×16 (varies by architecture)
```

On Ada Lovelace (RTX 4060 Ti):
- FP16 tensor core: 165 TFLOPS (vs 22 TFLOPS FP32 CUDA core = 7.5x)
- A single 16×16×16 tile: 16*16*16*2 = 8192 FLOPs in one instruction

### 5.2 The WMMA programming model

WMMA (Warp Matrix Multiply-Accumulate) is the C++ API for tensor cores.
A full warp (32 threads) cooperates on one tile:

```cuda
#include <mma.h>
using namespace nvcuda;

// Declare fragments (distributed across warp threads)
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// Initialize accumulator to zero
wmma::fill_fragment(c_frag, 0.0f);

// Load A and B tiles from memory into fragments
wmma::load_matrix_sync(a_frag, a_ptr, 16);  // 16 = leading dimension
wmma::load_matrix_sync(b_frag, b_ptr, 16);

// THE KEY INSTRUCTION: matrix multiply-accumulate
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result back to memory
wmma::store_matrix_sync(c_ptr, c_frag, 16, wmma::mem_row_major);
```

**What's happening under the hood:** the 32 threads in the warp each
hold a few elements of the fragment. The `mma_sync` instruction
triggers the tensor core hardware to multiply the 16x16 FP16 matrices
and accumulate into FP32 — one warp, one instruction, 8192 FLOPs.

### 5.3 Why inputs are FP16

Tensor cores multiply in reduced precision (FP16 or BF16) and
accumulate in FP32. This is safe because:
- The multiplication (A[i]*B[j]) is the source of most precision loss,
  but individual products rarely need > 16 bits of mantissa
- The accumulation (sum of products) uses FP32, preserving precision
  where it matters (catastrophic cancellation in sums)

This is the hardware basis for mixed-precision training (Gap 3): train
the forward/backward in FP16 on tensor cores, keep master weights
in FP32 for optimizer updates.

### 5.4 The demo kernel

`src/backend/cuda/kernels/wmma_demo.cu` implements a minimal WMMA
matmul:

```cuda
#include <mma.h>
using namespace nvcuda;

// Minimal WMMA demo: 16x16 @ 16x16 -> 16x16 (FP16 inputs, FP32 output)
__global__ void wmma_matmul_16x16(
    const half* A, const half* B, float* C)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
```

This is ONE warp, ONE tile. A real WMMA GEMM would tile across M, N, K
with shared memory (combining sections 3 and 5). CUTLASS does this
at production quality; our demo teaches the API.

### 5.5 How to run the WMMA demo

```bash
zig build run-example -Dexample=12_cuda_wmma_demo -Dcuda=true
```

Expected output:
```
WMMA 16x16x16 matmul (FP16 inputs, FP32 accumulator)
Reference (CPU FP32): C[0][0] = 128.000
WMMA result:          C[0][0] = 128.000
Max abs error: < 0.01  (FP16 rounding)
PASS
```

---

## 6. Kernel Fusion

### 6.1 What it is

Instead of launching separate kernels for each operation (load data
from global, compute, store to global, load again, compute, store
again...), combine multiple operations into a single kernel that
keeps data in registers/shared memory between operations.

### 6.2 Why it matters

Each kernel launch has:
- Fixed overhead: ~5-20 us of CPU-side submission
- Global memory round-trip: write result, then read it back for next op

For small tensors (like ours at D=64), kernel launches dominate.
Fusing N ops into 1 eliminates N-1 launches and N-1 global store/load
pairs.

### 6.3 Example: our fused cross-entropy kernel

`src/backend/cuda/kernels/ce_loss.cu` computes both the loss AND the
gradient in a single kernel. Without fusion this would be:

1. Kernel 1: softmax (read logits, write probs)
2. Kernel 2: log (read probs, write log_probs)
3. Kernel 3: gather + negate (read log_probs + targets, write loss)
4. Kernel 4: grad = (probs - one_hot) / B (read probs + targets, write grad)

Fused: ONE kernel reads logits + targets, writes loss + grad. Three
fewer launches, three fewer global round-trips.

### 6.4 When to fuse

Fuse when:
- Operations are elementwise (same data access pattern)
- Intermediate results are consumed only by the next op (no other users)
- Combined kernel fits in register budget

Don't fuse when:
- Operations have different parallelism patterns (reduction vs elementwise)
- Shared intermediate results are needed by multiple downstream ops
- Register pressure would reduce occupancy below ~25%

### 6.5 The AdamW kernel

`src/backend/cuda/kernels/adamw.cu` is another fusion example:
in one kernel launch it reads params + grads + m + v, and writes
updated params + m + v. Without fusion: 7 separate elementwise
kernels (m update, v update, m_hat, v_hat, denom, step, decay).

---

## 7. CUDA Streams and Graphs

### 7.1 The default stream

All our kernel launches go to the default stream (stream 0). Kernels
on the same stream execute in order — no overlap. This means:

```
HtoD copy → waits → kernel 1 → waits → kernel 2 → waits → DtoH copy
```

Total time = sum of all operations.

### 7.2 Multiple streams — overlap copy and compute

With two streams, you can overlap data transfer with computation:

```
Stream 1: HtoD(batch_1) → kernel(batch_1) → DtoH(result_1)
Stream 2:     HtoD(batch_2) → kernel(batch_2) → DtoH(result_2)
                  ↑ overlaps with kernel(batch_1)
```

Total time < sum of all operations (overlap saves time).

### 7.3 CUDA Graphs

For workloads with the same kernel launch sequence every iteration
(like training steps), CUDA Graphs let you:
1. **Record** the full sequence of launches once
2. **Replay** the recorded graph with minimal CPU overhead

This eliminates per-launch CPU overhead entirely for the recorded
portion. Measured benefit: 10-30% for launch-bound workloads (like
our small-tensor training).

### 7.4 Why we don't use these

At our scale (2/2/64 config, ~200 kernel launches per step), the
per-step wall-clock is 8.8 ms. Launch overhead is maybe 2-3 ms of
that. Streams and graphs would help, but the implementation complexity
isn't justified for a teaching codebase.

Where they become essential: production inference serving with
continuous batching (Section 8 of gap_map.md), where you process
thousands of requests per second and every microsecond matters.

---

## 8. Profiling with Nsight Compute

### 8.1 What to measure

- **Achieved occupancy**: fraction of maximum warps actually active.
  Low occupancy = threads idle = wasted SM resources.
- **Memory throughput (achieved)**: fraction of peak bandwidth used.
  If < 50%, your kernel has memory access pattern issues.
- **Compute throughput**: fraction of peak FLOPS achieved.
  If < 50% and memory throughput is also low, you have latency issues.
- **SM efficiency**: fraction of time at least one warp is active.
  < 100% means some SMs have no work (grid too small).

### 8.2 The roofline model

Plot your kernel on a graph of:
- X-axis: arithmetic intensity (FLOP/byte)
- Y-axis: achieved performance (GFLOPS)
- Ceiling: min(peak_compute, peak_bandwidth * intensity)

If your kernel is below the roof: optimization opportunity exists.
If it's on the memory-bandwidth slope: optimize memory access patterns.
If it's near the compute ceiling: you're compute-bound (good!).

### 8.3 Reading ncu output

```bash
# Run (requires root-equivalent permissions on our remote):
ncu --set full ./test_binary --specific-test

# Key metrics to look for:
# - sm__throughput.avg.pct_of_peak_sustained   (< 50% = problem)
# - dram__throughput.avg.pct_of_peak_sustained (close to 100% = memory bound)
# - launch__occupancy.avg                       (< 0.5 = register/shared pressure)
```

### 8.4 Our limitation

The remote box has `ERR_NVGPUCTRPERM` — Nsight Compute needs elevated
permissions (root or a modprobe.d override) to access GPU performance
counters. This is documented as a deferred item in `AGENTS.md`. Without
it, we can measure wall-clock time but not internal GPU metrics.

Workaround: use `nvidia-smi dmon` for coarse utilization numbers, and
estimate roofline position from theoretical FLOP counts + measured time.

---

## 9. Flash Attention — A Conceptual Walkthrough

Flash attention is the most important kernel optimization in modern
LLMs. This section explains WHY it works, not how to implement it
(that's a multi-week project; see `docs/extensions.md` H2).

### 9.1 The problem

Standard attention materializes the full scores matrix:

```
S = Q @ K^T      # shape (B, H, T, T)
P = softmax(S)   # shape (B, H, T, T)
O = P @ V        # shape (B, H, T, d_head)
```

For T=4096, H=32, B=1, FP16: the S and P matrices together are
2 * 4096^2 * 32 * 2 bytes = 2 GB. On a 16 GB GPU this leaves only
14 GB for model parameters, activations, and gradients. At T=8192
it's 8 GB for attention alone — infeasible.

### 9.2 The insight

Softmax doesn't need the full row at once. You can compute softmax
incrementally using the "online softmax" trick:

```
# Standard softmax: needs all values first
max_val = max(row)
exp_vals = exp(row - max_val)
sum_exp = sum(exp_vals)
softmax = exp_vals / sum_exp

# Online softmax: processes blocks incrementally
running_max = -inf
running_sum = 0
for block in blocks:
    new_max = max(running_max, max(block))
    # Rescale previous sum for the new max
    running_sum = running_sum * exp(running_max - new_max) + sum(exp(block - new_max))
    running_max = new_max
# Final: exp(x - running_max) / running_sum
```

### 9.3 The tiling strategy

Flash attention tiles along the sequence dimension:

```
For each block of Q rows (size B_r):
  For each block of K/V columns (size B_c):
    1. Load Q block into shared memory (or registers)
    2. Load K block into shared memory
    3. Compute local scores: S_local = Q_block @ K_block^T  (B_r x B_c)
    4. Update running softmax statistics (online max + sum)
    5. Load V block into shared memory
    6. Accumulate: O_block += rescaled_softmax @ V_block
  Write final O_block to global memory
```

**Key:** the T x T scores matrix is never fully materialized.
Only B_r x B_c tiles exist in shared memory at any moment.
Memory: O(T) instead of O(T^2).

### 9.4 Why it's faster too

Even ignoring the memory savings, flash attention is 2-3x faster
because:
1. Fewer global memory round-trips (scores stay in shared/registers)
2. Better arithmetic intensity (more compute per byte loaded)
3. The tiled pattern fits perfectly into the shared memory hierarchy

### 9.5 What's needed to implement it

- Shared memory tiling (section 3 of this chapter — you know this now)
- Online softmax with rescaling
- Careful accumulator management (O_block needs rescaling when max changes)
- Backward pass is even more complex (need to recompute attention during backward)

Tri Dao's FlashAttention paper (2022) gives the full algorithm. The
Triton implementation (Python-like GPU DSL) is more readable than the
raw CUDA version. If you want to implement this, see `docs/extensions.md` H2.

---

## 10. Where to Go Next

### For deeper GPU programming:
- **CUTLASS** (Nvidia's GEMM library source, C++ templates). Production
  GEMM kernels with register blocking, double buffering, and WMMA.
  Start with `examples/00_basic_gemm.cu`.
- **Triton** (OpenAI's GPU DSL). Write kernels in Python-like syntax;
  the compiler handles tiling and shared memory. Much faster to iterate
  than raw CUDA.
- **ThunderKittens** (Stanford). Recent work on simple, composable LLM
  kernel building blocks. More accessible than CUTLASS.

### For LLM-specific kernels:
- **Flash attention implementations** — both the CUDA version (Dao Lab)
  and the Triton version (in the triton repo).
- **Fused attention + RoPE** — combine position encoding with the
  attention kernel to avoid an extra launch.
- **Fused optimizer kernels** — our AdamW kernel already does this;
  study how DeepSpeed fuses more aggressively.

### For hardware-specific optimization:
- **Hopper architecture** (H100) — introduces TMA (Tensor Memory
  Accelerator) and FP8 instructions. The next frontier.
- **CUDA Graphs + CUDA Memory Pools** — eliminate allocation overhead
  in the hot loop.
- **Nsight Systems** (not Compute) — for system-level profiling
  (overlap analysis, stream utilization, CPU/GPU synchronization).

### The cutting edge:
- **FP8 training** (Nvidia, 2023) — even lower precision than FP16.
  Requires careful quantization and new hardware instructions.
- **Ring attention** — distributed flash attention across multiple GPUs.
- **Paged attention** (vLLM) — virtual memory for KV cache management.
