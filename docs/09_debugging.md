# 09 — Debugging: shape asserts, gradient checks, CUDA sanitizers

> Stage 8, Milestone 7. Companion code: `src/debug/*.zig`, `src/autograd/gradcheck.zig`,
> `examples/04_overfit_one_batch.zig`, `examples/10_train_deep.zig`.
>
> This chapter covers the "something is wrong, where is it" tooling built into the
> library and the systematic debugging playbook we use when a tensor pipeline breaks.
> The tools are generic; the playbook is opinionated.

The first thing a new contributor to a PyTorch-like runtime discovers is that
*silent* bugs outnumber *loud* bugs by an order of magnitude. A crash with a stack
trace is lucky — you fix it before lunch. A net that "trains" for 500 steps and
produces junk without any error message is what eats weekends.

Stage 8 formalises the debugging toolkit that grew up organically through Stages
1 through 7. It consists of four namespaces (`debug.shape`, `debug.finite`,
`debug.compare`, `debug.dump`), a gradient-checking harness
(`autograd.gradcheck`), and two external tools (`compute-sanitizer` and
`Nsight Compute`) that we invoke on Linux with CUDA.

The goal is turning every silent bug into a loud bug as close to its origin as
possible. Everything in this chapter is about moving the failure site from the
symptom (NaN loss at step 500) back towards the cause (a LayerNorm that divides
by standard deviation without an `eps`).

---

## 1 — Shape-assert driven development

Shape bugs are the bulk of what goes wrong when you write tensor code. An axis
is swapped, a batch dim is flattened too early, a broadcast expands when you
meant reduce, a transpose in one place isn't mirrored in its inverse. The
consequence is usually not an error — just a different-shaped tensor flowing
into downstream ops, which *also* silently accept it because they interpret the
data differently than you expected.

Our first line of defence is `src/debug/shape.zig`:

```zig
pub fn assertShape(t: Tensor, expected: Shape) void;
pub fn assertRank(t: Tensor, expected_rank: usize) void;
pub fn assertDim(t: Tensor, axis: usize, expected_size: usize) void;
```

All three print a rich message to stderr and `@panic` on mismatch. Use them
liberally at the entry points of your ops, at layer boundaries, and anywhere
you did something non-trivial with axes.

Worked example: you are implementing a fresh Q/K/V projection and making the
classic "forgot to reshape before the head split" mistake.

```zig
// Intended: (B, T, D) -> (B, T, H, d_head) -> (B, H, T, d_head)
// Actual:   q.reshape is wrong — transposed after reshape but kept dims swapped.
var q = try self.w_q.forward(x, tape);              // (B, T, D)
var q_heads = try ops_shape.reshapeTracked(
    alloc, q, Shape.init4D(B, T, n_head, d_head), tape);
// BUG: forgot transposeAxes12_4d, so q_heads is (B, T, H, d)
//      but downstream code expects (B, H, T, d).
var scores = try ops.matmul.matmulBatch(
    alloc, q_heads, k_heads_t, tape);
```

Without shape asserts, `matmulBatch` happily multiplies `(B, T, H, d) @ (B, H,
d, T)` broadcasting over the mismatched dims. You get a tensor out. It has the
wrong values. Training proceeds. Losses look weird but finite. A week later
you notice the model is not learning attention at all.

With asserts at the layer boundary:

```zig
debug.shape.assertShape(q_heads, Shape.init4D(B, n_head, T, d_head));
// shape assert: expected (4, 4, 16, 16), got (4, 16, 4, 16)
// thread panic: shape assertion failed
```

The panic fires the moment the wrong-shaped tensor leaves your new code, not
five ops downstream. You fix the missing `transposeAxes12_4d`, rerun, pass.

### When to use which

- `assertShape`: use when you know the exact expected shape. Most useful
  immediately before an op that would silently accept a wrong shape.
- `assertRank`: use when the shape's dims depend on runtime values but the
  rank is fixed. A common case: "this should be rank-3 (B, T, D)".
- `assertDim`: use when a single axis must be a specific size but the others
  are free. Example: "axis 2 must equal `vocab_size` for the final logits".

### Output format

All three print to stderr via `std.debug.print`, the same pattern used
throughout `src/autograd/backward.zig`. The message includes the expected and
observed shapes in their natural `(d0, d1, ...)` form so you can read them
without decoding field order. On the failing test suite:

```text
  assertShape FAIL: expected (3, 5), got (3, 5, 1)
  assertRank FAIL: expected rank 3, got rank 2 (shape (3, 5))
  assertDim FAIL: expected axis 0 = 5, got 2 (shape (2, 3, 4))
  assertDim FAIL: axis 4 out of range for rank-2 shape (3, 5)
```

(These are from the negative-path tests in `src/debug/shape.zig` itself —
`zig build test` exercises every failure mode.)

### Stage 8 M8 real-world use

The multi-head attention refactor in Milestone 4 added three new `assertShape`
calls at the boundaries of `splitHeads` and `mergeHeads`. Those asserts caught
two bugs during development that would have produced subtly wrong gradients
otherwise.

---

## 2 — NaN / Inf detection workflow

After shape, the second-most-common silent failure is numerical — a NaN
appears somewhere, propagates through linear operations, and eventually blows
up loss. The moment it first appears is usually not where it causes visible
damage, so a trip-wire that catches it early saves hours.

`src/debug/finite.zig`:

```zig
pub fn assertFinite(t: Tensor) void;
pub fn hasNaN(t: Tensor) bool;
pub fn hasInf(t: Tensor) bool;
```

All three are device-aware. On CUDA they DtoH-copy the tensor into a scratch
host buffer, scan it, and free. This is expensive (don't put it in a hot
loop), but it's exactly the right cost-for-signal ratio during debugging.

### Common sources

| Symptom | Where to look first |
|---|---|
| NaN appears in logits at step 50 | softmax overflow: some row of logits has a huge positive value |
| NaN in loss, finite in logits | targets index is out of range (OOB read into scratch buffer) |
| Inf in gradients | no gradient clipping + large learning rate + sparse embedding collision |
| NaN in LayerNorm output | variance went to zero, missing `eps` in the `1 / sqrt(var + eps)` |
| NaN in log-softmax | numerical underflow in the exp stage before log |

### Tripwire pattern in a training loop

```zig
for (0..max_steps) |step| {
    // ... forward pass ...

    var loss = try crossEntropy(alloc, logits, targets, &tape);

    // Trip-wire. If anything upstream went non-finite, crash here with
    // a concrete step number rather than letting it propagate into
    // backward where it will be much harder to localise.
    if (builtin.mode == .Debug) {
        debug.finite.assertFinite(loss);
    }

    // ... backward + step ...
}
```

### Softmax overflow, the worked example

The canonical case: your logits have an element around 100. `exp(100)` is
already `2.7e43`. On f32 the max finite value is `3.4e38` — softmax
immediately produces an Inf, which divides into a NaN.

The fix everyone writes eventually is the max-subtract trick:

```
p_i = exp(z_i - max_z) / sum_j exp(z_j - max_z)
```

Our `src/tensor/ops/softmax.zig` does this. If you're writing a fresh softmax
variant or using an op that does not protect against overflow, add the
`assertFinite` at the boundary so the failure mode is loud.

### Finding the source

When `assertFinite` fires deep in a network, the cause is almost always
upstream. Walk the chain backwards:

1. Is the gradient that produced this value finite?
2. Is the weight being updated by a finite amount?
3. Is the previous activation finite?
4. Are the inputs finite?

If you reach the inputs and they're finite, the problem is in one of the ops.
If an input is non-finite, it's either bad data or a bug in data loading.

---

## 3 — Gradient checking a new op

When you add a new op with a custom backward, the question is: "does the
backward I wrote actually compute the gradient of the forward I wrote?" The
only robust answer is a finite-difference comparison.

`src/autograd/gradcheck.zig` ships `gradCheckTwoSided` which runs
`f(x + h) - f(x - h) / (2h)` numerically for each input element, compares
against the analytical gradient from the backward pass, and returns a relative
error. You call it on a scalar-valued function wrapping your op.

```zig
const result = try gradcheck.gradCheckTwoSided(alloc, &inputs, &params, my_loss_fn, .{
    .h = 1e-3,
    .denom_floor = 1e-2,
});
try std.testing.expect(result.max_rel_err < 0.05 or result.max_abs_diff < 1e-2);
```

### The `denom_floor` rule

Gradient check relative errors are `|analytical - numerical| / max(|analytical|,
|numerical|, denom_floor)`. Without a floor, near-zero gradients produce
inflated relative errors that look like bugs but are actually just
finite-difference noise.

Our project's calibrated value is `denom_floor = 1e-2`. With smaller values
(e.g. `1e-8`) you get false-positive test failures on degenerate cases (e.g.
`sumAll(softmax(x))` which has an exact-zero gradient because softmax rows
always sum to 1).

The combined assertion pattern used everywhere:

```zig
try testing.expect(result.max_rel_err < 0.05 or result.max_abs_diff < 1e-2);
```

Either we got the gradient right to within 5 % relative error, or both
gradients are so small that the relative error is meaningless.

### Two-sided vs one-sided

Always prefer two-sided (`(f(x+h) - f(x-h)) / (2h)`). One-sided has O(h) error;
two-sided has O(h²). At `h = 1e-3` that's the difference between 0.1 %
accuracy (useful) and 0.001 % accuracy (extremely useful).

### When gradient check fails

Walk through these in order:

1. **Is the forward math what you think it is?** Write it out symbolically,
   differentiate it by hand, compare to your backward. More often than not
   the backward is a correct gradient of a slightly-wrong forward.
2. **Is `saved` data in SavedData a snapshot or a pointer?** Zig ops take
   Tensor by value, so `@constCast(&param)` produces a dangling pointer.
   Snapshot by value. (This tripped the Stage 3 implementation three times.)
3. **Is the reshape tracked?** An untracked `reshape` breaks the gradient
   graph — the backward re-enters at the wrong shape or tape ID. Use
   `ops_shape.reshapeTracked`.
4. **Is the accumulator correct?** If your op reads from an input multiple
   times (e.g. `x * x`), the backward must accumulate `dL/dx` from each path.

### Stage 8 M4 real-world use

The multi-head attention refactor introduced a new `transposeAxes12_4d` op.
Its backward is self-inverse (applying the same permutation twice is the
identity), but getting the stride math right took two gradcheck iterations.
The passing gradcheck is in `src/autograd/gradcheck.zig`'s test block; the
failure modes looked like plausible 5–10 % relative error until the
derivation was double-checked.

---

## 4 — CPU vs GPU output comparison

Stage 7 introduced a CUDA backend. The defining question for every CUDA op
is: "does it produce the same output as the CPU version, within some
tolerance?" The answer requires being able to dump and compare tensors across
devices.

Two helpers, both in `src/debug/`:

```zig
// Compare two tensors elementwise. Prints a CompareReport with
// max_abs_diff, max_rel_err, worst_idx, and pass/fail vs tolerances.
pub fn compare(alloc, a: Tensor, b: Tensor, opts: CompareOpts) !CompareReport;

// Dump a tensor to disk in the .ztlt format used by our PyTorch
// oracle, so it can be reloaded later (same device or cross-device)
// or inspected in Python.
pub fn dump(alloc, io, path, t: Tensor) !void;
pub fn load(alloc, io, path) !Tensor;
```

Both are device-aware. `compare` handles any combination of CPU/CUDA
operands via DtoH copy when needed. `dump` writes bytes that round-trip
losslessly; CUDA tensors DtoH-copy during dump.

### The `.ztlt` format

A trivial binary format:

```
[4]u8 magic      "ZTLT"
u8    rank
[3]u8 padding    (zeros)
[rank]u32 dims
[elems]f32 data  (little-endian)
```

Total header is 4+1+3+4·rank bytes. Files from this project can be loaded
with:

```python
import numpy as np, struct
data = open("my_tensor.ztlt", "rb").read()
assert data[:4] == b"ZTLT"
rank = data[4]
dims = struct.unpack(f"<{rank}I", data[8:8+4*rank])
tensor = np.frombuffer(data[8+4*rank:], dtype=np.float32).reshape(dims)
```

This is the same format `tools/oracle.py` uses to emit PyTorch fixtures. If
you're looking at a mysterious tensor divergence, a handy workflow is:

1. Dump the CPU tensor to `zig-out/cpu.ztlt`.
2. Dump the CUDA tensor to `zig-out/cuda.ztlt`.
3. Load both in a Python REPL.
4. Subtract, inspect `.max()`, `.argmax()`, visualise with matplotlib.

### CompareReport fields

```zig
pub const CompareReport = struct {
    max_abs_diff: f32,    // worst absolute difference
    max_rel_err: f32,     // worst relative error
    worst_idx: usize,     // flat index where the worst occurred
    passed: bool,         // did both diffs clear the tolerance?
};
```

The `worst_idx` is gold: it's the element in the flattened tensor where the
comparison failed hardest, so you can inspect it specifically. For a (B, T,
V) logits tensor, `worst_idx = 37` translates back to (b=0, t=2, v=5) via the
strides; print both tensors' values at that index and the difference will be
obvious (e.g. `inf` vs `42.3`).

### Tolerances

Our project's standard tolerances come from `docs/stage7_endgame_plan.md`
and apply throughout:

- Forward parity: `rel_tol = 1e-3, abs_tol = 5e-4`.
- Backward parity: `rel_tol = 1e-3, abs_tol = 1e-3`.
- One-step parity: `abs_tol = 2e-3`.

Use them as defaults; tighten only after you've confirmed the implementation
is actually more accurate than the default allows.

---

## 5 — Overfit-one-batch smoke test

Before trusting a training loop, run it on a batch of size B and a dataset of
size B and confirm the loss goes to zero. If it does, the forward +
backward + optimiser pipeline is correct end-to-end. If it does not, one of
them is broken and you can zoom in.

`examples/04_overfit_one_batch.zig` is this test baked in.

```
$ zig build run-example -Dexample=04_overfit_one_batch
=== 04: Overfit one batch ===
config: V=32 D=16 T=8 B=2
Step 0:  loss = 3.8321
Step 10: loss = 3.1290
Step 20: loss = 2.2541
...
Step 90: loss = 0.6203  Loss decreased!
Step 99: loss = 0.5375  Loss decreased!
=== Training Complete ===
Final loss: 0.5375
Training pipeline works!
```

### Why this works

With a dataset smaller than the model's capacity, the model can simply
memorise the data. Any pipeline bug that interferes with memorisation — a
broken gradient, a wrong shape, a numerical hole, an optimiser that doesn't
update — will show up as loss plateauing above a small threshold.

Conversely, if loss goes to zero, then your forward produces gradients that
flow backwards correctly, your backward accumulates them correctly, your
optimiser applies them correctly, and your data pipeline doesn't corrupt
anything. It doesn't prove the model learns distributions well on held-out
data — that's a bigger question — but it proves the plumbing is sound.

### When to run it

- After any non-trivial refactor that touches forward/backward/optimizer.
- After adding a new op with gradient.
- As a CI check (our `zig build test` exercises a miniature version at
  20 steps).
- Before shipping a stage commit.

### What to do when it plateaus

Loss plateau somewhere around initial entropy (~log(V)) means the model is
not learning. Usual causes:

1. **Gradient is zero**: check `param.grad.data` after backward. Zero
   gradient means either the loss doesn't depend on that param (bug in
   the forward graph) or the backward didn't accumulate into it (bug in
   your backward or tape).
2. **Gradient is non-zero but tiny**: check the optimiser's learning rate
   against the gradient magnitude. A 1e-8 gradient times a 1e-3 learning
   rate is a 1e-11 weight change — you'll need 1e9 steps to see anything.
3. **Loss is noisy around log(V)**: model is probably initialising to bad
   weights and failing to escape the saddle. Try a different seed; if that
   changes nothing, the model is numerically broken.
4. **Forward computes something, but backward doesn't flow**: check that
   every op in the forward was tracked (`*Tracked` variants for reshape /
   transpose). Untracked reshapes in a training loop are a silent-bug
   generator.

---

## 6 — compute-sanitizer walkthrough

`compute-sanitizer` is NVIDIA's equivalent of Valgrind — it wraps CUDA API
calls and kernel launches to catch illegal memory accesses, memory leaks,
race conditions, and similar device-side bugs. You run it on a binary; it
reports problems.

### Invocation

On our RTX 4060 Ti remote (Ubuntu 24.04 + CUDA 13.2):

```bash
$ /usr/local/cuda-13.2/bin/compute-sanitizer \
    --tool=memcheck \
    --leak-check=full \
    ./.zig-cache/o/<hash>/10_train_deep --sanitize
```

`memcheck` is the memory-safety tool. There are other tools (`racecheck`,
`synccheck`, `initcheck`); memcheck catches the bulk of real bugs.

`--leak-check=full` asks for per-allocation leak reports at exit. Without it
you get aggregate totals; with it you get stack traces to every leaked
allocation.

### Expected clean output

From the Stage 8 M8-f acceptance run of `examples/10_train_deep.zig`:

```text
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
```

That is the state we want on every Stage 7+ training run.

### Three broken kernels and what they look like

#### 6a. Illegal memory access (dropped bounds check)

A kernel that writes past the end of its output buffer:

```cuda
extern "C" __global__ void elw_add_bad(
    const float* a, const float* b, float* y, uint32_t n)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // BUG: missing `if (i < n) return;`
    y[i] = a[i] + b[i];
}
```

At grid-size = ceil(n / blockDim.x) threads, the last block has some
threads with `i >= n`. They write past the buffer. Sanitizer output:

```text
========= Invalid __global__ write of size 4 bytes
=========     at elw_add_bad+0xa0
=========     by thread (32,0,0) in block (3,0,0)
=========     Address 0x7f9b3a1000fc is out of bounds
```

The fix is the missing guard. Every production kernel in
`src/backend/cuda/kernels/` has one.

#### 6b. Non-coalesced vs. wrong-stride access (layernorm_rowwise)

A kernel that indexes through shared memory with the wrong stride — not
strictly illegal memory access, but each warp loads from addresses that
don't coalesce into a single transaction:

```cuda
extern "C" __global__ void ln_rowwise_bad(const float* x, float* y, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    // BUG: stride D means thread 0 reads x[row*D+0], thread 1 reads
    //      x[row*D+D], thread 2 reads x[row*D+2D] — not coalesced.
    float v = x[row * D + tid * D];
    y[row * D + tid] = v;
}
```

`memcheck` won't flag this (no illegal access), but a profile with
`Nsight Compute` will show 1/32 memory throughput.

#### 6c. Host/device pointer mixup

Passing a host pointer to a device kernel:

```cuda
extern "C" __global__ void read_x(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];  // crashes if x is a host pointer
}
```

Caller-side bug in Zig:

```zig
const host_array: [16]f32 = ... ;
// BUG: passing &host_array[0] where a CUdeviceptr is expected.
try kernel.launch(.{ &host_array[0], dev_y, 16 }, ...);
```

Sanitizer output:

```text
========= Invalid __global__ read of size 4 bytes
=========     at read_x+0x40
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7ffd... is not a device pointer
```

### What to do when sanitizer finds something

1. **Read the address**: is it an allocation we made, or out of bounds?
2. **Find the stack trace**: the `at kernel_name+0x<offset>` points at the
   instruction inside the compiled PTX. Disassembling that PTX
   (`cuobjdump --dump-sass`) lets you correlate with source.
3. **Reduce the input**: shrink the workload to the smallest repro.
4. **Compare against a known-good version**: `git bisect` if the bug was
   recent.

### Zig-specific quirks

- `tape.kept_alive_cuda` on the Tape stores DtoD-copied DeviceBuffers; a
  leak there shows up as "DeviceBuffer deinit never called". Our
  `Tape.deinit` iterates this list explicitly.
- `model.moveToCpu()` + `model.moveToCuda(&ctx)` round-trips must free
  the old-device storage; sanitizer catches if they don't.
- `CudaContext.deinit` must run AFTER every DeviceBuffer is freed. The
  Trainer's deinit ordering is explicit about this: `model.deinit()`
  first (releases all DeviceBuffers), then `ctx.deinit()`.

---

## 7 — Nsight Compute beginner pass

`ncu` (Nsight Compute) profiles individual kernel launches and tells you
about warp occupancy, memory throughput, compute utilisation, and dozens of
other metrics. It's a tool for perf work, not correctness — but an early
pass helps you catch "this kernel is running at 1% efficiency" before you
optimise the wrong thing.

### Invocation

```bash
$ /usr/local/cuda-13.2/bin/ncu \
    --target-processes all \
    --launch-count 1 \
    --kernel-name softmax_last \
    --set basic \
    ./.zig-cache/o/<hash>/10_train_deep --sanitize
```

`--launch-count 1` captures just the first launch of the named kernel.
`--set basic` picks a cheap metric set that gives you the essentials
without the overhead of a full profile.

### The kernel name filter

`ncu --target-processes all <binary>` without a filter lists every kernel
the binary launches. From the Stage 8 M8-f acceptance run of
`examples/10_train_deep`:

```
1.  adamw_step
2.  ampere_sgemm_128x128_nn
3.  ampere_sgemm_32x128_nn
4.  ampere_sgemm_64x32_sliced1x4_nn
5.  ampere_sgemm_64x64_nn
6.  bcast_copy
7.  ce_fused
8.  elw_add
9.  elw_add_scalar
10. elw_broadcast_add
11. elw_broadcast_div
12. elw_broadcast_mul
13. elw_broadcast_sub
14. elw_div
15. elw_mul
16. elw_mul_scalar
17. elw_neg
18. embedding_backward
19. embedding_forward
20. gemmSN_NN_kernel
21. reduce_sum_axis
22. softmax_last
23. splitKreduce_kernel
24. unary_gelu_exact
25. unary_gelu_exact_backward
26. unary_sqrt
```

`ampere_sgemm_*` and `gemmSN_NN_kernel` are cuBLAS internals, not our code.
The rest map 1:1 to kernels in `src/backend/cuda/kernels/*.cu`.

### What to look at

For `softmax_last` on a `(B·H, T, T) = (16, 16, 16)` tensor with the 2/2/64
config:

- **Achieved occupancy**: should be > 50 % for a memory-bound op, > 75 %
  for a compute-bound op. If it's 12 % the kernel is launching with too
  few blocks or too many registers per thread.
- **Memory throughput**: should approach the GPU's peak bandwidth (~448
  GB/s on the RTX 4060 Ti) for memory-bound ops.
- **SM efficiency**: indicates how well the kernel uses all streaming
  multiprocessors. Low SM efficiency means the kernel is launching too
  few blocks.

### Permissions gotcha

On the M8-f run, `ncu --set basic` returned:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

NVIDIA locked performance counters behind root in driver releases after
~2019. To unblock without sudo-ing every run, add a `/etc/modprobe.d/`
rule — see
<https://developer.nvidia.com/ERR_NVGPUCTRPERM> for the canonical
instructions. For a quick one-off, running `ncu` under `sudo` works.

Until we unblock this, Nsight Compute is an "inspect the kernel catalog"
tool on our remote, not a "measure the actual numbers" tool. The qualitative
pass (kernel list, launch counts, grid/block configs) is still useful for
catching "this kernel is being launched 3× more often than it should" bugs.

---

## 8 — Common CUDA errors catalog

This is a catalogue of six specific classes of bug we've actually
encountered across Stages 7 and 8, with their symptoms, failure signatures,
and fixes. Bookmark this section — it's the fastest shortcut when a CUDA
test fails on you.

### 8.1 Illegal memory access (dropped bounds check)

**Symptom.** Random crashes, sanitizer reports "Invalid __global__ read/write".

**Failure signature.** Test passes on small N, fails on N not divisible by the
block size. Example: kernel works at N=1024, crashes at N=1023.

**Fix.** Every kernel must begin with `if (i >= n) return;`. Verify by
reading the kernel source; if the early-return is missing, add it.

**Real example.** `tests/integration_cuda.zig: "cuda vector_add kernel:
bounds check prevents OOB for N not divisible by block size"` is a
regression test for this exact class.

### 8.2 Wrong grid / block dimensions (off-by-one on T)

**Symptom.** Last row / last column of output is stale (still contains the
previous run's data, or zeros).

**Failure signature.** Oracle parity test passes on all rows except the last.
Debug-dump the output and the last row is visibly different.

**Fix.** Grid dim should be `ceil(N / block_dim)`, not `N / block_dim`.
Integer division truncates. For a 2D kernel processing a (B, T) tensor,
the grid should be `(ceil(B / bx), ceil(T / by))`.

### 8.3 Non-coalesced memory access

**Symptom.** Correct output, but kernel runtime is 10–30× higher than
expected.

**Failure signature.** Nsight Compute shows memory throughput at a fraction
of peak. Warps spend most cycles stalled on memory.

**Fix.** Each warp's 32 threads should access 32 consecutive float
addresses. Rewrite the indexing so `threadIdx.x` varies the fastest
(innermost) dimension of the read.

**Real example.** An early draft of `layernorm_rowwise` had threads stride
across rows instead of down columns. Rewriting to row-major-sequential
brought runtime from 1.2 ms to 0.08 ms.

### 8.4 Missed `cudaStreamSynchronize`

**Symptom.** Reading loss scalar on host returns stale / zero values.

**Failure signature.** Loss appears to be exactly zero for step 0, then
updates to real values from step 1 onwards.

**Fix.** Either `ctx.synchronize()` before the host read, or use a blocking
DtoH copy. Our `Tensor.toCpu` is blocking (uses `cudaMemcpy`, not the
async variant), so reading via `loss.toCpu()` implicitly synchronises the
default stream.

### 8.5 Host / device pointer mixup

**Symptom.** "Invalid __global__ read/write" inside a kernel, but the
address is "not a device pointer".

**Failure signature.** Sanitizer reports an address in the high-stack
region (0x7ffd...) or the binary's .bss region (0x55...) rather than the
usual device-memory region (0x7f9b...).

**Fix.** Trace the pointer back to its origin. It's either (a) a CPU
`Tensor.data` slice where you meant `DeviceBuffer.ptr`, or (b) a stack
local that happened to be passed by address.

**Zig-specific variant.** `@constCast(&by_value_param)` creates a pointer
to a stack-local copy of a function parameter. When stored in SavedData
and dereferenced later (after the function returns), the stack is
overwritten and you get garbage. Snapshot by value instead.

### 8.6 Row-major vs column-major in cuBLAS

**Symptom.** GEMM output has the right shape but wrong values; every
entry looks transposed somehow.

**Failure signature.** `oracle matmul_2d forward parity` fails with
`max_abs_diff` on the order of the input magnitudes (not small floating
noise — structurally wrong).

**Fix.** cuBLAS GEMM is column-major; our tensors are row-major. The
trick we use in `src/backend/cuda/gemm.zig`:

```
For C = A @ B in row-major:
  C^T = B^T @ A^T  (column-major view of the same bytes)
So call cuBLAS with: op_A = B, op_B = A, output = C.
```

Operand swap. The tests in `src/backend/cuda/gemm.zig`'s test block
document the derivation.

### 8.7 (bonus) cuBLAS rejects non-contiguous transpose views

**Symptom.** `matmulBatch` returns `error.InvalidLayout` on the backward
pass of an attention op.

**Failure signature.** Forward parity passes; backward parity fails at the
matmulBatch in `backwardMatmulBatch` with a layout error.

**Fix.** cuBLAS batched GEMM requires all three operand tensors to be
contiguous in memory. A transpose view (produced by
`transpose2d` / `transposeInner2d`) is a stride-swapped view of the
original — the bytes are not in the order cuBLAS wants.

Materialise with `cuda_dispatch.broadcastTo(view, view.shape)` before the
matmul. This makes a contiguous copy. Precedent: `backwardMatmul` and
`backwardMatmulBatch` both do this; search for `broadcastTo` in
`src/autograd/backward.zig`.

---

## Further reading

- `docs/03_autograd.md` — gradient computation via tape; why gradcheck
  matters.
- `docs/07_cpu_training.md` — the CPU training loop this chapter's
  debugging tools support.
- `docs/08_backends_cuda.md` — CUDA backend architecture; where the
  kernels this chapter's tools help you debug are actually defined.
- `docs/stage7_endgame_plan.md` — the tolerance-band system that
  `compare` reports against.
- NVIDIA Compute Sanitizer docs:
  <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
- NVIDIA Nsight Compute CLI reference:
  <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>

---

*End of Chapter 9 — Debugging: shape asserts, gradient checks, CUDA sanitizers.*


---

## Exercises

**Exercise 1.** You've added a new op `my_op` and its gradient
check passes at tolerance `1e-4`. A week later, a refactor touches
the forward and the gradient check now fails with
`max_rel_err = 0.3` but `max_abs_diff = 2e-5`. Is this a
backward bug?

<details><summary>Solution</summary>

Probably not. `max_abs_diff = 2e-5` is tiny, which means the
gradient itself is near zero at the test input. The relative error
measure explodes for near-zero values (any small numerical noise
divided by a tiny denominator blows up). Our combined assertion
`max_rel_err < 0.05 OR max_abs_diff < 1e-2` exists for this
reason. If you see this pattern on a freshly failing test, look
at the absolute error and `denom_floor` before assuming the
backward is wrong.

If `max_abs_diff` were also large (say `1e-2`), that *would*
indicate a real backward bug.

</details>

**Exercise 2.** A CUDA training run passes all parity tests in your
suite but `examples/10_train_deep --sanitize` reports a memory
leak on step 20. What's the most likely cause?

<details><summary>Solution</summary>

A tape-owned tensor that isn't being freed when the tape deinit
fires. The per-step pipeline creates many intermediate CUDA
tensors; each goes through `Tape.cloneTensorData` which either
snapshots a CPU buffer or DtoD-copies a device one, both of which
are freed in `tape.deinit`. If a new op registered after the
main test suite doesn't route its allocation through the tape,
it leaks per step.

Debug approach: shrink `max_steps` to 1, then 2, then 5. If the
leak is N-bytes-per-step, linear scaling confirms per-step leak.
Then enable verbose logging in `Tape.deinit` to print every
buffer it releases; the count should match the count of buffers
allocated inside the step.

</details>
