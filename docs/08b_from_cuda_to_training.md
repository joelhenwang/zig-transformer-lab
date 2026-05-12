# 08b — From CUDA Backend to Training on GPU

> Bridges `docs/08_backends_cuda.md` (CUDA mechanics) to the ML/DL
> concepts the backend exists to serve. Assumes you've read that chapter
> and now want the conceptual picture: what does "training on GPU"
> actually mean, how do tensors move between devices, and what pitfalls
> trip up people moving from CPU-only training?
>
> Companion code: `src/lab/train.zig`, `src/nn/model.zig:moveToCuda`,
> `src/backend/cuda/*`, `examples/08_cuda_vs_cpu.zig`,
> `examples/09_cuda_benchmark.zig`, `examples/10_train_deep.zig`.

The Stage 7 CUDA backend and the Stage 8 Trainer CUDA path together mean
a single boolean — `TrainConfig.use_cuda = true` — switches your whole
training pipeline from CPU to GPU. This chapter unpacks what that
boolean actually does, step by step, from the moment you flip it to the
moment you call `trainer.deinit`.

---

## 1. What it means to move a model to GPU

The phrase "move the model to GPU" compresses a lot of bookkeeping. In
our library, that single line is:

```zig
try model.moveToCuda(&ctx);
```

Under the hood (`src/nn/model.zig:242`), it iterates every learnable
parameter and does three things per parameter:

1. Allocate a fresh `DeviceBuffer` on the CUDA context's device.
2. HtoD-copy the CPU `[]f32` into that buffer.
3. Deinit the CPU buffer, overwrite the `Tensor`'s `storage` field
   with the new `.cuda = DeviceBuffer{...}` variant, and flip its
   `device` tag to `.cuda`.

After this loop the model's parameter tensors no longer have any
CPU-side data — `tensor.data.len == 0`. The backing floats live on
the GPU.

### 1.1 Which tensors are "the model"?

`TinyWordTransformer.parameters(&list)` walks every learnable tensor:

```
tok_embed.weight            (V, D)
pos_embed.weight            (max_seq_len, D)
for each block i in [0, n_layer):
    blocks[i].ln1.gamma     (D,)
    blocks[i].ln1.beta      (D,)
    blocks[i].attn.w_q.weight
    blocks[i].attn.w_k.weight
    blocks[i].attn.w_v.weight
    blocks[i].attn.w_o.weight
    blocks[i].ln2.gamma     (D,)
    blocks[i].ln2.beta      (D,)
    blocks[i].mlp.fc1.weight
    blocks[i].mlp.fc2.weight
ln_f.gamma
ln_f.beta
lm_head.weight              (V, D)
```

That's `2 + 10 * n_layer + 3` tensors. For the Stage 8 2/2/64 config:
`2 + 20 + 3 = 25` tensors, total ~0.5 MB of f32 weights. All move to
the GPU in one `moveToCuda` call; all come back to CPU in one
`moveToCpu` call.

### 1.2 What stays on CPU

Deliberately left on CPU:

- The **model struct itself** (the `TinyWordTransformer` object
  containing pointers to parameter tensors). It's host-side
  bookkeeping.
- The **dataset's token buffer** (`dataset.tokens: []u32`). Too small
  to benefit from device residency; uploaded into device tensors per
  step on demand.
- The **optimizer state** (`AdamW.state` HashMap). The state values
  (moment vectors) end up on whichever device their paired parameter
  lives on, but the HashMap itself is host-side.
- The **tape** structure. Nodes and edges are host-side metadata; the
  `SavedData` payloads point at device memory when the op ran on GPU.

### 1.3 The asymmetry

The CPU side is always "live" (the model struct exists, the tape
exists, the trainer exists). The GPU side is "data only" — device
memory holding weight values, gradients, intermediate activations.
Kernel launches bridge the two: the CPU thread tells the GPU what to
do, the GPU does it, results land in device memory.

This asymmetry is why `CudaContext.init` on a machine without a GPU
fails at context creation (we can't even open the bridge), not at
model init.

---

## 2. Per-step device residency walk

Let's trace one training step of `examples/10_train_deep.zig`
(2/2/64 config) through device residency. Assume the CUDA context is
already built and `model.moveToCuda` has been called.

### 2.1 Before the step

```
CPU:  TinyWordTransformer struct (pointers + metadata)
      AdamW struct + state HashMap (ParamId -> {m, v, step})
      Tape (empty)
      Dataset.tokens: []u32 (the whole corpus)
      Batcher indices
GPU:  Parameter tensors (25 buffers, ~0.5 MB total)
      AdamW moment tensors (25 "m" + 25 "v", created lazily)
```

### 2.2 Per-step timeline

```
Step N enters the loop.

(a) batch = batcher.next()
      CPU: reads from dataset.tokens (borrowed view).
      No GPU action.

(b) ids_host = Tensor(alloc, (B,T)) + cast u32->f32
      CPU: 256 bytes allocated on host.

(c) targets_host = Tensor(alloc, (B*T,)) + cast u32->f32
      CPU: 256 bytes allocated on host.

(d) ids_cuda = ids_host.toCuda(&ctx)
      HtoD copy: 256 bytes CPU -> GPU.
      GPU: 256 bytes allocated.
      Block until memcpy completes (default stream is synchronous).

(e) targets_cuda = targets_host.toCuda(&ctx)
      Same: 256 bytes HtoD.

(f) tape = Tape.init(alloc)
      CPU: tape struct + arrays.

(g) trackLeaf each param
      CPU: tape nodes enumerate parameters.
      GPU: no action yet.

(h) model.forward(ids_cuda, &tape)
      GPU: sequence of kernel launches for every op:
        embedding_forward, add, layernorm (composed),
        matmul (cuBLAS), softmax_last, matmulBatch, ...
      Intermediate tensors allocated on GPU inside the tape.
      CPU: tape records OpKinds and SavedData snapshots.

(i) reshapeTracked logits_3d -> logits_2d
      View, no data movement. Tape records reshape op.

(j) crossEntropy(logits_2d, targets_cuda, &tape)
      GPU: ce_fused kernel produces loss scalar + grad_logits.
      loss tensor: 4 bytes on GPU.

(k) loss_val = loss.toCpu().data[0]
      DtoH copy: 4 bytes GPU -> CPU.
      Implicitly synchronises default stream; forward kernels must
      complete before the read returns.

(l) tape.backward(&loss)
      GPU: one kernel launch per op's backward.
      Each param's grad tensor is allocated on GPU.

(m) Gradient clipping (device-agnostic)
      sumOfSquaresAll per gradient (pure-GPU: mul + sumAll + 4-byte DtoH).
      If clip_coeff < 1: scaleInPlace per param (pure-GPU: mulScalar kernel).
      No per-element HtoD/DtoH — only scalar reads.

(n) opt.step(params)
      GPU: adamw_step kernel reads params.data + grads + m + v,
           updates all in place. One launch per param.

(o) opt.zeroGrad(params)
      GPU: memset each grad buffer to 0 (one cuMemset per param).

(p) defer tape.deinit()
      GPU: every tape-owned buffer freed via cuMemFree_v2.
      CPU: tape struct freed.

(q) defer ids_host, targets_host, ids_cuda, targets_cuda
      Host + device buffers freed.
```

Total HtoD per step: `~520 bytes` (ids + targets).
Total DtoH per step: `~4 bytes` (loss scalar) + `~4 bytes × N_params`
(grad clip scalar sums). Kernel launches per step: ~200 (rough count).

### 2.3 What dominates the wall-clock

At the 1/1/32 Shakespeare config (~4.7 ms/step), kernel launches are
the bottleneck, not data movement or compute. The GPU is largely
idle between launches; each launch has ~10-20 µs of submission
overhead. With ~150 launches per step × 15 µs = ~2.25 ms just in
launch overhead.

At 2/2/64 (12.3 ms/step), the matmul FLOPs start to matter, but
launch overhead is still a significant fraction. A larger model
would push the balance toward compute-bound; our scale is too small
to exit launch-bound territory. This is why the `gradient clip
DtoH` trick is cheap at our scale — data movement is overlapped
with kernel work already in flight.

---

## 3. Why HtoD/DtoH is usually wrong inside a hot loop

The most common CUDA-performance mistake for beginners: doing a
per-step data transfer that "shouldn't cost much" and then
discovering the transfer is the bottleneck.

### 3.1 The numbers

RTX 4060 Ti specs:

- Peak memory bandwidth (device local): 288 GB/s
- PCIe 4.0 x16 theoretical: 32 GB/s each direction
- PCIe 4.0 x16 realistic: ~25 GB/s each direction
- DtoH / HtoD transfer latency: ~5-20 µs minimum per memcpy

A 4 KB tensor transferring at 25 GB/s costs about 160 ns in bandwidth
alone — but the API call has fixed overhead on the order of 10 µs.
For small transfers, you pay the fixed cost, not the bandwidth cost.

### 3.2 Our per-step HtoD

We copy 256-byte `ids` + 256-byte `targets` per step. Cost: ~30 µs
fixed + negligible bandwidth = ~30 µs per step. At 12 ms per step
that's 0.25% overhead. Acceptable.

If we tried to copy a 16 MB token embedding lookup per step (naive
implementation of an on-device embedding cache), we'd pay ~700 µs
per step = 5.8% overhead. Borderline but noticeable.

If we did the whole dataset (1 MB Shakespeare) per step, we'd pay
~40 µs per step (100 KB/s effective, fixed-cost-limited) — still
cheap. The pattern that breaks things is many small transfers, not
one large one.

### 3.3 The "sanitize" mode exception

`examples/10_train_deep.zig --sanitize` runs 50 steps instead of 200.
Per-step wall-clock under `compute-sanitizer` jumps from ~12 ms to
~470 ms. Most of that is the sanitizer's wrapper overhead, not our
data movement. Moral: "optimise for the uninstrumented case"; don't
tune around sanitizer numbers.

### 3.4 When it does bite

real perf bugs in this class:

- Reading `loss.cpuData()[0]` before `toCpu`. `cpuData()` asserts
  `device == .cpu` in debug builds — so this now panics loudly rather
  than silently returning stale data.
- Debug code that prints intermediate activations each step.
  Innocent-looking but triggers a DtoH per print. 100 prints × 10 µs
  = 1 ms extra per step on a 4.7 ms/step baseline. That's 20% slower
  from one logging line.
- `debug.finite.assertFinite(tensor)` in a hot path on CUDA. The
  helper DtoH-copies the whole tensor to scan for NaN. Useful in
  debug mode; move outside the hot loop in release.

---

## 4. The Trainer CUDA path annotated

`src/lab/train.zig` is the Stage 8 integration point where
`use_cuda = true` flips the whole pipeline. Here's the annotated
flow.

### 4.1 `Trainer.init` with `use_cuda = true`

```zig
// Standard CPU init (dataset, windowing, batcher, model on CPU)
...
var model = try TinyWordTransformer.init(alloc, model_cfg, &rng);
errdefer model.deinit();

// CUDA branch
if (cfg.use_cuda) {
    const ctx = try allocator.create(CudaContext);
    errdefer allocator.destroy(ctx);
    ctx.* = try CudaContext.init(allocator);
    errdefer ctx.deinit();

    // Pre-load every PTX module the step needs. Missing a module
    // here would fail at first kernel launch with CUDA error 500.
    const ptx_modules = [_][:0]const u8{
        "elementwise", "reduce", "softmax", "unary",
        "embedding", "ce_loss", "adamw",
    };
    for (ptx_modules) |name| _ = try cuda_module.loadPtxFromFile(ctx, io, name);

    try model.moveToCuda(ctx);
    ctx_ptr = ctx;
}
```

Key design: the context is **heap-allocated**. Every `DeviceBuffer`
carries a `ctx: *const CudaContext` pointer; a stack-local ctx would
dangle as soon as `Trainer.init` returned.

### 4.2 `Trainer.train` per-step branches

Within the per-step loop (`src/lab/train.zig:290`):

```zig
// host-side ids + targets tensors (always CPU first)
var ids_host = try Tensor.init(alloc, Shape.init2D(B, T));
defer ids_host.deinit(alloc);
// ... fill from batch ...

var targets_host = try Tensor.init(alloc, Shape.init1D(B * T));
defer targets_host.deinit(alloc);
// ... fill from batch ...

// Upload if on CUDA; else use host directly.
var ids_cuda: ?Tensor = null;
var targets_cuda: ?Tensor = null;
defer if (ids_cuda) |*t| t.storage.deinit(alloc);
defer if (targets_cuda) |*t| t.storage.deinit(alloc);
if (self.ctx) |ctx| {
    ids_cuda = try ids_host.toCuda(ctx);
    targets_cuda = try targets_host.toCuda(ctx);
}

const ids = if (ids_cuda) |t| t else ids_host;
const targets = if (targets_cuda) |t| t else targets_host;

// Forward + backward + step: device-agnostic; dispatch layer routes.
```

The `ids` / `targets` aliases let the rest of the loop read
identically on both devices. Everything downstream (tape, forward,
loss, backward, optimizer) is device-aware internally.

### 4.3 `Trainer.deinit` ordering

```zig
pub fn deinit(self: *Trainer) void {
    self.model.deinit();    // frees every parameter DeviceBuffer
    self.batcher.deinit();
    self.dataset.deinit();
    if (self.ctx) |ctx| {
        ctx.deinit();          // destroys CUDA context
        self.allocator.destroy(ctx);
    }
}
```

The order matters. `cuMemFree_v2` requires a live context; freeing
device memory after `ctx.deinit()` would leak it silently. We free
every DeviceBuffer first, then tear down the context.

---

## 5. Checkpointing across devices

Stage 8 M8-c made `TinyWordTransformer.save` and `.load` device-aware
so that a CUDA-resident model can checkpoint without going through
`moveToCpu` first.

### 5.1 Save path

For each parameter in the model:

```zig
switch (t.storage) {
    .cpu => |s| try w.writeAll(std.mem.sliceAsBytes(s.data)),
    .cuda => |b| {
        const scratch = try self.allocator.alloc(f32, n_elems);
        defer self.allocator.free(scratch);
        try b.copyToHost(scratch);
        try w.writeAll(std.mem.sliceAsBytes(scratch));
    },
}
```

One DtoH per parameter, one scratch buffer at a time. Peak scratch
memory is the largest single parameter — the `lm_head.weight` at
`(V, D) = (2000, 64) = 512 KB`. Negligible.

### 5.2 Load path

Symmetric: read bytes into scratch, HtoD-copy into the param's
existing DeviceBuffer. The destination model must already have its
parameters on CUDA; the load path doesn't create new DeviceBuffers.

### 5.3 Cross-device checkpoints

A checkpoint file is device-agnostic. You can:

- Save from CUDA, load into a CPU-only model elsewhere (the test
  `cuda checkpoint: save from CUDA then load on a fresh CPU model`
  verifies this).
- Save from CPU, load into a CUDA model (a dev fine-tuning workflow).
- Save and load on the same device (the normal case).

The ZTLC v3 format stores raw f32 bytes in little-endian — no device
metadata in the header. Device residency is a property of the
runtime, not the checkpoint.

---

## 6. Compute-sanitizer and Nsight Compute: quick recipes

### 6.1 compute-sanitizer for correctness

```
/usr/local/cuda-13.2/bin/compute-sanitizer \
    --tool=memcheck --leak-check=full \
    ./.zig-cache/o/<hash>/10_train_deep --sanitize
```

Expected output on a clean run:

```
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
```

Use `--sanitize` to run in 50-step mode so the wrapper overhead
(10-50× slowdown) doesn't burn a whole afternoon.

Other tools:

- `--tool=racecheck` for intra-kernel race conditions (rarely
  relevant for us — our kernels don't use shared memory
  extensively).
- `--tool=synccheck` for incorrect synchronisation primitives.
- `--tool=initcheck` for uninitialised-memory reads. Useful when
  debugging new kernels.

### 6.2 ncu for kernel-level perf

```
/usr/local/cuda-13.2/bin/ncu \
    --target-processes all \
    --launch-count 1 \
    --kernel-name softmax_last \
    --set basic \
    ./.zig-cache/o/<hash>/10_train_deep --sanitize
```

Filters to the first launch of the named kernel, prints occupancy /
memory throughput / SM efficiency. Requires root or
`modprobe.d` override on most systems (ERR_NVGPUCTRPERM otherwise).

Without a filter, `ncu` lists every kernel the binary launched —
useful for "am I launching more kernels than I think?" questions.
See `docs/09_debugging.md` §7 for the kernel-list interpretation.

### 6.3 Useful one-off checks

- `nvidia-smi dmon -s u -c 10` — 10 seconds of GPU utilisation
  sampling. If your training step uses the GPU at < 50% utilisation,
  you're launch-bound and would benefit from larger batches or kernel
  fusion.
- `nvidia-smi --query-gpu=memory.used --format=csv` — current GPU
  memory used. Baseline + model memory tells you headroom for scaling.

---

## 7. Common mistakes

- **Forgetting to load a PTX module.** Every kernel a training step
  might launch must correspond to a loaded PTX module. A silent
  regression here surfaces as `cuda driver error 500: named symbol
  not found` at first launch. Keep the `ptx_modules` list in
  `Trainer.init` synced with the op set.
- **Mismatching CudaContext pointers.** If two tensors accidentally
  carry different `ctx` pointers (different contexts), cuBLAS may
  silently run on the wrong device or error with cross-context
  warnings. Every tensor in a graph must share one context.
- **Freeing DeviceBuffers after context destruction.** See §4.3. The
  Trainer explicitly frees the model before the context; if you
  hand-roll a training loop, mirror that ordering.
- **Running a CUDA test on Windows without SkipZigTest.** Our
  convention: `if (comptime builtin.os.tag != .linux) return
  error.SkipZigTest;` at the top of every CUDA test. Forget it and
  the Windows test binary fails at `bindings.load()`.
- **Reading `tensor.data[0]` on a CUDA tensor.** `data` is the empty
  stub slice; the read either returns 0.0 or out-of-bounds-reads.
  Always `tensor.toCpu(alloc)` first.
- **Assuming `ctx.synchronize()` is needed after every op.** The
  default stream is synchronous to the CPU thread, so most intra-step
  operations self-sync. Explicit sync is only needed before reading
  host memory from device pointers (and `toCpu` / `.data[0]` on
  device-resident tensors already handle this).

---

## 8. Exercises

**Exercise 1.** You have `examples/10_train_deep.zig` running at
12.3 ms/step. You add a single `std.debug.print("{d}", .{loss_val})`
call per step. Expected impact?

<details><summary>Solution</summary>

Negligible. `loss_val` is already computed on CPU via
`loss.toCpu().data[0]` (implicit sync). Printing to stderr is O(us).
Overall step time change: < 1%.

If you instead added `std.debug.print("{}", .{logits.toCpu()})`
inside the step, you'd trigger a full DtoH of the logits tensor
(`(B, T, V) = (4, 16, 2000) = 512 KB`). Cost: ~50 µs transfer + print
overhead. Still < 1% of the step, but now the cost scales with vocab
and sequence length.

</details>

**Exercise 2.** Why does `Trainer.init` with `use_cuda = true`
allocate the `CudaContext` on the heap instead of as a struct field?

<details><summary>Solution</summary>

`Trainer.init` returns a `Trainer` by value. Zig copies the return
struct into the caller's memory location. If the context were a
direct field of the struct, the `*CudaContext` pointers stored in
every `DeviceBuffer` would point at the pre-copy struct location and
dangle after the return. By heap-allocating the context, we get a
stable pointer that survives the struct move. The Trainer just stores
a `?*CudaContext` and frees it in `deinit`.

</details>

**Exercise 3.** After `model.moveToCuda(ctx)`, the model's parameter
tensors have `data.len == 0`. But `t.storage.cuda.len > 0`. What is
the invariant that makes this safe?

<details><summary>Solution</summary>

The invariant: `t.device == .cuda` implies that every operation on
`t` must go through the device-dispatch switch in op code, which
reads from `t.storage.cuda` instead of `t.data`. The `data` slice
is preserved as an empty stub for backward-compatibility with code
that does `t.data.len` as an element count (returns 0 instead of
crashing). Operations that *should not* run on CUDA (e.g. `data[i]`
direct index) hit the zero-length slice and either bounds-check-fail
(safe) or return zero silently (unsafe — caught by compute-sanitizer).

A belt-and-braces improvement would be to change `data`'s type to
`?[]f32` and force every CPU-specific read site to handle the `null`
case. Not yet done; the current convention works in practice.

</details>

**Exercise 4.** Suppose you want to add a new op to the library —
say, `abs` (element-wise absolute value). What are the minimum steps
to get it working on both CPU and CUDA?

<details><summary>Solution</summary>

Step-by-step:

1. Add `OpKind.abs` to `src/autograd/node.zig`.
2. Implement `cpuAbs` in `src/tensor/ops/unary.zig`: elementwise
   loop over `input.data`, write `@abs(x)` to output.
3. Implement `backwardAbs` in `src/autograd/backward.zig`: multiply
   grad_out by `sign(x)`. `sign(x) = 1 if x > 0, -1 if x < 0, 0 if
   x = 0`.
4. Add a kernel to `src/backend/cuda/kernels/unary.cu`:
   ```cuda
   extern "C" __global__ void unary_abs(
       const float* x, float* y, uint32_t n) {
       uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) y[i] = fabsf(x[i]);
   }
   ```
5. Add a dispatch function `cudaAbs` in
   `src/backend/cuda/dispatch.zig` that launches `unary_abs`.
6. Wire the device switch at the top of `ops_unary.abs` — one line
   choosing between `cpuAbs` and `cudaAbs`.
7. Tests: one CPU test, one CUDA test (Linux-gated), one gradcheck
   entry.
8. Optional: add to `tools/oracle.py` as a new case for PyTorch
   parity.

About 50 lines of Zig + 8 lines of CUDA. Each subsequent op follows
the same template.

</details>

---

## 9. PyTorch equivalents

| Our code | PyTorch equivalent |
|---|---|
| `CudaContext.init(alloc)` | `torch.cuda.init()` (implicit on first GPU use) |
| `model.moveToCuda(&ctx)` | `model.to("cuda:0")` |
| `model.moveToCpu()` | `model.to("cpu")` |
| `tensor.toCuda(&ctx)` | `tensor.to("cuda:0")` or `tensor.cuda()` |
| `tensor.toCpu(alloc)` | `tensor.to("cpu")` or `tensor.cpu()` |
| `loss.toCpu(alloc).data[0]` | `loss.item()` |
| `ctx.synchronize()` | `torch.cuda.synchronize()` |
| `cfg.use_cuda = true` | `device = torch.device("cuda:0")` + `model.to(device)` |

Key difference: PyTorch hides the module-load step; our
`loadPtxFromFile` calls are explicit because we control the kernel
compilation. In PyTorch, ATen does this for you during the first op
launch.

---

*End of Chapter 08b — From CUDA Backend to Training on GPU.*
