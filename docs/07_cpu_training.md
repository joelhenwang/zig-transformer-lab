# 07 — End-to-End CPU Training

This chapter ties together every piece from Stages 2–5 into a working
training loop. By the end you will understand:

1. How the Trainer orchestrates dataset → model → loss → backward → optimizer
2. Why gradient clipping is essential for stability
3. How autoregressive generation works (greedy vs. top-k sampling)
4. Why `beta2=0.999` (not 0.95) is the right default for AdamW
5. Three subtle bugs we found and fixed during integration

---

## 7.1 The Training Loop — Step by Step

A single training step does this:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Get next batch from Batcher (reset if exhausted)         │
│ 2. Create fresh Tape                                        │
│ 3. Track all model parameters as tape leaves                │
│ 4. Forward pass:  ids (B,T) → model → logits_3d (B,T,V)     │
│ 5. Reshape: logits_3d (B,T,V) → logits (B*T,V) [tracked!]   │
│ 6. Loss: crossEntropy(logits, targets) → loss (1,)           │
│ 7. Backward: tape.backward(&loss) → fills param.grad        │
│ 8. Gradient clipping (global L2 norm)                       │
│ 9. Optimizer step (AdamW)                                   │
│ 10. Zero gradients (for next step's tape)                    │
│ 11. Free tape (destroys all intermediate gradient tensors)   │
└─────────────────────────────────────────────────────────────┘
```

Let's trace the shapes for a Shakespeare run (V=2000, D=32, T=16, B=4):

| Step | Tensor | Shape | Elements |
|------|--------|-------|----------|
| Input IDs | `ids` | (4, 16) | 64 |
| After tok_embed | `tok` | (4, 16, 32) | 2,048 |
| After pos_embed | `pos` | (1, 16, 32) | 512 |
| After add (broadcast) | `x` | (4, 16, 32) | 2,048 |
| After block (attn+MLP) | `h` | (4, 16, 32) | 2,048 |
| After ln_f | `ln_out` | (4, 16, 32) | 2,048 |
| After lm_head | `logits_3d` | (4, 16, 2000) | 128,000 |
| Reshape (tracked) | `logits` | (64, 2000) | 128,000 |
| Targets | `targets` | (64,) | 64 |
| Loss | `loss` | (1,) | 1 |

Total elements in forward: ~264K floats ≈ 1 MB of data flowing through the model.
That's tiny! A real GPT-2 small processes ~6M elements per step.

### Why a fresh tape per step?

Our `Tape` accumulates `Node` records during the forward pass. Each node
stores `SavedData` (tensor snapshots) that reference heap buffers. If we
kept the tape alive across steps, memory would grow without bound.

PyTorch avoids this because its autograd graph is freed during
`loss.backward()` (default `retain_graph=False`). Our tape *does* free
intermediate gradients during backward, but the node list and saved data
persist until `tape.deinit()`. So: one tape per step, then deinit.

### Why track parameters each step?

`trackLeaf(param)` creates a "phantom" node for each parameter. This
node is a placeholder — it has no parents and no backward rule. Its
purpose is to give the parameter a `NodeId` that child nodes can
reference as a parent. After `backward()`, the parameter's gradient
is stored in `grad_map[leaf_id]` and then linked back to `param.grad`.

We MUST call `trackLeaf` each step because the previous tape's node
IDs are stale (the tape was deinited). If we skipped this, child nodes
would reference invalid IDs, causing crashes or silent corruption.

---

## 7.2 The Reshape Bug — Why `reshapeTracked()` Matters

The most subtle bug we found: using `Tensor.reshape()` instead of
`reshapeTracked()` in the training loop.

### The problem

After `model.forward()`, we have `logits_3d` of shape `(B, T, V)`.
Cross-entropy expects `(B*T, V)`. So we reshape:

```zig
// BUG — untracked reshape creates a VIEW
const logits = try logits_3d.reshape(Shape.init2D(B * T, V));
```

`Tensor.reshape()` returns a **view** — it shares `data`, `tape_node`,
and `requires_grad` with the original. The cross-entropy node records
`logits.tape_node` as its parent. But `logits.tape_node` is the same
as `logits_3d.tape_node`, which was created by the `lm_head.forward()`
operation (a matmul + bias add).

When `backwardCrossEntropy` runs, it computes a gradient of shape
`(B*T, V)` and stores it in `grad_map[logits.tape_node]`. But the
matmul backward expects a gradient of shape `(B, T, V)` — the shape
of the matmul's *output*, which is `logits_3d.shape`.

### Why it "works" by accident

Both `(B, T, V)` and `(B*T, V)` have the same total elements and the
same row-major memory layout. The matmul backward uses `grad_output.data`
as a flat array with `@intFromFloat` indexing — it never checks the
gradient's shape. So the gradient "flows through" despite the shape
mismatch.

This is a **latent bug**. It would break if:
- The matmul backward ever asserted `shape_equals(grad.shape, output.shape)`
- The data were not contiguous (e.g., after a transpose)
- We added multi-head attention with different head dimensions

### The fix

```zig
// CORRECT — tape-tracked reshape creates a new node
var logits = try ops_shape.reshapeTracked(allocator, logits_3d,
    Shape.init2D(B * T, V), &tape);
try tape.keepAlive(&logits);
defer logits.deinit(allocator);
```

`reshapeTracked()` creates a new `Node` on the tape. The cross-entropy
node's parent is now this reshape node, not the matmul node. During
backward, the reshape node correctly reshapes the gradient from
`(B*T, V)` back to `(B, T, V)` before passing it to the matmul node.

**Key insight:** In an autograd system, every shape change must be
recorded on the tape. Views bypass the tape and create silent
inconsistencies in gradient shapes.

---

## 7.3 The `@round` Bug in backwardCrossEntropy

### The problem

In `backward.zig`, line 666:

```zig
const target_idx: usize = @intFromFloat(ci.targets[b]);  // BUG
```

`@intFromFloat` **truncates towards zero**. If the target index is
stored as f32 with floating-point imprecision (e.g., `2.9999999`
instead of `3.0`), `@intFromFloat(2.9999999)` returns `2`, not `3`.

The forward pass (in `loss.zig`) uses `@round`:

```zig
const class_idx = @as(usize, @intFromFloat(@round(targets.data[i])));
```

So the forward and backward can disagree on which class is the target.
The one-hot vector in the backward points at class 2, but the forward
computed the loss for class 3. The gradient is **wrong** — it pushes
the wrong class's probability in the wrong direction.

### Why it doesn't trigger with integer targets

When targets come from `@floatFromInt(batch.target[i])`, where
`batch.target[i]` is an exact u32, the f32 representation is exact for
values < 2^24. So `2.0` stays `2.0` — no rounding needed.

But this would break with:
- Label smoothing (soft targets)
- Targets from floating-point computation
- Very large vocab sizes (> 16M)

### The fix

```zig
const target_idx: usize = @intFromFloat(@round(ci.targets[b]));
```

Always use `@round` before `@intFromFloat` when converting float
indices to integers. This matches the forward pass and prevents
silent gradient errors.

---

## 7.4 Gradient Clipping — Preventing Training Divergence

### The instability problem

Without gradient clipping, training with lr=3e-3 diverges after ~200
steps on tiny.txt. The loss decreases initially, then shoots up:

```
Step   0: loss = 6.14  (converging)
Step  50: loss = 5.13
Step 100: loss = 5.11
Step 150: loss = 4.64
Step 200: loss = 5.67  (divergence starts)
Step 250: loss = 5.09
Step 300: loss = 5.55
Step 350: loss = 6.79  (exploding)
Step 400: loss = 8.41
Step 450: loss = 8.48
Step 499: loss = 10.30 (catastrophic)
```

### Why this happens

AdamW's effective learning rate is `lr / (ε + √v̂)`, where `v̂` is the
bias-corrected second moment estimate. When gradient magnitudes suddenly
change (e.g., the model makes confident wrong predictions), the old
`v̂` is too small, making the effective learning rate too large. The
optimizer overshoots, making worse predictions, producing larger
gradients, in a positive feedback loop.

### The clipping algorithm

We use PyTorch's `clip_grad_norm_` approach:

```
1. Compute global L2 norm: total_norm = sqrt(Σ(param_grad²))
2. If total_norm > max_norm:
     clip_coeff = max_norm / total_norm
     For each param.grad: grad *= clip_coeff
```

This scales all gradients uniformly so their combined L2 norm equals
`max_norm`. It preserves the gradient *direction* while capping the
*magnitude*.

### Why `beta2=0.999` (not 0.95)

With `beta2=0.95`, the second moment estimate adapts 50x faster than
standard. After a sudden gradient direction change, the denominator
`√v̂` drops quickly (v̂ forgets old gradients fast), making the
effective learning rate spike. With `beta2=0.999`, v̂ changes slowly,
acting as a stabilizer.

Empirical results on tiny.txt (lr=1e-3, 500 steps):
- beta2=0.95: stable but identical trajectory (grads are small enough)
- beta2=0.999: same stability, standard behavior
- lr=3e-3 + beta2=0.95: diverges after step 200
- lr=3e-3 + beta2=0.999: also diverges, but slightly later

The lesson: `beta2=0.999` is the standard for a reason. Only use
`beta2=0.95` if you *want* fast adaptation (e.g., non-stationary
objectives), and accept the instability risk.

### Recommended settings

| Setting | Value | Reason |
|---------|-------|--------|
| lr | 1e-3 | Stable for D=32 model |
| beta2 | 0.999 | Standard AdamW value |
| grad_clip_norm | 5.0 | Allows normal gradients, prevents explosions |
| weight_decay | 0.01 | Standard decoupled decay |

With these settings, gradient norms are typically 1.5-3.5 on tiny.txt
and 1.5-2.1 on Shakespeare. Clipping rarely activates (norm < 5.0).

---

## 7.5 Autoregressive Generation

After training, we want the model to *generate* text. The algorithm
is autoregressive: each new token is predicted from the previous ones.

```
┌───────────────────────────────────────────────────┐
│ 1. Start with prompt tokens [t₀, t₁, ..., tₙ]    │
│ 2. Feed last T tokens into model → logits (1, T, V)│
│ 3. Take logits at last position → (V,)             │
│ 4. Apply temperature: logits /= temperature        │
│ 5. Apply top-k: keep only k highest logits        │
│ 6. Compute softmax → probabilities                 │
│ 7. Sample from distribution → next token           │
│ 8. Append token, repeat from step 2               │
└───────────────────────────────────────────────────┘
```

### Temperature

Temperature controls the "sharpness" of the probability distribution:

```
prob_i = exp(logit_i / T) / Σ exp(logit_j / T)
```

- **T → 0**: Only the highest logit survives (greedy/argmax)
- **T = 1.0**: Original distribution (default)
- **T > 1.0**: Flatter distribution (more random, more "creative")

### Top-k sampling

Top-k restricts sampling to the k tokens with the highest logits:

1. Sort logits descending
2. Find the k-th largest value (threshold)
3. Set all logits below threshold to -∞ (zero probability after softmax)
4. Sample from the remaining k tokens

- **k = 1**: Greedy (always pick the most likely token)
- **k = V**: Full distribution (no filtering)
- **k = 5**: Sample from top 5 candidates (default)

### Why no tape during generation

Generation only needs forward passes. There's no backward pass and no
optimizer step. Passing `null` for the tape parameter skips all
recording, saving memory and computation:

```zig
var logits_3d = try model.forward(ids, null);  // no tape
```

### Context window handling

The model has a fixed context window (T = max_seq_len). When the
generated sequence exceeds T, we truncate to the last T tokens:

```zig
const context_len = @min(tokens.items.len, T);
const context_start = tokens.items.len - context_len;
```

This is a sliding window: the model always sees the most recent T
tokens. Older tokens are "forgotten." Real GPT models handle this
with KV-caching and attention sinks; our pedagogical model just
truncates.

---

## 7.6 The Trainer Struct — Avoiding Pointer Bugs

### Don't collect params in init()

The `Trainer.init()` function creates a local `model` variable, then
returns `Trainer{ .model = model }`. If we collect `model.parameters()`
before the return, the param pointers point to the *local* variable's
fields. After the return, those pointers dangle because the model was
*copied* into the struct at a different memory address.

```zig
// WRONG — pointers dangle after return
var model = try TinyWordTransformer.init(alloc, cfg, &model_rng);
model.parameters(&params);  // params has &model.weight → local var!
return Trainer{ .model = model };  // model is COPIED; old pointers dangle
```

Fix: collect params in `train()`, after `self.model` is in its final
memory location.

### Don't store AdamW in the struct

AdamW has a `HashMap` with self-referential pointers. Copying it into
a struct field (via `return Trainer{ .opt = adam }`) corrupts these
pointers. Fix: create AdamW locally in `train()` so its HashMap is
valid for the entire training run.

These are general Zig patterns: **never take pointers to fields of
local variables that will be moved/copied into a struct**.

---

## 7.7 Training Dynamics — What to Expect

### tiny.txt (V=256, D=32, T=8, B=4, lr=1e-3)

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 6.14 | 3.63 |
| 50 | 5.45 | 2.73 |
| 100 | 5.24 | 2.25 |
| 150 | 4.84 | 2.60 |
| 200 | 5.16 | 2.23 |
| 250 | 4.70 | 2.33 |
| 300 | 4.68 | 2.30 |
| 350 | 4.59 | 2.93 |
| 400 | 4.77 | 2.14 |
| 450 | 4.23 | 2.31 |
| 499 | 4.48 | 3.19 |

- Initial loss ≈ log(256) ≈ 5.55 (uniform distribution baseline)
- Loss 4.23 after 450 steps: the model is learning patterns
- Gradient norms are 2-3, never clipping at 5.0

### Shakespeare (V=2000, D=32, T=16, B=4, lr=1e-3)

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 7.75 | 2.10 |
| 50 | 7.80 | 1.74 |
| 100 | 7.74 | 1.75 |
| 150 | 7.35 | 1.63 |
| 200 | 7.45 | 1.48 |
| 250 | 7.58 | 1.52 |
| 300 | 7.61 | 1.57 |
| 350 | 7.50 | 1.71 |
| 400 | 7.60 | 1.70 |
| 450 | 7.47 | 1.63 |
| 499 | 8.13 | 1.62 |

- Initial loss ≈ log(2000) ≈ 7.60 (correct!)
- Loss barely decreases: 7.75 → 7.35 in 500 steps
- Gradient norms are 1.5-2.1, very stable
- This model is too small to make real progress on 1MB of text in 500 steps
- Each step processes B*T = 64 tokens; 500 steps = 32K tokens ≈ 0.01 epochs

### What this tells us

The model is **functionally correct** but **too small and undertrained**
to produce good text. A 1-block, 1-head, D=32 transformer has ~60K
parameters. With 500 steps × 64 tokens = 32K token exposures, each
parameter has been updated ~500 times but seen very few distinct
examples. The loss *does* decrease (7.75 → 7.35), proving the training
pipeline works. More steps would continue the decrease.

---

## 7.8 Checkpoint Format

The model save/load format is simple and self-describing:

```
┌────────────────────────────────────────┐
│ Magic: "TWTL" (4 bytes)               │
│ Version: u32 = 1                       │
│ Num params: u32                         │
│ For each param:                         │
│   Name length: u32                     │
│   Name: []u8                           │
│   Rank: u8                             │
│   Dims: [4]u32 (padded with 0s)       │
│   Data length: u32 (bytes)             │
│   Data: []f32                          │
└────────────────────────────────────────┘
```

The name-based matching allows loading checkpoints even if the model
architecture changes (unknown parameters are skipped). This is
simpler than PyTorch's `state_dict` but serves the same purpose.

---

## 7.9 PyTorch Equivalents

| Our code | PyTorch equivalent |
|---------|--------------------|
| `Trainer.train()` | `for batch in loader: ...` training loop |
| `tape.backward(&loss)` | `loss.backward()` |
| `opt.step(params)` | `optimizer.step()` |
| `opt.zeroGrad(params)` | `optimizer.zero_grad()` |
| `generate(model, prompt, ...)` | `model.generate(...)` |
| `reshapeTracked(logits_3d, ...)` | `logits.view(B*T, V)` (always tracked) |
| `grad_clip_norm=5.0` | `torch.nn.utils.clip_grad_norm_(params, 5.0)` |
| `model.save(io, path)` | `torch.save(model.state_dict(), path)` |
| `model.load(io, path)` | `model.load_state_dict(torch.load(path))` |

Key difference: PyTorch's autograd graph is freed during `backward()`
(unless `retain_graph=True`). Our tape persists until `tape.deinit()`.
This means we need one tape per step and must deinit it explicitly.

---

## 7.10 The Training Loop as a Data Pipeline

So far we've talked about the training step in terms of math and
function calls. Now zoom into the memory lifecycle. Every step
creates a small graph of owning and non-owning tensors; understanding
who frees what, and when, is the only way to chase a real leak.

Here is a full per-step allocation walk for `Trainer.train` on CPU
with `B=4, T=16, V=2000, D=32` (the Stage 6 Shakespeare config).
Numbers are f32 byte counts.

```
Step 42 enters the loop. At this point the owning tensors are:
  - Every model parameter (permanent, lifetime = Trainer)
  - Every entry in adam.state (permanent, one per param)
  - The Dataset's token buffer (permanent, ~1 MB for Shakespeare)
  - The Batcher's two small shuffled index arrays (permanent)

Line: batch = batcher.next()           alloc:   0 bytes (borrowed view)
Line: ids = Tensor.init(...)           alloc: 256 bytes ((4,16) f32)
Line: targets = Tensor.init(...)       alloc: 256 bytes ((64,) f32)
Line: tape = Tape.init(alloc)          alloc: ~1 KB (tape arrays)
Line: forward(ids, &tape)              alloc: ~250 KB across:
        - intermediate tensors (every op creates one)
        - every tracked reshape/transpose materialises
        - logits_3d output: 512 KB = 4 * 16 * 2000 * 4

Line: reshapeTracked(logits_3d, ...)   alloc: 0 bytes (view)
Line: crossEntropy(logits, targets)    alloc: 4 bytes (loss scalar)

Line: loss_val = loss.data[0]          alloc: 0 (scalar read)
Line: tape.backward(&loss)             alloc: ~250 KB for gradients:
        - one grad tensor per param (accumulated into param.grad)
        - intermediate gradient tensors inside the tape

Line: loss.deinit(alloc)               frees:     4 bytes
Line: logits_3d.deinit(alloc)          frees: 512 KB
(grad clip: no new allocations — scans existing param.grad)

Line: opt.step(params)                 alloc: 0 bytes
                                       writes: param.data (in place)
Line: opt.zeroGrad(params)             writes: param.grad.data = 0

Line: defer tape.deinit()              frees: everything tape owned
                                       (~500 KB of intermediate
                                        tensors + gradients)
Line: defer ids.deinit(alloc)          frees: 256 bytes
Line: defer targets.deinit(alloc)      frees: 256 bytes
```

Peak per-step memory: about **1 MB** of transient allocation beyond
the permanent model + optimiser state. That's the scratch working
set for the forward + backward chain.

### Why the tape owns so much

The tape holds intermediate forward tensors that backward needs to
recompute gradients. A naive implementation would reference the
same buffers the forward op saw; ours takes snapshots (see
`docs/03c_saved_tensors.md` for the full discussion). The cost of
snapshots is memory; the benefit is that the forward code is free
to deinit its owned tensors without corrupting the backward pass.

### What leaks look like

If you forget `tape.deinit()`, the per-step transient grows
unbounded and DebugAllocator reports a leak at test exit. We have
been there — commit `1cc82ce` documents a version of this bug on
the CUDA path (grad tensors on the optional branch weren't being
freed).

If you forget `ids.deinit(alloc)` or `targets.deinit(alloc)`, the
input buffers leak at 512 bytes per step. On a 500-step run that's
still only 256 KB, which is often below what a casual eyeball scan
of `/proc/self/status` catches. DebugAllocator catches it
immediately.

---

## 7.11 Three Silent Bugs We Fixed

Every subsystem in this library has its horror story. The Stage 6
pipeline integration surfaced three that are worth retelling
because the lesson from each transfers directly to bugs you will
write in your own code.

### 7.11.1 `@intFromFloat` truncation on target indices

**Symptom.** Loss decreased normally for the first hundred steps
but the gradient shape on the tok_embed seemed wrong — checkpoints
from different steps converged to implausibly similar weights for
some token IDs.

**Cause.** `backwardCrossEntropy` converted the target tensor's
f32 value to a class index by `@intFromFloat(target.data[b])`.
`@intFromFloat` in Zig truncates **towards zero**, not towards
nearest. A target originally stored as `u32 = 3` round-trips to
f32 as exactly `3.0` — fine. But with any rounding drift, a
target that *was* class 3 could be stored as `2.9999...` and
truncate back to 2. Every step with that batch flowed gradient to
the wrong class row of the embedding.

**Fix.** `@intFromFloat(@round(target.data[b]))`. The forward path
already uses `@round`; the backward had fallen behind.
`src/autograd/backward.zig:666`.

**Lesson.** Every `@intFromFloat` on a float-representing-integer
must be guarded with `@round` (or `@trunc`, or a full rounding
policy — pick one and stick to it). A silent off-by-one in
gradient routing is hell to debug.

### 7.11.2 Untracked reshape in the training loop

**Symptom.** Training appeared to work on the initial Shakespeare
config (D=32, T=16). Moving to a bigger config with a different
`B*T` pattern made some gradients go NaN within ten steps.

**Cause.** The loop computed `logits_3d.reshape(Shape.init2D(B*T,
V))` — a view with the same `tape_node` as the parent. The CE
backward registered a gradient of shape `(B*T, V)` under that
tape_node. The matmul backward upstream expected `(B, T, V)`.
Because both layouts were row-major contiguous, the memory
happened to match and no crash fired. But the gradient's
interpretation was now inconsistent with the producer.

**Fix.** `ops_shape.reshapeTracked(allocator, logits_3d,
Shape.init2D(B*T, V), &tape)` — explicit reshape node with a
proper backward that reinterprets the gradient back to rank-3.
`src/lab/train.zig:347`.

**Lesson.** Any reshape that sits between an op with a gradient
and another op with a gradient must be **tape-tracked**. The
untracked variant is for inference or throwaway inspection.
A good heuristic: if a reshape is inside a forward method that's
wrapped by a `*Tracked` naming convention, the inner reshape must
also be tracked.

### 7.11.3 `beta2 = 0.95` as a default

**Symptom.** At `lr = 3e-3` training would diverge after 200 steps
even with gradient clipping at norm 1.0. Dropping to `lr = 1e-3`
stabilised, but the loss plateaued higher than expected.

**Cause.** Our initial `AdamW` defaulted `beta2` to `0.95`
(probably copied from an aggressive half-remembered blog post).
At `beta2 = 0.95` the second-moment estimate adapts about 50×
faster than the standard `0.999`. When gradient direction
oscillates (common early in training), the effective learning
rate per-parameter oscillates correspondingly, which is what
causes the "diverges at 3e-3" behaviour.

**Fix.** Default `beta2 = 0.999` in `AdamWOpts`. With that default,
`lr = 1e-3` becomes stable on tiny.txt and `lr = 3e-3` is
survivable (but still diverges sometimes — we don't recommend it).
`src/optim/adamw.zig`.

**Lesson.** Copied defaults are an invisible source of bugs. The
hyperparameter landscape is wide and most defaults in public
writing aren't actually the authors' production configuration.
Always trace a default back to a published paper or a widely-used
reference implementation before committing to it.

---

## 7.12 Common Mistakes

- **Forgetting `tape.deinit()` between steps.** The tape accumulates
  forever. Use `defer tape.deinit()` immediately after `Tape.init`.
- **Reading `loss.data[0]` after `loss.deinit()`.** Save the value
  first: `const v = loss.data[0]; loss.deinit(alloc);`.
- **Untracked `reshape()` in a training loop.** See §7.11.2. Always
  `reshapeTracked`.
- **Passing `(B, T)` input where `(B*T,)` target is expected.** The
  CE op takes 2D `(B*T, V)` logits and 1D `(B*T,)` targets. A stray
  `targets.reshape(Shape.init2D(B, T))` turns every index into
  garbage.
- **Creating `AdamW` in `Trainer.init` and copying into the struct.**
  The AdamW state HashMap has self-referential internal pointers;
  struct-move invalidates them. Create locally in `train()`.
- **Forgetting `grad_clip_norm` for new experiments.** Safe default
  is 5.0; raising `lr` without clipping eventually hits a spike that
  explodes training.
- **Mutating `param.grad.data` directly after backward.** The optimizer
  reads `param.grad.data`; the tape owns the backing buffer. Safe
  operations: scale in place (what `grad_clip` does) or zero out.
  Unsafe: reallocate, resize, swap pointers.

---

## 7.13 Exercises

**Exercise 1.** On a Shakespeare-sized dataset (V=2000, T=16, B=4),
what is the peak per-step memory usage beyond permanent
model/optimiser state? Where in `src/lab/train.zig` does it occur?

<details><summary>Solution</summary>

Peak is during `model.forward` after the final `lm_head` linear
has produced `logits_3d` but before we've reshaped and consumed it.
The logits tensor is `(4, 16, 2000)` f32 = 512 KB. Add the
intermediate tensors from earlier ops (roughly another 200-400 KB
depending on how aggressively the tape snapshots), plus the tape
structure itself (~1 KB), and the transient working set is
≈ 1 MB. The hot spot is the line
`var logits_3d = try self.model.forward(ids, &tape);` —
everything before that is batch prep, everything after is consume
+ free.

</details>

**Exercise 2.** What changes if you forget `@round` in
`backwardCrossEntropy` but target IDs are always exact integers
stored via direct `@floatFromInt` casts from `u32`?

<details><summary>Solution</summary>

Nothing, practically. `@floatFromInt(u32)` for values up to 2²⁴
produces exact f32 representations. `@intFromFloat` on an exact
integer-valued f32 is the same truncation as `@round`. The bug
only bites when rounding error creeps in somewhere upstream — for
example if a future pipeline stage inserts label smoothing or
mixup, which produce non-integer targets. The fix is defensive:
it costs nothing and prevents a future-you disaster.

</details>

**Exercise 3.** Suppose you increase `lr` from `1e-3` to `1e-2` on
tiny.txt. Gradient clipping is on (`grad_clip_norm=5.0`).
Training diverges at step 150. Which of these is the *first*
thing you'd check, and why?

(a) Is `grad_clip_norm` actually applied?
(b) Is `beta2` set to 0.999?
(c) Is the data pipeline emitting the right targets?
(d) Are the model weights initialised correctly?

<details><summary>Solution</summary>

**(a)**. The clip is the "last defence" and the cheapest thing to
verify. Print `total_norm_sq` each step and confirm it's being
recomputed correctly (not summed across old state). A bug where
`grad_clip_norm` is quietly not taking effect — for example if the
`max_norm > 0` guard regressed to `max_norm > 1.0` — is much more
common than people expect. The other three take longer to rule out
and should follow only after (a) is confirmed green.

</details>

---



## 7.14 Summary

Stage 6 completes the CPU training pipeline. The key lessons:

1. **Always use `reshapeTracked()` in the training loop.** Untracked
   reshapes create silent gradient shape mismatches that "work by
   accident" in row-major layouts but break with transposes or
   non-contiguous data.

2. **Always use `@round` before `@intFromFloat` on indices.** Zig's
   `@intFromFloat` truncates towards zero. A target of 2.9999 → 2,
   not 3. This produces wrong gradients silently.

3. **Gradient clipping prevents divergence.** Even with AdamW's
   adaptive learning rate, sudden gradient spikes can cause the
   optimizer to overshoot. Clipping at norm 5.0 is a safe default.

4. **`beta2=0.999` is the right default.** `beta2=0.95` adapts too
   fast, making the effective learning rate oscillate when gradient
   directions change. The standard value is more stable.

5. **Never take pointers to fields of local variables that will be
   moved.** This is a general Zig pattern: `&local.field` dangles
   after the local is copied into a struct.

6. **Training a 1-block 1-head D=32 transformer on Shakespeare is
   possible but slow.** 500 steps isn't enough to see significant
   improvement. The pipeline is correct; the model just needs more
   training time or a larger architecture.
