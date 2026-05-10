# 07b — Learning Guide: Understanding the Training Code and Its ML/DL Foundations

This guide bridges the Stage 6 code to the underlying ML/DL concepts. If you
understand Python but are new to Zig, or understand ML theory but haven't seen
it implemented from scratch, this is for you.

Read this alongside the source files — it references specific lines and
functions rather than reprinting code. Open `src/lab/train.zig`,
`src/nn/model.zig`, `src/tensor/ops/loss.zig`, and `src/optim/adamw.zig` in
another window.

---

## 0. The Big Picture: What Training a Transformer Actually Does

At the highest level, training a language model is:

```
Repeat many times:
    1. Show the model some text
    2. Ask it to predict the next word
    3. Measure how wrong it was (loss)
    4. Adjust its parameters to be less wrong (gradient + optimizer)
```

That's it. Everything else — the transformer architecture, the attention
mechanism, the embedding tables, the optimizer math — exists to make step 4
work well. The training loop in `src/lab/train.zig` is the literal
implementation of these four steps.

**PyTorch equivalent:**

```python
for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch.input)
    loss = F.cross_entropy(logits.view(-1, V), batch.target.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**Our Zig equivalent** (from `train.zig:250-366`):

```zig
for (0..max_steps) |step| {
    const batch = self.batcher.next() orelse ...;
    var tape = Tape.init(allocator);
    // ... track parameters, forward, loss, backward ...
    // ... gradient clipping ...
    opt.step(params.items);
    opt.zeroGrad(params.items);
}
```

Same four steps, same order. The difference is that PyTorch hides the
tape/autograd graph management, and Zig forces you to manage memory explicitly.

---

## 1. The Data Pipeline: From Raw Text to Training Batches

### ML concept: Tokenization and Windowing

Language models learn by predicting the next token given a sequence of previous
tokens. To create training examples from raw text:

1. **Tokenize** — convert text to a sequence of integer IDs
2. **Window** — slide a window of length T over the token sequence; each
   window produces an (input, target) pair where the target is shifted by 1
3. **Batch** — group B windows together so the model can process them in
   parallel (GPUs are designed for batched math)

### Code walkthrough

```
Raw text: "to be or not to be"
  ↓ tokenizer (Dataset.init, line 166)
Tokens: [4, 15, 7, 12, 4, 15]
  ↓ windowing (Windowing.init, line 175, T=4)
Windows: input=[4,15,7,12] target=[15,7,12,4]
         input=[15,7,12,4] target=[7,12,4,15]
  ↓ batching (Batcher.init, line 179, B=2)
Batch: input=[4,15,7,12, 15,7,12,4]  target=[15,7,12,4, 7,12,4,15]
```

### Why shuffle? (ML concept: stochastic gradient descent)

The Batcher shuffles windows before creating batches (`batcher.zig`). Without
shuffling, consecutive windows overlap by T-1 tokens, making the training
signal highly correlated within a batch. Shuffling decorrelates the batches,
helping the optimizer converge faster — this is why stochastic gradient
descent (SGD) works better than full-batch gradient descent.

### Shape flow into the model

```zig
// train.zig:258-269
var ids = try Tensor.init(allocator, Shape.init2D(B, T));
// ids.data[i] = @floatFromInt(batch.input[i]);
var targets = try Tensor.init(allocator, Shape.init1D(B * T));
// targets.data[i] = @floatFromInt(batch.target[i]);
```

**Why `floatFromInt`?** Our Tensor struct stores `f32` data only (decision
D9). Token IDs are integers, but we store them as `3.0`, `15.0`, etc. The
cross-entropy loss function converts them back to integers with `@round` and
`@intFromFloat` when it needs to index into the log-probability table.

**Why are targets shape `(B*T,)` instead of `(B, T)`?** Cross-entropy expects
logits of shape `(N, V)` where N is the total number of predictions. After
reshaping `(B, T, V)` logits to `(B*T, V)`, we need a flat target array of
length `B*T`. This is the same as PyTorch's `logits.view(-1, V)` pattern.

---

## 2. The Training Step: Line by Line

### 2.1 Fresh tape per step

```zig
// train.zig:273-280
var tape = Tape.init(allocator);
defer tape.deinit();
for (params.items) |param| {
    param.requires_grad = true;
    _ = try tape.trackLeaf(param);
}
```

**ML concept: computation graph.** PyTorch builds a dynamic computation graph
as you run the forward pass. Each operation (add, matmul, softmax, etc.)
creates a node that remembers its inputs and how to compute the gradient.
Our `Tape` is exactly this graph — an array of `Node` records.

**Why fresh each step?** PyTorch frees the graph during `loss.backward()`
(by default `retain_graph=False`). Our tape doesn't free itself during
backward — it only frees intermediate gradient tensors. The node list and
saved data persist until `tape.deinit()`. If we kept the tape across steps,
memory would grow without bound. One tape per step, then deinit.

**Why trackLeaf?** Each parameter needs a "phantom" node on the tape that
serves as a gradient sink. After `backward()`, the parameter's gradient
appears at this phantom node's ID. We must re-track each step because the
previous tape's node IDs are stale — they pointed into a destroyed array.

### 2.2 Forward pass

```zig
// train.zig:283-292
var logits_3d = try self.model.forward(ids, &tape);
try tape.keepAlive(&logits_3d);
var logits = try ops_shape.reshapeTracked(allocator, logits_3d, Shape.init2D(B * T, V), &tape);
try tape.keepAlive(&logits);
```

**ML concept: the forward pass.** The model takes token IDs and produces
logits — unnormalized scores for each possible next token. The shape
trace through the model:

```
ids (B, T) → tok_embed (B, T, D) → +pos_embed (B, T, D)
  → block (B, T, D) → ln_f (B, T, D) → lm_head (B, T, V)
```

**Why reshape from (B,T,V) to (B\*T,V)?** Cross-entropy loss expects a 2D
input: one row per prediction, one column per class. Each (batch, position)
pair is an independent next-token prediction, so we flatten B and T together.

**Why reshapeTracked instead of reshape?** This was our most subtle bug
(`train.zig:286-291` comments). The ordinary `Tensor.reshape()` returns a
VIEW — it shares the same `tape_node` as the 3D original. When cross-entropy
stores a gradient of shape `(B*T, V)` under this node ID, the matmul backward
later expects shape `(B, T, V)`. This "works by accident" because both shapes
are row-major contiguous, but it's a latent bug that would break with any
non-contiguous data. `reshapeTracked()` creates a proper tape node so the
gradient flows back with the correct shape.

**Why keepAlive?** Every intermediate tensor created inside `model.forward()`
is `defer`-freed before `backward()` runs. But the tape's `SavedData` holds
slices pointing into those freed buffers — use-after-free! `keepAlive()`
transfers buffer ownership to the tape, setting `tensor.owned = false` so the
`defer deinit` is a no-op. The tape frees all kept-alive buffers in
`tape.deinit()`.

### 2.3 Loss computation

```zig
// train.zig:297
var loss = try crossEntropy(allocator, logits, targets, &tape);
const loss_val = loss.data[0];
```

**ML concept: cross-entropy loss.** This is THE loss function for language
models. It measures how surprised the model is by the correct next token:

```
L = -(1/N) * Σᵢ log P(target_i | context_i)
```

Where P is the model's predicted probability (softmax of logits) and N is
the total number of predictions (B*T).

**Why cross-entropy?** It has the perfect gradient for classification:
`∂L/∂logits = softmax(logits) - one_hot(targets)`. This is simple, stable,
and well-understood. The gradient pushes UP the probability of the correct
class and pushes DOWN all others, proportional to their current probability.

**Implementation in `loss.zig:79-155`:**
1. Compute `log_softmax(logits)` for numerical stability (avoids `log(0)`)
2. Gather the log-probability at each target index
3. Negate and average

**The `@round` bug** (`backward.zig:666`): When converting a float target
index back to an integer, `@intFromFloat` truncates towards zero:
`@intFromFloat(2.9999) = 2`, not 3. This silently produced wrong gradients.
Fix: `@intFromFloat(@round(value))`.

**Why save `loss_val` before deinit?** `loss.data[0]` reads from a heap
buffer. After `loss.deinit()`, that buffer is freed — reading it is
use-after-free. Save the value first: `const loss_val = loss.data[0]`.

### 2.4 Backward pass

```zig
// train.zig:308
try tape.backward(&loss);
```

**ML concept: backpropagation.** The backward pass computes ∂L/∂w for every
parameter w by applying the chain rule in reverse through the computation
graph. Starting from `∂L/∂loss = 1.0` (a scalar loss), it propagates
gradients backward through each operation:

```
loss ← cross_entropy ← reshape ← lm_head ← ln_f ← block ← embed ← ids
```

Each operation's backward rule is a function in `backward.zig`. The key ones:

| Operation | Forward | Backward |
|-----------|---------|----------|
| Add | `c = a + b` | `∂L/∂a += ∂L/∂c, ∂L/∂b += ∂L/∂c` |
| Matmul | `c = a @ b` | `∂L/∂a += ∂L/∂c @ bᵀ, ∂L/∂b += aᵀ @ ∂L/∂c` |
| Softmax | `p = softmax(x)` | `∂L/∂x = p * (∂L/∂p - Σ(∂L/∂p * p))` |
| Cross-entropy | `L = -log(p[target])` | `∂L/∂logits = (softmax - one_hot) / B` |
| Embedding | `out = weight[idx]` | `∂L/∂weight[idx] += ∂L/∂out` (scatter-add) |
| Reshape | `b = view(a)` | `∂L/∂a += reshape(∂L/∂b, a.shape)` |

**How the tape works:** The tape is an array of nodes in forward order.
`backward()` iterates in reverse, computing each node's gradient and
accumulating it into a `grad_map` (HashMap from NodeId to gradient tensor).
Leaf nodes (parameters) have their gradients linked back to `param.grad`.

**Memory subtlety:** Intermediate gradients (for non-leaf nodes) are freed
as soon as all their children have been processed. This prevents unbounded
memory growth during backward. Only parameter gradients persist.

### 2.5 Gradient clipping

```zig
// train.zig:314-349
const max_norm = self.cfg.grad_clip_norm;
var total_norm_sq: f64 = 0;
if (max_norm > 0) {
    for (params.items) |param| {
        if (param.grad) |g| {
            for (g.data) |val| {
                const v: f64 = @floatCast(val);
                total_norm_sq += v * v;
            }
        }
    }
    const total_norm = @sqrt(total_norm_sq);
    if (total_norm > 0) {
        const clip_coeff: f32 = @floatCast(1.0 * max_norm / total_norm);
        if (clip_coeff < 1.0) {
            for (params.items) |param| {
                if (param.grad) |g| {
                    for (g.data) |*val| { val.* *= clip_coeff; }
                }
            }
        }
    }
}
```

**ML concept: gradient clipping.** Sometimes gradients are huge — a single
bad training example can produce a gradient that's 1000x larger than normal.
If the optimizer applies this gradient directly, the parameters jump to a
completely different part of the loss landscape, potentially destroying
everything the model has learned so far. This is called "training instability"
or "divergence."

**The algorithm** (same as PyTorch's `clip_grad_norm_`):
1. Compute the global L2 norm of ALL parameter gradients combined:
   `total_norm = sqrt(Σᵢ ||grad_i||²)`
2. Compute a scaling coefficient: `clip_coeff = max_norm / total_norm`
3. If `clip_coeff < 1` (i.e., the norm exceeds the limit), multiply ALL
   gradients by `clip_coeff`, scaling the total norm down to `max_norm`

**Why compute in f64?** The squared-norm sum can be large (thousands of
parameters, each with squared gradient). Accumulating in f32 would lose
precision. We accumulate in f64 and cast back to f32 only for the actual
scaling.

**Our empirically determined behavior:**
- With `grad_clip_norm=5.0` (our default after testing), lr=1e-3 is stable
- Gradient norms on Shakespeare are typically 1.5-2.1, so clipping rarely
  activates — it's a safety net, not a constant factor
- lr=3e-3 diverges even WITH clipping, because the optimizer step itself
  is too large

### 2.6 Optimizer step

```zig
// train.zig:360-366
opt.step(params.items) catch |err| { ... };
opt.zeroGrad(params.items);
```

**ML concept: AdamW.** AdamW is the standard optimizer for transformers.
It maintains two running averages per parameter:

```
m = β₁ * m + (1 - β₁) * grad       (1st moment — mean of gradients)
v = β₂ * v + (1 - β₂) * grad²      (2nd moment — mean of squared gradients)
```

Then updates the parameter using bias-corrected moments:

```
m̂ = m / (1 - β₁ᵗ)                  (correct for zero-initialization bias)
v̂ = v / (1 - β₂ᵗ)
param -= lr * (m̂ / (√v̂ + ε) + wd * param)
```

**Why AdamW instead of plain Adam?** In vanilla Adam, weight decay is added
to the gradient: `grad = grad + wd * param`. In AdamW, weight decay is
applied directly to the parameter: `param -= lr * (update + wd * param)`.
This decoupling produces better regularization because the weight decay
doesn't interact with the adaptive learning rate (the `m̂/√v̂` term).

**Why β₂=0.999 (not 0.95)?** We discovered during testing that β₂=0.95
causes training instability. The second moment estimate adapts 50x faster,
making the effective learning rate oscillate when gradient directions change.
The standard value (0.999) provides a much smoother estimate.

**Implementation in `adamw.zig:83-120`:**
- Each parameter is keyed by its data pointer address (`@intFromPtr`)
- First and second moments are stored in a HashMap (`state`)
- Bias correction is computed by raising β to the power of `t` (the step
  counter) and dividing: `1 / (1 - βᵗ)`

**Why zeroGrad after step?** Gradients accumulate across steps (the `+=`
in the backward accumulation loop). If we don't zero them, the next step's
gradients would be added on top of the previous step's, producing wrong
updates. PyTorch's `optimizer.zero_grad()` does the same thing.

---

## 3. The Model: Understanding What's Being Trained

### Architecture overview (`model.zig`)

```
ids (B, T) ─→ tok_embed ─→ (B, T, D)
                              ↓ + pos_embed (1, T, D) [broadcast over B]
                              (B, T, D)
                              ↓
                        TransformerBlock
                         ┌─────────────────────────────┐
                         │ LN₁ → Attention → +x (residual)│
                         │ LN₂ → MLP → +h (residual)    │
                         └─────────────────────────────┘
                              ↓ (B, T, D)
                         ln_f (LayerNorm)
                              ↓ (B, T, D)
                         lm_head (Linear: D → V)
                              ↓
                         logits (B, T, V)
```

### The 14 parameters being trained

From `model.zig:293-308` (`collectNamedParams`):

| Parameter | Shape | What it does |
|-----------|-------|-------------|
| `tok_embed.weight` | (V, D) | Maps each word to a D-dimensional vector |
| `pos_embed.weight` | (T, D) | Encodes position information (position 0 vs position 15) |
| `block.ln1.gamma` | (D,) | LayerNorm scale (pre-attention) |
| `block.ln1.beta` | (D,) | LayerNorm shift (pre-attention) |
| `block.attn.w_q.weight` | (D, D) | Projects input to Query for attention |
| `block.attn.w_k.weight` | (D, D) | Projects input to Key for attention |
| `block.attn.w_v.weight` | (D, D) | Projects input to Value for attention |
| `block.attn.w_o.weight` | (D, D) | Projects attention output back to D |
| `block.ln2.gamma` | (D,) | LayerNorm scale (pre-MLP) |
| `block.ln2.beta` | (D,) | LayerNorm shift (pre-MLP) |
| `block.mlp.fc1.weight` | (D, 4D) | First MLP layer: expands D → 4D |
| `block.mlp.fc2.weight` | (4D, D) | Second MLP layer: projects 4D → D |
| `ln_f.gamma` | (D,) | Final LayerNorm scale |
| `ln_f.beta` | (D,) | Final LayerNorm shift |
| `lm_head.weight` | (D, V) | Projects hidden state to vocabulary logits |

**Total parameters for V=2000, D=32, T=16:**
- tok_embed: 2000 × 32 = 64,000
- pos_embed: 16 × 32 = 512
- Attention weights: 4 × (32 × 32) = 4,096
- LayerNorm: 4 × 32 = 128 (gamma + beta for ln1, ln2)
- MLP: (32 × 128) + (128 × 32) = 8,192
- ln_f: 2 × 32 = 64
- lm_head: 32 × 2000 = 64,000
- **Total ≈ 141K parameters**

That's tiny — GPT-2 small has 117M. But it's enough to learn basic
next-word prediction on a small dataset.

### Why residual connections? (ML concept)

The `+ x` in the transformer block (`block.zig:100-101`) is a residual
connection. It creates a "skip path" for the gradient during backward:

```
∂L/∂x = ∂L/∂(x + f(x)) = ∂L/∂output + ∂L/∂output * ∂f/∂x
                                      ↑ identity path    ↑ transformation path
```

Without the residual, gradients would have to flow through ALL the
transformations (attention, MLP, LayerNorm) to reach the embedding layers.
Each transformation multiplies the gradient by its Jacobian, which can be
smaller than 1 — causing the gradient to vanish exponentially with depth.
The identity path guarantees that at least SOME gradient reaches every
parameter, regardless of depth.

---

## 4. Generation: Using the Trained Model

### ML concept: autoregressive generation

After training, we want the model to produce text. It does this
autoregressively — one token at a time, using its own previous outputs
as context:

```
Step 1: [the]         → predict "sailor"    → [the, sailor]
Step 2: [the, sailor]  → predict "with"    → [the, sailor, with]
Step 3: [the, sailor, with] → predict "a" → [the, sailor, with, a]
...
```

Each step runs a full forward pass, but we only use the logits from the
LAST position — that's the model's prediction for the next token.

### Implementation (`train.zig:428-565`)

```zig
for (0..max_new_tokens) |_| {
    // Take the last T tokens as context
    const context_len = @min(tokens.items.len, T);
    const context_start = tokens.items.len - context_len;

    // Forward pass (NO tape — generation doesn't need gradients)
    var logits_3d = try model.forward(ids, null);

    // Extract last position's logits, shape (V,)
    const last_pos = (context_len - 1) * V;

    // Apply temperature
    logits[v] = logits_3d.data[last_pos + v] / temp;

    // Top-k filtering + softmax sampling
    // ...
}
```

**Why no tape during generation?** We're not computing gradients — we're
just using the model for inference. Passing `null` for the tape parameter
skips all autograd recording, saving memory and computation. This is the
same as PyTorch's `with torch.no_grad():` context manager.

### Temperature: controlling randomness

```
logits = [2.0, 1.0, 0.5]

temp=0.5 (sharper):  logits/0.5 = [4.0, 2.0, 1.0] → softmax ≈ [0.84, 0.11, 0.04]
temp=1.0 (normal):   logits/1.0 = [2.0, 1.0, 0.5] → softmax ≈ [0.63, 0.23, 0.14]
temp=2.0 (flatter):  logits/2.0 = [1.0, 0.5, 0.25] → softmax ≈ [0.42, 0.32, 0.26]
```

Higher temperature flattens the probability distribution, making the model
more "creative" (sampling less likely tokens). Lower temperature makes it
more deterministic (concentrating on the most likely token).

### Top-k sampling: limiting choices

Top-k restricts the sampling pool to the k highest-scoring tokens. This
prevents the model from sampling extremely unlikely tokens (which can
produce gibberish). With k=5, the model only considers its top 5 guesses.

k=1 is always greedy (argmax) — the model always picks the single most
likely token. This produces repetitive but coherent text. k=V (full
vocabulary) allows any token, producing more diverse but potentially
lower-quality text.

### Why does greedy ignore the RNG?

When k=1, the sampling code (`train.zig:536-556`) builds a probability
distribution where all probability mass is on one token. Drawing a random
number and walking the CDF always lands on the same token. That's why
the test `"generate — greedy (top_k=1) is deterministic"` passes even
with different RNG seeds.

---

## 5. The Checkpoint System: Saving and Loading

### Code walkthrough (`model.zig:186-308`)

The checkpoint format (`save`/`load`) is a simple binary format:

```
"TWTL" (magic, 4 bytes)
version: u32 = 1
num_params: u32
For each parameter:
    name_len: u32
    name: []u8
    rank: u8 (number of dimensions)
    dims: [4]u32 (padded with zeros)
    data_len: u32 (in bytes)
    data: []f32 (raw weight values)
```

**ML concept: model persistence.** Training takes time (hours for large
models). Checkpoints let you:
- Resume training after an interruption
- Use the trained model for inference (generation)
- Share models (though our format is custom, not HuggingFace)

**Why save by name, not position?** If we saved parameters in a fixed
order, any architecture change would break old checkpoints. Matching by
name (`collectNamedParams`) makes checkpoints more robust — unknown
parameters are skipped, missing parameters retain their initial values.

---

## 6. Common Training Pathologies and How to Spot Them

### 6.1 Loss doesn't decrease

**Symptom:** Loss stays flat at `log(V)` (random-chance level).

**Causes:**
- Learning rate too low (try 1e-3)
- Gradients not flowing (check that `requires_grad = true` and
  `trackLeaf` is called — our bug where params weren't re-tracked)
- Optimizer not stepping (check that `opt.step` is called)

**Our code:** The test `"Trainer.train — loss decreases on tiny.txt"`
(`train.zig:617-638`) verifies loss decreases over 20 steps.

### 6.2 Loss diverges (goes to infinity or NaN)

**Symptom:** Loss increases rapidly, or becomes NaN.

**Causes:**
- Learning rate too high (lr=3e-3 diverges on our model)
- Missing gradient clipping (add `grad_clip_norm = 5.0`)
- Numerical instability in softmax or log (check max-subtraction trick)

**Our empirical finding:** lr=1e-3 with grad_clip_norm=5.0 is the safe
zone. lr=3e-3 diverges even with clipping at norm=1.0.

### 6.3 Gradient norm is huge

**Symptom:** `grad_norm` reported in training log is very large (>100).

**Causes:**
- Model is very wrong on some examples (large CE gradient)
- Bug in backward (wrong shape, wrong accumulation)
- Missing normalization (e.g., dividing by B in CE gradient)

**Our observed norms:** 1.5-3.5 on tiny.txt, 1.5-2.1 on Shakespeare.
These are healthy — well within the 5.0 clip threshold.

### 6.4 Training is very slow

**Symptom:** Loss barely moves after 500 steps.

**Causes:**
- Model too small (D=32 is tiny — can't capture complex patterns)
- Batch size too small (B=4 gives noisy gradients)
- Dataset too large for the model (Shakespeare with V=2000 is ambitious
  for a 141K-parameter model)
- Not enough steps (500 steps is very few — real training runs thousands)

**Our observation:** Shakespeare at V=2000, D=32 goes from 7.75→7.35
in 500 steps. The model IS learning, just very slowly. More steps or
a larger model would help.

---

## 7. PyTorch ↔ Zig Mapping Table

| Concept | PyTorch | Our Zig |
|---------|---------|---------|
| Tensor | `torch.Tensor` | `Tensor` struct (`tensor.zig`) |
| Autograd | `torch.autograd` (built-in) | `Tape` + `Node` (`tape.zig`, `node.zig`) |
| Forward with tracking | `y = f(x)` (auto-tracked) | `y = f(alloc, x, &tape)` (explicit tape) |
| Forward without tracking | `with torch.no_grad(): y = f(x)` | `y = f(alloc, x, null)` |
| Backward | `loss.backward()` | `tape.backward(&loss)` |
| Parameters | `model.parameters()` | `model.parameters(&list)` |
| Optimizer | `torch.optim.AdamW(params, lr=1e-3)` | `AdamW.init(alloc, config)` |
| Step | `optimizer.step()` | `opt.step(params)` |
| Zero grad | `optimizer.zero_grad()` | `opt.zeroGrad(params)` |
| Clip grad norm | `clip_grad_norm_(params, max_norm)` | Manual loop (train.zig:314-349) |
| Reshape with tracking | `x.view(B*T, V)` (auto-tracked) | `ops_shape.reshapeTracked(alloc, x, shape, &tape)` |
| Reshape without tracking | `x.reshape(...)` (no autograd) | `x.reshape(shape)` (returns VIEW) |
| Embedding | `nn.Embedding(V, D)` | `Embedding.init(alloc, V, D, rng)` |
| Linear | `nn.Linear(D_in, D_out)` | `Linear.init(alloc, D_in, D_out, bias, rng)` |
| LayerNorm | `nn.LayerNorm(D)` | `LayerNorm.init(alloc, D, eps, rng)` |
| Cross-entropy loss | `F.cross_entropy(logits, targets)` | `crossEntropy(alloc, logits, targets, tape)` |
| Model save | `torch.save(state_dict, path)` | `model.save(io, path)` |
| Model load | `model.load_state_dict(torch.load(path))` | `model.load(io, path)` |
| Generation | `model.generate(...)` | `generate(alloc, &model, prompt, ...)` |
| DataLoader | `DataLoader(dataset, batch_size=B, shuffle=True)` | `Batcher.init(alloc, windowing, B, &rng)` |
| RNG | `torch.manual_seed(42)` | `Rng.init(42)` |

---

## 8. Reading Order for Maximum Understanding

If you want to deeply understand how training works, read the source in
this order:

1. **`src/lab/train.zig:219-386`** — The `train()` method. This is the
   main loop. Everything else exists to serve this function.

2. **`src/tensor/ops/loss.zig`** — Cross-entropy loss. Understanding
   this function is the key to understanding what the model optimizes.

3. **`src/autograd/backward.zig:638-685`** — The cross-entropy backward
   (`backwardCrossEntropy`). The gradient `softmax - one_hot` is the
   mathematical heart of transformer training.

4. **`src/optim/adamw.zig:83-160`** — The AdamW step. This shows how
   moments, bias correction, and weight decay combine into the update.

5. **`src/nn/model.zig:110-151`** — The model forward pass. Trace the
   shape flow from `(B, T)` to `(B, T, V)`.

6. **`src/lab/train.zig:428-565`** — The `generate()` function. This is
   inference: the same forward pass, but one token at a time.

7. **`examples/06_train_shakespeare.zig`** — The user-facing entry point.
   See how all the pieces are configured and connected.

8. **`examples/07_generate.zig`** — How to load a checkpoint and produce
   text with different sampling strategies.

---

## 9. Exercises

### 9.1 Modify the learning rate

Edit `examples/06_train_shakespeare.zig` to use `lr = 3e-3` and run it.
You should see the loss decrease initially, then spike (diverge). Now set
`grad_clip_norm = 1.0` and try again. Still diverges? This shows that
gradient clipping alone can't save an overly aggressive learning rate.

### 9.2 Compare greedy vs. top-k generation

After training, run example 07. Compare the output for:
- `top_k=1, temp=1.0` (greedy)
- `top_k=5, temp=1.0` (default sampling)
- `top_k=50, temp=1.5` (creative)

Notice how greedy produces repetitive text while creative sampling produces
more varied (but less coherent) output.

### 9.3 Track gradient norms

The Trainer already logs `grad_norm` alongside loss. Run a few hundred
steps and observe:
- What's the typical gradient norm?
- Does it spike right before divergence?
- How often does clipping actually activate (clip_coeff < 1)?

### 9.4 Overfit on one batch

Create a tiny config with `batch_size=1, max_steps=200, lr=1e-3` and a
small dataset. The model should memorize the batch — loss should approach
zero. This verifies the training pipeline is correct end-to-end. Our
example 04 does exactly this.

### 9.5 Add a new training metric

Modify `train.zig` to also log the average parameter magnitude
`sqrt(mean(param²))` alongside loss and grad_norm. This helps detect
parameter explosion (weights growing without bound).

---

## 10. Key Takeaways

1. **Training is a loop:** batch → forward → loss → backward → clip → step →
   zero_grad. Everything else supports this loop.

2. **The loss function determines what the model learns.** Cross-entropy
   makes the model predict the next token. Change the loss, change the
   behavior.

3. **Gradients flow backward through the exact same path as the forward
   pass, but in reverse.** The tape records the forward; backward replays it.

4. **Gradient clipping is a safety net.** It doesn't help when things are
   going well, but prevents catastrophe when a bad batch produces huge
   gradients.

5. **Memory management is the hard part.** Three of our six Stage 6 bugs
   were memory issues (use-after-free, dangling pointers, corrupted
   HashMap pointers). PyTorch's garbage collector hides these; Zig forces
   you to think about them explicitly.

6. **Generation is just repeated forward passes.** No gradients, no
   optimizer — just feed the model's own outputs back as input.

7. **Small models learn slowly.** A 141K-parameter model on a 1MB dataset
   with 500 steps barely makes a dent. Real models are 1000x-1000000x
   larger and train for millions of steps.
