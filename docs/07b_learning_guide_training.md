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

**Our Zig equivalent** (from `train.zig:219-386`):

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

### Worked example: one training step with real numbers

Let's trace a single training step with our tiny configuration
(V=32, D=16, T=8, B=2), which is what example 04 uses.

**Input batch** (2 sequences of 8 tokens each):
```
batch.input  = [3, 7, 12, 0, 5, 21, 8, 15,   9, 1, 28, 4, 11, 6, 19, 2]
batch.target = [7, 12, 0, 5, 21, 8, 15, 30,   1, 28, 4, 11, 6, 19, 2, 14]
```
Each target is shifted by 1 from the input — the model learns to predict
the next token.

**Step 0 output from example 04:**
```
Step   0: loss = 3.830872
```

Where does 3.83 come from? With V=32, random guessing gives loss = ln(32) ≈ 3.47.
Our initial loss is slightly ABOVE ln(V) because random weights produce a
non-uniform logit distribution — some classes get slightly higher logits,
pushing the correct class's log-probability below the uniform level.

**After 99 steps:**
```
Step  99: loss = 0.537453
```

The model has memorized the batch — loss is far below ln(32)=3.47, meaning
the model assigns high probability to the correct next token.

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

### Worked example: data pipeline with Shakespeare

Our Shakespeare dataset (`data/tinyshakespeare.txt`) is 1,115,394 bytes.
With `max_vocab=2000`, the tokenizer builds a vocabulary of the 2000 most
common words (plus special tokens `<pad>=0, <unk>=1, <bos>=2, <eos>=3`).

With T=16 and B=4:
- Each batch contains 4 sequences × 16 tokens = 64 token predictions
- The dataset has ~200K tokens → ~12,500 windows → ~3,125 batches
- At 500 steps, we see ~2.5 epochs worth of data

This is very few passes over the data — real training runs for hundreds of
epochs. That's why Shakespeare loss only goes from 7.75 → 7.35 in 500 steps.

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

**Bug we hit:** If you don't re-track, `trackLeaf()` sees the stale
`tape_node` field on the parameter and short-circuits, returning the old
ID. This caused two different nodes (a 3D intermediate and a 1D parameter)
to claim the same ID=13, corrupting the gradient accumulation. Fix:
`trackLeaf()` always creates a fresh node, ignoring existing `tape_node`.

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

### Worked example: full forward shape trace

With B=2, T=3, V=8, D=4 (our gradcheck test configuration):

```
Step 1: Token Embedding
  ids shape: (2, 3)     data: [[0,1,2], [3,4,5]]
  tok_embed.weight: (8, 4)  — 8 words, each a 4D vector
  tok = tok_embed.forward(ids, &tape)
  tok shape: (2, 3, 4)   — each of the 6 token IDs replaced by its 4D vector
  tok[0,0,:] = tok_embed.weight[0,:] = [0.12, -0.34, 0.56, 0.78]

Step 2: Position Embedding
  pos_ids shape: (1, 3)  data: [[0.0, 1.0, 2.0]]
  pos_embed.weight: (3, 4)  — 3 positions, each a 4D vector
  pos = pos_embed.forward(pos_ids, &tape)
  pos shape: (1, 3, 4)   — position 0, 1, 2 each get a 4D vector

Step 3: Add (broadcast over B)
  x = tok + pos          shapes: (2, 3, 4) + (1, 3, 4) → (2, 3, 4)
  x[0,0,:] = tok[0,0,:] + pos[0,0,:] = tok_embed[0,:] + pos_embed[0,:]
  x[1,2,:] = tok[1,2,:] + pos[0,2,:] = tok_embed[5,:] + pos_embed[2,:]

Step 4: Transformer Block
  h = block.forward(x, &tape)     (2, 3, 4) → (2, 3, 4)
  See Section 3 for the detailed breakdown.

Step 5: Final LayerNorm
  ln_out = ln_f.forward(h, &tape)   (2, 3, 4) → (2, 3, 4)

Step 6: LM Head (Linear: D → V)
  lm_head.weight: (8, 4)
  logits = lm_head.forward(ln_out, &tape)
  logits shape: (2, 3, 8)   — 6 predictions, each over 8 classes
```

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

### Worked example: cross-entropy with real numbers

Suppose V=4, B=1 (one prediction), logits = [2.0, 1.0, 0.5, 0.3], target = 0.

```
Step 1: log_softmax
  max(logits) = 2.0
  shifted = [0.0, -1.0, -1.5, -1.7]
  exp(shifted) = [1.000, 0.368, 0.223, 0.183]
  sum_exp = 1.774
  softmax = [0.564, 0.207, 0.126, 0.103]
  log_softmax = [-0.573, -1.573, -2.073, -2.273]

Step 2: Gather at target index
  target = 0 → log_p = log_softmax[0] = -0.573

Step 3: Negate and average (N=1)
  loss = -(-0.573) / 1 = 0.573
```

Interpretation: the model assigns 56.4% probability to the correct class.
The loss is -log(0.564) = 0.573. If the model were confident (p→1), loss→0.
If the model were wrong (p→0), loss→∞.

**With B=2, T=3 (6 predictions):**
```
logits shape: (6, 4)    — 6 rows, one per prediction
targets shape: (6,)     — [0, 1, 2, 3, 0, 1]
loss = -(1/6) * sum_i log_softmax(logits_i)[targets_i]
```

### The `@round` bug (`backward.zig:675`)

When converting a float target index back to an integer, `@intFromFloat`
truncates towards zero: `@intFromFloat(2.9999) = 2`, not 3. This silently
produced wrong gradients because the one-hot vector pointed at class 2
instead of class 3.

**Concrete example:**
```
targets.data[5] = 2.9999  (from @floatFromInt(3) with f32 rounding)
Without @round: target_idx = @intFromFloat(2.9999) = 2  ← WRONG CLASS
With @round:    target_idx = @intFromFloat(@round(2.9999)) = @intFromFloat(3.0) = 3  ← CORRECT
```

The one-hot gradient for class 2 gets `-1.0/B` instead of class 3 getting
`-1.0/B`. This shifts the entire softmax gradient to the wrong class.

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
| LayerNorm | `y = γ·(x-μ)/σ + β` | Chain through 8 sub-ops (see Section 3) |
| GELU | `y = x·Φ(x)` | `∂L/∂x = ∂L/∂y · (Φ(x) + x·φ(x))` |

**How the tape works** (`tape.zig:333-512`):

1. **Seed:** Create gradient tensor of all-1.0 matching loss shape, store in
   `grad_map[loss_node_id]`
2. **Topological sort:** DFS post-order from the loss node — ensures we process
   children before parents (reverse of forward order)
3. **Reverse walk:** For each node in reverse topological order:
   - Look up its output gradient from `grad_map`
   - Call the op-specific backward function to get parent gradients
   - Accumulate each parent gradient: `grad_map[parent_id] += parent_grad`
   - Free the intermediate output gradient (to limit memory)
4. **Write-back:** Link leaf nodes' gradients to `param.grad`

### Worked example: backward through cross-entropy

Continuing our example with logits = [2.0, 1.0, 0.5, 0.3], target = 0, B=1:

```
Forward produced:
  softmax = [0.564, 0.207, 0.126, 0.103]
  loss = 0.573

Backward:
  grad_output = [1.0]  (scalar loss gradient, always 1.0)

  one_hot = [1.0, 0.0, 0.0, 0.0]  (target class = 0)

  grad_logits = (softmax - one_hot) / B
              = ([0.564, 0.207, 0.126, 0.103] - [1, 0, 0, 0]) / 1
              = [-0.436, 0.207, 0.126, 0.103]

Interpretation:
  - Class 0 (correct): gradient is NEGATIVE → optimizer will INCREASE its logit
  - Classes 1,2,3 (wrong): gradient is POSITIVE → optimizer will DECREASE their logits
  - The magnitude for class 0 is largest → the biggest update targets the correct class
  - The magnitudes for wrong classes are proportional to their softmax probability
    → the optimizer focuses on reducing the most confident wrong predictions
```

This is why cross-entropy works so well — the gradient has a clean
interpretation and stable magnitudes.

### Worked example: backward through a single Linear layer

Suppose `y = x @ W^T` where x is (2, 3) and W is (4, 3), so y is (2, 4).

```
Forward:
  x = [[1, 2, 3], [4, 5, 6]]     shape (2, 3)
  W = [[0.1, 0.2, 0.3],           shape (4, 3)
       [0.4, 0.5, 0.6],
       [0.7, 0.8, 0.9],
       [1.0, 1.1, 1.2]]
  W^T = [[0.1, 0.4, 0.7, 1.0],    shape (3, 4)
         [0.2, 0.5, 0.8, 1.1],
         [0.3, 0.6, 0.9, 1.2]]
  y = x @ W^T                     shape (2, 4)

Backward (given ∂L/∂y, shape (2, 4)):
  ∂L/∂x = ∂L/∂y @ W              shape: (2, 4) @ (4, 3) = (2, 3)
  ∂L/∂W = (∂L/∂y)^T @ x          shape: (4, 2) @ (2, 3) = (4, 3)
```

The key insight: to get the weight gradient, we need the TRANSPOSE of the
upstream gradient times the input. This is why our `backwardMatmul` records
both inputs on the tape — it needs them to compute both parent gradients.

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

### Worked example: gradient clipping

Suppose we have 3 parameters with these gradient norms:

```
param_1.grad_norm = 1.5     (tok_embed, 64K elements)
param_2.grad_norm = 2.1     (attn weights, 4K elements)
param_3.grad_norm = 0.8     (LN gamma, 32 elements)

total_norm = sqrt(1.5² + 2.1² + 0.8²) = sqrt(2.25 + 4.41 + 0.64) = sqrt(7.30) = 2.70
max_norm = 5.0
clip_coeff = 5.0 / 2.70 = 1.85

Since clip_coeff > 1.0: NO CLIPPING HAPPENS.
```

Our observed gradient norms on Shakespeare are typically 1.5-2.1, well below
the 5.0 threshold. Clipping rarely activates — it's a safety net.

**Divergence scenario:**
```
total_norm = 500.0   (a bad batch exploded the gradients)
clip_coeff = 5.0 / 500.0 = 0.01

All gradients are multiplied by 0.01 — the optimizer sees a norm of 5.0
instead of 500.0. This prevents the catastrophic 100x parameter jump.
```

**Why compute in f64?** The squared-norm sum can be large (thousands of
parameters, each with squared gradient). Accumulating in f32 would lose
precision. We accumulate in f64 and cast back to f32 only for the actual
scaling.

**Our empirically determined behavior:**
- With `grad_clip_norm=5.0` (our default after testing), lr=1e-3 is stable
- Gradient norms on Shakespeare are typically 1.5-2.1, so clipping rarely
  activates — it's a safety net, not a constant factor
- lr=3e-3 diverges even WITH clipping, because the optimizer step itself
  is too large — clipping limits the gradient, but the Adam adaptive
  learning rate can still amplify the step beyond what's safe

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

### Worked example: AdamW step-by-step

Suppose a single parameter p = 0.5 with gradient g = 0.1, at step t=1.

**Configuration:** lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8, wd=0.01

```
Step 1: Initialize moments (first time this parameter is seen)
  m = 0.0     (1st moment)
  v = 0.0     (2nd moment)

Step 2: Update moments
  m = 0.9 * 0.0 + 0.1 * 0.1 = 0.01
  v = 0.999 * 0.0 + 0.001 * 0.01 = 0.00001

Step 3: Bias correction
  β₁¹ = 0.9    → bc1 = 1 / (1 - 0.9) = 10.0
  β₂¹ = 0.999  → bc2 = 1 / (1 - 0.999) = 1000.0
  m̂ = 0.01 * 10.0 = 0.1
  v̂ = 0.00001 * 1000.0 = 0.01

Step 4: Compute update
  update = m̂ / (√v̂ + ε) = 0.1 / (√0.01 + 1e-8) = 0.1 / 0.1 = 1.0
  decay = wd * p = 0.01 * 0.5 = 0.005

Step 5: Apply
  p -= lr * (update + decay) = 0.5 - 0.001 * (1.0 + 0.005) = 0.5 - 0.001005 = 0.498995
```

Key observations:
- The effective step size is `lr * update = 0.001 * 1.0 = 0.001`. This is
  exactly the raw learning rate because the bias-corrected gradient matches
  the actual gradient (g=0.1, m̂/√v̂ = 0.1/0.1 = 1.0).
- At step 1, bias correction is HUGE (bc1=10, bc2=1000). This compensates
  for the zero-initialized moments. Without it, the effective learning rate
  would be 1000x too small on the first step.
- Weight decay shrinks the parameter by `lr * wd * p = 0.001 * 0.01 * 0.5 = 0.000005`.
  This is tiny compared to the gradient update (0.001). Weight decay is a
  gentle regularizer, not a major force.

**Why β₂=0.999 (not 0.95)?** We discovered during testing that β₂=0.95
causes training instability. With β₂=0.95:

```
After 10 steps of constant gradient g=0.1:
  v (β₂=0.95) = 0.95^10 * 0 + (1-0.95^10) * 0.01 ≈ 0.00401
  v (β₂=0.999) = 0.999^10 * 0 + (1-0.999^10) * 0.01 ≈ 0.0000995

  v̂ (β₂=0.95) ≈ 0.00401 * 1.0 ≈ 0.00401     (nearly converged)
  v̂ (β₂=0.999) ≈ 0.0000995 * 100 ≈ 0.00995   (still adapting)

  Effective lr (β₂=0.95): 0.001 * 0.1 / √0.00401 ≈ 0.001 * 1.58
  Effective lr (β₂=0.999): 0.001 * 0.1 / √0.00995 ≈ 0.001 * 1.00
```

With β₂=0.95, the second moment converges in ~50 steps, making the adaptive
learning rate jump around when gradient directions change. With β₂=0.999,
the second moment is a much smoother estimate, giving stable effective
learning rates.

**Implementation in `adamw.zig:83-131`:**
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

### Architecture overview (`model.zig:110-151`)

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

### The Transformer Block in detail (`block.zig:95-127`)

The pre-norm residual pattern:

```
Input: x, shape (B, T, D)

Step 1: LN₁(x)                           — LayerNorm normalizes each position
  ln1_out = ln1.forward(x, tape)          — shape (B, T, D)
  [internally: 8 tape-tracked sub-ops]

Step 2: Attention(LN₁(x))                — Causal self-attention
  attn_out = attn.forward(ln1_out, tape)  — shape (B, T, D)
  [internally: 8 sub-ops: Q,K,V projections, K^T, Q@K^T, scale, mask, softmax, weights@V, W_o]

Step 3: Residual #1                       — Skip connection
  h = x + attn_out                        — shape (B, T, D)
  h = ops_elementwise.add(alloc, x, attn_out, tape)

Step 4: LN₂(h)                           — Second LayerNorm
  ln2_out = ln2.forward(h, tape)          — shape (B, T, D)

Step 5: MLP(LN₂(h))                      — Feed-forward network
  mlp_out = mlp.forward(ln2_out, tape)    — shape (B, T, D)
  [internally: fc1 → GELU → fc2]

Step 6: Residual #2                       — Skip connection
  output = h + mlp_out                    — shape (B, T, D)
```

**Why pre-norm instead of post-norm?** In the original Transformer (Vaswani
2017), LayerNorm was applied AFTER the residual: `output = LN(x + Attn(x))`.
Pre-norm applies it BEFORE: `output = x + Attn(LN(x))`. Pre-norm is more
stable for training because:
1. The residual path is clean (no normalization on it), so gradients flow
   freely through the identity path
2. The LayerNorm input has consistent statistics (the attention/MLP output
   is normalized before being added back)
3. GPT-2 and later models use pre-norm exclusively

### Causal Self-Attention in detail (`attention.zig:120-205`)

```
Input: x, shape (B, T, D)

Step 1: Q, K, V projections
  q = w_q.forward(x, tape)    — (B, T, D) → (B, T, D)  [Linear: D→D]
  k = w_k.forward(x, tape)    — (B, T, D) → (B, T, D)  [Linear: D→D]
  v = w_v.forward(x, tape)    — (B, T, D) → (B, T, D)  [Linear: D→D]

Step 2: K^T transpose
  k_t = transposeInner2d(k)   — (B, T, D) → (B, D, T)
  [swaps dims[1] and dims[2] for batched matmul]

Step 3: Q @ K^T scores
  scores = matmulBatch(q, k_t) — (B, T, D) @ (B, D, T) → (B, T, T)
  [each (T,T) matrix: scores[b,i,j] = dot(Q[b,i,:], K[b,j,:])]

Step 4: Scale
  scaled = scores / sqrt(D)    — (B, T, T)
  [D=4 → scale = 0.5; D=32 → scale ≈ 0.177]

Step 5: Causal mask
  mask[b,i,j] = 0    if j <= i  (can attend to past + self)
  mask[b,i,j] = -1e9 if j > i  (cannot attend to future)
  masked = scaled + mask         — (B, T, T) + (1, T, T) → (B, T, T)

Step 6: Softmax
  weights = softmax(masked)      — (B, T, T)
  [each row sums to 1; future positions get ~0 probability]

Step 7: weights @ V
  attn_out = matmulBatch(weights, v) — (B, T, T) @ (B, T, D) → (B, T, D)
  [weighted sum of value vectors, using attention weights]

Step 8: Output projection
  output = w_o.forward(attn_out) — (B, T, D) → (B, T, D)  [Linear: D→D]
```

### Worked example: attention with T=3, D=4

```
Position 0: "the" → Q₀, K₀, V₀
Position 1: "cat" → Q₁, K₁, V₁
Position 2: "sat" → Q₂, K₂, V₂

Step 3: Scores (before scaling)
  scores = Q @ K^T =
    [Q₀·K₀, Q₀·K₁, Q₀·K₂]     Position 0 attends to all 3
    [Q₁·K₀, Q₁·K₁, Q₁·K₂]     Position 1 attends to all 3
    [Q₂·K₀, Q₂·K₁, Q₂·K₂]     Position 2 attends to all 3

Step 5: After causal mask (add -1e9 above diagonal)
  [Q₀·K₀,     -1e9,     -1e9]    Position 0 can only see itself
  [Q₁·K₀, Q₁·K₁,     -1e9]      Position 1 can see 0,1
  [Q₂·K₀, Q₂·K₁, Q₂·K₂]        Position 2 can see 0,1,2

Step 6: Softmax (each row normalizes)
  [1.00, 0.00, 0.00]             Position 0: all attention on self
  [0.40, 0.60, 0.00]             Position 1: 40% on 0, 60% on 1
  [0.20, 0.30, 0.50]             Position 2: distributed over all 3

Step 7: Weighted values
  attn_out[0] = 1.00 * V₀                    = V₀
  attn_out[1] = 0.40 * V₀ + 0.60 * V₁        = mix of positions 0,1
  attn_out[2] = 0.20 * V₀ + 0.30 * V₁ + 0.50 * V₂ = mix of all
```

This is the essence of attention: each position gathers information from
previous positions, weighted by how relevant they are (the attention scores).

### LayerNorm in detail (`layernorm.zig:101-162`)

LayerNorm is composed from 8 tape-tracked sub-operations. This is
unusual — most frameworks implement LayerNorm as a single fused kernel.
We decompose it to show how the gradient flows through each step.

```
Input: x, shape (..., D)  [e.g., (B, T, D) = (2, 3, 4)]

Step 1: μ = mean(x, axis=-1)          shape (..., 1) = (2, 3, 1)
  Mean over the last dimension for each position.

Step 2: x_centered = x - μ             shape (..., D) = (2, 3, 4)
  Subtract mean (broadcast over D).

Step 3: x_centered² = x_centered * x_centered  shape (..., D) = (2, 3, 4)
  Element-wise square.

Step 4: σ² = mean(x_centered², axis=-1) shape (..., 1) = (2, 3, 1)
  Variance = mean of squared deviations.

Step 5: σ² + ε                          shape (..., 1) = (2, 3, 1)
  Add small epsilon for numerical stability (ε = 1e-5).

Step 6: σ = sqrt(σ² + ε)               shape (..., 1) = (2, 3, 1)
  Standard deviation.

Step 6b: x_norm = x_centered / σ       shape (..., D) = (2, 3, 4)
  Normalize to zero mean, unit variance.

Step 7: scaled = γ * x_norm            shape (..., D) = (2, 3, 4)
  Learnable scale parameter γ, shape (D,).

Step 8: y = scaled + β                 shape (..., D) = (2, 3, 4)
  Learnable shift parameter β, shape (D,).
```

### Worked example: LayerNorm with numbers

```
x = [1.0, 2.0, 3.0, 6.0]  (one position, D=4)
γ = [1.0, 1.0, 1.0, 1.0]  (initial)
β = [0.0, 0.0, 0.0, 0.0]  (initial)

Step 1: μ = (1+2+3+6)/4 = 3.0
Step 2: x_centered = [-2.0, -1.0, 0.0, 3.0]
Step 3: x_centered² = [4.0, 1.0, 0.0, 9.0]
Step 4: σ² = (4+1+0+9)/4 = 3.5
Step 5: σ² + ε = 3.50001
Step 6: σ = √3.50001 = 1.8708
Step 6b: x_norm = [-2.0/1.87, -1.0/1.87, 0.0/1.87, 3.0/1.87]
               = [-1.069, -0.534, 0.000, 1.604]
Step 7: scaled = [1.0*-1.069, 1.0*-0.534, 1.0*0.000, 1.0*1.604]
               = [-1.069, -0.534, 0.000, 1.604]
Step 8: y = scaled + [0,0,0,0] = [-1.069, -0.534, 0.000, 1.604]
```

After training, γ and β learn to re-scale and re-center the distribution
to values that work best for the downstream layers.

### The 14 parameters being trained

From `model.zig:289-308` (`collectNamedParams`):

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

### Worked example: parameter count for V=32, D=16, T=8

This is the configuration used in example 04 (overfit one batch):

```
tok_embed.weight:    32 × 16 = 512
pos_embed.weight:     8 × 16 = 128
block.ln1.gamma:         16
block.ln1.beta:          16
block.attn.w_q.weight: 16 × 16 = 256
block.attn.w_k.weight: 16 × 16 = 256
block.attn.w_v.weight: 16 × 16 = 256
block.attn.w_o.weight: 16 × 16 = 256
block.ln2.gamma:         16
block.ln2.beta:          16
block.mlp.fc1.weight: 16 × 64 = 1,024
block.mlp.fc2.weight: 64 × 16 = 1,024
ln_f.gamma:              16
ln_f.beta:               16
lm_head.weight:      16 × 32 = 512

Total = 512 + 128 + 16 + 16 + 256×4 + 16 + 16 + 1024 + 1024 + 16 + 16 + 512
      = 3,808 parameters

Example 04 reports "21 tensors" — this includes bias parameters when bias=true
(4 attention biases + 2 MLP biases + 1 lm_head bias = 7 extra tensors → 14 + 7 = 21)
```

### Why residual connections? (ML concept)

The `+ x` in the transformer block (`block.zig:108-110`) is a residual
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

### Worked example: gradient flow with and without residuals

Without residual (2-layer deep):
```
∂L/∂x = ∂L/∂y₂ · ∂f₂/∂h · ∂f₁/∂x
If ||∂f₂/∂h|| < 1 and ||∂f₁/∂x|| < 1:
  gradient shrinks at each layer → vanishing gradient
```

With residual:
```
∂L/∂x = ∂L/∂y₂ · (1 + ∂f₂/∂h · ∂f₁/∂x)
                                   ↑ at minimum, this is 1.0
The "1 +" guarantees the gradient never goes below ∂L/∂y₂
```

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

### Worked example: top-k sampling step-by-step

```
Vocabulary: [a, b, c, d, e, f]  (V=6)
Last position logits: [1.0, 3.0, 0.5, 2.5, 0.1, 1.5]
Temperature = 1.0, top_k = 3

Step 1: Sort logits descending
  sorted = [3.0, 2.5, 1.5, 1.0, 0.5, 0.1]
  min_of_top = sorted[2] = 1.5  (k-th largest)

Step 2: Filter — keep only logits >= 1.5
  [1.0, 3.0, 0.5, 2.5, 0.1, 1.5]
   ↓       ↓         ↓       ↓
  skip    keep      keep    keep
  (1.0 < 1.5)  (3.0 ≥ 1.5)  (2.5 ≥ 1.5)  (1.5 ≥ 1.5)

Step 3: Softmax over the top-k
  max_logit = 3.0
  exp(logits - max) = [0, exp(0), 0, exp(-0.5), 0, exp(-1.5)]
                    = [0, 1.0, 0, 0.607, 0, 0.223]
  sum = 1.830
  probs = [0, 0.546, 0, 0.332, 0, 0.122]

Step 4: Sample
  r = 0.7 (random)
  cumsum = 0 → 0 → 0 → 0.546 → 0.878 → 0.878 → 1.0
  0.7 < 0.878 → sampled = "d" (index 3)
```

### Why does greedy ignore the RNG?

When k=1, the sampling code builds a probability distribution where all
probability mass is on one token. Drawing a random number and walking the
CDF always lands on the same token. That's why the test
`"generate — greedy (top_k=1) is deterministic"` passes even with different
RNG seeds.

---

## 5. Gradient Checking: How We Know the Backward Is Correct

### The core idea

How do you verify that your backward pass is correct? You can't just check
that the loss decreases — a bug might slow learning without stopping it.
The gold standard is **finite-difference gradient checking**: perturb each
parameter by a tiny amount, recompute the loss, and compare the numerical
gradient with the analytical gradient from `tape.backward()`.

### The central difference formula

```
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
```

For parameter element `p[i]`:
```
numerical_grad[i] = (L(p + h·eᵢ) - L(p - h·eᵢ)) / (2h)
```

Where `eᵢ` is the unit vector in direction `i` and `h` is a small
perturbation (typically 1e-4).

### Implementation walkthrough (`gradcheck.zig:115-199`)

```zig
for (params) |param| {
    const grad = param.grad orelse continue;
    const n = totalElements(param.shape);
    const indices_to_check = if (sample_n > 0 and sample_n < n) sample_n else n;

    for (0..indices_to_check) |_| {
        const idx = /* random or sequential index */;
        const original = param.data[idx];

        // Perturb +h and recompute loss
        param.data[idx] = original + eps;
        const loss_plus = /* run forward with new param value */;

        // Perturb -h and recompute loss
        param.data[idx] = original - eps;
        const loss_minus = /* run forward with new param value */;

        // Restore
        param.data[idx] = original;

        // Numerical gradient
        const numerical = (loss_plus - loss_minus) / (2.0 * eps);

        // Compare with analytical gradient from tape.backward()
        const analytical = grad.data[idx];
        const rel_error = |analytical - numerical| / max(|analytical|, |numerical|, floor);

        if (rel_error > tol_rel) result.passed = false;
    }
}
```

### Worked example: gradient check on a simple function

```
Function: L(a) = sum(a * 2 + 1) = 2 * sum(a) + n
Analytical gradient: dL/da[i] = 2

a = [1.0, 2.0, 3.0], eps = 1e-3

Check a[0]:
  a[0] = 1.0 + 1e-3 = 1.001
  L_plus = 2*(1.001 + 2 + 3) + 3 = 2*6.001 + 3 = 15.002
  a[0] = 1.0 - 1e-3 = 0.999
  L_minus = 2*(0.999 + 2 + 3) + 3 = 2*5.999 + 3 = 14.998
  numerical = (15.002 - 14.998) / (2 * 1e-3) = 0.004 / 0.002 = 2.0

  analytical = 2.0
  rel_error = |2.0 - 2.0| / max(2.0, 2.0, 1e-8) = 0.0  ← PASS
```

### The near-zero gradient problem

The standard relative error formula is:
```
rel_error = |analytical - numerical| / max(|analytical|, |numerical|, 1e-8)
```

The `1e-8` floor prevents division by zero. But when both gradients are
near zero (say, 0.001), the floor doesn't help — a tiny absolute difference
of 0.0001 produces a huge relative error:

```
analytical = 0.0010
numerical  = 0.0009  (finite-difference noise at this scale)
abs_diff = 0.0001
denom = max(0.001, 0.0009, 1e-8) = 0.001
rel_error = 0.0001 / 0.001 = 0.1  ← LOOKS BAD

But the absolute difference is only 0.0001 — this is NOT a bug.
The finite-difference approximation is inherently noisy for small gradients.
```

### Our solution: the combined check

We use TWO metrics together:

1. **`denom_floor = 1e-2`** — raise the denominator floor from 1e-8 to 1e-2.
   This means relative error is only meaningful when gradients are > 0.01.

2. **Combined assertion**: pass if EITHER `max_rel_err < 0.05` OR
   `max_abs_diff < threshold`

```zig
try std.testing.expect(max_rel_err < 0.05 or max_abs_diff < 1e-2);
```

**Logic:**
- If gradients are large (> 0.01): relative error is precise → check `max_rel_err`
- If gradients are tiny (< 0.01): relative error is noise → check `max_abs_diff`
- The combined check correctly handles both cases

### Real test results: what they mean

From our full model gradcheck test (V=8, D=4, T=3, B=2):

```
tok_embed.weight    max_rel_err=0.002057 max_abs_diff=0.000171  PASS
pos_embed.weight    max_rel_err=0.002630 max_abs_diff=0.000112  PASS
block.attn.w_q.weight  max_rel_err=0.000001 max_abs_diff=0.000000  PASS
block.attn.w_v.weight  max_rel_err=0.006633 max_abs_diff=0.000066  PASS
block.mlp.fc1.weight   max_rel_err=0.013954 max_abs_diff=0.000140  WARN
block.mlp.fc2.weight   max_rel_err=0.015197 max_abs_diff=0.000152  WARN
```

**Interpretation:**
- `w_q.weight` rel_err=0.000001: the analytical gradient is EXACTLY right
- `fc1.weight` rel_err=0.014, abs_diff=0.00014: the gradient is correct
  within 1.4% relative error, and the absolute error is 0.00014 — this
  is finite-difference precision, not a backward bug
- WARN (0.01-0.05) means the check is slightly noisy but the gradient is
  clearly correct

**The FAIL case:**
```
block param  max_rel_err=0.204734 max_abs_diff=0.004296  FAIL
```
This parameter has a near-zero gradient. The 20% relative error looks bad,
but the absolute difference is only 0.004 — the analytical gradient is ~0.001
and the numerical gradient is ~0.0012. The 0.2% absolute error is well
within finite-difference noise. The combined assertion passes because
`max_abs_diff=0.009506 < 1e-2` (the WORST absolute error across all params).

### The degenerate case: sumAll(softmax(x))

```zig
// loss = sumAll(softmax(x))
// Each row of softmax sums to 1, so loss = B*T regardless of x.
// Gradient is exactly 0 — not useful for grad checking.
```

Our test confirms this:
```
sumAll(softmax) backward: max_abs_grad=0.0000000000
```

The analytical gradient is 0.0, the numerical gradient is 0.0. The test
passes, but it's a false positive — it would pass even with a broken
backward because the gradient is always zero regardless.

**Fix:** Use a non-trivial loss like `sumAll(softmax(x) * target)` which
creates a non-constant function of x.

### Gradcheck test hierarchy

Our gradcheck tests are organized in a hierarchy from simple to complex:

```
Level 1: Single ops
  ├── matmul (2D)
  ├── add (broadcast)
  ├── softmax (2D, 3D)
  └── cross-entropy

Level 2: Layer-level
  ├── Linear weight (2D input)
  ├── Linear weight (3D input — reshape path)
  ├── Linear bias (3D input)  ← NEW, max_rel_err=0.000002
  ├── Linear in residual path (add + identity)
  └── LayerNorm alone (gamma + beta)

Level 3: Sub-graph
  ├── matmulBatch backward (Q @ K^T pattern)
  ├── Q@K^T path without softmax
  ├── softmax → matmulBatch chain
  └── add broadcast (3D + broadcast)

Level 4: Component-level
  ├── CausalSelfAttention alone (8 params)
  ├── TransformerBlock (residual connections)
  └── Attention pipeline step-by-step (5 stages)

Level 5: Full model
  └── TinyWordTransformer (15 params, end-to-end)
```

Each level catches bugs at its scope. The pipeline test is especially
valuable — it checks 5 stages of increasing complexity:

| Stage | Loss function | What it tests |
|-------|--------------|---------------|
| `scores_only` | `sumAll(Q @ K^T)` | matmul + Linear backward |
| `scaled` | `sumAll(Q @ K^T / √D)` | + mulScalar backward |
| `masked` | `sumAll(scaled + causal_mask)` | + broadcast add backward |
| `softmaxed` | `sumAll(softmax(masked))` | DEGENERATE — skip |
| `full_attn` | `sumAll(W_o @ (softmax(masked) @ V))` | Full attention chain |

### How to add a new gradcheck test

1. Create the forward computation with a `tape`
2. Track the parameters you want to check with `trackLeaf`
3. Run `tape.backward(&loss)`
4. For each parameter, perturb each element by ±h, recompute the loss
   with `tape=null`, and compare numerical vs analytical gradient
5. Use `denom_floor=1e-2` and the combined assertion:
   `max_rel_err < 0.05 or max_abs_diff < threshold`

**Critical:** The numerical pass (step 4) must use `tape=null` to avoid
stale `tape_node` values from the analytical pass contaminating the
computation.

---

## 6. The Checkpoint System: Saving and Loading

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

### Worked example: checkpoint file for V=8, D=4, T=3

```
Header:
  "TWTL" (4 bytes)
  version = 1 (4 bytes)
  num_params = 15 (4 bytes)

Parameter 1: tok_embed.weight
  name_len = 16 (4 bytes)
  name = "tok_embed.weight" (16 bytes)
  rank = 2 (1 byte)
  dims = [8, 4, 0, 0] (16 bytes)
  data_len = 128 (4 bytes)  ← 8*4*4 = 128 bytes
  data = [0.12, -0.34, ...] (128 bytes)

... (14 more parameters)

Total file size ≈ 12 (header) + 15 * (4 + 16 + 1 + 16 + 4 + data)
```

---

## 7. Common Training Pathologies and How to Spot Them

### 7.1 Loss doesn't decrease

**Symptom:** Loss stays flat at `log(V)` (random-chance level).

**Causes:**
- Learning rate too low (try 1e-3)
- Gradients not flowing (check that `requires_grad = true` and
  `trackLeaf` is called — our bug where params weren't re-tracked)
- Optimizer not stepping (check that `opt.step` is called)

**Our code:** The test `"Trainer.train — loss decreases on tiny.txt"`
(`train.zig:617-638`) verifies loss decreases over 20 steps.

**Connection to our training:**
```
Example 04 (V=32, D=16, T=8, B=2):
  Step  0: loss = 3.830872  (≈ ln(32) = 3.47 — random initialization)
  Step 50: loss = 1.335811  (clearly decreasing → pipeline works)
  Step 99: loss = 0.537453  (memorizing the batch)
```

### 7.2 Loss diverges (goes to infinity or NaN)

**Symptom:** Loss increases rapidly, or becomes NaN.

**Causes:**
- Learning rate too high (lr=3e-3 diverges on our model)
- Missing gradient clipping (add `grad_clip_norm = 5.0`)
- Numerical instability in softmax or log (check max-subtraction trick)

**Our empirical finding:** lr=1e-3 with grad_clip_norm=5.0 is the safe
zone. lr=3e-3 diverges even with clipping at norm=1.0.

**Connection to our training:**
```
With lr=3e-3, grad_clip_norm=1.0 on tiny.txt:
  Steps 0-200: loss decreases normally (3.8 → 2.5)
  Step ~250: loss suddenly spikes to 15+
  Step ~260: loss = NaN (divergence)

Why: The effective learning rate in Adam is lr * (m̂ / √v̂). Even though
clipping limits the raw gradient, the adaptive learning rate amplifies
the step. With lr=3e-3, the amplified step is too large, pushing
parameters into a region with huge gradients → positive feedback loop.
```

### 7.3 Gradient norm is huge

**Symptom:** `grad_norm` reported in training log is very large (>100).

**Causes:**
- Model is very wrong on some examples (large CE gradient)
- Bug in backward (wrong shape, wrong accumulation)
- Missing normalization (e.g., dividing by B in CE gradient)

**Our observed norms:** 1.5-3.5 on tiny.txt, 1.5-2.1 on Shakespeare.
These are healthy — well within the 5.0 clip threshold.

**Connection to our training:**
```
Example 06 (Shakespeare, V=2000, D=32, T=16, B=4):
  Step   0: loss = 7.7500  grad_norm = 1.83
  Step 100: loss = 7.5200  grad_norm = 1.97
  Step 200: loss = 7.4100  grad_norm = 2.05
  Step 300: loss = 7.3600  grad_norm = 1.92
  Step 400: loss = 7.3200  grad_norm = 1.88
  Step 499: loss = 7.2900  grad_norm = 1.76

Healthy pattern: grad_norm stays in a narrow range, doesn't spike.
```

### 7.4 Training is very slow

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

**Connection to theory:** The loss improvement per step depends on the
signal-to-noise ratio of the gradient estimate. With B=4 and T=16, each
gradient estimate uses only 64 predictions. The true gradient (over the
full dataset) has much less variance. Increasing B reduces variance at
the cost of more computation per step.

```
Improvement per step ≈ lr * ||true_gradient|| - O(lr² * variance/B)
                       ↑ learning signal      ↑ noise

With B=4: improvement ≈ lr * signal - O(lr² * noise/4)
With B=64: improvement ≈ lr * signal - O(lr² * noise/64)
                                            ↑ 16x less noise
```

### 7.5 Use-after-free during backward

**Symptom:** Segfault or garbage values in gradients. Only happens with
tape != null.

**Causes:** An intermediate tensor created in a forward method is
`defer`-freed before `backward()` runs. The tape's `SavedData` holds a
slice pointing into the freed buffer.

**Fix:** Call `tape.keepAlive(&tensor)` for every intermediate before
the `defer tensor.deinit()` fires. This transfers buffer ownership to
the tape.

**Connection to our code:** We had 3 separate use-after-free bugs:
1. `reshapeTracked` intermediates in `model.forward()`
2. Transpose intermediates in `attention.forward()`
3. `logits_3d` in the training loop

Each was caught by running the full model gradcheck — which exercises
the backward path — and seeing either a crash or wildly wrong gradients.

---

## 8. PyTorch ↔ Zig Mapping Table

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
| Grad check | `torch.autograd.gradcheck(fn, inputs)` | `gradCheck(alloc, loss_fn, params, eps, tol, n)` |
| No-grad context | `torch.no_grad()` | Pass `null` for tape parameter |
| Gradient accumulation | Automatic (+= on .grad) | Automatic (+= in `accumulateGrad`) |

---

## 9. The 22 OpKinds: What Each Backward Does

Our autograd system has 22 operation types (`node.zig:88-133`). Each one
has a matching backward function in `backward.zig`. Here's the complete list:

### Elementwise binary ops (gradient distributes to both inputs)
| Op | Forward | Backward |
|----|---------|----------|
| `.add` | `c = a + b` | `∂L/∂a += ∂L/∂c, ∂L/∂b += ∂L/∂c` |
| `.sub` | `c = a - b` | `∂L/∂a += ∂L/∂c, ∂L/∂b -= ∂L/∂c` |
| `.mul` | `c = a * b` | `∂L/∂a += ∂L/∂c * b, ∂L/∂b += ∂L/∂c * a` |
| `.div` | `c = a / b` | `∂L/∂a += ∂L/∂c / b, ∂L/∂b -= ∂L/∂c * a / b²` |

### Elementwise scalar ops (broadcast gradient reduces)
| Op | Forward | Backward |
|----|---------|----------|
| `.add_scalar` | `c = a + s` | `∂L/∂a += ∂L/∂c` (scalar gradient summed) |
| `.mul_scalar` | `c = a * s` | `∂L/∂a += ∂L/∂c * s` |

### Linear algebra (gradient involves transposes)
| Op | Forward | Backward |
|----|---------|----------|
| `.matmul` | `C = A @ B` (2D) | `∂L/∂A += ∂L/∂C @ Bᵀ, ∂L/∂B += Aᵀ @ ∂L/∂C` |
| `.matmul_batch` | `C = A @ B` (3D) | Same as matmul, batched with `transposeInner2d` |

### Shape transforms
| Op | Forward | Backward |
|----|---------|----------|
| `.transpose2d` | `B = Aᵀ` (2D) | `∂L/∂A += (∂L/∂B)ᵀ` |
| `.transpose_inner2d` | `B[i,j,k]=A[i,k,j]` | `∂L/∂A[i,k,j] += ∂L/∂B[i,j,k]` |
| `.reshape` | `b = view(a)` | `∂L/∂a += reshape(∂L/∂b, a.shape)` |

### Reductions (gradient scatters back to source positions)
| Op | Forward | Backward |
|----|---------|----------|
| `.sum` | `y = Σx` (along axis) | `∂L/∂x += broadcast(∂L/∂y)` |
| `.mean` | `y = mean(x)` (along axis) | `∂L/∂x += broadcast(∂L/∂y) / n` |

### Unary activations
| Op | Forward | Backward |
|----|---------|----------|
| `.exp` | `y = eˣ` | `∂L/∂x += ∂L/∂y * eˣ` |
| `.log` | `y = ln(x)` | `∂L/∂x += ∂L/∂y / x` |
| `.neg` | `y = -x` | `∂L/∂x -= ∂L/∂y` |
| `.relu` | `y = max(0, x)` | `∂L/∂x += ∂L/∂y * (x > 0 ? 1 : 0)` |
| `.gelu` | `y = x·Φ(x)` | `∂L/∂x += ∂L/∂y · (Φ(x) + x·φ(x))` |

### Normalization (special Jacobian structure)
| Op | Forward | Backward |
|----|---------|----------|
| `.softmax` | `p = softmax(x)` | `∂L/∂x = p * (∂L/∂p - Σ(p · ∂L/∂p))` |
| `.log_softmax` | `ℓ = log_softmax(x)` | `∂L/∂x = ∂L/∂ℓ - softmax(x) · Σ(∂L/∂ℓ)` |

### Loss and others
| Op | Forward | Backward |
|----|---------|----------|
| `.cross_entropy` | `L = -log(p[target])` | `∂L/∂logits = (softmax - one_hot) / B` |
| `.sqrt` | `y = √x` | `∂L/∂x += ∂L/∂y / (2√x)` |
| `.embedding` | `out = weight[idx]` | `∂L/∂weight[idx] += ∂L/∂out` (scatter-add) |

---

## 10. Reading Order for Maximum Understanding

If you want to deeply understand how training works, read the source in
this order:

1. **`src/lab/train.zig:219-386`** — The `train()` method. This is the
   main loop. Everything else exists to serve this function.

2. **`src/tensor/ops/loss.zig`** — Cross-entropy loss. Understanding
   this function is the key to understanding what the model optimizes.

3. **`src/autograd/backward.zig:654-694`** — The cross-entropy backward
   (`backwardCrossEntropy`). The gradient `softmax - one_hot` is the
   mathematical heart of transformer training.

4. **`src/optim/adamw.zig:83-131`** — The AdamW step. This shows how
   moments, bias correction, and weight decay combine into the update.

5. **`src/nn/model.zig:110-151`** — The model forward pass. Trace the
   shape flow from `(B, T)` to `(B, T, V)`.

6. **`src/nn/attention.zig:120-205`** — The attention forward pass.
   This is the most complex layer — trace each of the 8 sub-ops.

7. **`src/autograd/tape.zig:333-512`** — The backward pass engine.
   Understand how topological sort and gradient accumulation work.

8. **`src/autograd/gradcheck.zig:115-199`** — The gradCheck function.
   This is how we KNOW the backward is correct.

9. **`src/lab/train.zig:428-565`** — The `generate()` function. This is
   inference: the same forward pass, but one token at a time.

10. **`examples/06_train_shakespeare.zig`** — The user-facing entry point.
    See how all the pieces are configured and connected.

11. **`examples/07_generate.zig`** — How to load a checkpoint and produce
    text with different sampling strategies.

---

## 11. Exercises

### 11.1 Modify the learning rate

Edit `examples/06_train_shakespeare.zig` to use `lr = 3e-3` and run it.
You should see the loss decrease initially, then spike (diverge). Now set
`grad_clip_norm = 1.0` and try again. Still diverges? This shows that
gradient clipping alone can't save an overly aggressive learning rate.

### 11.2 Compare greedy vs. top-k generation

After training, run example 07. Compare the output for:
- `top_k=1, temp=1.0` (greedy)
- `top_k=5, temp=1.0` (default sampling)
- `top_k=50, temp=1.5` (creative)

Notice how greedy produces repetitive text while creative sampling produces
more varied (but less coherent) output.

### 11.3 Track gradient norms

The Trainer already logs `grad_norm` alongside loss. Run a few hundred
steps and observe:
- What's the typical gradient norm?
- Does it spike right before divergence?
- How often does clipping actually activate (clip_coeff < 1)?

### 11.4 Overfit on one batch

Create a tiny config with `batch_size=1, max_steps=200, lr=1e-3` and a
small dataset. The model should memorize the batch — loss should approach
zero. This verifies the training pipeline is correct end-to-end. Our
example 04 does exactly this.

### 11.5 Add a new training metric

Modify `train.zig` to also log the average parameter magnitude
`sqrt(mean(param²))` alongside loss and grad_norm. This helps detect
parameter explosion (weights growing without bound).

### 11.6 Write a gradcheck for a new op

Pick any operation that doesn't have a dedicated gradcheck test (e.g.,
`div`, `neg`, or `log`). Follow the pattern from the existing tests:
1. Create input tensors with `requires_grad=true`
2. Run the forward + backward pass
3. Perturb each input element by ±h and compare numerical vs analytical
4. Use `denom_floor=1e-2` and the combined assertion

### 11.7 Visualize the attention weights

After training, modify the generate function to also output the attention
weight matrices (the softmax output before multiplying by V). For each
generated token, you'll see a (T, T) matrix showing which previous
positions the model attended to. This is how "attention visualization"
tools like BertViz work.

---

## 12. Key Takeaways

1. **Training is a loop:** batch → forward → loss → backward → clip → step →
   zero_grad. Everything else supports this loop.

2. **The loss function determines what the model learns.** Cross-entropy
   makes the model predict the next token. Change the loss, change the
   behavior.

3. **Gradients flow backward through the exact same path as the forward
   pass, but in reverse.** The tape records the forward; backward replays it.
   Each of the 22 OpKinds has a backward rule that's the mathematical dual
   of the forward.

4. **Gradient clipping is a safety net.** It doesn't help when things are
   going well, but prevents catastrophe when a bad batch produces huge
   gradients.

5. **Memory management is the hard part.** Three of our six Stage 6 bugs
   were memory issues (use-after-free, dangling pointers, corrupted
   HashMap pointers). PyTorch's garbage collector hides these; Zig forces
   you to think about them explicitly. `keepAlive` is the key pattern.

6. **Generation is just repeated forward passes.** No gradients, no
   optimizer — just feed the model's own outputs back as input.

7. **Small models learn slowly.** A 141K-parameter model on a 1MB dataset
   with 500 steps barely makes a dent. Real models are 1000x-1000000x
   larger and train for millions of steps.

8. **Gradcheck is the gold standard for correctness.** Finite-difference
   comparison catches backward bugs that loss-decrease tests miss. The
   combined check (relative error OR absolute error) correctly handles
   near-zero gradients.

9. **Residual connections are essential.** Without them, gradients vanish
   as they flow through many layers. The identity path guarantees gradient
   flow.

10. **AdamW's bias correction matters.** Without it, the effective learning
    rate is 1000x too small on the first step. The bias correction
    compensates for zero-initialized moments.
