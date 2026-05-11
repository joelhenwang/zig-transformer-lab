# 04b — From Stage 4 NN Modules to Training a Transformer

This chapter bridges the gap between the nn module code you just read (Stage 4:
Linear, Embedding, LayerNorm, Attention, MLP, Block, Model, SGD, AdamW) and
the ML/DL concepts those pieces implement. By the end, you'll understand *why*
each layer exists, *how* it fits into a transformer training loop, and *what
exactly happens* when you call `model.forward()` then `optimizer.step()`.

We assume you've read `docs/04_nn.md` (the mechanical details of each layer
and optimizer) and now want the conceptual picture: how do these pieces
combine to *actually train a language model*?

---

## 1. The Big Picture: What Stage 4 Adds to Stages 2-3

Stages 2 and 3 gave us the *mechanics*: tensor operations and gradient
computation. Stage 4 gives us the *architecture*: a concrete neural network
that uses those mechanics to learn.

```
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 2: Tensors       "How to compute x @ W + b"                  │
│  Stage 3: Autograd      "How to compute ∂L/∂W for that computation" │
│  Stage 4: NN + Optim    "What to compute, what to optimize, and      │
│                           how to update the weights"                 │
│                                                                       │
│  Together:                                                            │
│    model.forward(ids, &tape)  → logits                               │
│    loss = crossEntropy(logits, targets, &tape)                        │
│    tape.backward(&loss)        → fills param.grad for every weight    │
│    optimizer.step(params)      → updates every weight using its grad  │
└───────────────────────────────────────────────────────────────────────┘
```

Stage 4 is the difference between "I can compute gradients" and "I can
train a model." The autograd engine from Stage 3 computes ∂L/∂W for *any*
computation; Stage 4 specifies *which* computation (the transformer
architecture) and *what to do* with the gradients (the optimizer).

### The Conceptual Stack

```
┌─────────────────────────────────────┐
│  Training Loop (example 04)         │  "Repeat forward-backward-update"
├─────────────────────────────────────┤
│  TinyWordTransformer (model.zig)   │  "The complete architecture"
├─────────────────────────────────────┤
│  TransformerBlock (block.zig)       │  "One layer of the architecture"
│    ├── LayerNorm (layernorm.zig)    │  "Normalize activations"
│    ├── Attention (attention.zig)    │  "Route information between positions"
│    ├── MLP (mlp.zig)               │  "Transform each position independently"
│    └── Residual (+ x)              │  "Let gradients flow through"
├─────────────────────────────────────┤
│  Linear (linear.zig)               │  "y = x @ W^T + b"
│  Embedding (embedding.zig)         │  "token ID → dense vector"
│  GELU (activations.zig)            │  "smooth nonlinearity"
├─────────────────────────────────────┤
│  Optimizer (adamw.zig)             │  "How to update weights using gradients"
├─────────────────────────────────────┤
│  Autograd (Stage 3)                 │  "How to compute gradients"
├─────────────────────────────────────┤
│  Tensor Ops (Stage 2)             │  "How to compute matmul, softmax, etc."
└─────────────────────────────────────┘
```

Each level uses the level below it. The training loop uses the model. The
model uses blocks. Blocks use layers. Layers use ops. The optimizer uses
gradients from autograd. Everything rests on tensor operations.

---

## 2. Why Each Layer Exists: The ML/DL Rationale

### 2.1 Embedding — From Discrete to Continuous

**The ML problem:** A language model processes *words*, which are discrete
symbols. Neural networks operate on *continuous vectors*. We need a mapping.

**The DL solution:** Learn a lookup table. Each word gets a dense vector
(called an "embedding"). The model learns these vectors during training —
similar words end up with similar vectors because they appear in similar
contexts.

**What PyTorch does:** `nn.Embedding(V, D)` — exactly the same thing.

**Our implementation:** `Embedding.forward(ids, tape)` copies rows from
the weight table into the output. The backward `scatter-add` accumulates
gradients back into the weight rows.

**Why two embeddings (token + positional)?**

Token embeddings encode *what* the word is. Positional embeddings encode
*where* it is. Without position information, the transformer is
permutation-invariant — it can't distinguish "the cat sat" from "sat
cat the." The model adds token + positional embeddings so each position
gets both semantic and positional information.

GPT-2 uses *learned* positional embeddings (our approach). The original
transformer used *fixed* sinusoidal embeddings. Learned embeddings are
simpler and work well for fixed-length sequences.

### 2.2 Linear — The Only Learnable Computation

**The ML problem:** We need a transformation that can *learn* arbitrary
linear relationships between input and output features.

**The DL solution:** `y = x @ W^T + b` — the only operation with
learnable parameters (weights and biases). Every learnable computation
in a neural network is ultimately a linear transformation.

**Why (D_out, D_in) and not (D_in, D_out)?**

PyTorch convention. The weight is stored as (D_out, D_in) so that:
- Forward: `y = x @ W^T` (one matmul, x is (N, D_in), W^T is (D_in, D_out))
- Backward for W: `∂L/∂W = (∂L/∂y)^T @ x` — the result is naturally
  (D_out, D_in), matching W's shape without an extra transpose.

**Kaiming initialization — why sqrt(6/D_in)?**

Random initialization matters enormously. If weights are too large, the
forward pass produces huge activations, and gradients explode. If weights
are too small, activations vanish, and gradients vanish too.

Kaiming init ensures the *variance* of the output equals the *variance*
of the input (assuming the input is zero-mean). For a linear layer with
D_in inputs:

```
Var(y_i) = D_in * Var(W_ij) * Var(x_j)
```

Setting `Var(y) = Var(x)` gives `Var(W) = 1/D_in`. For a uniform
distribution U(-a, a), Var = a²/3, so a = sqrt(3/D_in). Kaiming
uniform uses bound = sqrt(6/D_in) (the factor of 2 comes from
adjusting for ReLU, which zeros out half the inputs).

**What happens without proper initialization?** The model can't train.
Too-large weights → NaN losses. Too-small weights → loss doesn't
decrease. Kaiming init is one of those "invisible" design choices that
makes everything work.

### 2.3 LayerNorm — Training Stability

**The ML problem:** As data flows through many layers, the distribution
of activations shifts (internal covariate shift). This makes training
unstable — gradients can explode or vanish.

**The DL solution:** Normalize each position's activation to zero mean
and unit variance, then apply a learnable scale (gamma) and shift (beta).

```
x_norm = (x - mean(x)) / sqrt(var(x) + eps)
y = gamma * x_norm + beta
```

**Why gamma=1, beta=0 initially?** So LayerNorm starts as the identity
transform. The network learns to deviate from identity *as needed*.
If gamma started at 0, the entire signal would be zeroed out. If beta
started at a large value, the signal would be shifted unpredictably.

**Why compose from 7 ops instead of a fused backward?**

Pedagogical. Each tape node is a real operation with a real backward
that you can inspect. You can see the gradient flow through:

```
1. mean(x, axis)          → backward: broadcast grad / N
2. x - mean               → backward: grad (for x) + -sumToShape(grad) (for mean)
3. x_centered * x_centered → backward: 2 * grad * x_centered
4. mean(x_centered², axis) → backward: broadcast grad / N
5. variance + eps          → backward: grad passes through
6. sqrt(var + eps)         → backward: grad / (2 * sqrt(var + eps))
7. x_centered / sqrt      → backward: grad / sqrt (for x) + -grad * x / sqrt² (for sqrt)
8. gamma * x_norm          → backward: grad * x_norm (for gamma), grad * gamma (for x_norm)
9. scaled + beta           → backward: grad (for both, with sumToShape for beta)
```

That's 9 tape nodes for a single LayerNorm forward. Compare this to a
fused LayerNorm that would record 1 node. The fused version is ~10x
faster at runtime, but you lose the ability to see *how* the gradient
flows through each step.

**The cost:** With 3 LayerNorms per block (ln1, ln2, ln_f), that's
27 tape nodes just for normalization. For our tiny model, this is fine.
For production, you'd fuse this into a single kernel (as we will in
Stage 8 with `layernorm.cu`).

### 2.4 GELU — Why Not ReLU?

**The ML problem:** We need a nonlinearity. Without it, stacking linear
layers is equivalent to a single linear layer (because linear
transformations compose into linear transformations).

**The DL solution:** Apply a nonlinear function elementwise.

**ReLU vs GELU:**

```
ReLU(x)  = max(0, x)         — zero for negative, identity for positive
GELU(x)  = x * Φ(x)          — smooth curve, slightly negative for x < 0
```

ReLU has a hard zero for x < 0 — neurons that receive negative input
produce zero output and zero gradient ("dead neurons"). GELU has a small
but nonzero output for moderately negative inputs, which means:

1. **No dead neurons.** GELU neurons can recover from negative activations
   because the gradient is nonzero.
2. **Smooth gradient.** GELU's gradient is continuous at x=0, unlike
   ReLU's discontinuous jump.
3. **Better performance.** GPT-2 and most modern transformers use GELU.

**Why is it stateless?** GELU has no parameters — it's a fixed function.
We wrap it as an "nn layer" only for API consistency (so MLP can treat
it like any other sub-layer). The `parameters()` method is a no-op.

### 2.5 Attention — The Core Mechanism

**The ML problem:** In a sequence, each position needs information from
*other* positions. A fully-connected approach (every position connects to
every other) doesn't scale. A fixed approach (e.g., only look at the
previous position) is too limited.

**The DL solution:** Learn *where* to look. For each position, compute
a weighted average of all other positions, where the weights are learned.

**The three roles — Query, Key, Value:**

Think of attention as a retrieval system:

```
Query:  "What am I looking for?"     (the current position's information need)
Key:    "What do I contain?"         (each position's content descriptor)
Value:  "What information do I carry?" (the actual content to retrieve)

Attention = softmax(Query · Key^T) · Value
```

Positions whose keys match the query get high attention weights. Their
values contribute more to the output. This is the "attention" — the model
*pays attention* to relevant positions.

**Why scale by 1/sqrt(D)?**

The dot product Q·K grows with dimension D. For D=64, the typical dot
product magnitude is ~64. This pushes softmax into saturation (near-one-hot
distributions), which has tiny gradients. Dividing by sqrt(D) keeps the
softmax input in a reasonable range.

**Why causal masking?**

During language model training, position i should only attend to positions
j ≤ i. If position 3 could attend to position 5, the model would be
"cheating" — it would see future tokens that it's supposed to predict.

```
mask[i,j] = 0      if j ≤ i    (allowed — look at present and past)
mask[i,j] = -1e9   if j > i    (forbidden — don't look at future)
```

After adding this mask to the scores and applying softmax, positions j > i
get attention weight ≈ 0 (because e^{-1e9} ≈ 0).

**The four Linear sub-layers:**

```
W_q: (D, D) — projects input to queries
W_k: (D, D) — projects input to keys
W_v: (D, D) — projects input to values
W_o: (D, D) — projects attention output back to D dimensions
```

Each of these is a standard Linear layer with its own weight and bias.
Their gradients flow through `backwardMatmul` — the same mechanism that
works for every linear layer.

**The data flow through one attention forward:**

```
Input x: (B, T, D)

1. Q = W_q(x)      (B, T, D)    — 2 tape nodes: reshape + matmul
2. K = W_k(x)      (B, T, D)    — 2 tape nodes: reshape + matmul
3. V = W_v(x)      (B, T, D)    — 2 tape nodes: reshape + matmul
4. K^T = trans(K)   (B, D, T)    — 1 tape node: transposeInner2d
5. scores = Q @ K^T  (B, T, T)   — 1 tape node: matmulBatch
6. scores *= 1/√D               — 1 tape node: mulScalar
7. scores += mask               — 2 tape nodes: reshape + add
8. weights = softmax(scores)     (B, T, T) — 1 tape node: softmax
9. attn_out = weights @ V        (B, T, D) — 1 tape node: matmulBatch
10. output = W_o(attn_out)       (B, T, D) — 2 tape nodes: reshape + matmul

Total: ~15 tape nodes for one attention forward
```

Every intermediate is kept alive via `tape.keepAlive()` so the backward
can access its data.

### 2.6 MLP — The "Thinking" Layer

**The ML problem:** Attention routes information between positions, but
it doesn't *transform* that information. We need a layer that can learn
complex nonlinear relationships.

**The DL solution:** A position-wise feed-forward network — two linear
layers with a GELU activation between them.

```
MLP(x) = W_2(GELU(W_1(x) + b_1)) + b_2
         \___________/      \____/
          expand + activate   project back
```

**Why expand then contract?** The hidden dimension d_ff is typically 4×
the model dimension D. This gives the network more "working space" to
represent intermediate computations. The first linear layer projects into
a higher-dimensional space where it's easier to separate features
linearly, then GELU applies a nonlinearity, then the second layer
projects back to D dimensions.

**Where do most parameters live?** In the MLP. With d_ff = 4D:

```
MLP:  W1 (d_ff, D) + W2 (D, d_ff) = 2 * 4D² = 8D² parameters
Attn: Wq + Wk + Wv + Wo = 4 * D² parameters
```

The MLP has 2× the parameters of attention. This ratio (2:1 MLP:Attn)
is standard across transformer models.

### 2.7 TransformerBlock — Pre-Norm Residual Architecture

**The ML problem:** Deep networks are hard to train. Stacking many
layers without residual connections causes gradients to vanish — the
gradient from the loss gets multiplied by many small factors as it flows
backward through each layer.

**The DL solution:** Residual connections (skip connections) and pre-norm.

**Pre-norm vs post-norm:**

```
Pre-norm (our choice, GPT-2):
  h = x + Attention(LayerNorm(x))
  out = h + MLP(LayerNorm(h))

Post-norm (original transformer):
  h = LayerNorm(x + Attention(x))
  out = LayerNorm(h + MLP(h))
```

Pre-norm is more stable because:
1. The residual path `x → (+x)` is a *pure identity* — the gradient
   flows through without being transformed.
2. LayerNorm normalizes the input to attention/MLP, preventing the
   activations from growing or shrinking.
3. The gradient from the loss can reach early layers directly through
   the residual path, even if the sublayer's backward produces small
   gradients.

**How the residual affects the gradient:**

```
Forward:  h = x + Attention(LN(x))
Backward: ∂L/∂x = ∂L/∂h * 1 (from residual) + ∂L/∂h * ∂Attn/∂x (from sublayer)
```

The `* 1` is the identity gradient — it passes the upstream gradient
through unchanged. Even if `∂Attn/∂x` is tiny, the `* 1` ensures the
gradient reaches earlier layers. This is the "gradient highway."

### 2.8 TinyWordTransformer — The Complete Model

**The ML problem:** Given a sequence of tokens, predict the next token.

**The DL solution:** The complete transformer architecture:

```
ids → tok_embed + pos_embed → block → ln_f → lm_head → logits
       (B,T)    (B,T,D)                (B,T,D)  (B,T,V)
```

**The output projection (lm_head):** Maps from D dimensions back to V
(vocabulary size). The output at each position is a vector of V logits
— one score per vocabulary word. The highest-scoring word is the
model's prediction.

**Why no bias in lm_head?** Two reasons:
1. The token embeddings already encode per-token offsets (the mean of
   each embedding row acts as a bias).
2. Weight tying (sharing lm_head.weight and tok_embed.weight) requires
   the same shape, and embedding tables don't have biases.

**Save/load — why checkpoint?** Training takes many steps. If the
process crashes, you'd lose all progress. Checkpoints let you save and
resume. Our binary format includes parameter names for forward
compatibility — you can load a checkpoint even if the model has new
parameters that the checkpoint doesn't contain.

---

## 3. How the Optimizer Uses Gradients: The ML/DL Math

### 3.1 What Does `optimizer.step()` Actually Do?

After `tape.backward(&loss)`, every parameter has a `.grad` tensor of
the same shape. The optimizer reads `.grad` and updates `.data`:

```
SGD:     W.data[i] -= lr * W.grad.data[i]
AdamW:   W.data[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * W.data[i])
```

**The learning rate (lr) controls step size.** Too large → overshoot,
loss oscillates. Too small → slow convergence. Typical values:

```
SGD:    lr = 0.01 — 0.1
AdamW:  lr = 1e-4 — 3e-4 (for transformers)
```

AdamW uses smaller learning rates because its adaptive scaling
effectively increases the step size for parameters with small gradients.

### 3.2 SGD — The Simplest Optimizer

**The ML idea:** Walk downhill. The gradient points uphill (toward
higher loss), so subtract it.

```
W = W - lr * ∂L/∂W
```

**Momentum (why it helps):** The gradient is noisy — each batch gives
a different gradient estimate. Momentum smooths the noise by keeping a
running average of past gradients:

```
v = momentum * v + ∂L/∂W
W = W - lr * v
```

With momentum=0.9, the current update is a 90/10 mix of the previous
direction and the new gradient. This:
1. **Accelerates** along consistent gradient directions (the momentum
   builds up).
2. **Dampens** oscillations from noisy gradients (they cancel out).

**Coupled weight decay in SGD:**

```
v = momentum * v + grad + weight_decay * W
W = W - lr * v
```

The `weight_decay * W` term penalizes large weights (L2 regularization).
In SGD, it's "coupled" because it's added to the gradient before
momentum — the momentum also builds up from the decay term.

### 3.3 AdamW — The Transformer Optimizer

**Why AdamW and not Adam?**

Standard Adam adds weight decay to the gradient (coupled), then applies
adaptive scaling. This means the effective weight decay depends on the
adaptive learning rate — some parameters get more decay than others.

AdamW applies weight decay *directly* to the parameter (decoupled):

```
Adam:  W -= lr * m̂ / (√v̂ + ε)          (adaptive update only)
       where m̂, v̂ include weight_decay in the gradient

AdamW: W -= lr * (m̂ / (√v̂ + ε) + wd * W) (adaptive update + separate decay)
       where m̂, v̂ are computed from the raw gradient only
```

The difference is subtle but empirically important. Loshchilov &
Hutter (2017) showed that decoupled decay produces better
generalization, especially when you change the learning rate during
training (e.g., learning rate schedules).

**The per-parameter adaptation:**

```
m = β₁ * m + (1 - β₁) * grad    ← "What's the average gradient?"
v = β₂ * v + (1 - β₂) * grad²  ← "How variable is the gradient?"
```

If a parameter's gradient is consistently large (high m), Adam
takes a larger step. If the gradient is highly variable (high v),
Adam *reduces* the step size (because √v is in the denominator).
This is the "adaptive" part — each parameter gets its own effective
learning rate.

**Bias correction — why it matters early in training:**

m and v start at zero. At step 1:

```
m₁ = 0.1 * grad    (only 10% of the first gradient, because β₁=0.9)
v₁ = 0.05 * grad²  (only 5% of the first squared gradient, because β₂=0.95)
```

This underestimation biases m and v toward zero. The correction fixes it:

```
m̂₁ = m₁ / (1 - 0.9¹) = m₁ / 0.1 = grad    ← correct!
v̂₁ = v₁ / (1 - 0.95¹) = v₁ / 0.05 = grad²  ← correct!
```

Without correction, the first few steps would update parameters by only
a tiny fraction of the true gradient — the model would barely learn.

**Why β₂=0.95 instead of 0.999?**

The standard Adam uses β₂=0.999, which means v adapts very slowly — it
takes ~1000 steps for v to reflect the current gradient magnitude. For
our pedagogically short training runs (50-500 steps), the optimizer
would still be "warming up" and the loss wouldn't decrease noticeably.

β₂=0.95 adapts 20× faster, making the optimizer effective within
just a few dozen steps. In a real training run with millions of
steps, you'd use β₂=0.999.

### 3.4 Weight Decay — Implicit Regularization

**The ML idea:** Prevent the model from relying too heavily on any single
feature by penalizing large weights.

```
loss_total = loss_data + λ * Σ(W_i²)
```

The λ term (weight decay) adds a penalty for large weights. The
gradient of this penalty is 2λW, which the optimizer subtracts from
the weights, pulling them toward zero.

**Why does this help generalization?** Large weights can "memorize"
the training data (overfitting). Small weights produce smoother,
simpler functions that generalize better. Weight decay encourages
the model to find simple solutions.

**Decoupled decay in AdamW:**

```
W -= lr * (m̂ / (√v̂ + ε) + wd * W)
                 ↑                  ↑
           adaptive step        decay step
        (different per param)  (same rate for all)
```

The adaptive step handles the "direction" (sign and magnitude of the
gradient). The decay step handles the "regularization" (pulling toward
zero). They're independent — changing the learning rate doesn't change
the regularization strength.

---

## 4. The Complete Training Loop: Step by Step

Let's trace through one training step of example 04, connecting each
line of code to its ML/DL meaning.

### 4.1 Setup

```zig
var model = try TinyWordTransformer.init(allocator, cfg, &rng);
var adam = try AdamW.init(allocator, .{ .lr = 1e-3 });
var opt = adam.optimizer();

var params: std.ArrayList(*Tensor) = .empty;
model.parameters(&params);  // collect all *Tensor pointers
```

**What this does in ML terms:**
- Create the model architecture (random weights)
- Create the optimizer (with state buffers for m and v)
- Collect all learnable parameters into a flat list

**Why flat list?** The optimizer doesn't need to know the model's
structure — it just needs a list of `(data, grad)` pairs. This
decouples the optimizer from the model architecture. You can use
the same AdamW code for any model.

### 4.2 One Training Step

```zig
// 1. Fresh tape
var tape = Tape.init(allocator);
defer tape.deinit();
```

**ML meaning:** Reset the computation graph. Each step is a fresh
forward pass + backward pass. The old graph is discarded.

**Why fresh?** The graph records which operations used which inputs.
After backward, the graph has served its purpose. Keeping it around
wastes memory. PyTorch also frees the graph after backward by default.

```zig
// 2. Register parameters
for (params.items) |param| {
    param.requires_grad = true;
    _ = try tape.trackLeaf(param);
}
```

**ML meaning:** Tell the autograd engine which tensors need gradients.

**Why explicit?** Not every tensor needs gradients. Position IDs,
causal masks, and input tokens are constants — they don't get updated.
Only model weights (parameters) need gradients.

**Critical pitfall:** `trackLeaf` must create a *fresh* node each step.
The old `tape_node` from the previous step's tape is invalid — it
references a destroyed tape. Our implementation ignores stale values.

```zig
// 3. Forward pass
var logits_3d = try model.forward(ids, &tape);
try tape.keepAlive(&logits_3d);
```

**ML meaning:** Run the model on the input, computing predictions
(logits) for each position. The tape records every operation.

**What happens inside `model.forward()`:**

```
ids (B, T)
  → tok_embed: gather rows → (B, T, D)
  → pos_embed: gather rows → (1, T, D)
  → add tok + pos           → (B, T, D)
  → block:
      LN1(x)               → (B, T, D)    [9 tape nodes]
      Attn(LN1)            → (B, T, D)    [~15 tape nodes]
      x + Attn             → (B, T, D)    [1 tape node]
      LN2(h)               → (B, T, D)    [9 tape nodes]
      MLP(LN2)             → (B, T, D)    [~8 tape nodes]
      h + MLP              → (B, T, D)    [1 tape node]
  → ln_f                   → (B, T, D)    [9 tape nodes]
  → lm_head                → (B, T, V)    [~3 tape nodes]

Total: ~55 tape nodes for one forward pass
```

Each node saves the data it needs for backward (inputs, shapes, etc.).
The `keepAlive` calls transfer ownership of intermediate buffers to
the tape so they survive until backward runs.

```zig
// 4. Reshape logits for cross-entropy
const logits = try logits_3d.reshape(Shape.init2D(B * T, V));
const targets_flat = try targets.reshape(Shape.init1D(B * T));
```

**ML meaning:** Cross-entropy expects 2D logits (N, V) where N is the
number of predictions. We flatten (B, T, V) → (B*T, V) to treat each
position as an independent prediction.

**Why reshape (view) not reshapeTracked?** The reshape is a view — it
shares the same data as the 3D logits. The gradient flows through the
3D logits which are already on the tape from the model forward. Adding
another reshape node would create a redundant path.

```zig
// 5. Compute loss
var loss = try ztl.ops.loss.crossEntropy(allocator, logits, targets_flat, &tape);
```

**ML meaning:** Measure "how wrong" the model's predictions are. The
loss is a scalar — one number. Lower is better.

**What cross-entropy computes:**

```
For each position i:
  softmax = exp(logits[i]) / sum(exp(logits[i]))
  loss_i = -log(softmax[i, target_i])

loss = mean(loss_i) over all positions
```

**The beautiful gradient:** ∂L/∂logits = (softmax - one_hot) / B.
This drives all learning — it's the signal that flows backward through
every layer.

```zig
// 6. Backward pass
try tape.backward(&loss);
```

**ML meaning:** Compute the gradient of the loss with respect to every
parameter. This is the chain rule applied to the entire computation
graph, from loss back to each weight.

**What happens inside `tape.backward()`:**

```
1. Set ∂L/∂loss = 1.0 (seed gradient)

2. Walk tape in reverse (from last node to first):
   For each node with op kind:
     .cross_entropy → ∂L/∂logits = (softmax - one_hot) / B
     .matmul        → ∂L/∂x = ∂L/∂y @ W^T, ∂L/∂W = x^T @ ∂L/∂y
     .add           → ∂L/∂a = sumToShape(∂L/∂c, a.shape)
     .mul           → ∂L/∂a = ∂L/∂c * b
     .softmax       → ∂L/∂scores = S * (∂L/∂S - (∂L/∂S · S))
     .gelu          → ∂L/∂x = ∂L/∂y * GELU'(x)
     .embedding     → ∂L/∂weight[idx] += ∂L/∂output[idx]
     .sqrt          → ∂L/∂x = ∂L/∂y / (2 * sqrt(x))
     .mean          → ∂L/∂x = broadcastTo(∂L/∂y / N, x.shape)
     .reshape       → ∂L/∂x = reshape(∂L/∂y, x.shape)
     .transpose2d   → ∂L/∂x = transpose2d(∂L/∂y)
     ...

3. Accumulate gradients for each parameter:
   When multiple backward paths reach the same tensor, their
   gradients are summed (+=). This is the multivariable chain rule.

4. Store final gradients in param.grad for each leaf parameter.
```

**Gradient flow through the model (simplified):**

```
loss = 3.83
∂L/∂loss = 1.0

backwardCrossEntropy:
  ∂L/∂logits = (softmax(logits) - one_hot) / B    shape: (B*T, V)

backwardMatmul (lm_head):
  ∂L/∂ln_out = ∂L/∂logits @ W_head^T              shape: (B, T, D)
  ∂L/∂W_head = ln_out^T @ ∂L/∂logits              shape: (D, V)

backwardReshape (3D→2D):
  ∂L/∂ln_out_3d = reshape(∂L/∂ln_out, (B,T,D))    shape: (B, T, D)

backwardAdd (residual: h + mlp_out):
  ∂L/∂h += ∂L/∂block_out                          (residual path)
  ∂L/∂mlp_out += ∂L/∂block_out                     (sublayer path)

... and so on through every layer, all the way back to:

backwardEmbedding (tok_embed):
  ∂L/∂tok_weight[idx] += ∂L/∂embed_output[idx]    (scatter-add)
```

```zig
// 7. Save loss value, free tensors
const loss_val = loss.data[0];
loss.deinit(allocator);
logits_3d.deinit(allocator);
```

**ML meaning:** Record the loss for monitoring. Free memory.

**Why save before free?** `loss.data[0]` reads from the loss tensor's
heap buffer. `loss.deinit()` frees that buffer. Reading after free is
undefined behavior (use-after-free bug). Save the value first.

**Why is logits_3d.deinit() a no-op?** The `tape.keepAlive(&logits_3d)`
call transferred the data buffer ownership to the tape. It set
`logits_3d.owned = false`, so `deinit()` skips the free. The tape
will free the buffer in `tape.deinit()`.

```zig
// 8. Optimizer step
try opt.step(params.items);
opt.zeroGrad(params.items);
```

**ML meaning:** Update all weights using their gradients, then clear
the gradients for the next step.

**What happens inside `opt.step()` (AdamW):**

```
For each parameter:
  1. m = 0.9 * m + 0.1 * grad              (update 1st moment)
  2. v = 0.95 * v + 0.05 * grad²            (update 2nd moment)
  3. m̂ = m / (1 - 0.9^t)                   (bias-corrected 1st)
  4. v̂ = v / (1 - 0.95^t)                  (bias-corrected 2nd)
  5. W -= lr * (m̂ / (√v̂ + ε) + 0.01 * W)  (update + decay)
```

**Why zero gradients after?** Gradients *accumulate* (+=) during
backward. If we don't zero them, the next step's gradients would
be added to the current step's gradients, effectively doubling the
update. After zeroing, all `.grad` fields contain zeros, ready for
the next backward pass.

### 4.3 Loss Over 50 Steps

```
Step   0: loss = 3.830872    ← near log(V)=log(32)=3.47 (random chance)
Step  10: loss = 3.618151    ← starting to learn
Step  20: loss = 3.444025    ← improving
Step  30: loss = 3.300070    ← converging
Step  40: loss = 3.178837    ← nearly there
Step  49: loss = 3.086886    ← memorizing the batch
```

**What "overfitting one batch" proves:**

If the model can overfit a small batch (loss → 0), it proves:
1. Forward pass computes correct shapes
2. Backward pass computes correct gradients
3. Optimizer updates weights in the right direction
4. The keepAlive mechanism works (no use-after-free)
5. All layers compose correctly end-to-end

If *any* of these are broken, the loss won't decrease. This is the
"hello world" of deep learning — can the model learn *anything*?

**Why loss starts near log(V):** With random weights, the model's
predictions are uniform — it assigns equal probability to all V words.
The cross-entropy of a uniform distribution over V classes is log(V).
With V=32, log(32) ≈ 3.47. Our initial loss of 3.83 is slightly above
this, which is expected (random initialization isn't exactly uniform).

---

## 5. Memory Management: The Most Subtle Part of Stage 4

### 5.1 The Intermediate Tensor Lifetime Problem

Every nn layer's `forward()` creates temporary tensors:

```
Linear.forward():
  x_flat = reshapeTracked(input)     ← owned copy, heap-allocated
  wt = transpose2dTracked(weight)     ← owned copy, heap-allocated
  output = matmul(x_flat, wt)         ← owned result
  bias_2d = reshapeTracked(bias)     ← owned copy
  biased = add(output, bias_2d)      ← owned result
  output_3d = reshapeTracked(biased)  ← owned result
```

Each of these is `defer`-freed at the end of `forward()`. But the
tape's SavedData stores `data` slices pointing into these buffers.
If we free them before backward, the backward reads garbage.

### 5.2 The keepAlive Mechanism

**PyTorch's approach:** Reference counting. When a tensor is saved in
the computation graph, its reference count increments. When the graph
is freed, the count decrements. The tensor is freed only when both the
user and the graph release it.

**Our approach:** Buffer transfer. Before `defer deinit()`, we call
`tape.keepAlive(&tensor)`, which:

1. Appends `tensor.data` to `tape.kept_alive` (a list of `[]f32` slices)
2. Sets `tensor.owned = false` so subsequent `deinit()` is a no-op
3. The tape frees all kept-alive buffers in `tape.deinit()`

This is simpler than reference counting but has the same effect:
intermediate buffers live until the tape is destroyed.

### 5.3 When Does keepAlive NOT Matter?

**Parameters owned by the model.** `self.weight` in Linear, Embedding,
etc. These tensors outlive the tape — they exist across training steps.
Their data is never freed by `defer deinit()`, so SavedData references
are always valid.

**Input tensors from the training loop.** `ids`, `targets` — these
persist across the entire step and are freed after backward.

### 5.4 When Does keepAlive Matter?

**Owned intermediates created by ops.** The `reshapeTracked` and
`transpose2dTracked` functions create *owned copies* (not views).
These are the dangerous ones — they're freed by `defer deinit()`.

**Op results used by downstream ops.** `matmul(x, wt)` produces an
owned result. If the next op (like `add`) saves this result in its
SavedData, the result must survive until backward.

**The rule of thumb:** If you create a tensor in `forward()` and
`defer deinit()` it, call `keepAlive` first.

---

## 6. Shape Trace Through the Complete Model

For V=32, D=16, T=8, B=2:

```
TinyWordTransformer.forward(ids: (2, 8)):

  tok_embed.forward(ids: (2, 8))
    weight: (32, 16), ids → output: (2, 8, 16)
    tape node: .embedding (scatter-add backward)

  pos_embed.forward(pos_ids: (1, 8))
    weight: (8, 16), pos_ids → output: (1, 8, 16)
    tape node: .embedding

  add(tok, pos) → (2, 8, 16)
    tape node: .add (broadcast backward for pos)

  block.forward(x: (2, 8, 16)):
    ln1.forward(x) → (2, 8, 16)
      ~9 tape nodes (mean, sub, mul, sqrt, div, add, etc.)

    attn.forward(ln1_out: (2, 8, 16)):
      w_q.forward(ln1_out) → (2, 8, 16)    [3 nodes: reshape, matmul, reshape]
      w_k.forward(ln1_out) → (2, 8, 16)    [3 nodes]
      w_v.forward(ln1_out) → (2, 8, 16)    [3 nodes]
      k_t = transposeInner2d(k) → (2, 16, 8)  [1 node]
      scores = matmulBatch(q, k_t) → (2, 8, 8)  [1 node]
      scaled = mulScalar(scores, 1/√16=0.25) → (2, 8, 8)  [1 node]
      mask_3d = reshape(mask, (1, 8, 8))    [1 node, no tape]
      masked = add(scaled, mask_3d) → (2, 8, 8)  [1 node]
      weights = softmax(masked) → (2, 8, 8)  [1 node]
      attn_out = matmulBatch(weights, v) → (2, 8, 16)  [1 node]
      w_o.forward(attn_out) → (2, 8, 16)    [3 nodes]
      ~19 tape nodes

    h = add(x, attn_out) → (2, 8, 16)     [1 node: residual]

    ln2.forward(h) → (2, 8, 16)           [~9 nodes]
    mlp.forward(ln2_out):
      fc1.forward(ln2_out) → (2, 8, 64)   [3 nodes]
      gelu.forward(fc1_out) → (2, 8, 64)  [1 node]
      fc2.forward(gelu_out) → (2, 8, 16)  [3 nodes]
      ~7 tape nodes

    output = add(h, mlp_out) → (2, 8, 16)  [1 node: residual]
    ~47 tape nodes in block

  ln_f.forward(block_out) → (2, 8, 16)    [~9 nodes]
  lm_head.forward(ln_out) → (2, 8, 32)    [3 nodes (no bias)]

  logits: (2, 8, 32)
  Total: ~62 tape nodes
```

Each tape node consumes:
- 1 Node struct (~120 bytes: id, op, parents, saved data)
- Saved data (varies: 2 Tensor snapshots ≈ 2×(shape+data slice))
- Gradient tensor (same shape as the node's output)

For B=2, T=8, D=16, V=32, the total activation memory is modest
(~50 KB). But the pattern scales: a real GPT-2 with B=512, T=1024,
D=768 would need gigabytes of activation memory.

---

## 7. PyTorch Equivalents for the Full Training Loop

| Our Code | PyTorch Equivalent | Notes |
|----------|--------------------|-------|
| `model.forward(ids, &tape)` | `logits = model(ids)` | PyTorch records automatically if requires_grad |
| `tape.keepAlive(&tensor)` | (automatic via refcounting) | We must call explicitly |
| `loss = crossEntropy(alloc, logits, targets, &tape)` | `loss = F.cross_entropy(logits, targets)` | Same formula, same gradient |
| `tape.backward(&loss)` | `loss.backward()` | Same topological sort + chain rule |
| `opt.step(params.items)` | `optimizer.step()` | Same update rules |
| `opt.zeroGrad(params.items)` | `optimizer.zero_grad()` | Same purpose: clear gradients |
| `tape.deinit()` | (automatic GC) | Must call explicitly in Zig |
| `model.save("ckpt.bin")` | `torch.save(model.state_dict(), "ckpt")` | Binary vs pickle format |
| `model.load("ckpt.bin")` | `model.load_state_dict(torch.load("ckpt"))` | Name-based matching |

### Key Conceptual Differences

1. **Explicit vs implicit graph.** PyTorch builds the graph implicitly
   when tensors with `requires_grad=True` participate in operations.
   We pass `&tape` explicitly. This makes the recording decision visible
   and allows `null` for inference mode (zero overhead).

2. **Memory management.** PyTorch uses reference counting (with cycle
   collector). We use explicit `deinit()` and `keepAlive()`. This is
   more verbose but gives deterministic memory behavior.

3. **No Python GIL.** PyTorch's autograd is thread-safe because Python
   has the GIL. Our explicit tape can be used from any thread without
   locks — each thread has its own tape.

4. **No dynamic dispatch for layers.** PyTorch's `nn.Module` uses
   Python's dynamic dispatch for `forward()`. Our layers are concrete
   structs with direct function calls — no vtable, no indirection.

---

## 8. Common Mistakes When Connecting NN Code to ML/DL Concepts

### 1. Treating loss not decreasing as an optimizer bug

**More likely causes:**
- Learning rate too high (loss oscillates) or too low (loss flat)
- Forgetting `requires_grad = true` on parameters
- Forgetting `tape.trackLeaf()` before the forward pass
- Using β₂=0.999 for short training runs (optimizer still warming up)

### 2. Confusing parameter count with model quality

Our tiny model has ~4,464 parameters. GPT-2 has 1.5 billion. The
parameter count determines *capacity* (what the model *can* learn), not
*quality* (what the model *has* learned). A small model trained on the
right data can outperform a large model trained on the wrong data.

### 3. Ignoring the causal mask

If you remove the causal mask from attention, the model can "cheat" by
looking at future tokens during training. The loss will decrease faster,
but the model won't work at inference time (when future tokens aren't
available). This is a common beginner mistake that produces a model
that trains well but generates garbage.

### 4. Forgetting that embedding backward is scatter-add

Unlike most ops where the gradient is a simple elementwise or matmul
operation, embedding backward *scatters* gradients to specific rows.
If two input positions have the same token ID, their gradients
*accumulate* in the same weight row. This is correct behavior — the
word "the" appearing 50 times in a batch should get 50× the gradient
signal.

### 5. Thinking more layers = better model

Our model has 1 block. Adding more blocks increases capacity, but
also increases the risk of:
- Training instability (deeper = harder to train)
- Overfitting (more parameters can memorize training data)
- Slower convergence (gradients must flow through more layers)

The residual connections help, but they don't eliminate these issues.
For our pedagogical model, 1 block is the right choice (decision D5).

### 6. Not understanding why the optimizer has state

SGD with momentum has velocity buffers. AdamW has m and v buffers.
These persist across training steps — they're not part of the tape
(which is recreated each step). If you recreate the optimizer each
step, you lose all the momentum information and training degrades.

### 7. Treating weight decay as optional

For small models, weight decay barely matters. For large models, it's
critical. Without it, weights grow without bound, the model becomes
overconfident, and generalization suffers. Our AdamW default of
`weight_decay=0.1` is reasonable for a small transformer.

### 8. Confusing the tape's role with the optimizer's role

The tape computes *gradients* (how much each parameter contributes to
the loss). The optimizer computes *updates* (how much to change each
parameter). These are different operations:

```
Tape:  ∂L/∂W = 0.05     (the gradient says "increase W a bit")
SGD:   ΔW = -lr * 0.05 = -0.0005  (SGD decreases W by 0.0005)
AdamW: ΔW = -lr * m̂/(√v̂ + ε) - wd*W  (AdamW adjusts differently)
```

Same gradient, different updates. The optimizer's job is to interpret
the gradient signal intelligently.

---

## 9. Where to Go Next

- **Stage 5** — Tokenizer and data pipeline: real text → token IDs → batches
- **Stage 6** — Full CPU training loop on `data/tiny.txt`
- **Stage 7** — Generation: sample from the model's output distribution
- **Stage 8** — CUDA acceleration: same model, faster matmul
- **docs/05_transformer_math.md** — Detailed shape trace through the math


---

## Exercises

**Exercise 1.** AdamW's update rule includes a bias-correction
factor `1 / (1 - beta^t)`. At what step `t` does this factor
become negligible (say, within 1% of 1.0) for `beta1 = 0.9`?

<details><summary>Solution</summary>

`1 / (1 - 0.9^t) < 1.01` solves to
`0.9^t < 1 / 1.01 * (1 - 1) = ...`, or equivalently
`1 - 0.9^t > 0.990...`, so `0.9^t < 0.01`. Taking logs:
`t * log(0.9) < log(0.01)`, `t > log(0.01) / log(0.9) = 43.7`.
So by step 44, bias correction is within 1% of 1.0 and the update
is effectively uncorrected Adam.

For `beta2 = 0.999` the number is much larger: `t > log(0.01) /
log(0.999) = 4602`. This is why the first several thousand steps
of AdamW behave measurably differently from Adam - the second
moment is still warming up.

</details>

**Exercise 2.** A `Linear(64, 2000)` layer has weight matrix of
shape `(2000, 64)` = 128 000 floats. With AdamW each parameter
needs two moment buffers. What is the total memory footprint of
this single layer's trainable state?

<details><summary>Solution</summary>

Weight: `128 000 * 4 = 512 KB`.
First moment `m`: `512 KB`.
Second moment `v`: `512 KB`.
Gradient buffer: `512 KB` (transient per step).
Total: `~2 MB` for one layer while training.

At inference: weight only, 512 KB. This is the 4x-at-training
overhead AdamW is known for. If memory is tight, switching to SGD
drops this to weight + grad = 1 MB (2x inference).

</details>
