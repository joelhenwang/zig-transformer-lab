# 02b — From Stage 2 Tensors to Training a Transformer

This chapter bridges the gap between the code you just read (Stage 2: shapes,
strides, broadcasting, softmax, matmul) and the ML/DL concepts those pieces
were built for.  By the end, you'll understand *why* each Stage 2 function
exists and where it appears in a transformer training loop.

We assume you've read `docs/02_tensors.md` (the mechanical details) and now
want the conceptual picture: how do these pieces combine to train a model?

---

## 1. The Big Picture: What Training a Transformer Does

A transformer training loop has four phases that repeat thousands of times:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. FORWARD PASS                                               │
│     Input tokens → Embedding → Transformer blocks → Logits      │
│     "What does the model currently predict?"                    │
│                                                                 │
│  2. LOSS COMPUTATION                                            │
│     Logits + Targets → Cross-entropy loss (scalar)              │
│     "How wrong are the predictions?"                            │
│                                                                 │
│  3. BACKWARD PASS (autograd)                                     │
│     Loss → ∂L/∂W for every weight W                            │
│     "Which direction should each weight move to reduce loss?"   │
│                                                                 │
│  4. WEIGHT UPDATE (optimizer)                                   │
│     W = W - lr × ∂L/∂W                                        │
│     "Actually move the weights in that direction."              │
└─────────────────────────────────────────────────────────────────┘
```

**Stage 2 provides the operations for phase 1 and 2.** Phases 3 and 4 come
in Stage 3 (autograd) and Stage 4 (optimizer), but you need to understand
phases 1-2 deeply before those make sense.

### Where Stage 2 Ops Appear

| Training Phase | Stage 2 Ops Used |
|---------------|-----------------|
| Forward: embedding lookup | `fromSlice`, `reshape` |
| Forward: linear layers | `matmul`, `add` (broadcast bias) |
| Forward: activation | `geluExact`, `relu` |
| Forward: attention scores | `matmul`, `mul` (masking) |
| Forward: attention weights | `softmax` |
| Forward: weighted sum | `matmul` |
| Forward: residual connection | `add` |
| Loss: next-token prediction | `logSoftmax`, `crossEntropy` |
| Debugging any phase | `debugSummary`, `printValues` |

Matmul appears four times — that's why it's the most important operation.

---

## 2. Stage 2 Component → Transformer Role

### 2.1 Shape and Strides → Every Layer Needs Shape Reasoning

A transformer processes tensors with shape `(B, T, C)`:

- **B** = batch size (how many sequences we process in parallel)
- **T** = sequence length (how many tokens per sequence)
- **C** = channel dimension (embedding size, e.g., 64)

Every operation must reason about shapes: matmul requires inner dimensions to
match, broadcasting stretches biases across the batch, and reshape reinterprets
attention heads. If any shape is wrong, you get a `ShapeMismatch` error — not
a silent wrong answer.

```zig
// A typical shape flow through one transformer block:
//   input:        (B, T, C)        e.g., (4, 32, 64)
//   after linear: (B, T, C)        matmul with (C, C) weight
//   after GELU:   (B, T, C)        same shape, elementwise
//   after linear: (B, T, C)        second matmul
//   after resid:  (B, T, C)        add input back (residual)
```

Our `Shape` struct with `init1D` through `init4D` constructors covers every
shape a 1-block 1-head transformer needs:
- `(C,)` — bias vectors
- `(C, C)` — weight matrices
- `(B, T, C)` — activations
- `(B, T, T)` — attention score matrices

### 2.2 zeros, ones, full → Weight and Bias Initialization

Every trainable parameter starts at some initial value before training:

```zig
// Bias vectors start at zero (common practice)
const bias = try create.zeros(allocator, Shape.init1D(64));

// LayerNorm weights start at one (scale=1 means "pass through")
const ln_weight = try create.ones(allocator, Shape.init1D(64));

// Sometimes you need a constant fill (e.g., attention mask = -inf)
const mask = try create.full(allocator, Shape.init2D(32, 32), -std.math.inf(f32));
```

**Why zeros for biases?** A zero bias means the initial output of a linear
layer depends only on the input and the weight matrix. This is neutral — it
doesn't bias the network in any direction before training starts.

### 2.3 randn → Kaiming/Xavier Initialization

Weight matrices are initialized from a normal distribution, not from zeros:

```zig
// Xavier init for a (64, 64) weight matrix
const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(64 + 64)));
const W = try create.randn(allocator, Shape.init2D(64, 64), &rng, 0.0, std_dev);
```

**Why not zero weights?** If all weights are zero, every neuron in a layer
produces the same output, receives the same gradient, and updates identically —
the network never learns diverse features. Random initialization "breaks
symmetry."

**Why normal (randn) instead of uniform (randu)?** Normal distributions have
thin tails — extreme values are rare. This matches the statistical assumptions
behind Xavier and Kaiming initialization, which set the variance so that
signal magnitude is preserved across layers. Uniform distributions have hard
cutoffs that can cause edge effects.

Our `randn` uses the Box-Muller transform internally (see `rng.zig`), which
produces mathematically exact standard normals from uniform random inputs.

### 2.4 fromSlice → Loading Training Data and Targets

Training data comes from outside the tensor system — token IDs from a
tokenizer, target labels from a dataset. `fromSlice` bridges the gap:

```zig
// Token IDs for one batch (B=4, T=8)
const token_ids = try create.fromSlice(allocator, Shape.init2D(4, 8), &.{
    1, 5, 3, 9, 2, 7, 0, 4,
    3, 1, 6, 8, 0, 2, 5, 9,
    7, 0, 4, 1, 3, 6, 8, 2,
    9, 5, 0, 3, 7, 1, 4, 6,
});

// Target class indices for cross-entropy loss
const targets = try create.fromSlice(allocator, Shape.init1D(4), &.{ 0.0, 1.0, 2.0, 3.0 });
```

In a real training loop, these slices come from the data pipeline (Stage 5),
which tokenizes text into integer IDs and windows them into input/target pairs.

### 2.5 matmul → Every Linear Layer, Every Attention Score

Matmul is the single most-used operation in a transformer. It appears in:

1. **Linear layers** — `output = input @ W^T + bias`
   The weight matrix transforms the input from one dimension to another.

2. **Attention scores** — `scores = Q @ K^T / sqrt(d_k)`
   The dot product between query and key vectors measures how much each
   position should attend to every other position.

3. **Attention output** — `context = scores_weighted @ V`
   The weighted sum of value vectors produces the attention output.

4. **Output projection** — `output = context @ W_o`
   Projects the attention output back to the model dimension.

For our 1-block 1-head transformer with `C=64`:

```zig
// Linear layer: (B, T, C) @ (C, C) → (B, T, C)
// We reshape to (B*T, C) for 2D matmul, then reshape back
const x_flat = try x.reshape(Shape.init2D(B * T, C));   // (128, 64)
const out_flat = try ops.matmul.matmul(allocator, x_flat, W); // (128, 64)
const out = try out_flat.reshape(Shape.init3D(B, T, C)); // (4, 32, 64)
```

The ikj loop order in our matmul (see `matmul.zig`) is 2-3× faster than the
textbook ijk order for matrices that fit in L2 cache (~64KB), because both
`B[k,j]` and `C[i,j]` are accessed contiguously in the inner loop.

### 2.6 add (broadcast) → Bias Addition and Residual Connections

**Bias addition:** Every linear layer adds a bias vector to its output. The
bias has shape `(C,)` while the output has shape `(B, T, C)`. Broadcasting
stretches the bias across the batch and time dimensions:

```zig
// out = matmul_result + bias
// (B*T, C) + (C,) → (B*T, C) via broadcasting
const biased = try ops.elementwise.add(allocator, matmul_out, bias);
```

**Residual connections:** The defining feature of a transformer block. After
attention and MLP, the original input is added back:

```zig
// Residual connection: output = sublayer(x) + x
const residual = try ops.elementwise.add(allocator, sublayer_out, x);
```

Without residual connections, deep networks suffer from vanishing gradients —
the signal from the loss becomes too weak by the time it reaches early layers.
Residual connections provide a "highway" for gradient flow: the gradient can
pass through the addition unchanged.

### 2.7 mul (elementwise) → Attention Masking and Gradient Scaling

**Attention masking:** In a causal transformer, position i should not attend
to positions j > i (future tokens). We create a mask that is 0 for future
positions and 1 for allowed positions, then multiply:

```zig
// Causal mask: 1.0 where attention is allowed, 0.0 where it's not
// After softmax, masked positions get probability ~0
const mask = try create.fromSlice(allocator, Shape.init2D(T, T), mask_data);
const masked_scores = try ops.elementwise.mul(allocator, raw_scores, mask);
```

**Gradient scaling:** In mixed-precision training (not in our library — we're
f32-only), gradients are sometimes scaled by a constant to prevent underflow.
The `mulScalar` function does this:

```zig
const scaled_grad = try ops.elementwise.mulScalar(allocator, grad, 0.1);
```

### 2.8 softmax → Attention Weights

Attention scores are converted to probabilities via softmax, independently for
each query position:

```zig
// scores shape: (B, T, T) — each row is attention from one position
// softmax operates along the last axis (T), normalizing each row
const attn_weights = try ops.softmax.softmax(allocator, scores);
```

**Why numerical stability matters here:** Attention scores can be very large
when a query strongly matches a key. Without max-subtraction, `exp(50)` is
`5.18 × 10^21`, which still fits in f32 but `exp(100)` overflows. In a
trained model with large logits, the stable version prevents NaN.

**After softmax**, each row sums to 1.0 — it's a probability distribution
over positions. The model "decides" how much to attend to each position.

### 2.9 logSoftmax + crossEntropy → Next-Token Prediction Loss

The loss function measures how wrong the model's predictions are. For language
models, this is cross-entropy between predicted log-probabilities and the true
next token:

```zig
// logits: (B*T, vocab_size) — raw scores for each token in vocabulary
// targets: (B*T,) — integer IDs of the correct next tokens
const loss = try ops.loss.crossEntropy(allocator, logits, targets);
// loss.data[0] is the mean cross-entropy over the batch
```

**Why cross-entropy?** It's the negative log-probability of the correct class:
```
loss = -log(P(target_class))
```

This has a beautiful property: when the model assigns high probability to the
correct class, loss is near 0. When it assigns low probability, loss is high.
The gradient of cross-entropy w.r.t. logits is simply:
```
∂L/∂logit[j] = softmax(logit)[j] - (1 if j==target else 0)
```

This is clean and fast to compute — no second-order terms. That's why every
major framework uses cross-entropy for classification.

**Why logSoftmax instead of log(softmax)?** When softmax values are very small
(the model is confident about a different class), `log(tiny_number)` amplifies
floating-point rounding errors. `logSoftmax` computes the result directly from
the sum, avoiding the unstable intermediate step. See `docs/02_tensors.md`
Section 8 for the full derivation.

### 2.10 sum, mean, max → Batch-Mean Loss and Layer Normalization

**Batch-mean loss:** `crossEntropy` already averages over the batch internally,
but if you compute per-sample losses and want the mean yourself:

```zig
const loss_sum = try ops.reduce.sumAll(allocator, per_sample_losses);
const mean_loss = loss_sum.data[0] / @as(f32, @floatFromInt(B));
```

**Layer normalization** (Stage 4) uses `mean` and variance along the channel
axis:

```zig
// x shape: (B, T, C)
// mean along axis 2 (C) → shape (B, T, 1)
const x_mean = try ops.reduce.mean(allocator, x, 2);

// variance = mean((x - x_mean)^2)
const diff = try ops.elementwise.sub(allocator, x, x_mean);  // broadcast
const diff_sq = try ops.elementwise.mul(allocator, diff, diff);
const x_var = try ops.reduce.mean(allocator, diff_sq, 2);    // (B, T, 1)
```

LayerNorm stabilizes training by normalizing activations to zero mean and
unit variance at each position. This prevents internal covariate shift —
where the distribution of activations changes as training progresses, making
later layers chase a moving target.

**Max** is used in `softmax` (finding the maximum for the stability trick)
and in `LayerNorm`'s numerically stable variance computation.

### 2.11 exp, log → Softmax Internals and Loss Computation

These are the building blocks that `softmax` and `logSoftmax` use internally.
You rarely call them directly in a training loop, but understanding them
explains why those functions work:

```zig
// softmax = exp(logits - max) / sum(exp(logits - max))
// log_softmax = logits - max - log(sum(exp(logits - max)))
```

`log` also appears in learning rate schedulers (logarithmic decay) and in
perplexity computation (perplexity = exp(cross_entropy)).

### 2.12 relu, geluExact → MLP Activation

The MLP in a transformer block has two linear layers with an activation in
between. GPT-2 uses GELU:

```zig
// MLP: Linear → GELU → Linear
const hidden = try ops.matmul.matmul(allocator, x, W1); // (B*T, 4*C)
const activated = try ops.unary.geluExact(allocator, hidden);
const output = try ops.matmul.matmul(allocator, activated, W2); // (B*T, C)
```

**Why GELU instead of ReLU?** GELU smoothly transitions between 0 (for
negative inputs) and x (for positive inputs), unlike ReLU which has a sharp
kink at 0. This smoothness gives slightly better performance in transformers.

```
ReLU:  max(0, x)         — sharp cutoff at 0
GELU:  0.5x(1+erf(x/√2)) — smooth approximation to ReLU
```

Our `geluExact` uses the Abramowitz & Stegun erf approximation (see
`unary.zig`), which is accurate to ~1.5×10⁻⁷ — well within f32 precision.

### 2.13 transpose2d → Weight Transposition and Attention

**Weight transposition for backprop:** In the forward pass, a linear layer
computes `y = x @ W`. In the backward pass (Stage 3), the gradient w.r.t.
the input is `∂L/∂x = ∂L/∂y @ W^T`. We need the transposed weight matrix:

```zig
const W_T = try W.transpose2d();  // (C_out, C_in) → (C_in, C_out)
const grad_x = try ops.matmul.matmul(allocator, grad_y, W_T);
```

**Query-Key dot product in attention:** The attention score between a query
and a key is their dot product. Keys are transposed so that matmul computes
all pairwise dot products at once:

```zig
// Q: (B, T, d_k), K: (B, T, d_k)
// K^T: (B, d_k, T)  — transpose the last two dims
// scores = Q @ K^T: (B, T, T) — every query attends to every key
const K_T = try K.transpose2d();
const scores = try ops.matmul.matmul(allocator, Q, K_T);
```

`transpose2d` returns a **view** (owned=false), so no data is copied. This
matters in backprop where we transpose many matrices — copying would double
memory usage.

### 2.14 reshape, view → Flattening for Linear Layers and Head Reshaping

**Flattening for linear layers:** Matmul requires 2D inputs, but activations
are 3D. We reshape `(B, T, C)` → `(B*T, C)`, do the matmul, then reshape back:

```zig
const x_flat = try x.reshape(Shape.init2D(B * T, C));       // view, no copy
const out_flat = try ops.matmul.matmul(allocator, x_flat, W);
const out = try out_flat.reshape(Shape.init3D(B, T, C));    // view, no copy
```

Both reshapes are zero-copy because the data is contiguous. This is why
contiguity matters — non-contiguous tensors would require a copy, which is
expensive for large activations.

**Head reshaping** (when we generalize to multi-head in Stage 8):
`(B, T, C)` → `(B, T, n_head, d_head)` → transpose → compute attention per
head → transpose back → reshape to `(B, T, C)`.

### 2.15 isContiguous → GPU Transfer Readiness (Stage 7 Preview)

When we copy tensors to the GPU in Stage 7, we need a contiguous block of
memory. Non-contiguous tensors (e.g., after transpose) must be made
contiguous first:

```zig
if (!tensor.isContiguous()) {
    // Must copy to a new contiguous tensor before GPU transfer
    const contiguous = try Tensor.init(allocator, tensor.shape);
    try tensor.copyTo(allocator, &contiguous);
    // Now transfer contiguous to GPU
}
```

This is why `isContiguous` exists as a check — it tells you whether an
operation that requires contiguous data (reshape, GPU transfer, memcpy) can
proceed without a copy.

---

## 3. Mini Forward Pass Trace

Let's trace one forward pass through our 1-block 1-head transformer with
concrete shapes. We'll use `B=4` (batch size), `T=8` (sequence length),
`C=64` (embedding dimension), `vocab=100` (vocabulary size).

### 3.1 Token Embedding

```zig
// Input: token IDs from the tokenizer
// token_ids: shape (4, 8), values in [0, 99]
const token_ids = try create.fromSlice(allocator, Shape.init2D(4, 8), id_data);

// Embedding table: (vocab, C) = (100, 64)
// Each row is the vector representation of one token
const E = try create.randn(allocator, Shape.init2D(100, 64), &rng, 0.0, 0.02);

// Lookup: for each token ID, select the corresponding row from E
// This is a "fancy indexing" operation (Stage 4's Embedding layer)
// Result: (B, T, C) = (4, 8, 64)
// x[b, t, :] = E[token_ids[b, t], :]
```

**Stage 2 role:** `fromSlice` loads the token IDs. `randn` initializes the
embedding table. The actual lookup is an indexed gather (implemented in
Stage 4's `embedding.zig`).

### 3.2 First Linear Layer (QKV Projection)

In a self-attention block, we project the input into query, key, and value:
three separate linear transformations.

```zig
// Input x: (B*T, C) = (32, 64)  [reshaped from (4, 8, 64)]
// Weight W_q: (C, C) = (64, 64)
const W_q = try create.randn(allocator, Shape.init2D(64, 64), &rng, 0.0, 0.125);
const b_q = try create.zeros(allocator, Shape.init1D(64));

// Q = x @ W_q + b_q
const x_flat = try x.reshape(Shape.init2D(32, 64));
const Q_flat = try ops.matmul.matmul(allocator, x_flat, W_q);  // (32, 64)
const Q_biased = try ops.elementwise.add(allocator, Q_flat, b_q); // broadcast (64,) → (32, 64)
const Q = try Q_biased.reshape(Shape.init3D(4, 8, 64));        // (4, 8, 64)
```

Same pattern for K and V. **Three matmuls + three bias additions just for
the QKV projection.** That's 6 of our Stage 2 operations already.

### 3.3 Attention Scores

```zig
// Q: (4, 8, 64), K: (4, 8, 64)
// We need Q @ K^T for each batch element

// Reshape for batched matmul: (B, T, C) → keep B as batch
// Q: (4, 8, 64), K^T: (4, 64, 8)
const K_T = try ops.matmul.transpose2d(K_2d);  // (64, 8)
// In batched form: scores = Q @ K^T → (4, 8, 8)

// Scale by 1/sqrt(d_k) to prevent dot products from growing too large
const scale: f32 = 1.0 / @sqrt(@as(f32, 64.0));
const scores_scaled = try ops.elementwise.mulScalar(allocator, scores, scale);
```

**Why scale by 1/√d_k?** Dot products of d_k-dimensional vectors grow
proportionally to √d_k. Without scaling, large d_k means large logits,
which makes softmax extremely peaked (approaching a one-hot distribution).
Scaling keeps the variance of the logits at ~1 regardless of d_k, so
softmax produces meaningful probability distributions.

### 3.4 Attention Weights (Softmax)

```zig
// scores: (4, 8, 8) — raw attention scores
// Apply causal mask: future positions get -inf (cannot attend to future)
// Then softmax normalizes each row to sum to 1.0

// With mask applied:
const attn_weights = try ops.softmax.softmax(allocator, masked_scores);
// attn_weights: (4, 8, 8) — each row sums to 1.0
```

**Why this is critical:** If softmax receives a row with a very large value
(e.g., `[100, 1, 1, 1, ...]`), the result is essentially `[1, 0, 0, 0, ...]`
— the model attends exclusively to one position. The max-subtraction trick
ensures this computation is stable even with extreme scores.

### 3.5 Attention Output

```zig
// Weighted sum of values: attn_weights @ V
// attn_weights: (4, 8, 8), V: (4, 8, 64)
// Result: (4, 8, 64) — each position is a weighted average of values
const attn_out = try ops.matmul.matmulBatch(allocator, attn_weights, V);
```

Each output position is a mixture of all value vectors, weighted by how much
the corresponding query "attends" to each key.

### 3.6 Residual Connection

```zig
// Add the original input back: output = attn_out + x
// Both have shape (4, 8, 64)
const residual = try ops.elementwise.add(allocator, attn_out, x);
```

This is the "residual" or "skip connection." It allows gradients to flow
directly from the loss to earlier layers, preventing vanishing gradients.

### 3.7 MLP: Linear → GELU → Linear

```zig
// First linear: expand from C to 4*C (standard transformer ratio)
const W1 = try create.randn(allocator, Shape.init2D(64, 256), &rng, 0.0, 0.0625);
const b1 = try create.zeros(allocator, Shape.init1D(256));
const hidden = try ops.matmul.matmul(allocator, residual_flat, W1); // (32, 256)
const hidden_biased = try ops.elementwise.add(allocator, hidden, b1);

// GELU activation
const activated = try ops.unary.geluExact(allocator, hidden_biased);

// Second linear: project back from 4*C to C
const W2 = try create.randn(allocator, Shape.init2D(256, 64), &rng, 0.0, 0.0625);
const b2 = try create.zeros(allocator, Shape.init1D(64));
const mlp_out = try ops.matmul.matmul(allocator, activated, W2);  // (32, 64)
const mlp_biased = try ops.elementwise.add(allocator, mlp_out, b2);
```

**Two more matmuls + two bias additions + one GELU** for the MLP. That's
5 operations, bringing our total Stage 2 op count for one block to:

- QKV projection: 3 matmuls + 3 adds = 6
- Attention scores: 1 matmul + 1 mulScalar = 2
- Attention output: 1 matmul = 1
- Residual: 1 add = 1
- MLP: 2 matmuls + 2 adds + 1 GELU = 5
- **Total: 15 Stage 2 ops for one transformer block**

### 3.8 Output Projection and Loss

```zig
// Project to vocabulary size: (B*T, C) @ (C, vocab) → (B*T, vocab)
const W_out = try create.randn(allocator, Shape.init2D(64, 100), &rng, 0.0, 0.125);
const logits = try ops.matmul.matmul(allocator, block_out_flat, W_out); // (32, 100)

// Reshape logits to (B*T, vocab) and targets to (B*T,)
// Cross-entropy loss: measures how well logits predict the next token
const targets = try create.fromSlice(allocator, Shape.init1D(32), target_data);
const loss = try ops.loss.crossEntropy(allocator, logits, targets);
// loss.data[0] is a scalar: the mean -log(P(correct_token))
```

**One more matmul + one crossEntropy** (which internally uses logSoftmax).

### 3.9 Full Shape Summary

```
Token IDs          (4, 8)         fromSlice
  ↓ Embedding lookup
Embedded x         (4, 8, 64)     randn (for E)
  ↓ reshape
x_flat             (32, 64)       reshape (view)
  ↓ @ W_q + b_q
Q                  (4, 8, 64)     matmul + add
  ↓ (same for K, V)
  ↓ Q @ K^T / √d_k
Scores             (4, 8, 8)      matmul + mulScalar
  ↓ causal mask + softmax
Attn weights       (4, 8, 8)      softmax
  ↓ @ V
Attn output        (4, 8, 64)     matmulBatch
  ↓ + x (residual)
Residual           (4, 8, 64)     add
  ↓ @ W1 + b1 → GELU → @ W2 + b2
MLP output         (4, 8, 64)     matmul×2 + add×2 + geluExact
  ↓ + residual (another residual)
Block output       (4, 8, 64)     add
  ↓ reshape → @ W_out
Logits             (32, 100)      matmul
  ↓ crossEntropy
Loss               (1,)           crossEntropy (uses logSoftmax internally)
```

**Stage 2 operation count:**
- `matmul`: 6 calls (QKV×3, scores, attn_out, W_out) + 2 (MLP) = 8
- `add` (broadcast): 6 calls (QKV bias×3, MLP bias×2, residual×2) = 7
- `softmax`: 1 call (attention weights)
- `crossEntropy`: 1 call (loss, internally uses logSoftmax)
- `geluExact`: 1 call (MLP activation)
- `mulScalar`: 1 call (attention scaling)
- `reshape`: ~4 calls (flatten/unflatten for matmul)
- `randn`: ~8 calls (weight init)
- `zeros`: ~7 calls (bias init)
- **Total: ~36 Stage 2 function calls for one forward pass**

And this is a *tiny* model. A 12-layer GPT-2 would multiply by 12.

---

## 4. Why Design Decisions Matter for Training

### 4.1 Row-Major → Cache-Friendly Matmul

Matmul is O(M×K×N) — the dominant cost in a transformer. Our ikj loop order
is 2-3× faster than ijk because it accesses both B and C contiguously in the
inner loop. This is only possible because data is in row-major order.

**Impact:** For our (32, 64) @ (64, 64) matmul, ikj processes ~131K elements.
A cache miss costs ~100 cycles; a cache hit costs ~4 cycles. The difference
between 2-3× fewer cache misses translates to real training speedup.

### 4.2 Broadcasting → No Unnecessary Bias Copies

Without broadcasting, adding a bias of shape (C,) to an output of shape
(B*T, C) would require expanding the bias to a (B*T, C) tensor — copying
C values B*T times. For B*T=32 and C=64, that's 2048 extra elements per
bias addition, and we do 7 bias additions per forward pass. Total: ~14K
extra f32 values copied per forward pass.

With broadcasting, we add the bias "virtually" — no copy, no extra memory.
The `broadcastIndex` function maps each output position to the correct bias
element at runtime.

### 4.3 Max-Subtraction in Softmax → No NaN in Attention

Attention scores can be very large (see Section 3.4). Without
max-subtraction, `exp(large_score)` overflows to Inf, and `Inf / Inf = NaN`.
Once NaN appears, it propagates through every subsequent operation — the
entire forward pass produces NaN, and training collapses.

The max-subtraction trick costs one extra pass over each row (to find the
max) but guarantees every exponent is ≤ 0, so no overflow. This is not
optional — every production deep learning framework uses this technique.

### 4.4 log-softmax in Cross-Entropy → No log(0)

When the model confidently predicts the wrong class, the correct class's
softmax probability can underflow to 0.0 in f32. `log(0.0) = -Inf`, and
`-(-Inf) = +Inf` — the loss becomes infinite, and the gradient is undefined.

`logSoftmax` avoids this by computing the log directly from the sum:
```
log_softmax[j] = x[j] - max - log(sum(exp(x - max)))
```

The sum is always ≥ 1 (because `exp(0) = 1` is always one term), so
`log(sum)` is always ≥ 0, and the result is finite. This is why
`crossEntropy` uses `logSoftmax` internally instead of `log(softmax(...))`.

### 4.5 ikj Matmul Order → 2-3× Speedup for Our Sizes

For our typical matmul size (32×64) @ (64×64), the total elements touched
is ~131K × 2 (reads of A and B) + ~2K (writes to C) ≈ 264K. With L2 cache
of 256KB on a modern CPU, ijk order would thrash (B's columns don't fit),
while ikj order keeps B's row and C's row in cache.

**Rough numbers for our sizes:**
- ijk: ~50 cache misses per row of B → ~50 × 64 × 32 = ~100K stalls
- ikj: ~2 cache misses per row of B → ~2 × 64 × 32 = ~4K stalls
- Speedup: ~25× fewer cache stalls

Even though the total FLOPs are identical, the memory access pattern makes
the difference.

### 4.6 Owned vs View → Memory Savings in Backprop

`transpose2d` returns a view (owned=false). In backprop, we need transposed
weights for computing `∂L/∂x = ∂L/∂y @ W^T`. If transpose created a copy,
we'd allocate a second (C, C) weight matrix for every layer — that's 64×64×4
= 16KB per layer. For 12 layers (GPT-2 small), that's 192KB of extra memory
per training step.

By returning a view, we use zero extra memory for the transpose. The data
is shared with the original weight matrix. This matters even more for large
models: GPT-2 medium has (1024, 4096) weight matrices — a copy would be
16MB per layer.

---

## 5. The Training Loop Skeleton

Here's what the complete training loop will look like once all stages are
implemented. Stage 2 operations are marked **[S2]**; future stages are
marked with their stage number.

```
for each training step:

    # --- 1. Get a batch of data ---
    # [S5: tokenizer + data pipeline]
    inputs, targets = data_batcher.next()

    # --- 2. Forward pass ---
    # [S2: tensor operations throughout]
    x = embed(inputs)                    # [S4: nn.Embedding]
    x = transformer_block(x)            # [S4: nn.TransformerBlock]
    logits = output_projection(x)        # [S4: nn.Linear]

    # --- 3. Compute loss ---
    # [S2: crossEntropy uses logSoftmax internally]
    loss = crossEntropy(logits, targets)  # [S2]

    # --- 4. Backward pass ---
    # [S3: autograd tape replays forward in reverse]
    loss.backward()                       # [S3]
    # Now every parameter W has W.grad filled in

    # --- 5. Update weights ---
    # [S4: optimizer uses gradients to update parameters]
    for param in model.parameters():      # [S4: Module protocol]
        optimizer.step(param)             # [S4: SGD or AdamW]
        param.grad = 0                   # [S2: fill with 0.0]

    # --- 6. Log and checkpoint ---
    print(f"step {i}: loss = {loss:.4f}") # [S2: debugSummary]
```

**What Stage 2 gives you right now:**
- All the tensor operations in step 2 (matmul, add, softmax, gelu, etc.)
- The loss function in step 3 (crossEntropy)
- Debug printing for step 6 (debugSummary, printValues)

**What's coming:**
- Step 1: Stage 5 (tokenizer + data pipeline)
- Step 2's nn layers: Stage 4 (nn.Linear, nn.Embedding, etc.)
- Step 4: Stage 3 (autograd tape — the `loss.backward()` magic)
- Step 5: Stage 4 (optimizers — SGD, AdamW)

### The Training Loop in Our API (Preview)

Once all stages are done, actual code will look like:

```zig
// Stage 6: lab/train.zig
var model = try TransformerBlock.init(allocator, &rng, .{ .d_model = 64, .vocab = 100 });
var optimizer = try AdamW.init(allocator, model.parameters(), .{ .lr = 3e-4 });

for (0..num_steps) |step| {
    const batch = try batcher.next(allocator);
    const logits = try model.forward(allocator, batch.inputs);
    const loss = try ops.loss.crossEntropy(allocator, logits, batch.targets);

    try tape.backward(allocator, loss);
    try optimizer.step(allocator);
    try optimizer.zeroGrad(allocator);

    if (step % 100 == 0) {
        try tensor_print.debugSummary(loss, writer);
    }
}
```

Every function in that loop is built on Stage 2 primitives:
- `model.forward` calls `matmul`, `add`, `softmax`, `geluExact`, etc.
- `crossEntropy` calls `logSoftmax` internally
- `tape.backward` (Stage 3) calls `matmul` with transposed weights, `mul`,
  `sub`, etc. — the same Stage 2 ops, but in reverse order
- `optimizer.step` calls `addInPlace`, `mulScalar`, `mul` — Stage 2 ops
  applied to weight tensors and their gradients

---

## 6. PyTorch Equivalents

If you've seen PyTorch before, here's how our Stage 2 API maps to `torch`:

| Our Library | PyTorch Equivalent | Notes |
|-------------|--------------------|-------|
| `Shape.init2D(2, 3)` | `torch.Size([2, 3])` | No slice constructor |
| `create.zeros(alloc, shape)` | `torch.zeros(2, 3)` | Explicit allocator |
| `create.ones(alloc, shape)` | `torch.ones(2, 3)` | |
| `create.full(alloc, shape, val)` | `torch.full((2,3), val)` | |
| `create.randn(alloc, shape, &rng, μ, σ)` | `torch.randn(2,3) * σ + μ` | Explicit RNG |
| `create.arange(alloc, start, stop, step)` | `torch.arange(start, stop, step)` | |
| `create.fromSlice(alloc, shape, data)` | `torch.tensor(data)` | |
| `elementwise.add(alloc, a, b)` | `a + b` | No operator overloading |
| `elementwise.sub(alloc, a, b)` | `a - b` | |
| `elementwise.mul(alloc, a, b)` | `a * b` | Elementwise, not matmul |
| `elementwise.div(alloc, a, b)` | `a / b` | |
| `elementwise.addScalar(alloc, a, s)` | `a + s` | |
| `elementwise.mulScalar(alloc, a, s)` | `a * s` | |
| `reduce.sum(alloc, t, axis)` | `t.sum(dim=axis, keepdim=True)` | Axis always kept |
| `reduce.mean(alloc, t, axis)` | `t.mean(dim=axis, keepdim=True)` | |
| `reduce.max(alloc, t, axis)` | `t.max(dim=axis, keepdim=True)` | |
| `reduce.sumAll(alloc, t)` | `t.sum()` | Returns shape (1,) |
| `matmul.matmul(alloc, a, b)` | `a @ b` or `torch.mm(a, b)` | 2D only |
| `matmul.matmulBatch(alloc, a, b)` | `a @ b` or `torch.bmm(a, b)` | 3D only |
| `matmul.transpose2d(t)` | `t.T` or `t.transpose(0, 1)` | Returns view |
| `unary.exp(alloc, t)` | `torch.exp(t)` | |
| `unary.log(alloc, t)` | `torch.log(t)` | Natural log |
| `unary.relu(alloc, t)` | `torch.relu(t)` or `F.relu(t)` | |
| `unary.geluExact(alloc, t)` | `F.gelu(t, approximate='none')` | |
| `softmax.softmax(alloc, t)` | `F.softmax(t, dim=-1)` | Last axis only |
| `softmax.logSoftmax(alloc, t)` | `F.log_softmax(t, dim=-1)` | |
| `loss.crossEntropy(alloc, logits, targets)` | `F.cross_entropy(logits, targets)` | Fused log_softmax + NLL |
| `tensor.reshape(new_shape)` | `t.view(3, 2)` or `t.reshape(3, 2)` | Returns view |
| `tensor.view()` | `t.detach()` (approximate) | Shares data |
| `tensor.isContiguous()` | `t.is_contiguous()` | |
| `tensor_print.debugSummary(t, w)` | `print(t.shape, t.min(), t.max())` | Our version is richer |
| `tensor_print.printValues(t, w, n)` | `print(t[:n])` | |

### Key Differences

1. **No operator overloading.** PyTorch lets you write `c = a + b`. We write
   `c = try elementwise.add(allocator, a, b)`. This is more verbose but makes
   every operation explicit — no hidden allocations, no hidden kernel launches.

2. **Explicit allocator.** Every allocation function takes `allocator` as the
   first parameter. PyTorch uses a global allocator. This matters for:
   - Arena allocation (free everything at once — great for forward passes)
   - Leak detection (GPA detects leaks — essential for testing)
   - Future CUDA allocator (different allocator for device memory)

3. **Explicit RNG.** Every random function takes `&rng`. PyTorch uses a global
   RNG. This makes our training deterministic — the same seed produces the
   exact same weight initialization and data ordering.

4. **No autograd yet.** PyTorch tensors track gradients automatically. Our
   tensors have `requires_grad` and `grad` fields (declared but unused), and
   the autograd tape comes in Stage 3.

5. **Error handling vs exceptions.** PyTorch throws Python exceptions. We
   return error unions (`!Tensor`). Every fallible call uses `try` or `catch`.
   No surprises — if a function can fail, you see it at the call site.

---

## 7. What to Read Next

- **Stage 3 (`docs/03_autograd.md`):** How `loss.backward()` actually works —
  the tape records forward ops, then replays them in reverse to compute
  gradients. Every Stage 2 operation gets a corresponding backward function.

- **Stage 4 (`docs/04_nn.md`):** How `nn.Linear`, `nn.Embedding`, and
  `nn.LayerNorm` compose Stage 2 ops into reusable layers. How `AdamW`
  updates weights using gradients from Stage 3.

- **Stage 6 (`docs/07_cpu_training.md`):** The complete training loop —
  all the pieces from Stages 2-5 assembled into a working program that
  trains a transformer and generates text.
