# Chapter 05: Transformer Math — A Complete Shape Trace

## What This Chapter Covers

This chapter walks through **every single tensor operation** in one forward pass
of our tiny transformer. We trace the **shape** of every intermediate tensor,
show the **math** behind each operation, and explain **why** each step exists.

By the end you'll understand:
- How a sequence of token IDs becomes a sequence of probability distributions
- Why each layer preserves or changes the tensor shape
- How many floating-point numbers live in memory at each stage
- What "shape mismatch" errors really mean (and how to prevent them)

### Concrete Configuration

Throughout this chapter we use these fixed dimensions:

| Symbol | Meaning | Value |
|--------|---------|-------|
| B | Batch size | 2 |
| T | Sequence length | 8 |
| D | Model dimension (d_model) | 32 |
| V | Vocabulary size | 64 |
| d_ff | Feed-forward hidden dim | 128 (= 4 × D) |

These match the model in `src/nn/model.zig` and example 04.

---

## 1. Why Shape Tracing Matters

If you've ever seen a PyTorch error like:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x8x32 and 64x32)
```

...you know the pain. Shape errors are the #1 debugging headache in deep learning.
They happen because every operation expects its inputs in a very specific layout,
and one wrong transpose or one-off dimension can cascade through dozens of layers.

**Shape tracing** is the practice of writing down the exact dimensions at every
step. It's tedious for real models (GPT-3 has 96 layers), but for our tiny
1-block model it's perfectly feasible — and deeply illuminating.

The pattern:
1. Write down the input shape
2. Apply the operation's rule
3. Write down the output shape
4. Check: does the next layer accept this shape?

If every shape lines up, your forward pass will run without crashes. If there's
a mismatch, you'll know exactly which operation is wrong.

### Notation Convention

```
(B, T, D)  ←  a rank-3 tensor with dims[0]=B, dims[1]=T, dims[2]=D

W: (D_out, D_in)  ←  a weight matrix stored row-major

x @ W^T  ←  matrix multiply x with the transpose of W
```

We always write shapes left-to-right as `(dim0, dim1, dim2, ...)`. In row-major
storage (C order), the rightmost dimension is contiguous in memory.

---

## 2. Input: Token IDs

Everything starts with a batch of integer sequences. Each integer is a **token ID**
— an index into the vocabulary.

```
Input tensor shape: (B, T) = (2, 8)

Example data:
  Batch 0: [5, 12, 3, 45, 7, 22, 9, 1]   ← 8 tokens
  Batch 1: [8, 33, 17, 2, 50, 11, 6, 41]  ← 8 tokens
```

This is a 2D integer tensor. No gradients flow through it — it's the **ground
truth input**, not a learned parameter.

```
        Token IDs (B=2, T=8)
        ┌─────────────────────────────────┐
  B=0:  │  5   12   3   45   7   22   9   1 │
  B=1:  │  8   33  17    2  50   11   6  41 │
        └─────────────────────────────────┘
```

**Memory:** 2 × 8 = 16 integers (i32) = 64 bytes. Tiny.

---

## 3. Token Embedding Lookup

The token embedding is a **lookup table** — a matrix where row `i` holds the
D-dimensional vector for word `i`.

### Weight Shape

```
tok_embed.weight: (V, D) = (64, 32)
```

Row 0 is the embedding for word 0, row 1 for word 1, ..., row 63 for word 63.
Each row is a learned 32-dimensional vector.

### Operation: Gather Rows

For each token ID, we **copy** the corresponding row out of the weight matrix.

```
input:  token_ids  (B, T) = (2, 8)     ← integer indices
weight: tok_embed  (V, D) = (64, 32)   ← lookup table
output: tok_embeds (B, T, D) = (2, 8, 32)
```

The operation is `output[b][t] = weight[token_ids[b][t]]` — a simple array index.

```
  token_ids[0][0] = 5  →  copy weight[5, :]  →  output[0, 0, :]  (32 floats)
  token_ids[0][1] = 12 →  copy weight[12, :] →  output[0, 1, :]  (32 floats)
  ...
  token_ids[1][7] = 41 →  copy weight[41, :] →  output[1, 7, :]  (32 floats)
```

### Memory After This Step

```
tok_embeds: 2 × 8 × 32 = 512 floats = 2048 bytes
```

### Why Not One-Hot + Matmul?

Mathematically, embedding lookup is identical to:
1. One-hot encode the token ID → (V,) vector with a 1 at position `i`
2. Multiply by the weight matrix → (V,) × (V, D) = (D,)

The lookup is just the **faster** version — you skip the V-sized one-hot vector
and go straight to the D-sized row. Same result, O(D) instead of O(V×D).

In our code (`src/nn/embedding.zig`), the forward pass does:
```zig
// For each (b, t), copy weight[token_id] into output[b, t, :]
for (output_data, 0..) |*val, flat_idx| {
    const b = flat_idx / (T * D);
    const t = (flat_idx / D) % T;
    const d = flat_idx % D;
    const vocab_idx = token_ids_data[b * T + t];
    val.* = weight_data[vocab_idx * D + d];
}
```

---

## 4. Position Embedding Lookup

The position embedding tells the model **where** each token sits in the sequence.
Without it, "the cat sat" and "sat cat the" would look identical to the attention
mechanism — same set of vectors, different order.

### Weight Shape

```
pos_embed.weight: (T, D) = (8, 32)    ← one row per position
```

Wait — our code actually stores this as `(1, T, D)` because the Embedding layer
wraps a 2D weight of shape `(T, D)`, and when we do a forward pass with a
`(1, T)` input of positions `[0, 1, ..., T-1]`, we get output shape `(1, T, D)`.

### Operation

```
input:  positions  (1, T) = (1, 8)     ← [0, 1, 2, 3, 4, 5, 6, 7]
weight: pos_embed  (T, D) = (8, 32)    ← lookup table
output: pos_embeds (1, T, D) = (1, 8, 32)
```

Position 0 → row 0 of the weight matrix, position 1 → row 1, etc.

```
  positions[0][0] = 0 → weight[0, :] → output[0, 0, :]  (32 floats)
  positions[0][1] = 1 → weight[1, :] → output[0, 1, :]  (32 floats)
  ...
  positions[0][7] = 7 → weight[7, :] → output[0, 7, :]  (32 floats)
```

### Memory After This Step

```
pos_embeds: 1 × 8 × 32 = 256 floats = 1024 bytes
```

---

## 5. Embedding Addition: Broadcast Over Batch

Now we add the token embeddings and position embeddings:

```
tok_embeds:  (B, T, D) = (2, 8, 32)
pos_embeds:  (1, T, D) = (1, 8, 32)
───────────────────────────────────────
x:          (B, T, D) = (2, 8, 32)    ← output of addition
```

### Broadcasting

The position embeddings have shape `(1, 8, 32)` while the token embeddings have
shape `(2, 8, 32)`. The `1` in the batch dimension **broadcasts** — the same
position vectors are added to every sample in the batch.

```
  x[0, t, d] = tok_embeds[0, t, d] + pos_embeds[0, t, d]
  x[1, t, d] = tok_embeds[1, t, d] + pos_embeds[0, t, d]  ← same pos_embeds!
       ↑                                   ↑
  batch 0                           batch 0 of pos (the only one)
```

Visually:

```
  tok_embeds (2, 8, 32)      pos_embeds (1, 8, 32)       x (2, 8, 32)
  ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
  │ batch 0: 8×32    │  +    │ pos 0-7: 8×32    │  =    │ batch 0: 8×32    │
  │ batch 1: 8×32    │  +    │ (same 8×32 used   │  =    │ batch 1: 8×32    │
  └──────────────────┘       │  for both batches) │       └──────────────────┘
                             └──────────────────┘
```

**Why shared position embeddings?** Position 3 means the same thing regardless
of which batch sample it's in. The model learns "position 3 tends to have
certain syntactic roles" (e.g., after a determiner in English).

### Memory After This Step

```
x: 2 × 8 × 32 = 512 floats = 2048 bytes  (newly allocated)
```

---

## 6. Transformer Block: The Heart of the Model

Our model has exactly **one** transformer block. Here's the full architecture:

```
        x (B, T, D) = (2, 8, 32)
              │
              ▼
    ┌───── LayerNorm 1 ──────┐
    │   Welford algorithm     │
    │   (B,T,D) → (B,T,D)    │
    └────────────────────────┘
              │
              ▼
    ┌── CausalSelfAttention ──┐
    │  W_q: (D,D) Linear     │
    │  W_k: (D,D) Linear     │
    │  W_v: (D,D) Linear     │
    │  Q @ K^T / sqrt(D)     │
    │  Causal mask + softmax │
    │  Attn @ V              │
    │  W_o: (D,D) Linear     │
    │  (B,T,D) → (B,T,D)     │
    └────────────────────────┘
              │
              ▼
         x + attn_out          ← residual connection
              │
              ▼
    ┌───── LayerNorm 2 ──────┐
    │   (B,T,D) → (B,T,D)    │
    └────────────────────────┘
              │
              ▼
    ┌──────── MLP ───────────┐
    │  fc1: (D, 4D) Linear   │
    │  GELU activation        │
    │  fc2: (4D, D) Linear    │
    │  (B,T,D) → (B,T,D)     │
    └────────────────────────┘
              │
              ▼
         h + mlp_out           ← residual connection
              │
              ▼
        out (B, T, D) = (2, 8, 32)
```

We'll now trace through each operation in detail.

---

## 6.1 LayerNorm 1 (Pre-Norm)

LayerNorm normalizes each token's D-dimensional vector independently. It computes
the mean and variance **across the D dimension**, then rescales to zero mean and
unit variance, and finally applies learned scale (gamma) and shift (beta).

### Welford Algorithm

For a single token vector of length D=32:

```
Step 1: Compute mean
  μ = (1/D) × Σ x[d]           for d = 0..31

Step 2: Compute variance (Welford's online algorithm — numerically stable)
  M₂ = Σ (x[d] - μ)²          for d = 0..31
  σ² = M₂ / D

Step 3: Normalize
  x_hat[d] = (x[d] - μ) / sqrt(σ² + ε)     ε = 1e-5

Step 4: Scale and shift (learned parameters)
  y[d] = γ[d] × x_hat[d] + β[d]
```

### Shape Trace

```
Input x:     (B, T, D) = (2, 8, 32)
μ:           (B, T, 1)  = (2, 8, 1)    ← one mean per token
σ²:          (B, T, 1)  = (2, 8, 1)    ← one variance per token
x_hat:       (B, T, D) = (2, 8, 32)   ← normalized
γ (gamma):   (D,)       = (32,)         ← learned scale, broadcast to (1, 1, 32)
β (beta):    (D,)       = (32,)         ← learned shift, broadcast to (1, 1, 32)
Output:      (B, T, D) = (2, 8, 32)    ← same shape as input
```

### Why Per-Token, Not Per-Batch?

BatchNorm normalizes across the batch dimension. LayerNorm normalizes across
the feature dimension. For language models:

- Different sequences in a batch may have very different statistics
- At inference time you process one sequence, so batch statistics are meaningless
- LayerNorm works the same regardless of batch size → no train/test discrepancy

### Implementation in Our Code

In `src/nn/layernorm.zig`, LayerNorm is **composed** from tape-tracked primitives
rather than implemented as a single fused op. The sequence:

```
x → mean → sub(x, mean) → square → mean → sqrt → div → mul(gamma) → add(beta)
```

Each of these operations is individually recorded on the autograd tape, so
gradients flow correctly through every step. This is pedagogically clearer than
a monolithic LayerNorm backward (which is what PyTorch does for performance).

### Intermediate Tensors Created by LayerNorm (composed version)

```
x_mean:     (B, T, 1)  = (2, 8, 1)     ← 16 floats
x_centered: (B, T, D)  = (2, 8, 32)   ← 512 floats
x_sq:       (B, T, D)  = (2, 8, 32)   ← 512 floats
var:        (B, T, 1)  = (2, 8, 1)     ← 16 floats
std:        (B, T, 1)  = (2, 8, 1)     ← 16 floats
x_hat:      (B, T, D)  = (2, 8, 32)   ← 512 floats
scaled:     (B, T, D)  = (2, 8, 32)   ← 512 floats
ln1_out:    (B, T, D)  = (2, 8, 32)   ← 512 floats  (final output)
```

Total intermediate memory: 512 × 5 + 16 × 4 = 2624 floats ≈ 10.5 KB

All intermediates must be kept alive via `tape.keepAlive()` so backward can read
their data. Without keepAlive, the defer-freed intermediates would be
use-after-free when backward runs.

---

## 6.2 QKV Projections

The attention mechanism projects the input into three vectors per token: a **Query**,
a **Key**, and a **Value**. Each projection is a linear (fully connected) layer.

### Linear Layer Math

A linear layer with weight W: (D_out, D_in) computes:

```
output = input @ W^T
```

For input shape (B, T, D_in), the matmul happens on the last two dimensions:

```
(B, T, D_in) @ (D_in, D_out) → (B, T, D_out)
```

But wait — in our implementation, we **reshape** the 3D input to 2D first:

```
Input:  (B, T, D_in) = (2, 8, 32)
Reshape to 2D: (B*T, D_in) = (16, 32)
Matmul: (16, 32) @ (D_out, D_in)^T = (16, 32) @ (32, D_out) = (16, D_out)
Reshape back: (B, T, D_out) = (2, 8, D_out)
```

This reshaping trick lets us reuse the 2D matmul kernel. The data layout is
unchanged — we just reinterpret the shape.

### Q Projection

```
W_q weight:  (D, D) = (32, 32)       ← stored row-major
Input:       (B, T, D) = (2, 8, 32)  ← ln1_out
Reshape:     (16, 32)
Q = input @ W_q^T = (16, 32) @ (32, 32) = (16, 32)
Reshape:     (B, T, D) = (2, 8, 32)
```

### K Projection

```
W_k weight:  (D, D) = (32, 32)
Input:       (B, T, D) = (2, 8, 32)  ← same ln1_out
K = input @ W_k^T = (16, 32) @ (32, 32) = (16, 32)
Reshape:     (B, T, D) = (2, 8, 32)
```

### V Projection

```
W_v weight:  (D, D) = (32, 32)
Input:       (B, T, D) = (2, 8, 32)  ← same ln1_out
V = input @ W_v^T = (16, 32) @ (32, 32) = (16, 32)
Reshape:     (B, T, D) = (2, 8, 32)
```

### Intuition: What Are Q, K, V?

Think of a library:

- **Query (Q):** "I'm looking for information about X" — what this token wants
- **Key (K):** "I contain information about Y" — what this token offers
- **Value (V):** The actual content — what this token contributes if selected

The dot product Q·K measures "how much does what I want match what you offer."
High score → you're relevant → I attend to you → I incorporate your Value.

---

## 6.3 Attention Scores: Q @ K^T

Now we compute how much each token "attends to" every other token.

### The Key Transpose

K has shape (B, T, D) = (2, 8, 32). We transpose the last two dimensions:

```
K:  (B, T, D) = (2, 8, 32)
K^T: (B, D, T) = (2, 32, 8)     ← swap dims[1] and dims[2]
```

This is `transposeInner2d` in our code — it swaps dims[1] and dims[2] of a 3D
tensor, leaving the batch dimension (dims[0]) alone.

### Batched Matmul

```
Q:   (B, T, D)  = (2, 8, 32)
K^T: (B, D, T)  = (2, 32, 8)
──────────────────────────────
scores = Q @ K^T: (B, T, T) = (2, 8, 8)
```

For each batch element b:

```
scores[b] = Q[b] @ K[b]^T    ← (8, 32) @ (32, 8) = (8, 8)
```

The result `scores[b][i][j]` is the **raw attention score** from token i to
token j. Higher values mean token i wants to attend more to token j.

### Why (T, T)?

The attention matrix is T×T because we have T tokens, each potentially attending
to T tokens (including itself). This is the **quadratic cost** of attention —
doubling the sequence length quadruples the attention computation. For T=8 it's
trivial (64 elements), but for T=2048 it's 4M elements per batch.

### Memory After This Step

```
scores: 2 × 8 × 8 = 128 floats = 512 bytes
```

---

## 6.4 Scaling: Divide by sqrt(D)

Raw dot products grow with the dimension D. A dot product of two D-dimensional
vectors has expected magnitude proportional to sqrt(D). If we don't scale, the
softmax in the next step will be dominated by a few large scores (near-one-hot
output), making gradients vanish.

### The Fix

```
scaled_scores = scores / sqrt(D)
             = scores / sqrt(32)
             = scores / 5.657
```

### Shape Trace

```
scores:          (B, T, T) = (2, 8, 8)
sqrt(D):         scalar = 5.657
scaled_scores:   (B, T, T) = (2, 8, 8)    ← same shape, elementwise div
```

### Numerical Example

If `scores[b][0][3] = 22.6`, then after scaling:
```
scaled = 22.6 / 5.657 ≈ 4.0
```

Without scaling, scores like 22.6 would make softmax output nearly one-hot
(e.g., [0.97, 0.01, 0.01, 0.01]), giving tiny gradients on the non-max positions.
After scaling to ~4.0, softmax produces softer distributions like [0.60, 0.10,
0.10, 0.10, 0.10], which have meaningful gradients everywhere.

### Why sqrt(D) and Not D?

The variance of the dot product of two random D-dimensional vectors is D, so
the standard deviation is sqrt(D). Dividing by sqrt(D) makes the scaled scores
have unit variance regardless of D — keeping the softmax in a healthy regime.

---

## 6.5 Causal Mask: Prevent Looking Into the Future

In **autoregressive** language models, token i must not attend to tokens j > i
(tokens that come after it in the sequence). Otherwise the model could "cheat"
by looking at the next word during training.

### The Mask

```
Mask matrix (T=8):
       j=0  j=1  j=2  j=3  j=4  j=5  j=6  j=7
i=0  [  0,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞  ]
i=1  [  0,   0,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞  ]
i=2  [  0,   0,   0,  -∞,  -∞,  -∞,  -∞,  -∞  ]
i=3  [  0,   0,   0,   0,  -∞,  -∞,  -∞,  -∞  ]
i=4  [  0,   0,   0,   0,   0,  -∞,  -∞,  -∞  ]
i=5  [  0,   0,   0,   0,   0,   0,  -∞,  -∞  ]
i=6  [  0,   0,   0,   0,   0,   0,   0,  -∞  ]
i=7  [  0,   0,   0,   0,   0,   0,   0,   0  ]
```

- `0` means "allowed" (add 0 to the score — no change)
- `-∞` means "forbidden" (add -infinity → softmax output = 0)

Lower triangle = allowed. Upper triangle = forbidden.

### After Masking

```
masked_scores[b][i][j] = scaled_scores[b][i][j] + mask[i][j]

Where mask[i][j] = 0     if j <= i    (look at self and past)
                = -inf  if j > i     (don't look at future)
```

### Shape Trace

```
scaled_scores:   (B, T, T) = (2, 8, 8)
mask:            (T, T)    = (8, 8)    ← broadcast over B
masked_scores:   (B, T, T) = (2, 8, 8)
```

### Why Lower Triangular?

Token 0 can only see itself. Token 1 can see tokens 0 and 1. Token 7 can see all
8 tokens. This is the **causal** (left-to-right) constraint — the model predicts
the next token given only past context, just like it would during generation.

---

## 6.6 Softmax: Row-Wise Normalization

Softmax converts each row of the attention matrix into a probability distribution
that sums to 1.

### The Math

For row i (the attention distribution from token i to all tokens):

```
attn_weights[i][j] = exp(masked_scores[i][j]) / Σ_k exp(masked_scores[i][k])
```

### Numerically Stable Version

Naive softmax overflows for large scores. The stable version subtracts the row
maximum first:

```
m[i] = max_j(masked_scores[i][j])       ← row max
shifted[i][j] = masked_scores[i][j] - m[i]
attn_weights[i][j] = exp(shifted[i][j]) / Σ_k exp(shifted[i][k])
```

Subtracting the max doesn't change the result (exp cancels) but keeps all
exponent arguments ≤ 0, preventing overflow.

### Shape Trace

```
masked_scores:   (B, T, T) = (2, 8, 8)
row_max:         (B, T, 1) = (2, 8, 1)    ← one max per row, broadcast
shifted:         (B, T, T) = (2, 8, 8)
exp_shifted:     (B, T, T) = (2, 8, 8)
row_sum:         (B, T, 1) = (2, 8, 1)    ← one sum per row, broadcast
attn_weights:    (B, T, T) = (2, 8, 8)    ← each row sums to 1.0
```

### Example Row

For token i=3 after masking, scores might be: [1.2, 0.5, -0.3, 2.1, -∞, -∞, -∞, -∞]

```
Row max: 2.1
Shifted: [-0.9, -1.6, -2.4, 0.0, -∞, -∞, -∞, -∞]
Exp:     [0.407, 0.202, 0.091, 1.000, 0, 0, 0, 0]
Sum:     1.700
Attn:    [0.239, 0.119, 0.053, 0.588, 0, 0, 0, 0]  ← sums to 1.0
```

Token 3 attends most to itself (0.588) and some to tokens 0-2. Tokens 4-7 get
zero attention — they're in the future.

### Each Row Sums to 1.0

This is critical. Every row of `attn_weights` is a probability distribution over
the T tokens. Token i distributes its "attention budget" among tokens 0..i.
The total attention is always 1.0 — attending more to one token means attending
less to others.

---

## 6.7 Weighted Sum: Attention Weights @ V

Now we use the attention weights to combine the Value vectors.

### The Matmul

```
attn_weights: (B, T, T)  = (2, 8, 8)     ← which tokens to attend to
V:            (B, T, D)  = (2, 8, 32)    ← what each token contributes
─────────────────────────────────────────
attn_out:     (B, T, D)  = (2, 8, 32)    ← weighted combination
```

For each token i:

```
attn_out[b][i] = Σ_j attn_weights[b][i][j] × V[b][j]
```

Token i's output is a **weighted average** of all Value vectors, with weights
given by the attention distribution. If token i attends 60% to token j, then
60% of attn_out[i] comes from V[j].

### Intuition

```
V[0] = "the"  → [0.2, -0.5, 1.3, ...]   ← what "the" contributes
V[1] = "cat"  → [1.1, 0.3, -0.7, ...]    ← what "cat" contributes
V[2] = "sat"  → [-0.4, 0.8, 0.2, ...]    ← what "sat" contributes

If attn_weights for token 2 = [0.1, 0.3, 0.6]:
attn_out[2] = 0.1 × V[0] + 0.3 × V[1] + 0.6 × V[2]
            = 0.1×[0.2,-.5,1.3,...] + 0.3×[1.1,.3,-.7,...] + 0.6×[-.4,.8,.2,...]
            = [-0.01, 0.44, 0.01, ...]    ← a mix of context
```

### Memory After This Step

```
attn_out: 2 × 8 × 32 = 512 floats = 2048 bytes
```

---

## 6.8 Output Projection: W_o

The attention output goes through one more linear layer — the **output projection**
— which mixes the D dimensions and produces the final attention result.

### Shape Trace

```
W_o weight:   (D, D) = (32, 32)
attn_out:     (B, T, D) = (2, 8, 32)
Reshape:      (16, 32)
proj = attn_out @ W_o^T = (16, 32) @ (32, 32) = (16, 32)
Reshape:      (B, T, D) = (2, 8, 32)
```

### Why W_o?

With a single head, W_o is technically redundant — you could fuse it with V's
weight matrix. But in multi-head attention, W_o concatenates the outputs from
all heads and projects back to D dimensions. We keep it for architectural
consistency with the standard transformer design.

---

## 6.9 Residual Connection: x + attn_out

The **residual (skip) connection** adds the block's input back to its output.

### Shape Trace

```
x:          (B, T, D) = (2, 8, 32)   ← saved from before LayerNorm
attn_out:   (B, T, D) = (2, 8, 32)   ← output of W_o projection
──────────────────────────────────────
h:          (B, T, D) = (2, 8, 32)   ← elementwise add
```

### Why Residuals?

Without residuals, each layer must learn the **full transformation** from input
to output. With residuals, each layer only learns the **delta** (residual) — the
difference between input and desired output. This is much easier:

```
h = x + F(x)    ← F only needs to learn the "correction" to x
```

If F(x) ≈ 0 (e.g., early in training when weights are small), the output is
just x — the identity function. The network starts as "pass-through" and
gradually learns meaningful transformations.

This is called the **"identity initialization"** intuition. It's why deep
residual networks (ResNets) can be trained with 100+ layers — each layer only
needs to make a small adjustment.

### In Our Code

```zig
// In src/nn/block.zig:
const ln1_out = try self.ln1.forward(alloc, x, tape);
const attn_out = try self.attn.forward(alloc, ln1_out, tape);
const h = try ops_binary.add(alloc, x, attn_out, tape);  // residual
```

Note: we use `x` (pre-LayerNorm input) for the residual, not `ln1_out`. This is
**pre-norm** architecture — LayerNorm is applied before the sub-layer, and the
residual connects around the entire [LN → sub-layer] block.

---

## 6.10 LayerNorm 2 (Pre-Norm for MLP)

Same operation as LayerNorm 1, applied to the residual output h.

### Shape Trace

```
Input h:     (B, T, D) = (2, 8, 32)
μ:           (B, T, 1)  = (2, 8, 1)
σ²:          (B, T, 1)  = (2, 8, 1)
h_hat:       (B, T, D) = (2, 8, 32)
γ₂:          (D,)       = (32,)
β₂:          (D,)       = (32,)
ln2_out:     (B, T, D) = (2, 8, 32)
```

LayerNorm 2 has its **own** gamma and beta parameters, separate from LayerNorm 1.
The model learns different normalization adjustments for the attention and MLP
sub-layers.

---

## 6.11 MLP: Feed-Forward Network

The MLP (Multi-Layer Perceptron) is a two-layer feed-forward network with a
non-linear activation in between.

### Architecture

```
ln2_out ──→ fc1 ──→ GELU ──→ fc2 ──→ mlp_out
(D=32)     (4D=128)         (D=32)
```

### fc1: Expand D → 4D

```
W_fc1 weight: (d_ff, D) = (128, 32)
Input:        (B, T, D) = (2, 8, 32)
Reshape:      (16, 32)
fc1_out = input @ W_fc1^T = (16, 32) @ (32, 128) = (16, 128)
Reshape:      (B, T, d_ff) = (2, 8, 128)
```

The expansion to 4× the model dimension gives the MLP more "working space" to
learn complex patterns. The factor of 4 is a design choice from the original
Transformer paper (Vaswani et al., 2017).

### GELU Activation

```
fc1_out:  (B, T, d_ff) = (2, 8, 128)
gelu_out: (B, T, d_ff) = (2, 8, 128)    ← same shape, elementwise
```

GELU (Gaussian Error Linear Unit) is defined as:

```
GELU(x) = x × Φ(x)    where Φ is the standard normal CDF

Approximation (used in practice):
GELU(x) ≈ 0.5 × x × (1 + tanh(sqrt(2/π) × (x + 0.044715 × x³)))
```

GELU is a smooth approximation of ReLU that:
- Is ~0 for large negative values (like ReLU)
- Is ~x for large positive values (like ReLU)
- Has a **smooth** transition around 0 (unlike ReLU's sharp cutoff)
- Allows small negative values to pass through (unlike ReLU which clips to 0)

The smooth transition helps gradient flow during training.

### fc2: Contract 4D → D

```
W_fc2 weight: (D, d_ff) = (32, 128)
Input:        (B, T, d_ff) = (2, 8, 128)
Reshape:      (16, 128)
fc2_out = input @ W_fc2^T = (16, 128) @ (128, 32) = (16, 32)
Reshape:      (B, T, D) = (2, 8, 32)
```

fc2 projects back down from 128 dimensions to the model dimension D=32. The
MLP "expands, transforms, contracts" — it temporarily works in a
higher-dimensional space where it can learn richer representations.

### MLP Summary

```
ln2_out:   (2, 8, 32)    ← input
fc1_out:   (2, 8, 128)   ← expanded
gelu_out:  (2, 8, 128)   ← activated
fc2_out:   (2, 8, 32)    ← contracted back
```

### Memory at Peak (within MLP)

The MLP's peak memory is at `gelu_out` — we hold both `fc1_out` and `gelu_out`
(temporaries for backward) and the fc2 weight. Total activation memory:
2×8×128 × 2 = 4096 floats = 16 KB. This is the **most memory-intensive**
sub-layer in our tiny model.

In large models, the MLP's 4× expansion dominates memory: for GPT-3 with
D=12288, the MLP hidden dim is 49152, consuming ~192 MB per layer per batch
sample for activations alone.

---

## 6.12 Second Residual Connection: h + mlp_out

```
h:          (B, T, D) = (2, 8, 32)   ← saved from after first residual
mlp_out:    (B, T, D) = (2, 8, 32)   ← fc2 output
──────────────────────────────────────
block_out:  (B, T, D) = (2, 8, 32)   ← final output of the transformer block
```

This is the second residual connection, wrapping the [LN2 → MLP] block.

### Full Block Input → Output

```
x:          (2, 8, 32)    ← input to the block
block_out:  (2, 8, 32)    ← output of the block

Shape unchanged. The transformer block is a shape-preserving transformation.
```

This is a fundamental design property: **every sub-layer preserves the shape
(B, T, D)**. Residual connections only work when shapes match. This is why
Linear layers have D_in = D_out = D for Q/K/V/O projections.

---

## 7. Final LayerNorm

After the transformer block, we apply one more LayerNorm. This is the **final
layer normalization** before the language model head.

### Shape Trace

```
block_out:    (B, T, D) = (2, 8, 32)
γ_f:          (D,)       = (32,)
β_f:          (D,)       = (32,)
ln_f_out:     (B, T, D) = (2, 8, 32)
```

This LayerNorm has its own gamma and beta (separate from the two inside the block).

### Why a Final LayerNorm?

The original post-norm Transformer (Vaswani et al.) placed LayerNorm **after**
the residual add. In pre-norm (our architecture), the output of the last block
hasn't been normalized, so we need a final LN to stabilize the values before
feeding into the language model head.

---

## 8. lm_head: Project to Vocabulary

The language model head projects each token's D-dimensional representation to
a V-dimensional logit vector, one score per vocabulary word.

### Shape Trace

```
W_lm weight:  (V, D) = (64, 32)       ← one row per vocab word
ln_f_out:     (B, T, D) = (2, 8, 32)
Reshape:      (16, 32)
logits = input @ W_lm^T = (16, 32) @ (32, 64) = (16, 64)
Reshape:      (B, T, V) = (2, 8, 64)
```

### What Are Logits?

`logits[b][t][v]` is the **unnormalized score** for vocabulary word v at position
t in batch sample b. Higher logit → the model thinks word v is more likely to
appear at position t.

To convert to probabilities:

```
P(word v at position t) = exp(logits[t][v]) / Σ_k exp(logits[t][k])
```

This is just softmax over the V dimension. But we don't compute it explicitly —
the cross-entropy loss function does it internally (more numerically stable).

### Relationship to Token Embedding

Notice that `W_lm` has shape (V, D) — the same shape as `tok_embed.weight`.
In many implementations (including ours), `lm_head.weight` is a **separate**
parameter from `tok_embed.weight`. Some architectures tie them (share the same
weight matrix), but we keep them independent for clarity.

### Memory After This Step

```
logits: 2 × 8 × 64 = 1024 floats = 4096 bytes
```

---

## 9. Cross-Entropy Loss

The loss function measures how badly the model's predictions match the targets.

### The Targets

For language modeling, the target for position t is the **next token** at
position t+1. Given input `[5, 12, 3, 45, 7, 22, 9, 1]`, the targets are
`[12, 3, 45, 7, 22, 9, 1, ?]`. (The last target is typically a special end
token or ignored.)

```
targets shape: (B, T) = (2, 8)     ← integer token IDs
```

### Reshape for Loss Computation

Our cross-entropy implementation expects 2D logits and 1D targets:

```
logits:   (B, T, V) = (2, 8, 64)  → reshape → (B*T, V) = (16, 64)
targets:  (B, T)    = (2, 8)      → reshape → (B*T,)   = (16,)
```

Flattening batch and sequence dimensions is safe because the loss is computed
independently for each (b, t) pair.

### Cross-Entropy Formula

For a single position with logits `z` (length V) and target class `y`:

```
loss = -log(softmax(z)[y])
     = -z[y] + log(Σ_k exp(z[k]))
```

The **log-sum-exp trick** prevents overflow:

```
m = max_k(z[k])                      ← max for stability
loss = -z[y] + m + log(Σ_k exp(z[k] - m))
```

### Total Loss

We average over all B×T positions:

```
loss = (1 / (B × T)) × Σ_{b,t} CE(logits[b,t], targets[b,t])
     = (1 / 16) × (sum of 16 per-position losses)
```

### Shape Trace

```
logits_flat:  (B*T, V) = (16, 64)
targets_flat: (B*T,)   = (16,)
per_pos_loss: (B*T,)   = (16,)         ← one loss per position
loss:         scalar   = ()            ← mean over 16 positions
```

### What the Scalar Loss Means

- **loss ≈ ln(V) = ln(64) ≈ 4.16**: random predictions (chance level)
- **loss ≈ 0**: perfect predictions (the model is certain and correct)
- **loss between 0 and 4.16**: the model is learning

In our example 04, the loss drops from 3.83 (near-random) to 3.09 (somewhat
better than random) over 50 training steps. This is expected for a tiny model
with a small training set — the model is barely starting to learn patterns.

---

## 10. Full Shape Trace Table

Every operation in one forward pass, in order:

| # | Operation | Input Shape | Output Shape | Notes |
|---|-----------|-------------|-------------|-------|
| 1 | Token Embed | (2, 8) | (2, 8, 32) | Gather rows from (64, 32) |
| 2 | Position Embed | (1, 8) | (1, 8, 32) | Gather rows from (8, 32) |
| 3 | Add | (2,8,32) + (1,8,32) | (2, 8, 32) | Broadcast over B |
| 4 | LayerNorm 1 | (2, 8, 32) | (2, 8, 32) | Per-token normalization |
| 5 | Q = Linear(D, D) | (2, 8, 32) | (2, 8, 32) | Reshape→matmul→reshape |
| 6 | K = Linear(D, D) | (2, 8, 32) | (2, 8, 32) | Same input as Q |
| 7 | V = Linear(D, D) | (2, 8, 32) | (2, 8, 32) | Same input as Q |
| 8 | K^T = transpose | (2, 8, 32) | (2, 32, 8) | Swap inner dims |
| 9 | Scores = Q @ K^T | (2,8,32)×(2,32,8) | (2, 8, 8) | Batched matmul |
| 10 | Scale ÷ √D | (2, 8, 8) | (2, 8, 8) | ÷5.657, elementwise |
| 11 | Causal mask | (2, 8, 8) | (2, 8, 8) | Add -inf to upper tri |
| 12 | Softmax | (2, 8, 8) | (2, 8, 8) | Row-wise, rows sum to 1 |
| 13 | Attn @ V | (2,8,8)×(2,8,32) | (2, 8, 32) | Weighted sum |
| 14 | O = Linear(D, D) | (2, 8, 32) | (2, 8, 32) | Output projection |
| 15 | Residual add | (2,8,32)+(2,8,32) | (2, 8, 32) | x + attn_out |
| 16 | LayerNorm 2 | (2, 8, 32) | (2, 8, 32) | Separate γ₂, β₂ |
| 17 | fc1 = Linear(D, 4D) | (2, 8, 32) | (2, 8, 128) | Expand |
| 18 | GELU | (2, 8, 128) | (2, 8, 128) | Smooth activation |
| 19 | fc2 = Linear(4D, D) | (2, 8, 128) | (2, 8, 32) | Contract |
| 20 | Residual add | (2,8,32)+(2,8,32) | (2, 8, 32) | h + mlp_out |
| 21 | Final LayerNorm | (2, 8, 32) | (2, 8, 32) | Separate γ_f, β_f |
| 22 | lm_head = Linear(D, V) | (2, 8, 32) | (2, 8, 64) | Logits |
| 23 | Reshape logits | (2, 8, 64) | (16, 64) | Flatten B, T |
| 24 | Reshape targets | (2, 8) | (16,) | Flatten B, T |
| 25 | Cross-entropy | (16, 64) + (16,) | () scalar | Mean over positions |

**25 operations** from raw token IDs to a single loss scalar.

---

## 11. Memory Accounting: Parameters and Activations

### Parameter Count

| Parameter | Shape | Elements | f32 Bytes |
|-----------|-------|----------|-----------|
| tok_embed.weight | (64, 32) | 2,048 | 8,192 |
| pos_embed.weight | (8, 32) | 256 | 1,024 |
| ln1.gamma | (32,) | 32 | 128 |
| ln1.beta | (32,) | 32 | 128 |
| W_q.weight | (32, 32) | 1,024 | 4,096 |
| W_q.bias | (32,) | 32 | 128 |
| W_k.weight | (32, 32) | 1,024 | 4,096 |
| W_k.bias | (32,) | 32 | 128 |
| W_v.weight | (32, 32) | 1,024 | 4,096 |
| W_v.bias | (32,) | 32 | 128 |
| W_o.weight | (32, 32) | 1,024 | 4,096 |
| W_o.bias | (32,) | 32 | 128 |
| ln2.gamma | (32,) | 32 | 128 |
| ln2.beta | (32,) | 32 | 128 |
| fc1.weight | (128, 32) | 4,096 | 16,384 |
| fc1.bias | (128,) | 128 | 512 |
| fc2.weight | (32, 128) | 4,096 | 16,384 |
| fc2.bias | (32,) | 32 | 128 |
| ln_f.gamma | (32,) | 32 | 128 |
| ln_f.beta | (32,) | 32 | 128 |
| lm_head.weight | (64, 32) | 2,048 | 8,192 |
| lm_head.bias | (64,) | 64 | 256 |

**Total parameters:** 20,832 elements = 83,328 bytes ≈ 81.5 KB

### Optimizer State (AdamW)

AdamW stores two additional tensors per parameter (first and second moment):

```
Optimizer state = 2 × total_params = 2 × 20,832 = 41,664 floats = 166,656 bytes ≈ 163 KB
```

**Total with optimizer:** 83,328 + 166,656 = 249,984 bytes ≈ 244 KB

### Activation Memory (Peak)

The maximum activation memory is reached inside the MLP when both `fc1_out` and
`gelu_out` are alive simultaneously:

```
Peak activations ≈ 2 × 8 × 128 × 2 + 2 × 8 × 8 + 2 × 8 × 32 × (several)
                 ≈ 4096 + 128 + ~2000
                 ≈ 6000 floats ≈ 24 KB
```

For comparison, a GPT-3 layer with D=12288, B=1, T=2048:

```
Peak MLP activations ≈ 2 × 1 × 2048 × 49152 × 2 = ~400M floats ≈ 1.6 GB per layer
```

This is why large model training requires gradient checkpointing — recomputing
activations during backward instead of storing them all.

---

## 12. PyTorch Equivalent

For readers familiar with PyTorch, here's the equivalent architecture:

```python
import torch
import torch.nn as nn
import math

class TinyWordTransformer(nn.Module):
    def __init__(self, V=64, D=32, T=8, d_ff=128):
        super().__init__()
        self.tok_embed = nn.Embedding(V, D)
        self.pos_embed = nn.Embedding(T, D)
        self.ln1 = nn.LayerNorm(D)
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, d_ff)
        self.fc2 = nn.Linear(d_ff, D)
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_embed(idx)               # (B, T, D)
        pos = self.pos_embed(torch.arange(T))    # (T, D) → (1, T, D)
        x = tok + pos                            # (B, T, D)

        # Attention sub-block
        h = self.ln1(x)                          # (B, T, D)
        Q = self.W_q(h)                          # (B, T, D)
        K = self.W_k(h)                          # (B, T, D)
        V = self.W_v(h)                          # (B, T, D)
        scores = Q @ K.transpose(-2, -1)         # (B, T, T)
        scores = scores / math.sqrt(D)           # (B, T, T)
        mask = torch.triu(torch.ones(T, T), diagonal=1) * float('-inf')
        scores = scores + mask                   # (B, T, T)
        attn = torch.softmax(scores, dim=-1)     # (B, T, T)
        attn_out = attn @ V                      # (B, T, D)
        attn_out = self.W_o(attn_out)            # (B, T, D)
        x = x + attn_out                         # residual

        # MLP sub-block
        h = self.ln2(x)                          # (B, T, D)
        mlp = self.fc1(h)                        # (B, T, d_ff)
        mlp = nn.functional.gelu(mlp)            # (B, T, d_ff)
        mlp = self.fc2(mlp)                      # (B, T, D)
        x = x + mlp                              # residual

        # Output
        x = self.ln_f(x)                         # (B, T, D)
        logits = self.lm_head(x)                 # (B, T, V)
        return logits

model = TinyWordTransformer(V=64, D=32, T=8, d_ff=128)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# → Parameters: 20,832
```

---

## 13. Common Mistakes

### Mistake 1: Forgetting the Transpose on Weight Matrices

**Wrong:**
```
output = input @ W       ← (B*T, D_in) @ (D_out, D_in) = SHAPE MISMATCH
```

**Right:**
```
output = input @ W^T     ← (B*T, D_in) @ (D_in, D_out) = (B*T, D_out)
```

In our code, `Linear.weight` is stored as `(D_out, D_in)` (Kaiming init
convention), and the forward pass does `input @ W^T`. If you forget the
transpose, you get a shape mismatch error — or worse, if D_in = D_out, it
silently computes the wrong thing.

### Mistake 2: Using transpose2d on a 3D Tensor

**Wrong:**
```
K_transposed = K.transpose2d()   ← crashes on (B, T, D) — expects 2D
```

**Right:**
```
K_transposed = transposeInner2d(K)  ← swaps dims[1] and dims[2] of 3D tensor
```

This is a recurring bug in our codebase (discovered in Stage 4). The
`transpose2d` function expects a rank-2 tensor. For 3D tensors like K of
shape (B, T, D), use `transposeInner2d` which swaps the inner two dimensions
while leaving the batch dimension alone.

### Mistake 3: Forgetting keepAlive on Forward Intermediates

**Wrong:**
```zig
const transposed = try transposeInner2d(alloc, K);
defer transposed.deinit();
// ... later, backward reads transposed.data → USE-AFTER-FREE
```

**Right:**
```zig
const transposed = try transposeInner2d(alloc, K);
try tape.keepAlive(&transposed);  // tape takes ownership of the buffer
defer transposed.deinit();        // now a no-op (owned=false after keepAlive)
```

Without `keepAlive`, the tape's SavedData holds a slice into a freed buffer.
During backward, reading that slice is undefined behavior (use-after-free).
The `keepAlive` call transfers buffer ownership to the tape, which frees it
after backward completes.

### Mistake 4: Reading Loss After Deinit

**Wrong:**
```zig
const loss = try crossEntropy(alloc, logits_flat, targets, tape);
loss.deinit();
const loss_val = loss.data[0];  // USE-AFTER-FREE
```

**Right:**
```zig
const loss = try crossEntropy(alloc, logits_flat, targets, tape);
const loss_val = loss.data[0];  // save before deinit
loss.deinit();
```

Tensor `deinit()` frees the underlying buffer. Any read after deinit is
use-after-free. Always extract scalar values before calling deinit.

### Mistake 5: Missing Causal Mask

**Wrong:**
```
attn_weights = softmax(scores)  ← token 0 can attend to token 7 (the future!)
```

**Right:**
```
masked_scores = scores + causal_mask  ← -inf in upper triangle
attn_weights = softmax(masked_scores) ← future tokens get 0 probability
```

Without the causal mask, the model can "cheat" by looking at future tokens
during training. It will learn faster (lower training loss) but fail
completely at generation time (it never learned to predict without seeing
the answer).

### Mistake 6: Wrong Scaling Factor in Attention

**Wrong:**
```
scores = Q @ K^T / D        ← divide by 32 (too much scaling)
scores = Q @ K^T            ← no scaling (softmax saturates)
```

**Right:**
```
scores = Q @ K^T / sqrt(D) ← divide by 5.657 (correct)
```

The scaling factor is sqrt(D), not D. Dividing by D makes the scores too small
(softmax output becomes nearly uniform → no attention). Not scaling at all
makes scores too large (softmax becomes one-hot → vanishing gradients).

### Mistake 7: Confusing Pre-Norm and Post-Norm Residual Placement

**Pre-norm (ours):**
```
x = x + SubLayer(LayerNorm(x))
```

**Post-norm (original Transformer):**
```
x = LayerNorm(x + SubLayer(x))
```

These look similar but behave very differently:

- **Pre-norm**: LayerNorm is applied **before** the sub-layer. The residual
  path is "clean" (no normalization on it). Gradients flow easily through
  the residual path. Training is more stable.

- **Post-norm**: LayerNorm is applied **after** the residual add. The
  residual path goes through LayerNorm. Gradients must pass through
  LayerNorm on every skip, which can destabilize training in deep models.

Mixing up the two leads to wrong gradient computation and training instability.

### Mistake 8: Forgetting That Attention Is Per-Token, Not Per-Feature

LayerNorm normalizes across D (features). Softmax in attention normalizes
across T (tokens). Confusing the two:

- LayerNorm: each token's D-dimensional vector is normalized independently
- Attention softmax: each row (source token's attention) is normalized
  independently over the T target tokens

If you accidentally softmax over D instead of T, you'd get a probability
distribution over features, not over tokens — the model would learn to
weight feature dimensions, not to attend to context.

---

## Appendix A: Shape Changes at a Glance

```
(B, T) = (2, 8)                   ← token IDs
  │
  ├── tok_embed ──→ (B, T, D) = (2, 8, 32)
  ├── pos_embed ──→ (1, T, D) = (1, 8, 32)
  │
  ├── add ────────→ (B, T, D) = (2, 8, 32)
  │
  ├── LN1 ───────→ (B, T, D) = (2, 8, 32)
  │     │
  │     ├── Q ───→ (B, T, D) = (2, 8, 32)
  │     ├── K ───→ (B, T, D) = (2, 8, 32) ──→ K^T (B, D, T) = (2, 32, 8)
  │     ├── V ───→ (B, T, D) = (2, 8, 32)
  │     │
  │     ├── Q@K^T → (B, T, T) = (2, 8, 8)
  │     ├── ÷√D ──→ (B, T, T) = (2, 8, 8)
  │     ├── mask ──→ (B, T, T) = (2, 8, 8)
  │     ├── softmax → (B, T, T) = (2, 8, 8)
  │     ├── attn@V → (B, T, D) = (2, 8, 32)
  │     └── W_o ──→ (B, T, D) = (2, 8, 32)
  │
  ├── +residual ──→ (B, T, D) = (2, 8, 32)
  │
  ├── LN2 ───────→ (B, T, D) = (2, 8, 32)
  │     │
  │     ├── fc1 ──→ (B, T, 4D) = (2, 8, 128)
  │     ├── GELU ─→ (B, T, 4D) = (2, 8, 128)
  │     └── fc2 ──→ (B, T, D) = (2, 8, 32)
  │
  ├── +residual ──→ (B, T, D) = (2, 8, 32)
  │
  ├── LN_f ──────→ (B, T, D) = (2, 8, 32)
  │
  ├── lm_head ───→ (B, T, V) = (2, 8, 64)
  │
  ├── reshape ────→ (B*T, V) = (16, 64)
  │
  └── cross_entropy → scalar ()
```

---

## Appendix B: Dimension Symbols Quick Reference

| Symbol | Name | Our Value | Meaning |
|--------|------|-----------|---------|
| B | batch_size | 2 | Number of sequences processed simultaneously |
| T | seq_len | 8 | Number of tokens per sequence (context window) |
| D | d_model | 32 | Width of each token's representation vector |
| V | vocab_size | 64 | Number of unique words in the vocabulary |
| d_ff | feed-forward dim | 128 | Hidden size of the MLP (typically 4×D) |
| √D | attention scale | 5.657 | Prevents softmax saturation |
| B×T | total positions | 16 | Total predictions per batch |
| B×T×D | total activations | 512 | Per-tensor activation count (before MLP expansion) |
| B×T×d_ff | MLP hidden | 2048 | Per-tensor activation count (MLP expansion) |
| B×T×T | attention matrix | 128 | Per-batch attention score elements |

---

## Appendix C: Where in Our Code

| Concept | File | Key Function/Struct |
|---------|------|---------------------|
| Token embedding | `src/nn/embedding.zig` | `Embedding.forward()` |
| Position embedding | `src/nn/model.zig` | `TinyWordTransformer.forward()` |
| LayerNorm | `src/nn/layernorm.zig` | `LayerNorm.forward()` |
| Linear (QKV, MLP) | `src/nn/linear.zig` | `Linear.forward()` |
| CausalSelfAttention | `src/nn/attention.zig` | `CausalSelfAttention.forward()` |
| Transformer block | `src/nn/block.zig` | `TransformerBlock.forward()` |
| MLP | `src/nn/mlp.zig` | `MLP.forward()` |
| GELU | `src/nn/activations.zig` | `gelu()` |
| Full model | `src/nn/model.zig` | `TinyWordTransformer.forward()` |
| Cross-entropy | `src/tensor/ops/reduction.zig` | `crossEntropy()` |
| Matmul | `src/tensor/ops/matmul.zig` | `matmul()`, `matmulBatch()` |
| Softmax | `src/tensor/ops/reduction.zig` | `softmax()` |
| Transpose inner | `src/tensor/ops/shape_ops.zig` | `transposeInner2d()` |
| keepAlive | `src/autograd/tape.zig` | `Tape.keepAlive()` |
| Reshape tracked | `src/tensor/ops/shape_ops.zig` | `reshapeTracked()` |


---

## Exercises

**Exercise 1.** A transformer block with `D = 64` and `n_head = 4`
produces per-head attention scores of what shape? For `B = 2, T = 8`.

<details><summary>Solution</summary>

Per-head dimension: `d_head = D / n_head = 64 / 4 = 16`. Q, K, V
after split-heads: `(B, n_head, T, d_head) = (2, 4, 8, 16)`. The
attention scores `Q @ K^T` produce `(B, n_head, T, T) = (2, 4, 8, 8)`.
After softmax and multiplication by V, back to `(B, n_head, T, d_head)`.
Reverse merge-heads returns to `(B, T, D) = (2, 8, 64)`.

</details>

**Exercise 2.** Why is the scale factor in attention `1 / sqrt(d_head)`
rather than `1 / sqrt(D)`?

<details><summary>Solution</summary>

Each attention head operates in its own `d_head`-dimensional
subspace. The dot products `Q @ K^T` have magnitude proportional
to `sqrt(d_head)` (central limit argument: sum of `d_head`
roughly-independent products). Dividing by `sqrt(d_head)` keeps
the softmax input at unit magnitude regardless of head size. If you
used `1 / sqrt(D)` with `D = 64` and `d_head = 8` you'd be
scaling by `1/8` when only `1/2.83` is needed, squashing the
attention distribution toward uniform.

This is a Stage 8 M4 detail: early drafts of the multi-head code
used `D` and the attention outputs collapsed to near-uniform
softmax distributions until the scale was fixed.

</details>
