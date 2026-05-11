# 03b — From Stage 3 Autograd to Training a Transformer

This chapter bridges the gap between the autograd code you just read (Stage 3:
tape, nodes, backward functions, gradient checking) and the ML/DL concepts
those pieces implement. By the end, you'll understand *why* each backward
function exists, *where* it fires during transformer training, and *how*
the pieces compose into the backward pass of a real model.

We assume you've read `docs/03_autograd.md` (the mechanical details of the
tape and backward functions) and now want the conceptual picture: how does
this autograd engine actually train a transformer?

---

## 1. The Big Picture: What Backprop Does for a Transformer

The forward pass produces a loss — one number that says "how wrong is the
model?" The backward pass answers a different question: *if I tweak each
parameter by a tiny amount, how much does the loss change?*

That's the **gradient**: ∂L/∂W for every weight W in the model.

```
┌───────────────────────────────────────────────────────────────────┐
│  FORWARD (Stage 2 ops)                                           │
│                                                                   │
│  tokens → embed → linear → attention → mlp → logits → loss        │
│                                                                   │
│  Each op records itself on the tape: "I did a matmul with these   │
│  inputs, producing this output, and here's the data backward      │
│  needs."                                                          │
│                                                                   │
│  BACKWARD (Stage 3 autograd)                                     │
│                                                                   │
│  Walk the tape in reverse. For each node:                         │
│    1. Look up the gradient of the output (∂L/∂output)            │
│    2. Apply the backward rule to get ∂L/∂each_input              │
│    3. Accumulate into the input's gradient slot                   │
│                                                                   │
│  After backward, every parameter has .grad filled in.             │
│  The optimizer (Stage 4) uses these to update the weights.        │
└───────────────────────────────────────────────────────────────────┘
```

### Where Each Backward Function Fires in a Transformer

| Training Phase | Forward Op | Backward Function | What It Computes |
|---------------|-----------|-------------------|-----------------|
| Linear layer | `matmul(x, W)` | `backwardMatmul` | ∂L/∂x = ∂L/∂y @ Wᵀ, ∂L/∂W = xᵀ @ ∂L/∂y |
| Bias addition | `add(out, bias)` | `backwardAdd` | ∂L/∂bias = sumToShape(∂L/∂y, bias.shape) |
| Residual | `add(sublayer, x)` | `backwardAdd` | ∂L/∂x += ∂L/∂y (gradient passes through) |
| GELU activation | `geluExact(hidden)` | `backwardGelu` | ∂L/∂hidden = ∂L/∂y * GELU'(hidden) |
| Attention weights | `softmax(scores)` | `backwardSoftmax` | ∂L/∂scores = S*(∂L/∂S - (∂L/∂S·S)) |
| Attention scaling | `mulScalar(scores, 1/√d)` | `backwardMulScalar` | ∂L/∂scores = ∂L/∂y * (1/√d) |
| Attention masking | `mul(scores, mask)` | `backwardMul` | ∂L/∂scores = ∂L/∂y * mask |
| Output loss | `crossEntropy(logits, targets)` | `backwardCrossEntropy` | ∂L/∂logits = softmax - one_hot |
| LayerNorm (Stage 4) | `mean`, `sub`, `div` | backwardMean/Sub/Div | Per-channel normalization gradient |

---

## 2. The Tape Is the Computation Graph

If you've used PyTorch, you've seen `loss.backward()`. Under the hood,
PyTorch builds a **dynamic computation graph** — a DAG where each node is
a tensor and each edge is an operation. Our tape is the same thing, but
explicit: we store the graph as an `ArrayList(Node)`.

### Why "Tape" and Not "Graph"?

The name comes from **Wengert lists** (1964), which described automatic
differentiation as recording operations on a "tape" and then replaying
them in reverse. Our implementation matches this metaphor:

- **Record phase** (forward): ops append nodes to the tape
- **Replay phase** (backward): walk the tape in reverse, computing gradients

### What PyTorch Hides, We Make Explicit

| Concept | PyTorch | Our Library |
|---------|---------|-------------|
| Graph construction | Automatic (autograd) | Explicit `tape` parameter |
| Gradient tracking | `requires_grad=True` on tensor | `requires_grad=true` + `tape.trackLeaf()` |
| Backward invocation | `loss.backward()` | `tape.backward(&loss)` |
| Gradient storage | `tensor.grad` | `tensor.grad` (same!) |
| Graph lifetime | Freed after backward by default | `tape.deinit()` frees gradients |
| Multiple uses of a tensor | Handled by autograd | Handled by `accumulateGrad` |

### The tape Parameter on Every Op

In PyTorch, you write `c = a + b` and autograd records the operation
automatically. In our library, you pass the tape explicitly:

```zig
// PyTorch:   c = a + b              (autograd records automatically)
// Our lib:   var c = try elementwise.add(allocator, a, b, &tape);
```

Why explicit? Two reasons:

1. **No hidden globals.** PyTorch uses a thread-local gradient tracker.
   We pass the tape as a parameter — no global state, no thread-safety
   issues, and you can have multiple independent tapes.

2. **Zero-cost when not training.** Pass `null` for the tape parameter
   and no nodes are recorded, no gradient tracking happens, no overhead.
   This matters for inference (forward-only) where you don't need gradients.

---

## 3. Backward Functions: The Calculus Behind Training

Each backward function implements a specific calculus rule. Understanding
*which* rule and *why* tells you what the optimizer will do with the
resulting gradient.

### 3.1 backwardMatmul — The Workhorse of Transformer Training

**Forward:** `C = A @ B` (linear layer: `output = input @ W`)

**Backward:**
```
∂L/∂A = ∂L/∂C @ Bᵀ    (gradient w.r.t. input)
∂L/∂W = Aᵀ @ ∂L/∂C    (gradient w.r.t. weight)
```

**Where it fires in a transformer:** Every linear layer — QKV projection,
output projection, MLP layers. With 8 matmuls per block (see docs/02b), that's
8 backwardMatmul calls per block per training step.

**What the optimizer does with ∂L/∂W:** SGD updates `W -= lr * ∂L/∂W`.
If the gradient says "increasing W[3,7] would increase the loss," the
optimizer decreases W[3,7].

**Why the transpose?** The matmul `C = A @ B` means each element `C[i,j]`
depends on an entire *row* of A and an entire *column* of B. The gradient
∂L/∂A[i,k] is the sum over all j of ∂L/∂C[i,j] * B[k,j] — that's exactly
the matmul `∂L/∂C @ Bᵀ`. The transpose aligns the dimensions for the
dot product.

```zig
// Our implementation in backward.zig:
const bt = try tp.b.transpose2d();       // Bᵀ (view, zero-copy)
const da_val = try ops_matmul.matmul(allocator, grad_output.*, bt, null);
// da_val = ∂L/∂C @ Bᵀ = ∂L/∂A

const at = try tp.a.transpose2d();       // Aᵀ (view, zero-copy)
const db_val = try ops_matmul.matmul(allocator, at, grad_output.*, null);
// db_val = Aᵀ @ ∂L/∂C = ∂L/∂B
```

**Cost:** The backward of a matmul is *two more matmuls* of similar size.
This is why matmul dominates both forward and backward time — the backward
is at least as expensive as the forward.

### 3.2 backwardAdd — Residual Connections and Bias Gradients

**Forward:** `c = a + b` (bias addition, residual connection)

**Backward:**
```
∂L/∂a = sumToShape(∂L/∂c, a.shape)
∂L/∂b = sumToShape(∂L/∂c, b.shape)
```

**Bias addition:** If `a` has shape `(B*T, C)` and `b` (the bias) has shape
`(C,)`, then `b` was broadcast from `(C,)` to `(B*T, C)` during the forward
pass. In backward, we need to collapse the gradient back to `(C,)`:

```
∂L/∂bias[c] = Σ_{i} ∂L/∂c[i,c]    (sum over all positions)
```

This is `sumToShape`: it sums along the axes that were broadcast (expanded
from dim 1 to a larger dim).

**Residual connections:** When `x` is used in both the sublayer and the
residual addition `output = sublayer(x) + x`, the gradient of `x` gets two
contributions — one from the sublayer backward, one from the addition backward.
The `accumulateGrad` function adds them: `∂L/∂x += ∂L/∂output`.

**Why is this important for training?** Residual connections provide a "gradient
highway." Even if the sublayer's backward produces a tiny gradient (vanishing
gradients), the addition backward passes the upstream gradient through unchanged
to the residual path. This is why transformers can be trained with dozens of
layers — the residual path ensures gradients reach early layers.

### 3.3 backwardMul — Attention Masking and Squared Loss

**Forward:** `c = a * b` (elementwise multiplication)

**Backward:**
```
∂L/∂a = sumToShape(∂L/∂c * b, a.shape)
∂L/∂b = sumToShape(∂L/∂c * a, b.shape)
```

**Attention masking:** In a causal transformer, we mask future positions by
multiplying scores with a 0/1 mask before softmax. During backward, the
mask acts as a gate: masked positions (mask=0) get zero gradient — they
don't contribute to the loss, so there's no learning signal for them.

```
∂L/∂scores = ∂L/∂masked_scores * mask
```

This is mathematically correct: if `masked_score[i,j] = score[i,j] * mask[i,j]`
and `mask[i,j] = 0`, then `∂masked_score/∂score[i,j] = 0` — the output
doesn't depend on that input, so the gradient is zero.

**Squared loss:** When computing `loss = sumAll(probs * probs)`, the backward
of `mul(probs, probs)` uses the **product rule**:

```
∂L/∂probs = ∂L/∂squared * probs (from input a) + ∂L/∂squared * probs (from input b)
          = 2 * ∂L/∂squared * probs
```

Since `probs` is used as both inputs to the mul, `accumulateGrad` adds the
two contributions. This is the multivariable chain rule in action.

### 3.4 backwardGelu — The Smooth Gradient of GELU

**Forward:** `c = GELU(a) = 0.5 * a * (1 + erf(a / √2))`

**Backward:** `∂L/∂a = ∂L/∂c * GELU'(a)`

```
GELU'(x) = 0.5 * (1 + erf(x/√2)) + x * exp(-x²/2) / √(2π)
```

**Where it fires:** In the MLP, between the two linear layers.

**Why GELU's smooth gradient matters:** ReLU has a gradient of exactly 0 for
negative inputs (dead neurons). GELU has a small but nonzero gradient for
moderately negative inputs — the `exp(-x²/2)` term provides a "soft leak."
This means:

- Dead neurons can **recover** during training (they're not permanently dead)
- The gradient is **continuous** at x=0 (no sharp discontinuity like ReLU)
- Training is slightly **more stable** because the gradient doesn't vanish
  abruptly

Our implementation uses the Abramowitz & Stegun polynomial approximation
for erf, accurate to ~1.5×10⁻⁷ (within f32 precision).

### 3.5 backwardSoftmax — The "Diag Minus Outer" Formula

**Forward:** `S = softmax(x)` along the last axis

**Backward:** `∂L/∂x = S * (∂L/∂S - (∂L/∂S · S) * 1)`

**Where it fires:** After attention score computation. The softmax converts
raw scores to attention weights; its backward tells us how to adjust the
scores to change the attention pattern.

**Intuition:** If position i currently attends strongly to position j
(S[i,j] is high), and the gradient says "the loss would decrease if
S[i,j] went down," then ∂L/∂score[i,j] is negative — the optimizer will
decrease score[i,j], making position i attend less to position j.

**The "diag minus outer" structure:** The Jacobian of softmax has a special
form. For a single row of softmax:

```
∂S_i/∂x_j = S_i(δ_ij - S_j)
```

- **Diagonal term** (i=j): `S_i(1 - S_i)` — positive, meaning increasing
  x_i increases S_i (as long as S_i < 1).
- **Off-diagonal term** (i≠j): `-S_i * S_j` — negative, meaning increasing
  x_j decreases S_i (because softmax is a competition — more probability
  for one class means less for others).

This competitive structure is what makes softmax backward different from
a simple elementwise backward — each output depends on *all* inputs.

### 3.6 backwardCrossEntropy — The Elegant "Softmax Minus One-Hot"

**Forward:** `loss = -log_softmax(logits)[target_class] / B`

**Backward:** `∂L/∂logits = (softmax(logits) - one_hot(targets)) / B`

**Where it fires:** At the end of every training step, when computing the
loss gradient.

**Why this is the most beautiful backward in the library:** The gradient of
cross-entropy w.r.t. the logits is incredibly simple:

```
∂L/∂logits[i,j] = (S[i,j] - 1{j == target[i]}) / B
```

Where `S = softmax(logits)` and `1{j == target[i]}` is 1 if j is the target
class for sample i, else 0.

**Intuition:**
- If the model assigns high probability to the correct class (S[j] ≈ 1 for
  j=target), the gradient is near zero — the model is already right, no
  update needed.
- If the model assigns high probability to the *wrong* class (S[j] ≈ 1 for
  j≠target), the gradient for that class is large and positive — the
  optimizer will push logits[j] down to reduce its probability.
- The gradient for the correct class is negative (S - 1 < 0 when S < 1),
  so the optimizer pushes the correct logit up.

**This single backward function drives all of language model training.** The
gradient flows from the loss through logits, through the output projection,
through the transformer blocks, all the way back to the embedding table.
Every parameter gets updated based on this signal.

### 3.7 backwardSum / backwardMean — Gradient Broadcasting

**Forward:** `y = sum(x, axis)` or `y = mean(x, axis)`

**Backward:**
```
∂L/∂x = broadcastTo(∂L/∂y, x.shape)         [for sum]
∂L/∂x = broadcastTo(∂L/∂y / N, x.shape)     [for mean, where N = axis size]
```

**Where it fires:** In LayerNorm (Stage 4), which computes mean and variance
along the channel axis. Also in `sumAll`, which reduces a tensor to a scalar
loss.

**Intuition:** When you sum along an axis, every element along that axis
contributed equally. So the gradient flows back equally to all of them.
For mean, there's an additional 1/N factor — each element contributed 1/N
of the output, so it gets 1/N of the gradient.

**The broadcastTo function is the inverse of sumToShape:**
- `sumToShape`: collapse gradient from broadcast shape to original shape
- `broadcastTo`: expand gradient from reduced shape to original shape

They form a pair: forward broadcast → backward sumToShape; forward sum →
backward broadcastTo.

---

## 4. How the Backward Pass Flows Through a Transformer Block

Let's trace the backward pass through the same transformer block we traced
in `docs/02b`, showing how gradients flow from loss back to parameters.

### 4.1 Setting Up the Tape

```zig
var tape = Tape.init(allocator);
defer tape.deinit();

// Register all parameters that need gradients
_ = try tape.trackLeaf(&W_q);   // Query projection weight
_ = try tape.trackLeaf(&W_k);   // Key projection weight
_ = try tape.trackLeaf(&W_v);   // Value projection weight
_ = try tape.trackLeaf(&W_o);   // Output projection weight
_ = try tape.trackLeaf(&W1);    // MLP first linear weight
_ = try tape.trackLeaf(&W2);    // MLP second linear weight
_ = try tape.trackLeaf(&b_q);   // Query bias
// ... etc for all biases
```

Each `trackLeaf` creates a phantom node (n_parents=0) so the tape has a
place to accumulate the gradient for that parameter.

### 4.2 Forward Pass (Recording)

Each operation that involves a parameter with `requires_grad=true` records
a node on the tape. Let's count nodes for one block:

```
Q = x @ W_q + b_q       → 2 nodes: matmul, add
K = x @ W_k + b_k       → 2 nodes: matmul, add
V = x @ W_v + b_v       → 2 nodes: matmul, add
scores = Q @ K^T / √d   → 2 nodes: matmul, mulScalar
masked = scores * mask   → 1 node:  mul
attn = softmax(masked)   → 1 node:  softmax
context = attn @ V       → 1 node:  matmul
attn_out = context + x   → 1 node:  add (residual)
hidden = attn_out @ W1 + b1 → 2 nodes: matmul, add
activated = GELU(hidden)   → 1 node:  gelu
mlp_out = activated @ W2 + b2 → 2 nodes: matmul, add
block_out = mlp_out + attn_out → 1 node: add (residual)
logits = block_out @ W_out → 1 node: matmul
loss = crossEntropy(logits, targets) → 1 node: cross_entropy

Total: ~19 nodes on the tape
```

Each node stores its parent IDs and the data backward needs (saved tensors,
shapes, etc.). The total memory for the tape is proportional to the number
of nodes — O(graph size).

### 4.3 Backward Pass (Gradient Flow)

The tape walks these 19 nodes in reverse order. Here's the gradient flow:

```
∂L/∂loss = 1.0                    [seed]

∂L/∂logits = softmax(logits) - one_hot(targets)    [backwardCrossEntropy]
  │
  ├→ ∂L/∂W_out = block_outᵀ @ ∂L/∂logits          [backwardMatmul]
  ├→ ∂L/∂block_out = ∂L/∂logits @ W_outᵀ           [backwardMatmul]
  │
  ├→ ∂L/∂mlp_out += ∂L/∂block_out                   [backwardAdd — residual]
  │  ├→ ∂L/∂b2 = sumToShape(∂L/∂mlp_out, b2.shape) [backwardAdd — bias]
  │  ├→ ∂L/∂activated = ∂L/∂mlp_out @ W2ᵀ          [backwardMatmul]
  │  │  └→ ∂L/∂hidden = ∂L/∂activated * GELU'(hidden) [backwardGelu]
  │  │     ├→ ∂L/∂b1 = sumToShape(∂L/∂hidden, b1.shape) [backwardAdd]
  │  │     └→ ∂L/∂W1 = attn_outᵀ @ ∂L/∂hidden      [backwardMatmul]
  │  │        └→ ∂L/∂attn_out += ∂L/∂hidden @ W1ᵀ  [backwardMatmul]
  │
  ├→ ∂L/∂attn_out += ∂L/∂block_out                  [backwardAdd — residual]
  │  └→ ∂L/∂context = ∂L/∂attn_out                   [backwardAdd — passes through]
  │     └→ ∂L/∂attn_weights = ∂L/∂context @ Vᵀ      [backwardMatmul]
  │        ├→ ∂L/∂V = attn_weightsᵀ @ ∂L/∂context   [backwardMatmul]
  │        └→ ∂L/∂masked = backwardSoftmax(...)      [backwardSoftmax]
  │           └→ ∂L/∂scores = ∂L/∂masked * mask     [backwardMul]
  │              └→ ∂L/∂Q, ∂L/∂K = backwardMatmul(...) [QKV backward]
```

**Key observations:**

1. **Residual connections split the gradient.** At the `add` node for the
   residual, the gradient flows to both branches. Each branch gets a copy
   of the upstream gradient.

2. **The gradient passes through add unchanged.** `backwardAdd` doesn't
   scale the gradient — it just routes it (with sumToShape for broadcasting).
   This is why residuals work: the gradient from the loss reaches early
   layers without being multiplied by small numbers.

3. **Matmul backward is expensive.** Each matmul in the forward pass
   generates two matmuls in the backward pass (one for each input). With
   8 forward matmuls, that's 16 backward matmuls. **Backward is ~2× the
   cost of forward.**

4. **Cross-entropy's gradient is simple but powerful.** The "softmax minus
   one-hot" formula produces the signal that drives all parameter updates.

---

## 5. The Multivariable Chain Rule in Practice

When a tensor is used in multiple operations, its gradient is the **sum**
of contributions from each use. This is the multivariable chain rule, and
it's critical for understanding two transformer features:

### 5.1 Residual Connections

In `block_out = mlp_out + attn_out`, `attn_out` is used in both the
addition and the MLP (as input to the MLP's first linear layer). The
gradient of `attn_out` gets two contributions:

```
∂L/∂attn_out = ∂L/∂block_out (from the add) + ∂L/∂hidden @ W1ᵀ (from the MLP)
```

Our `accumulateGrad` function handles this automatically — when a second
gradient arrives for a tensor that already has a gradient, we add them.

### 5.2 Self-Attention: Q, K, V Share the Same Input

In a single-head transformer, Q, K, and V are all computed from the same
input x:

```
Q = x @ W_q
K = x @ W_k
V = x @ W_v
```

The gradient of x is the sum of gradients from all three paths:

```
∂L/∂x = ∂L/∂Q @ W_qᵀ + ∂L/∂K @ W_kᵀ + ∂L/∂V @ W_vᵀ
```

This is three backwardMatmul calls, all accumulating into the same gradient
slot for x. The tape handles this via `accumulateGrad`.

---

## 6. Memory Management: Why Stage 3 Is the Memory Bottleneck

The backward pass requires storing the forward pass's intermediate tensors
(the "activations") so that backward functions can access them. This is the
major memory cost of training.

### 6.1 What the Tape Stores

For a 1-block transformer with B=4, T=8, C=64:

| Saved Data | Size | Why backward needs it |
|-----------|------|----------------------|
| Input x for QKV matmul | 4×8×64 = 2048 f32 | ∂L/∂W = xᵀ @ ∂L/∂output |
| Q, K, V for attention | 3×2048 = 6144 f32 | ∂L/∂scores = Q, K for matmul backward |
| Attention weights | 4×8×8 = 256 f32 | ∂L/∂scores uses S in softmax backward |
| MLP activations (for GELU) | 4×8×256 = 8192 f32 | ∂L/∂hidden = ∂L/∂output * GELU'(hidden) |
| **Total activations** | **~16,640 f32** | **~65 KB** |

For comparison, the model parameters are ~50K f32 (~200 KB). So the
activations are about 1/3 of the parameter memory — manageable for a tiny
model, but for larger models (GPT-2: 1.5B parameters), activations can
exceed parameter memory.

### 6.2 Gradient Memory

The tape also allocates gradient tensors:

| Gradient | Size | Who allocates it |
|---------|------|------------------|
| ∂L/∂logits (2,4) | 8 f32 | backwardCrossEntropy |
| ∂L/∂W_q (64,64) | 4096 f32 | backwardMatmul |
| ∂L/∂W_k (64,64) | 4096 f32 | backwardMatmul |
| ... (one per parameter) | ... | ... |
| **Total gradient memory** | **~50K f32** | **~200 KB** |

Gradients are freed when `tape.deinit()` is called (or eagerly for
intermediates when `retain_graph=false`).

### 6.3 The `retain_graph=false` Optimization

By default, the tape frees intermediate gradient tensors as soon as all
children have been processed. This bounds gradient memory to O(depth)
instead of O(width):

```
Processing node 5: free ∂L/∂node5 after accumulating into parents
Processing node 4: free ∂L/∂node4 after accumulating into parents
...
Only leaf gradients (for parameters) survive until tape.deinit()
```

---

## 7. Gradient Checking: Proving Your Backward Is Correct

The `gradCheck` function uses **finite differences** to verify that the
analytical gradients (from backward functions) match numerical gradients:

```
numerical_grad[i] = (L(x + h·e_i) - L(x - h·e_i)) / (2h)
```

### 7.1 Why This Matters for Transformers

Transformer backward passes are complex — matmul backward involves transposes,
softmax backward involves a "diag minus outer" formula, cross-entropy backward
fuses softmax and NLL loss. Any bug in these formulas produces silently wrong
gradients, which means the optimizer updates parameters in the wrong direction,
and the model fails to train.

**Gradient checking catches these bugs.** It's the gold standard for autograd
correctness. Every major framework (PyTorch, JAX, TensorFlow) has a built-in
gradcheck function, and we use it the same way.

### 7.2 How to Use It in Our Library

```zig
// 1. Define a loss function
fn myLoss(allocator: std.mem.Allocator, params: []*Tensor, tape: *Tape) !*Tensor {
    var logits = try ops.matmul.matmul(allocator, X, params[0].*, tape);
    var loss = try ops.reduce.sumAll(allocator, logits, tape);
    const ptr = try allocator.create(Tensor);
    ptr.* = loss;
    return ptr;
}

// 2. Run forward + backward
var tape = Tape.init(allocator);
_ = try tape.trackLeaf(&W);
var loss = try myLoss(allocator, &params, &tape);
try tape.backward(loss);

// 3. Check gradients numerically
var params = [_]*Tensor{&W};
const result = try gradCheck(allocator, myLoss, &params, 1e-3, 1e-2, 5);
try std.testing.expect(result.passed);
```

### 7.3 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All indices fail | Forward bug, wrong loss shape | Verify loss is scalar (1,) |
| Specific op fails | Backward formula bug | Check the backward rule against calculus |
| High error with small eps | Finite-difference precision | Increase eps to 1e-3 or 1e-2 |
| High error with large eps | Second-order effects | Decrease eps to 1e-4 or 1e-5 |
| Intermittent failures | NaN/Inf in forward | Check for numerical instability |

---

## 8. How Autograd Connects to the Optimizer (Stage 4 Preview)

After `tape.backward(&loss)`, every parameter tensor has its `.grad` field
filled in. The optimizer (Stage 4) uses these gradients to update the
weights:

### SGD (Stochastic Gradient Descent)

```
W = W - lr × ∂L/∂W
```

Each weight moves in the direction that reduces the loss, scaled by the
learning rate. Simple, but effective for small models.

### AdamW (what GPT-2 uses)

AdamW maintains per-parameter running averages of gradients and squared
gradients, and adjusts the learning rate for each parameter individually:

```
m = β₁ × m + (1-β₁) × ∂L/∂W          (first moment — momentum)
v = β₂ × v + (1-β₂) × (∂L/∂W)²       (second moment — adaptive lr)
W = W - lr × m / (√v + ε) - lr × λ × W (weight decay)
```

All of these operations are implemented using Stage 2 tensor ops:
- `mulScalar` for learning rate scaling
- `add` for momentum accumulation
- `mul` for squared gradients
- `div` for the adaptive scaling

### The Full Training Loop with Autograd

```zig
// Pseudocode for the complete training loop
for (0..num_steps) |step| {
    // 1. Create a fresh tape for this step
    var tape = Tape.init(allocator);
    defer tape.deinit();

    // 2. Register parameters
    for (model.parameters()) |param| {
        _ = try tape.trackLeaf(param);
    }

    // 3. Forward pass (ops record on tape automatically)
    var logits = try model.forward(allocator, batch.inputs, &tape);
    var loss = try ops.loss.crossEntropy(allocator, logits, batch.targets, &tape);

    // 4. Backward pass (fills .grad on all parameters)
    try tape.backward(&loss);

    // 5. Optimizer step (uses .grad to update weights)
    try optimizer.step(model.parameters());

    // 6. Zero gradients for next step
    tape.zeroGrad(model.parameters());

    // 7. Tape is deinited (frees gradient tensors)
    // 8. Next iteration creates a fresh tape
}
```

The autograd engine is the bridge between "the model is wrong" (loss) and
"here's how to fix it" (gradients → optimizer updates).

---

## 9. PyTorch Equivalents for Autograd

If you've used PyTorch's autograd, here's how our Stage 3 API maps:

| Our Library | PyTorch Equivalent | Notes |
|-------------|--------------------|-------|
| `tape = Tape.init(allocator)` | (implicit — PyTorch creates graph automatically) | Explicit tape, explicit allocator |
| `tape.trackLeaf(&W)` | `W.requires_grad = True` | We also require trackLeaf for tape bookkeeping |
| `try elementwise.mul(alloc, a, b, &tape)` | `c = a * b` | PyTorch records if any input requires grad |
| `try elementwise.mul(alloc, a, b, null)` | `with torch.no_grad(): c = a * b` | Pass `null` to skip recording |
| `try tape.backward(&loss)` | `loss.backward()` | Same semantics, same gradient computation |
| `W.grad` | `W.grad` | Same field, same meaning |
| `tape.zeroGrad(&params)` | `optimizer.zero_grad()` | We split tape and optimizer duties |
| `tape.deinit()` | (automatic garbage collection) | Must call explicitly — no GC in Zig |
| `tape.retain_graph = true` | `loss.backward(retain_graph=True)` | Needed for double backward |
| `gradCheck(alloc, fn, &params, eps, tol, n)` | `torch.autograd.gradcheck(fn, params, eps, atol)` | Same algorithm |
| `SavedData.tensor_pair` | `ctx.saved_tensors` (PyTorch Function) | Our version stores by value, not pointer |
| `OpKind.mul` | `torch.autograd.Function` subclass | We use enum dispatch, not inheritance |
| `accumulateGrad(dst, src)` | `grad += new_grad` (automatic) | We must call explicitly |

### Key Differences

1. **Enum dispatch vs virtual dispatch.** PyTorch uses Python's dynamic
   dispatch (or C++ virtual functions) to route backward calls. We use a
   simple `switch (node.op)` on an enum. This gives compile-time safety
   (missing cases are errors) and better performance (no vtable lookup).

2. **Explicit tape vs implicit graph.** PyTorch builds the graph implicitly
   whenever a tensor with `requires_grad=True` is involved in an operation.
   We require passing `&tape` to every op. This is more verbose but makes
   the recording decision explicit and allows `null` for inference mode.

3. **By-value SavedData vs by-reference.** PyTorch's `ctx.save_for_backward`
   stores tensor references that the autograd engine keeps alive. Our
   SavedData stores Tensor structs by value — the `data` slice shares the
   original buffer, but the shape/strides are copies. This prevents dangling
   pointers from by-value function parameters.

4. **No Python GC.** PyTorch relies on Python's garbage collector to free
   gradient tensors. We free them explicitly in `tape.deinit()` or eagerly
   during backward (when `retain_graph=false`).

---

## 10. Common Mistakes

1. **Forgetting `trackLeaf` before the forward pass.** If you call
   `tape.trackLeaf` after the op that uses the parameter, the parameter
   won't have a `tape_node` at recording time, so no gradient will be
   computed for it. Call `trackLeaf` *before* any op that uses the tensor.

2. **Using the same tape for multiple backward calls.** By default
   (`retain_graph=false`), intermediate gradient tensors are freed during
   backward. Calling `tape.backward()` twice is an error. Create a fresh
   tape for each training step.

3. **Deiniting forward tensors before backward.** The SavedData snapshots
   share the `data` buffer with the original tensors. If you `deinit` an
   intermediate tensor before `tape.backward()`, the saved data becomes
   dangling — you'll get garbage gradients or a segfault.

4. **Not freeing intermediate tensors after backward.** The tape owns the
   *gradient* tensors, but the *forward* tensors (intermediate results like
   `logits`, `hidden`, etc.) are owned by the caller. You must deinit them
   after backward to avoid memory leaks. Use `defer deinit(allocator)`.

5. **Passing `null` tape when you need gradients.** If you pass `null`
   instead of `&tape` to any op, that op won't be recorded on the tape,
   and the gradient chain will be broken. The downstream gradients will
   be computed as if that op doesn't exist — silently wrong.

6. **Expecting gradients for non-leaf tensors.** Only tensors registered
   via `trackLeaf` get their `.grad` field populated. Intermediate tensors
   (results of ops) don't get `.grad` — their gradients live in the tape's
   `grad_map` and are freed after backward. If you need an intermediate's
   gradient, use `retain_graph=true` and access it from the tape.

7. **Confusing sumToShape and broadcastTo.** These are inverses:
   - Forward `broadcast` → backward `sumToShape` (collapse gradient)
   - Forward `sum/mean` → backward `broadcastTo` (expand gradient)
   Swapping them produces shape errors or silently wrong gradients.

8. **Modifying tensor data after recording on tape.** Because SavedData
   snapshots share the `data` buffer, modifying a tensor after it's been
   recorded changes what the backward function sees. Always complete the
   forward pass before modifying any tensor data.


---

## Exercises

**Exercise 1.** During backward through a residual branch
`h = x + layer(x)`, the gradient flowing back into `x` is what?

<details><summary>Solution</summary>

Both branches contribute, summed. `dh/dx_left = 1` (identity
through the add), so `dL/dx` from the left branch is just
`dL/dh`. `dh/dx_right = dLayer/dx` (whatever `layer` does),
so `dL/dx` from the right branch is
`dL/dh * dLayer/dx`. Total: `dL/dx = dL/dh + dL/dh * dLayer/dx`.

This is why residual connections are powerful - they guarantee at
least the identity-path gradient reaches every layer, so very deep
stacks can still train. Without the skip, only the right branch's
`dLayer/dx` contributes and that shrinks geometrically with
depth.

</details>

**Exercise 2.** The tape stores `SavedData` snapshots for each op's
backward. For `softmax` specifically, what does the backward need
to save from the forward?

<details><summary>Solution</summary>

The softmax output itself. Given `y = softmax(x)`, the backward
with respect to `x` is:

`
dL/dx_i = y_i * (dL/dy_i - sum_j dL/dy_j * y_j)
`

This depends on `y`, not on `x`. So the forward stashes a
snapshot of `y`, and the backward reads it. Stashing `x` would
also work (then you'd recompute `y` in the backward) but costs
two extra exponentials per element - wasteful.

</details>
