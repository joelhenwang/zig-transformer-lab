# Chapter 3: Tape-Based Autograd

This chapter explains how `zig-transformer-lab` implements reverse-mode automatic
differentiation — the algorithm that makes neural network training possible.
Every gradient computed during training flows through the autograd engine
described here.

## Why Autograd?

Training a neural network means adjusting its parameters to minimize a loss
function. To know *which direction* to adjust, we need the **gradient** — the
derivative of the loss with respect to each parameter.

For a network with millions of parameters, computing these derivatives by
hand is impossible. **Automatic differentiation** (autograd) computes them
mechanically, using the chain rule of calculus applied to the computation
graph.

There are two main approaches:

| Approach | Direction | Memory | Use case |
|----------|-----------|--------|----------|
| Forward-mode | Input → Output | O(1) | Jacobian-vector products |
| **Reverse-mode** | Output → Input | O(graph size) | **Gradient of scalar loss** |

We use **reverse-mode** (backpropagation) because our loss is always a scalar
— one number that measures "how wrong" the model is. Reverse-mode computes
*all* parameter gradients in a single backward pass, which is exactly what
training needs.

## The Big Picture

```
Forward pass:                  Backward pass:
  a ──┐                          dL/da ◄──┐
      ├── mul ── d ──┐              dL/dd ◄──┤
  b ──┘              ├── add ── e ──┤        ├── dL/de ◄──┐
                     c ──────────────┘        │           ├── dL/dloss = 1
                           loss = e * e ──────┘
```

**Forward**: Operations execute in order, recording what they did on a **tape**.

**Backward**: The tape is walked in reverse. Each operation's **backward rule**
computes how the gradient flows from the output back to the inputs.

The key mathematical insight is the **chain rule**:

```
dL/dx = dL/dy * dy/dx
```

If `y = f(x)` and we know `dL/dy` (the gradient coming from downstream),
then `dL/dx = dL/dy * f'(x)` (multiply by the local derivative).

For a tensor operation, this becomes a **vector-Jacobian product (VJP)**:
the gradient of the loss with respect to each input element is computed by
multiplying the upstream gradient by the operation's local Jacobian matrix.

## The Tape

The tape is a list of **nodes**, each recording one operation from the forward
pass. Think of it as a flight data recorder: it captures everything needed to
replay the computation in reverse.

### Tape Lifecycle

```zig
// 1. Create a fresh tape for this training step
var tape = Tape.init(allocator);
defer tape.deinit();

// 2. Register leaf tensors that need gradients
_ = try tape.trackLeaf(&weight);

// 3. Forward pass: ops record themselves on the tape automatically
var logits = try ops.matmul(allocator, input, weight, &tape);
var loss = try ops.crossEntropy(allocator, logits, targets, &tape);

// 4. Backward pass: compute all gradients
try tape.backward(&loss);

// 5. Access gradients
if (weight.grad) |g| {
    // g.data[i] contains dL/d_weight[i]
}

// 6. After optimizer step, deinit the tape (frees all gradient tensors)
```

**One tape per training step** is the intended usage. After the optimizer
updates the parameters, create a fresh tape for the next step.

### Why Not Reuse the Tape?

The tape stores **borrowed references** to the forward-pass tensors (via
`SavedData`). If we reused the tape across steps, those references would
dangle when the forward-pass tensors are freed. A fresh tape each step
guarantees all references are valid for the duration of the backward pass.

## Nodes and SavedData

Each node on the tape records:

1. **What operation** was performed (`OpKind` — add, mul, matmul, relu, etc.)
2. **Which tensors** were the inputs (parent node IDs)
3. **What data** the backward pass needs (saved tensors, shapes, scalars)

### OpKind Enum

```zig
pub const OpKind = enum {
    add, sub, mul, div,           // elementwise binary
    add_scalar, mul_scalar,       // elementwise scalar
    matmul, matmul_batch,         // linear algebra
    transpose2d, reshape,         // shape transforms
    sum, mean,                    // reductions
    exp, log, neg, relu, gelu,    // unary activations
    softmax, log_softmax,         // normalization
    cross_entropy,                // loss
    embedding,                    // lookup (Stage 4)
};
```

The exhaustive switch in the backward dispatch gives **compile-time safety**:
adding a new `OpKind` without a corresponding backward case is a compilation
error. You cannot forget to implement a backward.

### SavedData: Storing What Backward Needs

Different operations need different data for their backward pass:

| OpKind | SavedData variant | What's stored | Why backward needs it |
|--------|-------------------|---------------|----------------------|
| add, sub | tensor_pair | Both input tensor snapshots | Shapes for sumToShape |
| mul, div | tensor_pair | Both input tensor snapshots | Values for element-wise gradient |
| matmul | tensor_pair | Both input tensor snapshots | Values for A^T, B^T |
| exp, relu, gelu | tensor_ref | Input tensor snapshot | Values for derivative formula |
| neg | nothing | (nothing) | Backward is just negate |
| sum, mean | reduce_info | Input shape + axis | Shape for broadcastTo |
| add_scalar | tensor_scalar | Input shape + scalar value | Shape for sumToShape |
| cross_entropy | ce_info | Logits snapshot + target slice | Recompute softmax for gradient |

**Critical design decision: Tensor snapshots by value, not by pointer.**

The `tensor_ref` and `tensor_pair` variants store entire `Tensor` structs **by
value** (snapshots), not `*Tensor` pointers. This is essential because:

1. All ops take `Tensor` by value (Zig convention for struct parameters).
2. Storing `@constCast(&parameter)` would create a dangling pointer — the
   parameter is a stack-local copy that's destroyed when the op returns.
3. By storing the whole struct, we capture the `data` slice header (which
   still points to the original heap buffer) along with shape and strides.

The `data` slice in the snapshot shares the **same heap buffer** as the
original tensor. This buffer is alive as long as the caller's tensor hasn't
been deinited. The contract: **tensors that participate in the computation
graph must outlive the tape.** This is the same contract as PyTorch's autograd.

## trackLeaf: Registering Parameters

Leaf tensors (parameters like weights and biases) start with `tape_node = null`.
Without a node ID, the backward pass has nowhere to accumulate their gradient.
`trackLeaf()` creates a phantom node for the leaf:

```zig
var w = try Tensor.init(allocator, Shape.init2D(3, 4));
w.requires_grad = true;
const w_id = try tape.trackLeaf(&w);
// w.tape_node is now set; backward will populate w.grad
```

The phantom node has `n_parents = 0`, so the backward pass skips it (no
backward rule to run). But its entry in `grad_map` gets populated by child
nodes, and after backward, `w.grad` points to the computed gradient tensor.

## Backward Pass: Step by Step

The `tape.backward(loss)` function implements reverse-mode autograd:

### Step 1: Seed the Gradient

The gradient of the loss with respect to itself is 1.0:

```zig
var seed = try Tensor.init(allocator, loss.shape);
seed.fill(1.0);
try grad_map.put(loss_id, seed_ptr);
```

For a scalar loss (shape `(1,)`), the seed is `[1.0]`. For a non-scalar loss,
every element gets gradient 1.0 (which corresponds to the gradient of the
*sum* of the loss elements).

### Step 2: Topological Sort

Starting from the loss node, we perform a depth-first search to collect all
reachable nodes. The post-order traversal gives us a valid topological order
where a node appears after all its dependencies.

```
DFS from loss:
  visit loss → visit its parents → ... → visit leaves

Post-order: [leaf_a, leaf_b, op_mul, leaf_c, op_add, op_mul_loss]
```

### Step 3: Walk in Reverse

We walk the topological order **backwards** — from the loss node to the leaves.
This ensures that when we process a node, all nodes that contribute to its
gradient have already been processed.

```zig
var i: usize = topo_order.items.len;
while (i > 0) {
    i -= 1;
    const node_id = topo_order.items[i];
    const node = self.nodes.items[node_id];

    // Skip leaves (no backward rule)
    if (node.n_parents == 0) continue;

    // Look up the gradient of this node's output
    const grad_out = self.grad_map.get(node_id) orelse continue;

    // Compute input gradients
    const parent_grads = try backward_mod.backward(allocator, node, grad_out);

    // Accumulate into parents' gradient slots
    for (0..node.n_parents) |pi| {
        const parent_id = node.parents[pi] orelse continue;
        const parent_grad = parent_grads[pi] orelse continue;

        if (self.grad_map.get(parent_id)) |existing| {
            // Multivariable chain rule: gradient += contribution
            try backward_mod.accumulateGrad(existing, parent_grad);
            parent_grad.deinit(allocator);
            allocator.destroy(parent_grad);
        } else {
            try self.grad_map.put(parent_id, parent_grad);
        }
    }
}
```

### Step 4: Write Gradients to Leaves

After processing all nodes, we copy gradient pointers from `grad_map` back to
the leaf tensors' `.grad` fields:

```zig
var leaf_iter = self.leaf_map.iterator();
while (leaf_iter.next()) |entry| {
    const leaf_id = entry.key_ptr.*;
    const tensor_ptr = entry.value_ptr.*;
    if (self.grad_map.get(leaf_id)) |grad| {
        tensor_ptr.grad = grad;
    }
}
```

### Step 5: Free Intermediate Gradients

When `retain_graph = false` (the default), we eagerly free intermediate
gradient tensors after their children have been processed. This bounds memory
usage to O(depth) instead of O(width):

```zig
if (node.n_parents > 0 and node_id != loss_id) {
    if (self.grad_map.fetchRemove(node_id)) |kv| {
        kv.value.deinit(allocator);
        allocator.destroy(kv.value);
    }
}
```

## Gradient Formulas

Each backward function implements a specific calculus rule. Here are the most
important ones:

### Elementwise Operations

**add(a, b) → c**: The gradient flows through unchanged.

```
dL/da = sumToShape(dL/dc, a.shape)
dL/db = sumToShape(dL/dc, b.shape)
```

The `sumToShape` reduces the gradient along broadcast dimensions. If `a` was
broadcast from `(1,3)` to `(2,3)`, we sum along axis 0 to get back to `(1,3)`.

**mul(a, b) → c = a * b**: The product rule.

```
dL/da = sumToShape(dL/dc * b, a.shape)
dL/db = sumToShape(dL/dc * a, b.shape)
```

Each input's gradient is the upstream gradient times the other input (the
"other factor" in the product rule), then reduced to the input's shape.

**div(a, b) → c = a / b**: Quotient rule.

```
dL/da = sumToShape(dL/dc / b, a.shape)
dL/db = sumToShape(-dL/dc * a / b², b.shape)
```

### Matrix Multiplication

**matmul(A, B) → C = A @ B**: The most important backward in transformers.

```
dL/dA = dL/dC @ Bᵀ
dL/dB = Aᵀ @ dL/dC
```

This follows directly from the matrix calculus identity. Every linear layer
and attention score computation uses this backward. The transpose swaps the
inner dimension to align the dot products correctly.

### Activation Functions

**relu(a) → c = max(0, a)**: Gradient is 1 for positive inputs, 0 otherwise.

```
dL/da = dL/dc * (a > 0 ? 1 : 0)
```

**gelu(a) → c = 0.5 * a * (1 + erf(a / √2))**: Smooth gate.

```
GELU'(x) = 0.5 * (1 + erf(x/√2)) + x * exp(-x²/2) / √(2π)
```

The first term is the CDF of the standard normal; the second is the PDF
times x. We use the Abramowitz & Stegun polynomial approximation for erf.

**exp(a) → c = e^a**: Self-referential derivative.

```
dL/da = dL/dc * e^a
```

### Softmax

**softmax(x) → S**: The "diag - outer" formula.

```
dL/dx = S * (dL/dS - (dL/dS · S) * 1)    [row-wise, last axis]
```

This comes from the Jacobian of softmax:

```
∂S_i/∂x_j = S_i(δ_ij - S_j)
```

So the VJP is `dL/dx_j = S_j * (dL/dS_j - Σ_i S_i * dL/dS_i)`.

We compute `dot = sum(dL/dS * S, last_axis)`, then `dL/dx = S * (dL/dS - dot)`.

### Cross-Entropy Loss

**cross_entropy(logits, targets) → loss**: The famous "softmax - 1" backward.

```
dL/dlogits = (softmax(logits) - one_hot(targets)) / B
```

Where `B` is the batch size. This is incredibly efficient — no Jacobian
needed! The gradient is just the softmax probabilities minus a one-hot
vector at the target class.

Intuition: if the model assigns high probability to the wrong class, the
gradient is large and negative for that class (pushing it down), while the
correct class gets a positive gradient (pushing it up).

## Broadcasting in Backward

Broadcasting in the forward pass creates a complication in the backward
pass: the output shape may be larger than an input shape. The gradient of
the output has the larger shape, but we need to reduce it back to the
input's shape.

### sumToShape: Collapsing Broadcast Dimensions

When `a` of shape `(1,3)` is broadcast to `(2,3)` in the forward pass, the
gradient of the output has shape `(2,3)`. To get the gradient for `a`, we
sum along the broadcast dimension (axis 0):

```
grad_a = sumToShape(grad_out: (2,3), target: (1,3))
       = sum(grad, axis=0)  →  (1,3)
```

### broadcastTo: Expanding Reduced Dimensions

When `sum(a, axis=0)` reduces `(2,3)` to `(1,3)`, the backward needs to
expand the gradient back to `(2,3)` by repeating values along the reduced
axis:

```
dL/da = broadcastTo(dL/d_sum: (1,3), target: (2,3))
```

Every element along the reduced axis contributed equally to the sum, so the
gradient flows back equally to all of them.

## The Multivariable Chain Rule

If a tensor is used in **multiple operations**, its gradient is the **sum**
of contributions from each use. This is the multivariable chain rule:

```
dL/da = dL/da (from op1) + dL/da (from op2) + ...
```

Example: `loss = e * e` where `e` is used as both inputs to the multiplication.

```
dL/de = dL/dloss * e (from input a) + dL/dloss * e (from input b)
      = 2 * e
```

The `accumulateGrad` function implements this: `dst += src`. When the tape
processes a node whose parent already has a gradient, we add the new
contribution instead of overwriting.

## Memory Management

### Who Owns What?

| Data | Owner | Freed when |
|------|-------|------------|
| Forward-pass tensors | The caller (user code) | After backward + optimizer step |
| Gradient tensors | The tape (grad_map) | In `tape.deinit()` or eagerly during backward |
| SavedData tensor snapshots | The tape (stored by value in nodes) | When the tape's ArrayList is freed |
| Backward temp tensors | The backward functions | With `defer deinit` inside each backward |

### The `heapAlloc` Pattern

Backward functions return `[2]?*Tensor` — an array of pointers. Every
gradient tensor must be heap-allocated so it can be stored in `grad_map`
or accumulated into an existing entry:

```zig
fn heapAlloc(allocator: std.mem.Allocator, tensor: Tensor) !*Tensor {
    const ptr = try allocator.create(Tensor);
    ptr.* = tensor;
    return ptr;
}
```

### Gradient Cleanup for Null Parents

When an input doesn't require a gradient (e.g., fixed input `X` in a matmul),
its `tape_node` is null. The backward function still computes the gradient
for this input (the backward rule doesn't know about grad requirements), but
the accumulation loop skips null parent IDs. Without explicit cleanup, these
gradient tensors would leak. After the accumulation loop, we free any
parent gradients whose parent ID was null:

```zig
for (0..node.n_parents) |pi| {
    const parent_grad = parent_grads[pi] orelse continue;
    const parent_id = node.parents[pi] orelse {
        parent_grad.deinit(allocator);
        allocator.destroy(parent_grad);
        continue;
    };
    _ = parent_id;
}
```

## Gradient Checking

The `gradCheck` function verifies analytical gradients against numerical
gradients computed via **central finite differences**:

```
numerical_grad[i] = (L(x + h·e_i) - L(x - h·e_i)) / (2h)
```

Where `e_i` is the unit vector in direction `i` and `h` is a small
perturbation (typically 1e-3 to 1e-5).

The relative error is:

```
rel_error = |analytical - numerical| / max(|analytical|, |numerical|, 1e-8)
```

The `max(..., 1e-8)` in the denominator avoids division by zero when both
gradients are near zero.

### How to Use gradCheck

```zig
// 1. Run forward + backward to get analytical gradients
var tape = Tape.init(allocator);
_ = try tape.trackLeaf(&w);
var loss = try myLoss(allocator, &w, &tape);
try tape.backward(&loss);

// 2. Run gradient check
var params = [_]*Tensor{&w};
const result = try gradCheck(allocator, myLossFn, &params, 1e-3, 1e-2, 5);
//                        allocator, loss_fn, params, eps,  tol,  n_samples

if (result.passed) {
    print("All {} sampled indices passed!\n", .{result.n_sampled});
}
```

### When Gradients Disagree

If `gradCheck` reports failures:

1. **Check the loss function**: Is it a scalar? `gradCheck` reads only
   `loss.data[0]`, so a non-scalar loss will give wrong numerical gradients.
2. **Check broadcasting**: Mismatches between `sumToShape`/`broadcastTo` and
   the actual forward-pass broadcast are the #1 source of gradient bugs.
3. **Reduce epsilon**: If `h` is too large, the finite-difference approximation
   has O(h²) error. Try 1e-4 or 1e-5.
4. **Check for in-place modifications**: If a tensor is modified after being
   recorded on the tape, the saved data will reflect the new values, not the
   values at the time of the forward pass.

## Worked Example: Scalar Autograd

```
loss = (a * b + c)²

With a=2, b=3, c=1:
  a*b = 6, loss = (6+1)² = 49

Gradients (by hand):
  d/da = 2*(a*b + c) * b = 2*7*3 = 42
  d/db = 2*(a*b + c) * a = 2*7*2 = 28
  d/dc = 2*(a*b + c)     = 2*7   = 14
```

The tape records:
1. Node 0: leaf (a, tape_node=0)
2. Node 1: leaf (b, tape_node=1)
3. Node 2: leaf (c, tape_node=2)
4. Node 3: mul, parents=[0,1], saved={a: a, b: b}
5. Node 4: add, parents=[3,2], saved={a: d, b: c}
6. Node 5: mul, parents=[4,4], saved={a: e, b: e}

Backward (reverse order: 5, 4, 3):

**Node 5** (loss = e * e):
```
grad_out = [1.0]
tp.a = e (value 7.0), tp.b = e (value 7.0)
dL/de (from a) = 1.0 * 7.0 = 7.0
dL/de (from b) = 1.0 * 7.0 = 7.0
grad_map[4] += 7.0 + 7.0 = 14.0
```

**Node 4** (e = d + c):
```
grad_out = [14.0]
tp.a = d, tp.b = c
dL/dd = sumToShape([14.0], d.shape) = [14.0]
dL/dc = sumToShape([14.0], c.shape) = [14.0]
grad_map[3] = [14.0]
grad_map[2] = [14.0]
```

**Node 3** (d = a * b):
```
grad_out = [14.0]
tp.a = a (value 2.0), tp.b = b (value 3.0)
dL/da = sumToShape([14.0 * 3.0], a.shape) = [42.0]
dL/db = sumToShape([14.0 * 2.0], b.shape) = [28.0]
grad_map[0] = [42.0]
grad_map[1] = [28.0]
```

Final: a.grad = [42.0], b.grad = [28.0], c.grad = [14.0] ✓

## Worked Example: Matmul + Softmax

For a 2x3 input X and 3x4 weight W:

```
logits = X @ W           (2,4)
probs   = softmax(logits) (2,4)
loss    = sumAll(probs²)  (1,)
```

Backward chain:

1. **sumAll backward**: broadcast [1.0] to (2,4) → gradient of squared is (2,4) all-ones
2. **mul(probs, probs) backward**: dL/dprobs = 2 * probs * grad (product rule for a²)
3. **softmax backward**: dL/dlogits = probs * (dL/dprobs - sum(dL/dprobs * probs, last_axis))
4. **matmul backward**: dL/dW = Xᵀ @ dL/dlogits

The matmul backward is the most expensive step — it's another matmul! This
is why matrix multiplication dominates training time.

## The Code Structure

```
src/autograd/
├── node.zig        OpKind enum, SavedData union, Node struct
├── tape.zig        Tape struct: init, record, trackLeaf, backward, zeroGrad
├── backward.zig    All backward implementations + helpers
└── gradcheck.zig   Finite-difference gradient checker
```

### Key Invariants

1. **Exhaustive switch**: Adding a new `OpKind` without a backward case is
   a compile error. The compiler is your safety net.

2. **SavedData by value**: Tensor snapshots (not pointers) prevent dangling
   references to stack-local by-value parameters.

3. **One tape per step**: The tape borrows forward-pass tensor data. After
   the step, deinit the tape and create a fresh one.

4. **Gradient accumulation**: If a tensor feeds into multiple ops, its
   gradient is the sum of contributions. This implements the multivariable
   chain rule automatically.

5. **Leaves have phantom nodes**: `trackLeaf` creates a zero-parent node
   so the accumulation loop has a place to store the gradient.

## Common Mistakes

1. **Forgetting `trackLeaf`**: If you don't call `trackLeaf(&param)`, the
   parameter won't get a node ID, and backward won't compute its gradient.
   The symptom: `param.grad` is null after backward.

2. **Deiniting tensors before backward**: If you free an intermediate
   tensor before calling `tape.backward()`, the SavedData snapshots'
   `data` slices point to freed memory. The symptom: garbage gradients
   or segfaults.

3. **Using non-scalar loss with gradCheck**: `gradCheck` reads only
   `loss.data[0]`. If your loss is shape (2,1) instead of (1,), the
   numerical gradient considers only the first element while the
   analytical gradient considers the sum. The symptom: gradcheck
   fails with high relative error.

4. **Not freeing intermediate gradient tensors for null parents**: When
   an input doesn't require grad (`tape_node = null`), the backward
   function still computes its gradient, but the accumulation loop
   skips it. Without explicit cleanup, these gradient tensors leak.
   Fixed by checking for null parent IDs after the accumulation loop.

5. **Storing pointers to by-value parameters in SavedData**: This was
   our most subtle bug. When an op takes `Tensor` by value, storing
   `@constCast(&parameter)` creates a dangling pointer — the parameter
   is a stack-local copy destroyed when the function returns. The fix:
   store Tensor by value (snapshot) in SavedData, capturing the `data`
   slice header which still points to the original heap buffer.

6. **Topological sort order**: The topo sort must visit children before
   parents. Walking in reverse topological order ensures that each node's
   output gradient has been fully accumulated by the time we process it.

7. **Broadcasting backward shape mismatch**: Using `broadcastTo` where
   `sumToShape` is needed (or vice versa) is the #1 source of shape
   errors. Rule: forward `broadcast` → backward `sumToShape`; forward
   `sum/reduce` → backward `broadcastTo`.

8. **Mutating forward-pass tensors after recording**: If you modify a
   tensor's data after it's been recorded on the tape, the SavedData
   snapshot shares the same `data` buffer, so the backward will see the
   modified values. This silently produces wrong gradients.

9. **Double backward without `retain_graph`**: Calling `tape.backward()`
   twice with `retain_graph = false` is an error — intermediate gradient
   tensors have been freed. Set `retain_graph = true` if you need multiple
   backward passes.

10. **Missing `requires_grad` on intermediates**: The `recordBinaryOp`
    function sets `out.requires_grad = true` and `out.tape_node = node_id`
    only if at least one input requires grad. If you forget to set
    `requires_grad = true` on your leaf parameters, no nodes will be
    recorded, and `tape.backward()` will find an empty graph.
