# 02 — Tensors, Strides, Broadcasting, and Softmax Stability

This chapter explains how tensors work at the memory level — the foundation
of every deep learning framework.  We derive row-major strides from first
principles, show how broadcasting avoids unnecessary copies, and explain why
the "max subtraction" trick makes softmax numerically stable.

All examples use the actual code from `zig-transformer-lab` (Stage 2).

---

## 1. What Is a Tensor?

A tensor is three things:

1. **A flat buffer of f32 values** — `[]f32`, stored contiguously in memory
2. **A shape** — the logical dimensions, e.g., `(2, 3)` for a 2×3 matrix
3. **Strides** — how many f32 elements to skip to move one step along each
   dimension

That's it.  There is no magic.  PyTorch's `torch.Tensor`, NumPy's
`ndarray`, and our `Tensor` struct all boil down to these same three
components plus some metadata (dtype, device, ownership).

```zig
pub const Tensor = struct {
    data: []f32,      // the flat buffer
    shape: Shape,      // logical dimensions
    strides: Strides,  // memory layout
    dtype: DType,      // always .f32 for now
    device: Device,    // .cpu or .cuda
    owned: bool,       // true = we must free data; false = view
};
```

The key insight: **the shape tells you what the tensor *means*
(logically), and the strides tell you how the data is *laid out* in memory.
The same logical tensor can have different strides depending on how it was
created.**

---

## 2. Row-Major (C-Contiguous) Layout

### 2.1 The Convention

Our library uses row-major order (also called C-contiguous).  This means:
**the rightmost index varies fastest in memory.**

For a 2D tensor with shape `(M, N)`, element `[i, j]` is stored at:

```
flat_index = i * N + j
```

Example: shape `(2, 3)`, data = `[10, 20, 30, 40, 50, 60]`

```
Memory:  [10, 20, 30, 40, 50, 60]
Index:    0   1   2   3   4   5

Logical:
  Row 0: [10, 20, 30]    ← indices [0,0], [0,1], [0,2]
  Row 1: [40, 50, 60]    ← indices [1,0], [1,1], [1,2]
```

Row 0 occupies memory positions 0–2; row 1 occupies positions 3–5.
Elements within a row are contiguous. This is why row-major is also called
"row-contiguous."

### 2.2 Deriving Strides

The stride for each dimension is the number of f32 elements to skip when
you increment that dimension's index by 1:

```
For shape (M, N):
  stride[1] = 1                          (move one column: skip 1 element)
  stride[0] = N                          (move one row: skip N elements)

For shape (D, M, N):
  stride[2] = 1
  stride[1] = N
  stride[0] = M * N

For shape (B, D, M, N):
  stride[3] = 1
  stride[2] = N
  stride[1] = M * N
  stride[0] = D * M * N
```

**General formula** (right-to-left accumulation):

```
stride[ndim-1] = 1
stride[i] = stride[i+1] * dims[i+1]     for i = ndim-2 ... 0
```

Our implementation in `shape.zig`:

```zig
pub fn computeStrides(shape: Shape) Strides {
    var strides = Strides{
        .values = [_]usize{0} ** max_rank,
        .rank = shape.rank,
    };
    const n = shape.ndim();
    strides.values[n - 1] = 1;
    if (n >= 2) {
        var i: usize = n - 1;
        while (i > 0) : (i -= 1) {
            strides.values[i - 1] = strides.values[i] * shape.dims[i];
        }
    }
    return strides;
}
```

### 2.3 Worked Examples

**Shape (2, 3):**

```
stride[1] = 1
stride[0] = 1 * 3 = 3

Element [1, 2] → 1*3 + 2*1 = 5
Data[5] = 60 ✓
```

**Shape (2, 3, 4):**

```
stride[2] = 1
stride[1] = 1 * 4 = 4
stride[0] = 4 * 3 = 12

Element [1, 2, 3] → 1*12 + 2*4 + 3*1 = 23
```

**Shape (2, 3, 4, 5):**

```
stride[3] = 1
stride[2] = 1 * 5 = 5
stride[1] = 5 * 4 = 20
stride[0] = 20 * 3 = 60
```

---

## 3. The Flat Index Formula

The fundamental indexing operation that everything builds on:

```
flat_index = Σ (indices[i] * strides[i])    for i = 0 ... ndim-1
```

This single formula handles every rank, every stride layout — including
non-contiguous strides from transposed tensors.

Our implementation in `tensor.zig`:

```zig
pub fn flatIndex(self: Tensor, indices: []const usize) usize {
    const ndim = self.shape.ndim();
    std.debug.assert(indices.len == ndim);
    var idx: usize = 0;
    for (indices, 0..) |dim_idx, axis| {
        std.debug.assert(dim_idx < self.shape.dims[axis]);
        idx += dim_idx * self.strides.values[axis];
    }
    return idx;
}
```

### 3.1 Why Strides, Not Just Shape?

If we only had row-major data, we could compute the flat index from the
shape alone. But strides allow us to represent **views** that share data
without copying.

Example: transpose a `(2, 3)` matrix:

```
Original:  shape (2, 3), strides [3, 1]
  [10, 20, 30, 40, 50, 60]

Transposed: shape (3, 2), strides [1, 3]   ← swapped!
  Row 0: data[0*1 + 0*3] = 10
         data[0*1 + 1*3] = 40
  Row 1: data[1*1 + 0*3] = 20
         data[1*1 + 1*3] = 50
  Row 2: data[2*1 + 0*3] = 30
         data[2*1 + 1*3] = 60

Logical view:
  [10, 40]
  [20, 50]
  [30, 60]
```

Same 6 elements, different logical interpretation — zero data movement.

---

## 4. Contiguity

### 4.1 Definition

A tensor is **contiguous** if its strides match what `computeStrides`
would produce for its shape.  This means the data is laid out in the
standard row-major order with no gaps.

```zig
pub fn isContiguous(shape: Shape, strides: Strides) bool {
    if (shape.rank != strides.rank) return false;
    const expected = computeStrides(shape);
    const n = shape.ndim();
    for (0..n) |i| {
        if (strides.values[i] != expected.values[i]) return false;
    }
    return true;
}
```

### 4.2 Why Contiguity Matters

1. **reshape() is zero-copy for contiguous tensors.** If data is in
   row-major order, we can simply reinterpret the same buffer with a
   new shape and new strides. No copy needed.

2. **Non-contiguous reshape requires a copy.** A transposed tensor's data
   isn't in row-major order, so reshaping it would require rearranging
   elements. Our library returns `error.NotImplemented` instead of
   silently copying — making the cost visible.

3. **GPU transfers need contiguous data.** When we copy a tensor to the
   GPU in Stage 7, we need a contiguous block of memory. Non-contiguous
   tensors must be made contiguous first.

4. **Some operations assume contiguity.** Our `softmax` and `copyTo`
   have fast paths for contiguous tensors that use `@memcpy` instead of
   per-element iteration.

### 4.3 What Operations Break Contiguity

| Operation | Result contiguous? | Why |
|-----------|-------------------|-----|
| `Tensor.init` | Yes | Fresh allocation, row-major |
| `zeros`, `ones`, `full`, `randn` | Yes | Wraps `Tensor.init` |
| `transpose2d` | **No** | Swaps strides |
| `reshape` | Yes | Recomputes strides for new shape |
| `view()` | Same as source | Shares source's strides |
| Elementwise ops | Yes | Allocates new contiguous output |
| `add`, `sub`, `mul`, `div` | Yes | Allocates new contiguous output |
| `softmax` | Yes | Allocates new contiguous output |
| `matmul` | Yes | Allocates new contiguous output |

**The only operation that breaks contiguity in Stage 2 is `transpose2d`.**
Every other operation either allocates fresh contiguous memory or preserves
the source's contiguity.

---

## 5. Broadcasting

### 5.1 The Problem

We want to add a shape `(1, 3)` row vector to a shape `(2, 3)` matrix:

```
row:  [[100, 200, 300]]       shape (1, 3)
mat:  [[  1,   2,   3],       shape (2, 3)
       [  4,   5,   6]]

Wanted result: shape (2, 3)
  [[101, 202, 303],
   [104, 205, 306]]
```

The "naive" approach: explicitly copy `row` into a `(2, 3)` tensor, then
add.  But that wastes memory and time — we'd allocate 6 extra f32 values
just to hold a copy of data we already have.

### 5.2 The NumPy Broadcasting Rules

Instead of copying, we **broadcast**: the size-1 dimension is "virtually
stretched" to match the other tensor.  The rules:

1. **Align shapes from the right.** Trailing dimensions are compared first.
2. **Two dims are compatible if** they are equal, OR one of them is 1.
3. **The output dim is `max(d_a, d_b)`.**
4. **A dim of size 1 "stretches"** — the same value is reused for every
   position along that dimension.
5. **If any pair is incompatible, it's an error** (ShapeMismatch).

### 5.3 Examples

```
(1, 3) + (2, 3) → (2, 3)     # row stretched to 2 rows
(2, 1) + (2, 3) → (2, 3)     # column stretched to 3 columns
(1, 1) + (2, 3) → (2, 3)     # both dims stretched
(3,)   + (2, 3) → (2, 3)     # 1D treated as (1, 3), then stretched
(3,)   + (4,)   → ERROR       # 3 != 4, neither is 1
(2, 3) + (3, 2) → ERROR       # aligned: (3,3) and (2,2) → both mismatch
```

### 5.4 ASCII Diagram: (1,3) + (2,3)

```
    a (1,3)        b (2,3)        out (2,3)

  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │100 200 300│  │  1   2   3│  │101 202 303│
  └───────────┘  │  4   5   6│  │104 205 306│
      ↑          └───────────┘  └───────────┘
      │               ↑
   stretched       original
   (reused)     (not stretched)
```

The single row of `a` is "stretched" conceptually. In memory, no
copy happens — when computing `out[1, j]`, we read `a[0, j]` (index 0
for the broadcast dimension, index j for the real dimension).

### 5.5 Our Implementation

The key insight is the `broadcastIndex` function, which maps a flat
output index to a flat input index:

```zig
fn broadcastIndex(out_idx: usize, out_shape: Shape, in_shape: Shape, in_strides: Strides) usize {
    var result: usize = 0;
    var i: usize = out_shape.ndim();
    var remaining = out_idx;
    while (i > 0) {
        i -= 1;
        const dim_size = out_shape.dims[i];
        const coord = remaining % dim_size;
        remaining = remaining / dim_size;
        // If input has dim=1 at this axis, use index 0 (broadcast)
        const in_coord: usize = if (i < in_shape.ndim() and in_shape.dims[i] == 1) 0 else coord;
        if (i < in_shape.ndim()) {
            result += in_coord * in_strides.values[i];
        }
    }
    return result;
}
```

**How it works:**

1. Decompose the flat output index into per-axis coordinates.
2. For each axis: if the input has dim=1, use coordinate 0 (the broadcast
   value). Otherwise, use the output's coordinate.
3. Multiply by the input's stride and accumulate.

This means: for a `(1, 3)` input and a `(2, 3)` output, when computing
`out[1, j]`, we get `coord=1` for axis 0, but `in_coord=0` because
`in_shape.dims[0]=1`. So we read `a[0, j]` — the same value for every
row of the output.

### 5.6 broadcastShapes: Computing the Output Shape

```zig
pub fn broadcastShapes(a: Shape, b: Shape) !Shape {
    const ndim_a = a.ndim();
    const ndim_b = b.ndim();
    const ndim_out = @max(ndim_a, ndim_b);
    var result = Shape{ .dims = [_]usize{1} ** max_rank, .rank = @intCast(ndim_out - 1) };

    var i: usize = 0;
    while (i < ndim_out) : (i += 1) {
        const da = if (i < ndim_a) a.dims[ndim_a - 1 - i] else 1;
        const db = if (i < ndim_b) b.dims[ndim_b - 1 - i] else 1;

        if (da == db) {
            result.dims[ndim_out - 1 - i] = da;
        } else if (da == 1) {
            result.dims[ndim_out - 1 - i] = db;
        } else if (db == 1) {
            result.dims[ndim_out - 1 - i] = da;
        } else {
            return LabError.ShapeMismatch;
        }
    }
    return result;
}
```

The right-to-left walk (using `ndim_a - 1 - i`) implements the
right-alignment rule. Missing dimensions (when one shape has fewer
dims) are treated as size 1, which is automatically broadcastable.

---

## 6. Views: Zero-Copy Tensor Sharing

### 6.1 The `owned` Flag

Every tensor has an `owned` field:

- `owned = true`: this tensor allocated its data buffer and must free it
  in `deinit()`.
- `owned = false`: this tensor is a view sharing another tensor's buffer.
  It must NOT free the data.

### 6.2 view(): Create an Alias

```zig
pub fn view(self: Tensor) Tensor {
    return Tensor{
        .data = self.data,      // same pointer!
        .shape = self.shape,
        .strides = self.strides,
        .owned = false,         // not responsible for freeing
        // ... other fields
    };
}
```

The view shares the EXACT same data pointer. Writing to the original
modifies the view, and vice versa.

### 6.3 reshape(): Zero-Copy When Contiguous

```zig
pub fn reshape(self: Tensor, new_shape: Shape) LabError!Tensor {
    if (totalElements(self.shape) != totalElements(new_shape)) return error.ShapeMismatch;
    if (!self.isContiguous()) return error.NotImplemented;
    return Tensor{
        .data = self.data,             // same buffer!
        .shape = new_shape,            // new logical shape
        .strides = computeStrides(new_shape), // new strides
        .owned = false,                // not responsible for freeing
    };
}
```

Example: reshape `(2, 3)` → `(3, 2)`:

```
Before: shape (2,3), strides [3,1], data [10,20,30,40,50,60]
After:  shape (3,2), strides [2,1], data [10,20,30,40,50,60]

Interpretation changes:
  [10, 20]     ← was row 0's first two elements
  [30, 40]     ← was row 0's third + row 1's first
  [50, 60]     ← was row 1's last two elements
```

Same 6 bytes, different logical grid. Zero allocation, zero copy.

### 6.4 transpose2d(): Swap Strides, Not Data

```zig
pub fn transpose2d(self: Tensor) LabError!Tensor {
    const new_shape = Shape.init2D(self.shape.dims[1], self.shape.dims[0]);
    var new_strides = Strides{
        .values = .{ self.strides.values[1], self.strides.values[0], 0, 0 },
        .rank = new_shape.rank,
    };
    return Tensor{
        .data = self.data,
        .shape = new_shape,
        .strides = new_strides,
        .owned = false,
    };
}
```

Why swap strides instead of copying? Because the transpose of a row-major
matrix is column-major. Swapping strides captures this exactly:
- What was a "row step" (stride[0]=3) becomes a "column step" in the
  transposed view.
- What was a "column step" (stride[1]=1) becomes a "row step."

This is O(1) instead of O(M*N) for a copy.

### 6.5 Lifetime Rule

**The owning tensor must outlive all of its views.** If you deinit the
owner first, the views become dangling pointers.

```zig
var t = try Tensor.init(alloc, Shape.init2D(2, 3));
defer t.deinit(alloc);        // t is freed at scope exit

const v = t.view();
// v is valid HERE
// At scope exit, t.deinit() frees the data, and v becomes invalid
```

This is the same contract as a slice referencing freed memory. It's
intentionally low-level — no reference counting, no garbage collection.
The caller is responsible for ordering.

---

## 7. Softmax and Numerical Stability

### 7.1 The Naive Formula

Softmax converts raw scores (logits) into probabilities:

```
softmax(x)[j] = exp(x[j]) / Σ_k exp(x[k])
```

This works fine for small values. But what happens when x[j] = 100?

```
exp(100) = 2.69 × 10^43  ← exceeds f32 max (3.4 × 10^38)
```

The result is `+Inf`. Then `Inf / Inf = NaN`, and every subsequent
computation is contaminated. This is the **softmax overflow problem**.

### 7.2 The Max-Subtraction Trick

The key insight: softmax is invariant to adding a constant:

```
softmax(x)[j] = exp(x[j]) / Σ_k exp(x[k])
             = exp(x[j] - c) / Σ_k exp(x[k] - c)    for any c
```

**Proof:**

```
exp(x[j] - c)         exp(x[j]) * exp(-c)         exp(x[j])
─────────────── = ─────────────────────────── = ──────────────
Σ exp(x[k] - c)   Σ exp(x[k]) * exp(-c)     Σ exp(x[k])
```

The `exp(-c)` factors cancel top and bottom. So we can choose c = max(x)
without changing the result:

```
softmax(x)[j] = exp(x[j] - max(x)) / Σ_k exp(x[k] - max(x))
```

Now every exponent is ≤ 0 (because x[j] ≤ max(x)), so every exp value
is in [0, 1]. No overflow possible.

### 7.3 Worked Example

```
x = [1, 2, 3]
max(x) = 3

Naive:  exp([1,2,3]) = [2.718, 7.389, 20.086]
        sum = 30.193
        softmax = [0.0900, 0.2447, 0.6652]

Stable: x - max = [-2, -1, 0]
        exp([-2,-1,0]) = [0.1353, 0.3679, 1.0]
        sum = 1.5032
        softmax = [0.0900, 0.2447, 0.6652]   ← same result!
```

Same result, but the stable version never computes `exp(3) = 20.086`.
For `x = [100, 101]`, the naive version overflows while the stable
version computes `exp([-1, 0]) = [0.368, 1.0]` — perfectly fine.

### 7.4 Underflow Is Harmless

What about `exp(-very_large_number)`? This underflows to 0.0, but that's
fine:

```
softmax([1000, 0]) = [exp(0)/sum, exp(-1000)/sum]
                   = [1.0/1.0, 0.0/1.0]
                   = [1.0, 0.0]
```

The second probability is 0, which correctly represents "the model is
extremely confident that class 0 is the right one." No NaN, no Inf.

### 7.5 Our Implementation

```zig
pub fn softmax(allocator: std.mem.Allocator, tensor: Tensor) LabError!Tensor {
    const ndim = tensor.shape.ndim();
    const C = tensor.shape.dims[ndim - 1];     // size of last axis
    const num_groups = totalElements(tensor.shape) / C;

    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    for (0..num_groups) |g| {
        // ... decode group index into base_offset ...

        // Step 1: find max
        var max_val: f32 = -std.math.inf(f32);
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last];
            max_val = @max(max_val, val);
        }

        // Step 2: compute exp(x - max) and sum
        var sum_exp: f32 = 0;
        for (0..C) |c| {
            const val = tensor.data[base_offset + c * stride_last] - max_val;
            const ev = @exp(val);
            out.data[g * C + c] = ev;
            sum_exp += ev;
        }

        // Step 3: normalize
        for (0..C) |c| {
            out.data[g * C + c] /= sum_exp;
        }
    }
    return out;
}
```

Three-pass per group (max, exp+sum, normalize). For a group of C=128
elements (typical vocabulary size in our transformer), this is three
loops of 128 — negligible.

---

## 8. log-softmax: Why Not log(softmax)?

### 8.1 The Instability of log(softmax)

Cross-entropy loss requires `log(softmax(x))`. The naive approach:

```
probs = softmax(x)           # each prob in (0, 1]
log_probs = log(probs)       # each in (-inf, 0]
```

The problem: when `softmax(x)[j]` is very close to 0 (the model is very
confident about a different class), `log(tiny_number)` amplifies rounding
errors. In f32, the smallest positive normal number is ~1.2×10⁻³⁸, and
log of that is ~-87. But softmax values can underflow to 0.0, and
`log(0.0) = -Inf`.

### 8.2 The Direct Formula

Instead, compute log-softmax directly:

```
log_softmax(x)[j] = x[j] - max(x) - log(Σ_k exp(x[k] - max(x)))
```

This avoids the division-then-log pattern. We compute the log of the sum
directly (the sum is always >= 1 because `exp(0) = 1` is one of the terms),
so `log(sum)` is always >= 0, and the subtraction is numerically stable.

### 8.3 Worked Example

```
x = [1, 2, 3], max = 3

exp_shifted = [0.1353, 0.3679, 1.0]
sum = 1.5032
log(sum) = 0.4076

log_softmax = [1-3-0.4076, 2-3-0.4076, 3-3-0.4076]
            = [-2.4076, -1.4076, -0.4076]

Verification: exp and log of softmax should match:
  log(0.0900) = -2.4076  ✓
  log(0.2447) = -1.4076  ✓
  log(0.6652) = -0.4076  ✓
```

Same result, but no intermediate `softmax → log` step, so no risk of
`log(0)` or precision loss from tiny softmax values.

---

## 9. Cross-Entropy Loss

### 9.1 The Formula

Cross-entropy measures how well predicted log-probabilities match the
true class:

```
loss = -(1/B) * Σ_i log_softmax(logits[i, target[i]])
```

Where:
- `logits` has shape `(B, C)` — B samples, C classes
- `targets` has shape `(B,)` — integer class indices
- `loss` is a scalar (mean over the batch)

### 9.2 Why Mean Over Batch?

Dividing by B makes the loss scale-independent of batch size. Without
this, a batch of 32 would have ~32× the loss of a batch of 1, making
the learning rate batch-size-dependent. Mean normalization keeps the
loss in a consistent range regardless of batch size.

### 9.3 Special Cases

- **All logits equal** → uniform softmax → loss = log(C). For C=3,
  loss = ln(3) ≈ 1.099. This is the maximum entropy (most uncertain)
  prediction.

- **Correct class has dominant logit** → loss → 0. The model is
  confidently correct.

- **Correct class has very low logit** → loss → +Inf. The model is
  confidently wrong about the right class.

### 9.4 Our Implementation

```zig
pub fn crossEntropy(allocator: std.mem.Allocator, logits: Tensor, targets: Tensor) LabError!Tensor {
    // Validate shapes
    if (logits.shape.ndim() != 2) return LabError.InvalidArgument;
    if (targets.shape.ndim() != 1) return LabError.InvalidArgument;
    const B = logits.shape.dims[0];
    const C = logits.shape.dims[1];

    // Compute log_softmax (numerically stable)
    var log_probs = try logSoftmax(allocator, logits);
    defer log_probs.deinit(allocator);

    // Gather the log-probability of each target class
    var loss_sum: f32 = 0;
    for (0..B) |i| {
        const class_idx = @as(usize, @intFromFloat(@round(targets.data[i])));
        const log_p = log_probs.data[i * C + class_idx];
        loss_sum += -log_p;      // negate: loss = -log(P(target))
    }

    // Mean over the batch
    const mean_loss = loss_sum / @as(f32, @floatFromInt(B));
    var out = try Tensor.init(allocator, Shape.init1D(1));
    out.data[0] = mean_loss;
    return out;
}
```

Key design choice: we use `logSoftmax` internally instead of computing
`log(softmax(...))`. This is the numerically stable path.

---

## 10. Matmul: The ikj Loop Order

### 10.1 The Textbook (ijk) Order

```
for i in 0..M:
    for j in 0..N:
        for k in 0..K:
            C[i,j] += A[i,k] * B[k,j]
```

The inner loop iterates over `k` while `j` is fixed. In row-major order,
`B[k,j]` accesses elements from different rows of B (different k values)
while j stays the same. This means the access pattern for B jumps by
`stride[0]` elements on each iteration — potentially cache-missing for
large matrices.

### 10.2 The Cache-Friendly (ikj) Order

```
for i in 0..M:
    for k in 0..K:
        a_ik = A[i,k]        // load ONCE, reuse for all j
        for j in 0..N:
            C[i,j] += a_ik * B[k,j]
```

Now the inner loop iterates over `j` while `k` is fixed:
- `B[k,j]` walks along a row of B → contiguous access, cache-friendly.
- `C[i,j]` walks along a row of C → contiguous access, cache-friendly.
- `A[i,k]` is loaded once per `(i,k)` pair, then reused N times.

### 10.3 Worked Example

```
A = [[1,2,3],[4,5,6]]    shape (2,3)
B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]    shape (3,4)
C = A @ B = shape (2,4)

ikj order for C[0,0]:
  k=0: a[0,0]=1, C[0,0] += 1*B[0,0]=1 → 1
  k=1: a[0,1]=2, C[0,0] += 2*B[1,0]=10 → 11
  k=2: a[0,2]=3, C[0,0] += 3*B[2,0]=27 → 38

Result: C[0,0] = 38 ✓
```

### 10.4 Why Not Tile or Use BLAS?

For Stage 2, the naive ikj matmul is:
- **Correct** — easy to verify against hand computation
- **Understandable** — the loop structure is transparent
- **Sufficient** — our transformer has tiny dimensions (vocab ~100, d_model ~64)

Stage 7 replaces this with cuBLAS for GPU tensors. Stage 8 may add tiled
matmul for CPU if performance is a bottleneck.

---

## 11. Reductions: Summing Along an Axis

### 11.1 What "Reduce Along Axis" Means

For a `(2, 3)` tensor:

```
[[1, 2, 3],
 [4, 5, 6]]

sum(axis=0) → [[5, 7, 9]]     shape (1, 3)  — collapse rows
sum(axis=1) → [[6], [15]]     shape (2, 1)  — collapse columns
```

The output keeps the reduced axis as dim=1 (not removed entirely). This
preserves the rank, which makes it easy to broadcast the result back
against the original tensor.

### 11.2 The Index Mapping

For `sum(axis=0)` of a `(2, 3)` tensor:
- Output has shape `(1, 3)`, so output index `(0, j)` maps to input
  indices `(0, j)` and `(1, j)`.

For `sum(axis=1)` of a `(2, 3)` tensor:
- Output has shape `(2, 1)`, so output index `(i, 0)` maps to input
  indices `(i, 0)`, `(i, 1)`, and `(i, 2)`.

### 11.3 Why Keep the Axis?

Keeping the reduced axis as size-1 (instead of removing it) enables:

```zig
// Compute mean along axis 1, then broadcast-subtract
const row_mean = try reduce.mean(alloc, x, 1);     // shape (B, 1)
const centered = try elementwise.sub(alloc, x, row_mean);  // broadcasts!
```

If `row_mean` had shape `(B,)` instead of `(B, 1)`, the broadcast would
align differently and give the wrong result. Keeping the axis makes
broadcasting predictable.

---

## 12. The Shape Struct: Why u2 for Rank?

### 12.1 The rank Field

```zig
pub const Shape = struct {
    dims: [4]usize,
    rank: u2,        // stores ndim - 1
};
```

`u2` can hold values 0, 1, 2, 3, representing ndim 1, 2, 3, 4.

### 12.2 Why Not Just Store ndim?

A `u3` could store ndim directly (values 1-4), but `u2` is more compact:
- `u2` occupies exactly 2 bits, leaving 62 bits of padding after the
  `[4]usize` array on a 64-bit platform.
- More importantly, `u2` enforces the constraint at the type level:
  you literally cannot create a Shape with ndim=5 or ndim=0.

### 12.3 The ndim() Method

```zig
pub fn ndim(self: Shape) usize {
    return @as(usize, self.rank) + 1;
}
```

This conversion from `u2` to `usize` happens at every call site. The cost
is negligible (one zero-extend instruction), and the type safety is worth
it — we can never accidentally create an impossible ndim value.

### 12.4 Unused Dims Are Always 1

```zig
// Shape.init2D(2, 3) produces:
.dims = [2, 3, 1, 1]
```

This convention means `totalElements()` can simply multiply all four dims
without checking which ones are "real." A dim of 1 doesn't affect the
product. It also makes broadcasting logic simpler — unused dims are
already broadcastable.

---

## 13. The Full Data Flow: From Creation to Loss

Let's trace a complete mini-batch through the tensor layer:

```
1. Create weight matrix:
   W = randn(alloc, Shape.init2D(64, 100), &rng, 0, 0.02)

2. Create input batch:
   x = fromSlice(alloc, Shape.init2D(4, 64), input_data)

3. Matrix multiply:
   logits = matmul(alloc, x, W)           // (4, 64) @ (64, 100) → (4, 100)

4. Compute loss:
   targets = fromSlice(alloc, Shape.init1D(4), target_indices)
   loss = crossEntropy(alloc, logits, targets)  // (4, 100), (4,) → (1,)

5. Print results:
   debugSummary(logits, writer)           // shape, min/mean/max, NaN count
   print("loss = {d:.4}\n", .{loss.data[0]})
```

Each step allocates a new tensor, uses it, and eventually deinits it.
No hidden state, no global caches, no implicit copies. Every allocation
is explicit, every free is explicit.

---

## 14. Common Mistakes

1. **Forgetting max-subtraction in softmax.** Without it, `exp(100)`
   overflows to Inf and the result is NaN. Always subtract max(x) before
   exp.

2. **Computing log(softmax(x)) instead of log_softmax(x).** The direct
   formula avoids the unstable `log(tiny_number)` path. Use `logSoftmax`
   whenever you need log-probabilities.

3. **Assuming transpose creates a contiguous tensor.** Transpose swaps
   strides — the result is NOT contiguous. Reshaping a transposed tensor
   returns `error.NotImplemented`. Call `copyTo` first if you need a
   contiguous version.

4. **Broadcasting with wrong axis alignment.** Shapes are aligned from
   the RIGHT. `(3,) + (2,3) = (2,3)` works (3 aligns with last dim), but
   `(3,) + (3,2) = ERROR` (3 aligns with first dim, which is 3≠2 and
   neither is 1).

5. **Deiniting a view tensor.** Views (owned=false) share another
   tensor's buffer. Only the owner should deinit. Double-freeing causes
   undefined behavior.

6. **Using the wrong strides for a transposed tensor.** After
   `transpose2d()`, the strides are swapped. If you compute flat indices
   using the old strides, you'll read wrong elements. Always use the
   tensor's own strides field.

7. **Ignoring NaN in debugSummary.** NaN comparisons are always false,
   so NaN values never update min/max. The `nan=` count in debugSummary
   tells you how many elements are NaN — check it before trusting min/max.

8. **Assuming `sumAll` returns a scalar.** Our system represents scalars
   as shape `(1,)` tensors, not 0D tensors. Access the value with
   `result.data[0]`.

9. **Forgetting errdefer in functions that allocate.** If you allocate a
   tensor and then call a fallible function, use `errdefer t.deinit(alloc)`
   to prevent leaks on error paths.

10. **Mixing row-major and column-major.** Our library is row-major
    throughout. When we add CUDA (Stage 7), cuBLAS expects column-major
    inputs. The wrapping in `src/backend/cuda/gemm.zig` handles the
    transpose — don't try to do it manually in the tensor layer.


---

## Exercises

**Exercise 1.** A row-major tensor of shape `(2, 3, 4)` has what
strides? At flat index 11, what are the logical `(i, j, k)`
coordinates?

<details><summary>Solution</summary>

Strides: `(12, 4, 1)`. Each step along axis 0 jumps 12 elements
(one full `(3, 4)` slab); each step along axis 1 jumps 4 (one row);
each step along axis 2 jumps 1 (one element).

Flat index 11: `11 = 0*12 + 2*4 + 3*1`, so `(i=0, j=2, k=3)`.

</details>

**Exercise 2.** Consider `a + b` where `a: (1, 4)` and `b: (3, 4)`.
Does broadcasting succeed? What is the output shape, and which element
of `a` is read for output position `(2, 1)`?

<details><summary>Solution</summary>

Broadcasting succeeds: the `1` in `a`'s shape aligns with the
`3` in `b`'s shape and is stretched. Output shape `(3, 4)`.

For output `(2, 1)`, we read `a[0, 1]` (the size-1 axis always
reads index 0 regardless of the output index) plus `b[2, 1]`. The
broadcast is implemented via stride-0 axes in `a`; no copy is
made.

</details>
