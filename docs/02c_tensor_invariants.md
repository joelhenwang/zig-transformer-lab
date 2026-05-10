# 02c — Tensor Invariants and `LabError`

This chapter is about the rails we put under the tensor system in PR-γ:
the five structural properties every `Tensor` value must satisfy, the
single function that enforces them, and the expanded `LabError` set
that gives callers a vocabulary for talking about *how* a tensor went
wrong.

Invariants are a design tool. Every invariant in this chapter was
chosen because its absence would turn a silent bug — wrong numbers,
corrupted memory, zero gradients — into an obvious crash. For a
library whose product is correctness, a loud panic is always better
than a quiet wrong answer.

Read `docs/02_tensors.md` first if you have not seen the shape /
strides / ownership model.

---

## 1. What an invariant is, and why we care

An **invariant** is a property of a value that is true every time the
value is looked at. Code that maintains an invariant can rely on it
without re-checking. Code that *creates* values has to prove the
invariant holds.

For a `Tensor`, the invariants are what we mean when we say "this is a
valid tensor". A pile of fields that individually typecheck is not
automatically valid. For example:

```zig
var data: [6]f32 = .{ 0, 0, 0, 0, 0, 0 };
const bogus = Tensor{
    .data    = &data,
    .shape   = Shape.init2D(2, 3),
    // Strides (6, 2) would reach physical offset (1*6 + 2*2) = 10.
    // The buffer is only 6 elements long. Reading index 10 is UB.
    .strides = Strides{ .values = .{ 6, 2, 0, 0 }, .rank = 1 },
    .dtype   = .f32,
    .device  = .cpu,
    .owned   = false,
    .storage = .{ .cpu = .{ .data = &data, .owned = false } },
    .offset  = 0,
    // ... the rest
};
```

Every field typechecks. Zig is happy. But logically this value is
broken: an inner loop that iterates logical indices `0..6` will read
memory that does not belong to the tensor. In debug builds this
crashes; in release-fast builds it returns garbage floats that become
garbage gradients that become a silently-wrong training run.

The invariant we want is:

> The largest physical offset any logical element of a view can
> reach is less than `storage.len()`.

Once that invariant holds, the inner loop is safe. Once we say so
*everywhere* a `Tensor` is produced, we never have to think about it
inside an op implementation.

---

## 2. The five `Tensor` invariants

`src/tensor/tensor.zig:Tensor.checkInvariants` encodes five checks.
They are intentionally minimal: each one prevents a class of bug we
actually hit or nearly hit during Stages 2–6.

### 2.1 Rank is between 1 and 4

```zig
// 1. Rank is intrinsically in [0, 3] (u2), so ndim is [1, 4].
//    Nothing to assert beyond that.
```

We encode this in the type system rather than at runtime:
`Shape.rank: u2` holds `ndim - 1`, so rank ∈ {0, 1, 2, 3} and
`ndim ∈ {1, 2, 3, 4}`. A `u2` cannot be 4 or 5. Invalid ranks are
rejected at compile time or struct-construction time, and
`checkInvariants` has nothing to do here.

Why 4? Because our model never needs more: a transformer activation
is `(batch, seq, heads, head_dim)` at worst. If you later need 5D,
widening `rank` to `u3` and `max_rank` to 5 is a one-line change,
but every op that switches on rank must add the case.

### 2.2 No dimension is zero

```zig
// 2. No dim is zero.
const n = self.shape.ndim();
for (0..n) |axis| {
    if (self.shape.dims[axis] == 0) return error.ShapeMismatch;
}
```

A zero-sized tensor is legal in NumPy and PyTorch; they spent serious
effort making sure reductions, softmax, and autograd behave sensibly
on them. We chose not to. Rejecting zero dims at construction means:

- `totalElements(shape)` is always ≥ 1 (never 0)
- `maxLogicalOffset` never underflows (it computes `dims[i] - 1`; if
  `dims[i] == 0` the subtraction wraps to `usize::MAX`)
- Reductions never have to answer "what is `max` of an empty set?"
- Softmax never has to decide what `exp` / `sum = 0` means
- Kernel launches (future CUDA work) never have zero grid dimensions

The debug asserts in `Shape.initND` catch construction; `checkInvariants`
catches tensors built by hand. Both return `error.ShapeMismatch`
rather than panicking, so callers can recover if they want.

### 2.3 Stride rank matches shape rank

```zig
// 3. Stride and shape rank agree.
if (self.shape.rank != self.strides.rank) return error.InvalidLayout;
```

`Shape` and `Strides` both store a `rank: u2`. They should always
match. A mismatch means someone forgot to update one of them when
building a view, and subsequent stride-based indexing will read
garbage (`strides.values[i]` for `i >= strides.rank` is defined to be
`0`, but only because we never rely on it).

This check caught two bugs during the PR-γ itself: one in a manually
constructed test tensor, one in an earlier draft of `transpose2d`
that set the wrong rank on the new strides.

### 2.4 Every logical element fits in the backing storage

```zig
// 4. Every reachable logical element fits in the backing storage
//    (accounting for offset).
const max_off = maxLogicalOffset(self.shape, self.strides);
if (self.offset + max_off >= self.storage.len()) return error.InvalidLayout;
```

This is the most important invariant and the most subtle one. Read
it carefully.

`maxLogicalOffset` returns `Σᵢ (dims[i] - 1) * strides[i]`: the
physical offset of the "last" logical element under row-major
iteration. The formula is derived in `src/tensor/shape.zig`:

```
A logical element is at physical offset:
    offset_phys = Σᵢ coord[i] * strides[i]

Each coord[i] ranges over [0, dims[i] - 1]. Strides are non-negative.
So the maximum of the sum is the sum of the maxima:
    max_offset_phys = Σᵢ (dims[i] - 1) * strides[i]
```

Adding `self.offset` gives the largest offset the view actually
touches. It must be strictly less than `storage.len()` (the length of
the backing buffer in f32 elements).

**Why `<` and not `<=`?** Because offsets are 0-indexed. A buffer of
length `n` has valid indices `[0, n-1]`. If `offset + max_off == n`,
the last access reads index `n`, which is one past the end.

#### A worked example

Take a `(2, 3)` tensor with row-major strides `(3, 1)`, offset 0,
backing buffer of length 6:

```
max_off = (2-1)*3 + (3-1)*1 = 3 + 2 = 5
offset + max_off = 0 + 5 = 5
storage.len()    = 6
assertion        = 5 < 6  ✓
```

Now the `(3, 2)` transposed view of the same buffer, strides `(1, 3)`:

```
max_off = (3-1)*1 + (2-1)*3 = 2 + 3 = 5
offset + max_off = 0 + 5 = 5
storage.len()    = 6
assertion        = 5 < 6  ✓
```

And the bogus tensor from §1 with strides `(6, 2)`:

```
max_off = (2-1)*6 + (3-1)*2 = 6 + 4 = 10
offset + max_off = 0 + 10 = 10
storage.len()    = 6
assertion        = 10 < 6  ✗   → error.InvalidLayout
```

#### Why the offset term matters

In PR-δ we added `Tensor.offset: usize`, pre-populated to 0 but
reserved for future sub-view (slice) operations. A slice `t[2..5]` of
a length-10 tensor would have `offset = 2` and `storage.len() = 10`,
so the invariant becomes:

```
offset + max_off < storage.len()
2 + (shape[0]-1) * strides[0] < 10
```

Without the offset term, the check would be too permissive — a
one-element slice at offset 9 with shape `(1,)` and strides `(1,)`
would satisfy `max_off = 0 < 10` but actually dereferences offset 9,
which is still inside the buffer (fine here), but a slice at offset 10
would also pass. The offset-aware version rejects it.

### 2.5 CPU/CUDA storage alias consistency

```zig
// 5. CPU compat alias must agree with storage.
switch (self.storage) {
    .cpu => |s| {
        if (self.data.ptr != s.data.ptr or self.data.len != s.data.len) {
            return error.InvalidLayout;
        }
        if (self.owned != s.owned) return error.InvalidLayout;
        if (self.device != .cpu) return error.DeviceMismatch;
    },
    .cuda => {
        if (self.device != .cuda) return error.DeviceMismatch;
    },
}
```

This one exists because of PR-δ's migration strategy. We introduced
`Storage` as a union and `Tensor.storage` as the source of truth, but
kept `Tensor.data` and `Tensor.owned` as convenience aliases so that
the ~800 existing `tensor.data[i]` call sites compile unchanged.

For that to be safe, the alias must not drift from the truth:

- `tensor.data.ptr` must equal `tensor.storage.cpu.data.ptr`
- `tensor.data.len` must equal `tensor.storage.cpu.data.len`
- `tensor.owned` must equal `tensor.storage.cpu.owned`
- `tensor.device` must match the storage tag (CPU tensors have CPU
  storage; CUDA tensors have CUDA storage)

If any of these drifts — for example, because an op updated `owned`
but forgot to mirror it into `storage.cpu.owned` — the next call to
`deinit` will either double-free or leak, depending on which copy the
code reads. The invariant makes drift immediately visible.

PR-ε hit exactly this bug: `tape.keepAlive` set `tensor.owned = false`
but left `storage.cpu.owned = true`, and the storage-aware `deinit`
freed the buffer the tape had already claimed. Invariant 2.5 plus a
test would have caught it in minutes instead of an hour.

### 2.6 Gradient shape and device match, when present

```zig
// 6. Attached gradient, if any, matches shape and device.
if (self.grad) |g| {
    if (!shape_equals(g.shape, self.shape)) return error.ShapeMismatch;
    if (g.device != self.device) return error.DeviceMismatch;
}
```

After `tape.backward(&loss)`, every leaf tensor that required a
gradient has a `grad` pointer attached. That gradient must be the same
shape as the tensor (otherwise `optimizer.step` will mis-subtract) and
on the same device (otherwise the optimizer will try to write CPU
bytes into CUDA memory). Both conditions are true by construction if
the autograd engine is correct; this invariant is a smoke test for
the engine.

---

## 3. The `LabError` set after PR-γ

```zig
// src/core/errors.zig
pub const LabError = error{
    ShapeMismatch,
    OutOfMemory,
    InvalidArgument,
    InvalidLayout,     // new in PR-β, documented in PR-γ
    InvalidIndex,      // new in PR-γ
    DeviceMismatch,    // new in PR-γ
    CudaError,
    IoError,
    NotImplemented,
    NumericalError,
};
```

Three new variants, each for a specific failure mode:

### `InvalidLayout`

> The tensor's layout (contiguity, aliasing, offset, stride-rank
> agreement) is not supported by this operation.

Examples of where it fires:

- `addInPlace(a, b)` when `a` or `b` is non-contiguous. In-place
  writes plus non-contiguous strides plus aliasing between `a` and `b`
  is a four-way correctness minefield; we reject instead of handling.
- Invariant violations (§2.3, §2.4, §2.5) in `checkInvariants`.
- Future: CUDA ops that require contiguous inputs when given a
  transposed view.

Contrast with `ShapeMismatch`, which is about *dimensions not
matching* (two tensors can't be added because one is 2×3 and the other
is 3×2). Shape and layout are independent concerns.

### `InvalidIndex`

> A token, class, or axis index is outside its valid range.

The operations that consume integer-like indices are:

- `embedding(ids, weight)`: if any id ≥ vocab_size, we can't look up
  the row. Returning `InvalidIndex` is better than silently reading
  past the end of the weight matrix.
- `crossEntropy(logits, targets)`: targets must be in
  `[0, vocab_size)`. Outside that range, the one-hot backward has no
  meaning.
- Future: `gather`, `scatter`, axis-based reductions with
  out-of-range axis.

Because we do not yet have an `IndexTensor` type (see §5), these
checks happen inside the ops rather than at tensor-construction time.

### `DeviceMismatch`

> A binary op was given tensors on different devices.

Today with only CPU, this can only fire via `requireSameDevice` on
manually-constructed `.cuda` tensors (as in the PR-γ unit tests). Once
PR-ι introduces real CUDA storage, `DeviceMismatch` is the single
place the "no implicit transfer" rule is enforced.

The PyTorch equivalent:

```python
>>> x = torch.randn(3)
>>> y = torch.randn(3, device='cuda')
>>> x + y
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

Our version:

```zig
var y = ops.add(allocator, x_cpu, z_cuda, null);
// → error.DeviceMismatch
```

---

## 4. Where the invariants get checked

### 4.1 `debugCheckInvariants`

```zig
pub fn debugCheckInvariants(t: Tensor) void {
    if (std.debug.runtime_safety) {
        t.checkInvariants() catch |err| {
            std.debug.panic("Tensor invariants violated: {s}", .{@errorName(err)});
        };
    }
}
```

This is the wrapper every tensor-producing function calls at the end.
In debug and safe-release builds (`std.debug.runtime_safety` is true
in `.Debug` and `.ReleaseSafe`), a violation panics with the error
name. In `.ReleaseFast` and `.ReleaseSmall`, the whole function
compiles away — zero-cost in production.

This is the same pattern as `std.debug.assert`: belt-and-suspenders
during development, invisible in release.

Call sites (current):

- `Tensor.init` — after successful allocation
- `Tensor.view` — before returning the view
- `Tensor.reshape` — before returning the reshaped view
- `Tensor.transpose2d` — before returning the transposed view

Every op that constructs a new tensor goes through `Tensor.init`, so
this covers every op transitively.

### 4.2 Why panic instead of return an error?

`debugCheckInvariants` panics; `checkInvariants` returns an error. The
distinction is intentional:

- `checkInvariants` is a *query*: "is this tensor valid?" Callers that
  suspect a bug can ask and recover.
- `debugCheckInvariants` is an *assertion*: "this tensor came out of
  our constructor, it had better be valid, and if it is not we cannot
  continue safely."

A constructor returning an invalid tensor is a bug in the constructor,
not a recoverable failure mode. Panicking forces the bug to be fixed
rather than papered over with a `catch` in user code.

### 4.3 `requireSameDevice`

```zig
pub fn requireSameDevice(a: Tensor, b: Tensor) LabError!void {
    if (a.device != b.device) return error.DeviceMismatch;
}
```

This is a pre-condition check rather than an invariant. Every binary
op calls it before reading data. Today it's a no-op because every
tensor is CPU; the function exists to reserve the spot where CUDA
enforcement lands.

---

## 5. Things we considered and rejected

### 5.1 Returning `LabError` from `Shape.initND`

We could have made the constructors:

```zig
pub fn init2D(d0: usize, d1: usize) LabError!Shape { ... }
```

and had them return `error.ShapeMismatch` on zero dims. Every call
site — dozens of them across the library — would have to `try` the
result. For a library whose goal is pedagogical clarity, we judged
that the noise wasn't worth it: zero dims are a programmer error, not
a runtime failure mode the user can recover from.

Instead we debug-assert:

```zig
pub fn init2D(d0: usize, d1: usize) Shape {
    std.debug.assert(d0 > 0 and d1 > 0);
    return .{ .dims = .{ d0, d1, 1, 1 }, .rank = 1 };
}
```

In debug builds, a zero-dim shape panics immediately at the point of
construction. In release-fast builds the check compiles away. Call
sites are unchanged.

### 5.2 A dedicated `IndexTensor` type

Our cross-entropy targets and embedding ids are stored in `f32`
tensors, with `@round` / `@intFromFloat` used to recover integer
values at consumption time. This is genuinely ugly — class labels
are not floating-point quantities.

The clean fix is a parallel `IndexTensor` type holding `[]u32`. We
chose to defer that to a later stage because:

1. It touches every tokenizer, dataset, and batching surface.
2. It adds a second tensor type to the library, doubling the op
   surface we have to document.
3. The `InvalidIndex` error path plus an `@round` in forward and
   backward closes the same bug class in two lines.

When Stage 8 or 9 pulls in a wider data pipeline, an `IndexTensor` is
the right refactor. For Stage 6.5 hardening, two checks are enough.

### 5.3 A full gradcheck as part of `checkInvariants`

Some frameworks hide a `debug_check_grad` mode that runs every
backward with finite differences and compares against the analytic
gradient. We already have this as a separate test harness
(`src/autograd/gradcheck.zig`); folding it into `checkInvariants`
would make debug builds unbearably slow. The two tools are
complementary — invariants catch structural bugs, gradcheck catches
mathematical bugs.

---

## 6. PyTorch parallels

Each of our invariants has a PyTorch counterpart. Knowing where to
look in PyTorch lets you cross-reference behaviour when debugging.

| Our invariant | PyTorch counterpart | Where to find it |
|---|---|---|
| §2.1 Rank 1-4 | `TENSOR_MAX_DIMS` (set to 25 in recent PyTorch) | `c10/core/TensorImpl.h` |
| §2.2 No zero dims | **PyTorch allows zero dims**; we diverge here | `torch.empty(0, 3)` returns a valid tensor |
| §2.3 Stride/shape rank agree | PyTorch enforces this structurally — `sizes` and `strides` are the same `SmallVector` length | `Tensor::sizes()` / `Tensor::strides()` |
| §2.4 Logical elements fit in storage | `Tensor.storage()`, `Tensor.storage_offset()`, and the assertion `max_offset < storage.nbytes / element_size` | `at::native::check_size_nonnegative` |
| §2.5 Storage alias consistency | PyTorch has only one representation (`StorageImpl`); there's nothing to drift | N/A |
| §2.6 Grad shape/device match | `torch.autograd.gradcheck` and the runtime check in `Variable::grad()` | `torch/csrc/autograd/variable.cpp` |
| `requireSameDevice` | `Expected all tensors to be on the same device` | `TensorIterator::check_all_same_device` |

The biggest divergence from PyTorch is §2.2: they support zero dims,
we reject them. That is a *scope* decision (see §5.1), not a
correctness issue. If you later port code that creates zero-dim
tensors from PyTorch, you will need to guard it at the boundary.

---

## 7. Common mistakes

### "My test constructs a Tensor by hand and it panics in debug builds"

That is the invariant system working correctly. Double-check that
every field of your manual `Tensor{ ... }` literal is consistent:

- `data.len == storage.cpu.data.len` and the pointers match
- `owned == storage.cpu.owned`
- `strides.rank == shape.rank`
- `offset + maxLogicalOffset(shape, strides) < storage.len()`

If you are deliberately testing an invalid tensor (as we do for PR-γ
failure-mode tests), call `checkInvariants()` directly and assert on
the returned error — don't let the value escape into the rest of the
system.

### "I added a new view op and it sometimes crashes in debug builds"

Your op is producing a tensor with strides or offset that violate
§2.3 or §2.4. Call `debugCheckInvariants` at the end of your op and
read the panic message; it names which error variant fired. Then
recompute the strides / offset on paper for your worked example.

Most often, the bug is a `strides.rank` that wasn't updated when the
shape rank changed.

### "My CI passes but `zig build run-example` panics"

CI likely runs `zig build test` in debug mode, which runs the
invariant checks; release-fast example runs compile them out. If the
panic happens only in debug, the invariants caught something the
release build would have silently run with. Do not "fix" this by
removing the check — fix the constructor that produced the bad
tensor.

### "I want to see the invariant message for all my tensors"

Add a debug print after `debugCheckInvariants` in the constructor you
suspect, or run the program under a debugger with a breakpoint on
`std.debug.panic`. Because `checkInvariants` names the specific
variant (`error.ShapeMismatch` vs `error.InvalidLayout` vs
`error.DeviceMismatch`), the error name alone is usually enough to
locate the bug.

---

## 8. Exercises

### Exercise 1 — Trigger each invariant

Write four tests that construct a `Tensor` by hand and make each of
the first four invariants fire. Check that the error variant returned
by `checkInvariants()` is the one we documented.

Starting point:

```zig
test "invariant exercise 1 — zero dim" {
    // ...
}
```

**Hint:** `Shape` is a plain struct; you can bypass `initND` by
writing the literal `Shape{ .dims = .{ 0, 3, 1, 1 }, .rank = 1 }`.

### Exercise 2 — A slice op

Add a function `pub fn slice(t: Tensor, start: usize, end: usize)
LabError!Tensor` that returns a view of `t` whose first dimension is
restricted to `[start, end)`. The view shares storage but has a
non-zero offset.

Required behaviour:

- `start < end <= t.shape.dims[0]` — otherwise `InvalidArgument`.
- The returned view has `.shape.dims[0] = end - start`, same strides,
  and `.offset = t.offset + start * t.strides.values[0]`.
- Every call to `slice` runs `debugCheckInvariants` on the result.

Write a test that verifies slicing a `(10, 3)` tensor with
`slice(2, 5)` produces a view whose invariants hold and whose logical
elements match rows 2, 3, 4 of the original.

### Exercise 3 — Finding an invariant violation in the wild

Grep the codebase for places that construct a `Tensor{ ... }` literal
outside `Tensor.init` / `Tensor.view` / `Tensor.reshape` /
`Tensor.transpose2d` / `transposeInner2d`. For each one, argue why
the invariants hold by construction. If you find one that does not,
file a bug.

---

## 9. File reference

| File | Lines | What to read |
|---|---|---|
| `src/core/errors.zig` | 1–36 | Full `LabError` set with one-line descriptions |
| `src/tensor/shape.zig` | `maxLogicalOffset` and `logicalOffsetFromLinear` helpers |
| `src/tensor/tensor.zig` | `Tensor.checkInvariants`, `requireSameDevice`, `debugCheckInvariants` |
| `src/tensor/tensor.zig` | Tests `Tensor.checkInvariants passes on ...`, `... rejects ...`, `requireSameDevice ...` |

---

## 10. Test commands

```bash
# Run the full test suite — invariants fire in every debug test
zig build test

# Run only the invariant-specific tests
# (they are inlined in tensor.zig so you need to filter by name)
zig build test -- --test-filter "checkInvariants"

# Run in release-fast to confirm the checks vanish
# (invariant panics should disappear; all tests should still pass)
zig build test -Drelease-fast
```

Next chapter: `02d_storage_and_views.md` — the storage/view split, the
`Storage` union, and why `[]f32` cannot represent CUDA memory.
