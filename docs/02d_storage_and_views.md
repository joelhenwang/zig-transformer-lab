# 02d — Storage, Views, and the Backend Seam

This chapter explains the single most important architectural move in
Stage 6.5: splitting a tensor's *bytes* from its *view over those
bytes*. Before PR-δ we conflated the two inside a single `[]f32`
slice. After PR-δ they are distinct concepts, represented by separate
fields (`storage` and `shape`/`strides`/`offset`), with a union-based
device tag that makes it impossible for a CUDA tensor to masquerade as
a host-addressable slice.

This is the seam on top of which Stage 7's CUDA backend will be
built. Understanding it is a prerequisite for understanding how any
non-trivial tensor library — PyTorch, JAX, NumPy, TensorFlow — models
memory.

Read `docs/02_tensors.md` and `docs/02c_tensor_invariants.md` first.

---

## 1. Three ideas hiding inside one field

Pre-PR-δ, `Tensor.data: []f32` tried to represent three distinct
things simultaneously:

1. **Who owns the bytes.** For an owned tensor, `data` pointed at a
   heap allocation we had to free. For a view, `data` pointed at
   someone else's heap allocation and we must not free it. The
   `owned: bool` field disambiguated.
2. **How many bytes there are.** `data.len` was the size of the
   backing buffer — which for views is bigger than the view's logical
   element count.
3. **Where the bytes live.** The `device: Device` field claimed
   CPU/CUDA, but `[]f32` is definitionally a host-side slice. A
   "`device = .cuda`" tensor with `data = <some slice>` was a lie the
   type system could not detect.

Those three ideas are independent. Lumping them into one field forced
every op to re-derive which was which from context. The `Storage`
union in PR-δ breaks them apart.

### A small motivating example

```zig
var owned = try Tensor.init(allocator, Shape.init2D(3, 2));
defer owned.deinit(allocator);

const view1 = owned.view();       // shares owned's bytes
const view2 = try owned.transpose2d();  // shares owned's bytes, swapped strides
```

All three tensors point at the same six f32s. Owned's bytes must be
freed exactly once. Views must not be freed. `view2`'s strides disagree
with row-major, so indexing through `.data[i]` naively gives wrong
answers.

The pre-δ model:

```
owned.data   → [f32; 6] heap       owned.owned = true
view1.data   → same slice          view1.owned = false
view2.data   → same slice          view2.owned = false
```

Works, but the `data` field has lost its meaning: is it "the buffer I
can iterate" (true for `owned` and `view1`), or "the buffer I can
iterate with the help of strides" (true for `view2`), or "a host-side
slice I can safely read" (false in any future CUDA case)?

The post-δ model:

```
owned.storage = .{ .cpu = { .data = [f32; 6] heap, .owned = true  } }
view1.storage = .{ .cpu = { .data = [f32; 6] heap, .owned = false } }
view2.storage = .{ .cpu = { .data = [f32; 6] heap, .owned = false } }

owned.offset  = 0,  strides = (2, 1)
view1.offset  = 0,  strides = (2, 1)   (same as owned)
view2.offset  = 0,  strides = (1, 2)   (transposed)
```

Ownership is now one field (`storage.cpu.owned`). The backing buffer
is one field (`storage.cpu.data`). The view over that buffer is three
fields (`shape`, `strides`, `offset`). Each field has exactly one
job.

---

## 2. The three-way split: storage vs view vs device

### 2.1 Storage — physical bytes

> A *storage* is a block of memory. It knows how many elements it
> holds, which device it lives on, and whether it should be freed.

```zig
pub const CpuStorage = struct {
    data: []f32,   // the flat buffer
    owned: bool,   // true = free me in deinit
};

pub const Storage = union(Device) {
    cpu: CpuStorage,
    cuda: void,    // populated by PR-ι
};
```

Two things to notice:

- `Storage` is a tagged union over `Device`. The tag and the payload
  are the same type, so mixing them is structurally impossible. You
  can't have a `.cuda` storage holding `CpuStorage`.
- `cuda = void` is a placeholder. Today the CUDA branch carries no
  data because CUDA isn't implemented. In PR-ι it becomes
  `cuda: CudaStorage` with a `CUdeviceptr`, a length, a device id, and
  an owned flag. Every `switch` on `Storage` already has a `.cuda =>`
  branch, so that PR adds data without reshaping the union.

Storage is *always* non-negotiable about what it represents. If you
have a `Storage.cpu`, you have a host-addressable `[]f32`. If you
have a `Storage.cuda`, you do not. The type system prevents accidents.

### 2.2 View — shape, strides, offset

> A *view* is a way of interpreting a sub-region of a storage as a
> multi-dimensional array.

The view is described by three fields on `Tensor`:

```zig
shape:   Shape,      // logical dimensions, e.g. (3, 2)
strides: Strides,    // step sizes per axis, e.g. (2, 1) for row-major
offset:  usize,      // starting offset into storage, in f32 elements
```

Physical offset of logical element `(c0, c1, ..., c_{n-1})` is:

```
phys = offset + Σᵢ cᵢ · strides[i]
```

A freshly `init`'d tensor has `offset = 0` and row-major strides. A
`view()` has the same shape/strides/offset. A `transpose2d()` has the
same offset but swapped strides. A future `slice(start, end)` would
set a non-zero offset and (potentially) a smaller shape.

The view is *cheap*: all three fields are small values stored by
copy. Creating a view never allocates.

### 2.3 Device — placement

```zig
pub const Device = enum { cpu, cuda };
```

Tensor.device duplicates information already encoded in the storage
union's tag. Why keep both?

- `device` is a convenience: `if (t.device == .cpu)` reads more
  naturally than `if (t.storage == .cpu)`.
- The invariant in §2.5 of `02c_tensor_invariants.md` ensures they
  agree. If they disagree, `checkInvariants` panics.

Once PR-ε and PR-δ stabilise and we fully migrate the ~800 `data[i]`
call sites, the top-level `device` alias may be removed; for now, the
dual representation keeps the diff small.

---

## 3. Why `[]f32` cannot represent CUDA memory

This is the single biggest reason PR-δ exists. Stage 7 cannot proceed
without it.

### 3.1 What `[]f32` means

A Zig slice is a host pointer plus a length:

```
[]f32 ≡ struct { ptr: [*]f32, len: usize }
```

Indexing `data[i]` compiles to a host load instruction: move `data.ptr
+ i * 4` into a register, dereference, write into `xmm0`. On x86 and
ARM, this works because `data.ptr` refers to memory the CPU can
address directly.

### 3.2 What a CUDA device pointer is

CUDA allocates memory with `cuMemAlloc_v2` and returns a
`CUdeviceptr`:

```c
typedef unsigned long long CUdeviceptr;
```

That's right — `CUdeviceptr` is *not* a C pointer at all. It's a
64-bit handle that refers to memory in the GPU's address space. The
CPU cannot dereference it. Loads and stores only work from inside a
CUDA kernel, or indirectly via `cuMemcpyHtoD_v2` /
`cuMemcpyDtoH_v2` copies.

If we tried to represent a CUDA allocation as a `[]f32`:

```zig
// DON'T DO THIS
const fake = @as([*]f32, @ptrFromInt(cuda_ptr))[0..len];
Tensor{ .data = fake, .device = .cuda, ... };
```

…every subsequent `fake[i]` would be a host load of an invalid host
address. On Linux it segfaults immediately. On Windows it reads
whatever happens to live at that virtual address in the process's
page tables — garbage floats, unrelated library state, whatever. The
bug manifests as "training suddenly diverges" or "model weights
become NaN after the first CUDA forward pass" with no hint about why.

This is the kind of bug that eats a weekend.

### 3.3 How `Storage` prevents it

With `Storage = union(Device) { cpu: CpuStorage, cuda: void }`, there
is no slot in a CUDA-tagged storage for a `[]f32`. The invariant in
§2.5 of chapter 02c requires:

```
storage == .cpu  ⇔  device == .cpu
storage == .cuda ⇔  device == .cuda
```

A CUDA tensor that tries to expose `tensor.data` sees a zero-length
compat slice (set by `Tensor.init` only in the `.cpu` branch). Any
host code that walks `tensor.data[i]` on a CUDA tensor iterates zero
times — the loop body never executes. A bug, but a loud one: the
"forward pass produced no output" is a failure, not a silent wrong
answer.

When PR-ι fleshes out `Storage.cuda`, every op will branch on the
storage tag at its entry point:

```zig
pub fn add(alloc: Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    try requireSameDevice(a, b);
    return switch (a.storage) {
        .cpu => try cpu_ops.add(alloc, a, b, tape),
        .cuda => try cuda_ops.add(alloc, a, b, tape),
    };
}
```

CPU code paths will never see a `.cuda` storage. CUDA code paths will
never see a `.cpu` storage. The type system makes the dispatch
exhaustive.

---

## 4. ASCII diagrams: the memory picture

### 4.1 An owned tensor

Shape `(3, 2)`, strides `(2, 1)`, offset `0`, storage is a length-6
heap buffer.

```
storage.cpu.data:   [ 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 ]
physical index:        0     1     2     3     4     5

Row-major iteration, logical index → physical:
  (0,0) → 0     (0,1) → 1
  (1,0) → 2     (1,1) → 3
  (2,0) → 4     (2,1) → 5

Tensor {
    data    = <points at storage.cpu.data>
    shape   = (3, 2)
    strides = (2, 1)
    offset  = 0
    storage = .{ .cpu = .{ .data = <heap>, .owned = true } }
}
```

### 4.2 A transpose view

Same storage. New tensor shares the bytes, different strides.

```
storage.cpu.data:   [ 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 ]
physical index:        0     1     2     3     4     5

Logical (2, 3) view with strides (1, 2):
  (0,0) → 0     (0,1) → 2     (0,2) → 4
  (1,0) → 1     (1,1) → 3     (1,2) → 5

Tensor {
    data    = <same slice as parent>
    shape   = (2, 3)
    strides = (1, 2)         ← note: swapped from (2, 1)
    offset  = 0
    storage = .{ .cpu = .{ .data = <heap>, .owned = false } }
    //                                     ^^^^^^
    //                                     view does not own the bytes
}
```

Same six floats. Different logical matrix. No copy.

### 4.3 A slice view (future)

`slice(t, 1, 3)` of a length-4 tensor.

```
storage.cpu.data:   [ 1.0 | 2.0 | 3.0 | 4.0 ]
physical index:        0     1     2     3

Tensor {
    shape   = (2,)
    strides = (1,)
    offset  = 1                  ← start at index 1
    storage = <same buffer>
}

Logical (0,) → phys = offset + 0*1 = 1   → 2.0
Logical (1,) → phys = offset + 1*1 = 2   → 3.0
```

The invariant check at §2.4 of chapter 02c ensures
`offset + max_logical_offset < storage.len()`:

```
offset + (2-1)*1 = 1 + 1 = 2 < 4  ✓
```

A slice `slice(t, 3, 5)` of the same length-4 tensor:

```
offset + (2-1)*1 = 3 + 1 = 4 < 4  ✗
                                   ^^^^^
                                   error.InvalidLayout
```

The invariant catches the off-by-one before it reads past the buffer.

### 4.4 A 3D inner transpose

`transposeInner2d` swaps the last two axes of a rank-3 tensor:

```
Input shape:  (B, M, K) = (2, 3, 4)
Input strides: (12, 4, 1)   row-major

Output shape:  (B, K, M) = (2, 4, 3)
Output strides: (12, 1, 4)   inner two swapped
Offset:        0  (unchanged — just reinterpreting strides)
```

The outer (batch) stride stays the same because the batch dimension
isn't being reordered; only the inner 2D matrix is transposed per
batch. This is exactly the operation attention needs when computing
`K^T` before `Q @ K^T`.

---

## 5. The `Storage.deinit` dispatch

Pre-PR-δ, `Tensor.deinit` was a two-liner:

```zig
pub fn deinit(self: *Tensor, allocator: Allocator) void {
    if (self.owned) allocator.free(self.data);
    self.* = undefined;
}
```

Post-PR-δ:

```zig
pub fn deinit(self: *Tensor, allocator: Allocator) void {
    self.storage.deinit(allocator);
    self.* = undefined;
}
```

with:

```zig
pub fn deinit(self: *Storage, allocator: Allocator) void {
    switch (self.*) {
        .cpu => |*s| {
            if (s.owned) allocator.free(s.data);
        },
        .cuda => {},  // will call cuMemFree_v2 once PR-ι fleshes this out
    }
}
```

The dispatch is explicit on the storage tag. When Stage 7 adds the
real `Storage.cuda`, we update *one function*, not every `deinit`
call site in the library.

This is the payoff of the seam. Every "CPU today, CUDA tomorrow"
change is a single-switch-case edit, not an audit of the whole
codebase.

---

## 6. `nonOwningStorage`: the view helper

Every view-constructing op needs to produce a `Storage` value that
references the parent's bytes but promises not to free them. Rather
than duplicate that logic four times (`view`, `reshape`,
`transpose2d`, `transposeInner2d`), we centralise it:

```zig
pub fn nonOwningStorage(src: Storage) Storage {
    return switch (src) {
        .cpu => |s| Storage{ .cpu = .{ .data = s.data, .owned = false } },
        .cuda => Storage{ .cuda = {} },
    };
}
```

The pattern every view op follows:

```zig
const v = Tensor{
    .data    = self.data,
    .shape   = new_shape,
    .strides = new_strides,
    // ...
    .storage = nonOwningStorage(self.storage),
    .offset  = self.offset,
    // ...
};
debugCheckInvariants(v);
return v;
```

One source of truth for "this is a view". If we ever need to change
what "non-owning" means — say, adding a reference count — we change
it in one function.

---

## 7. `Tensor.data` and `Tensor.owned` — removed (arch-phase-3)

The PR-δ diff originally kept `Tensor.data` and `Tensor.owned` as
transition aliases alongside the new `Storage` union. That migration
has now been completed in `arch-phase-3-complete`:

- **Phase 1a** added `cpuData()` and `isOwned()` accessor methods.
- **Phase 1b** migrated all ~900 call sites from `tensor.data[i]` to
  `tensor.cpuData()[i]` and removed the duplicate fields entirely.

The Tensor struct went from 14 fields to 12. `Storage` is now the
**single source of truth** for buffer ownership and location. There
is no longer a sync invariant between top-level fields and storage —
because the top-level fields don't exist.

### Accessing CPU data

```zig
// Read:
const val = tensor.cpuData()[i];

// Write:
tensor.cpuData()[i] = 42.0;

// Iterate:
for (tensor.cpuData()) |v| { ... }

// Ownership check (works for both CPU and CUDA):
if (tensor.isOwned()) { ... }
```

`cpuData()` asserts `device == .cpu` in debug builds. Calling it on a
CUDA tensor panics immediately rather than silently iterating an empty
slice. This catches device-confusion bugs at the call site.

The aliases are:

```zig
data:  []f32    // points at the same buffer as storage.cpu.data
owned: bool     // mirrors storage.cpu.owned
```

The invariant in §2.5 of chapter 02c enforces agreement. Any op that
mutates `owned` must also mutate `storage.cpu.owned` (as PR-ε's
`tape.keepAlive` does). Any op that reads `data` gets the same slice
a reader of `storage.cpu.data` would get.

**What this means for you:** new code should prefer the storage
accessors. Old code continues to work. In a future PR we will delete
the aliases and update every call site.

---

## 8. Why we did not do X

### 8.1 Why not a `StorageRef` / `Rc<Storage>` refcount?

PyTorch uses reference-counted storage so multiple tensors can share
the same bytes and the storage is freed when the last reference goes
away. It is the elegant solution.

Zig does not have a built-in Rc. Implementing one for f32 buffers is
perfectly doable — it is one struct and two functions — but:

1. Our view lifetime is already manageable without it. Every view is
   created in a scope that is shorter than the parent tensor's scope.
   Rust-style lifetime-elision rules are the contract.
2. Adding Rc would introduce a type parameter (`Storage(T)` for the
   refcount bookkeeping), complicating every switch.
3. The educational benefit is modest: Rc is a standard pattern, well
   documented elsewhere; a tensor library is not the place to teach
   it.

When Stage 8 adds model-parallel tensors or gradient accumulation,
Rc becomes attractive. Stage 6.5 does not need it.

### 8.2 Why not make `Tensor.data` private?

We considered it. The answer is the 800-call-site number again. Going
from a public field to an accessor method would rename every access.
Accessor methods also generate more lines of code (`tensor.dataPtr()`
vs `tensor.data`) in inner loops where conciseness matters for
readability. We will revisit this in a follow-up PR once the
storage-based accessors cover all of the codebase.

### 8.3 Why `cuda: void` instead of `cuda: ?CudaStorage`?

Two reasons:

1. An optional type would let us construct a `.cuda` storage with no
   payload, which is a nonsense state. Explicit `void` means the
   branch simply doesn't carry data; there is no "null CUDA" to
   accidentally produce.
2. Replacing `void` with a real `CudaStorage` in PR-ι is a
   one-character type edit. Every `switch` branch today handles the
   `.cuda` variant explicitly, so turning on the payload is a
   mechanical change.

### 8.4 Why `offset: usize` and not a bare slice?

A contiguous slice encodes offset implicitly — `data[2..]` is a slice
starting at element 2. We could represent views that way. But
non-contiguous views cannot use a slice: a transposed view needs
custom strides. Once we have to store shape and strides separately
from the underlying buffer, we also need to store *where in the
buffer* the view starts. `offset` is that field.

Keeping `offset: usize` also makes sub-view composition associative:
`slice(slice(t, 2, 6), 1, 3)` is a tensor with offset
`t.offset + 2 + 1 = t.offset + 3`. No intermediate allocation.

---

## 9. PyTorch parallels

The storage/view split is where our design most closely follows
PyTorch. The correspondence is almost one-to-one.

| Our field | PyTorch equivalent | PyTorch reference |
|---|---|---|
| `Storage.cpu.data: []f32` | `Storage::data_ptr()` returning `void*` | `c10/core/Storage.h` |
| `Storage.cpu.owned` | `Storage::use_count()` via Rc | `c10/core/StorageImpl.h` |
| `Storage.len()` | `Storage::nbytes()` | `c10/core/Storage.h` |
| `Tensor.shape` | `Tensor.sizes()` returning `IntArrayRef` | `ATen/core/TensorBase.h` |
| `Tensor.strides` | `Tensor.strides()` returning `IntArrayRef` | `ATen/core/TensorBase.h` |
| `Tensor.offset` | `Tensor.storage_offset()` returning `int64_t` | `ATen/core/TensorBase.h` |
| `Tensor.device` | `Tensor.device()` returning `Device` | `c10/core/Device.h` |
| `Storage.cuda` (future) | `Storage` on a CUDA device | `c10/cuda/CUDACachingAllocator.h` |
| `nonOwningStorage` | `Tensor::alias()` | `ATen/core/TensorBase.h` |

A PyTorch snippet that exercises all of these:

```python
>>> t = torch.arange(12).reshape(3, 4).float()
>>> t.storage()
 0.0
 1.0
 2.0
 ...
 11.0
>>> t.sizes()        # our .shape.dims
[3, 4]
>>> t.stride()       # our .strides.values, in elements
(4, 1)
>>> t.storage_offset()  # our .offset
0
>>> view = t[1:, 1:].t()
>>> view.storage() is t.storage()  # same underlying bytes
True
>>> view.sizes(), view.stride(), view.storage_offset()
([3, 2], (1, 4), 5)
```

Same mechanical result as our `slice` + `transpose2d` composition
would produce. The conceptual model is identical; only the language
differs.

---

## 10. Things we did not fix in PR-δ

PR-δ is intentionally scope-controlled. These follow-ups are
documented here so you know they exist:

### 10.1 The 800 `tensor.data[i]` call sites

**Completed** in `arch-phase-3-complete` (Phase 1a/1b). All sites now
use `tensor.cpuData()[i]`. The `.data` field no longer exists.

### 10.2 No `slice` op

The `offset` field is populated but no op currently produces a
non-zero offset. The slice op is a one-function addition (see the
Exercise below); we deferred it because nothing in Stage 6 actually
needed sliced views.

### 10.3 `cpuData()` accessor on `Tensor`

**Completed** in `arch-phase-3-complete` (Phase 1a). The accessor is:

```zig
pub inline fn cpuData(self: Tensor) []f32 {
    std.debug.assert(self.device == .cpu);
    return self.storage.cpu.data;
}
```

It asserts `device == .cpu` in debug builds, replacing the old pattern
where `tensor.data` on a CUDA tensor silently returned an empty slice.

### 10.4 No multi-device storage

Our `Device` enum has exactly two values: `.cpu` and `.cuda`. No
multi-GPU, no hip / rocm / metal, no mixed precision. That is a
deliberate scope boundary.

---

## 11. Common mistakes

### "I added a view op and forgot to call `nonOwningStorage`"

Your new view tensor has `storage.cpu.owned = true`. When the view
goes out of scope and `deinit` fires, it frees the parent's buffer;
the parent's own `deinit` then frees it again → double free → GPA
panic or allocator corruption.

Fix: always pass the parent's storage through `nonOwningStorage`:

```zig
.storage = nonOwningStorage(self.storage),
.offset  = self.offset,    // don't forget this!
```

The invariant check in §2.5 of chapter 02c catches the resulting
drift between `owned` and `storage.cpu.owned` in debug builds.

### "My op returns a tensor and `checkInvariants` panics on it"

Most likely, you forgot to update `strides.rank` alongside a change
to the shape, or you computed a stride that makes the last logical
element reach past the buffer. Work through §2.4 of chapter 02c on
paper for your specific shape/strides.

### "I want to write a CUDA tensor by hand and I don't know what to put in `storage.cuda`"

You can't yet — `cuda = void` carries no payload today. The
PR-γ failure-mode test constructs such a tensor by explicitly setting
`storage = .{ .cuda = {} }` and `device = .cuda`, and only tests the
device-mismatch path (never the data path). Once PR-ι introduces
`CudaStorage`, the hand-built form will require real fields
(`ptr`, `len`, `device_id`, `owned`).

### "How do I access the CPU data buffer?"

Use `tensor.cpuData()` — it returns `[]f32` and asserts `device == .cpu`
in debug builds. For ownership checks, use `tensor.isOwned()` which
works for both CPU and CUDA tensors. The old `tensor.data` field no
longer exists (removed in arch-phase-3).

---

## 12. Exercises

### Exercise 1 — Build a slice op

Write `pub fn slice(self: Tensor, start: usize, end: usize)
LabError!Tensor` that returns a view of `self` restricted to the
first-dimension range `[start, end)`. Signature:

```zig
pub fn slice(self: Tensor, start: usize, end: usize) LabError!Tensor {
    if (end <= start or end > self.shape.dims[0]) {
        return error.InvalidArgument;
    }
    // 1. Compute new shape: dims[0] shrinks, others unchanged.
    // 2. Compute new offset: self.offset + start * strides[0].
    // 3. Keep strides and storage unchanged.
    // 4. nonOwningStorage + debugCheckInvariants.
}
```

Add a test that slices a `(10, 3)` tensor with `slice(2, 5)`, checks
the result has shape `(3, 3)`, offset `2 * strides[0] = 2 * 3 = 6`,
and that `view.at(&.{0, 0}) == parent.at(&.{2, 0})`.

### Exercise 2 — Make `tensor.cpuData()` an accessor

Add `pub fn cpuData(self: Tensor) LabError![]f32` that returns the
storage slice or `error.DeviceMismatch`. Migrate five call sites
(pick them from `src/tensor/ops/reduce.zig`) from `tensor.data` to
`try tensor.cpuData()`. Run `zig build test` to confirm nothing
regresses.

Reflect: how many `try` statements did you add? How readable is the
result? Is this a migration you want to do in one PR or several?

### Exercise 3 — Predict the output

Given:

```zig
var t = try Tensor.init(alloc, Shape.init3D(2, 3, 4));
for (t.data, 0..) |*v, i| v.* = @floatFromInt(i);

const inner = try ops_shape.transposeInner2d(t);
// inner.shape = (2, 4, 3), strides = (12, 1, 4), offset = 0

const row = inner.at(&.{ 1, 2, 0 });
```

Work out `row` by hand using the offset formula, without running the
code. Then run it and verify.

Hint: `phys = offset + 1*12 + 2*1 + 0*4 = 14`. What was in
`t.data[14]` before the transpose?

---

## 13. File reference

| File | What to read |
|---|---|
| `src/tensor/tensor.zig` | `CpuStorage`, `Storage`, `Tensor` struct (fields), `Tensor.init`, `deinit`, `view`, `reshape`, `transpose2d`, `nonOwningStorage` |
| `src/tensor/ops/shape_ops.zig` | `transposeInner2d` (the only other view op outside the core Tensor methods) |
| `src/core/device.zig` | The `Device` enum that tags `Storage` |

---

## 14. Test commands

```bash
# Run the storage-specific tests
zig build test -- --test-filter "storage"

# Run the view-construction tests
zig build test -- --test-filter "view"

# Confirm the alias invariants
zig build test -- --test-filter "checkInvariants"
```

Next chapter: `03c_saved_tensors.md` — the single biggest conceptual
change in Stage 6.5. How the tape owns its own copies of backward-
relevant data, why `keepAlive` is no longer needed, and why this
matches PyTorch's `ctx.save_for_backward`.
