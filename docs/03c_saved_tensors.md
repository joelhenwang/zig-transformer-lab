# 03c — Saved Tensors and Why `keepAlive` Died

This is the most conceptually important chapter in Stage 6.5. It
documents the refactor that moved ownership of backward-relevant data
from "whoever happens to call keepAlive correctly" to "the tape
itself". If you understand this chapter, you understand how PyTorch's
`ctx.save_for_backward` works under the hood, why manual buffer
management for autograd is a footgun, and why removing 86 lines of
hand-written bookkeeping made the code simultaneously shorter, safer,
and easier to teach.

Read `docs/03_autograd.md` first to see the tape model this chapter
refines.

---

## 1. The problem, stated once

A backward pass needs data from the forward pass. Consider `mul`:

```
Forward:   c = a * b
Backward:  ∂c/∂a = b,   ∂c/∂b = a
```

To compute the gradient of `a`, backward needs the value of `b`. To
compute the gradient of `b`, backward needs the value of `a`. Neither
can be reconstructed from `c` alone (`c / b` would give `a`, but only
if `b` has no zeros, and `div` is numerically painful).

So when the forward records the `mul` node, it must also stash `a`
and `b` somewhere that survives until backward runs.

The question this chapter answers is: *where exactly does that
stashing happen, and who owns the bytes?*

---

## 2. The `keepAlive` era (Stages 3–6, before PR-ε)

Our pre-PR-ε tape stored saved tensors *by value*:

```zig
// node.zig
pub const SavedData = union(enum) {
    nothing,
    tensor_ref: Tensor,
    tensor_pair: struct { a: Tensor, b: Tensor },
    // ... etc.
};
```

A `Tensor` struct held by value includes its shape, strides, device,
ownership flag — and crucially a `.data: []f32` slice pointer. The
slice pointer refers to bytes allocated by someone else's
`allocator.alloc(f32, n)` call.

### 2.1 The dangling-slice bug

The forward pass in a real training loop looks like this:

```zig
pub fn forward(self: *Layer, input: Tensor, tape: ?*Tape) !Tensor {
    var h = try ops.matmul(allocator, input, self.weight, tape);
    defer h.deinit(allocator);                       // (*)
    var y = try ops.add(allocator, h, self.bias, tape);
    return y;                                        // caller frees y
}
```

At line `(*)`, `h.deinit` fires when `forward` returns. `h` had
`owned = true`, so its heap buffer is freed.

But `tape.record` was called from inside `ops.add` with
`saved = .{ .tensor_pair = .{ .a = h, .b = self.bias } }`. The saved
tensor snapshot's `.data` slice still points at the heap buffer that
was just freed.

Later, `tape.backward(&loss)` tries to read `saved.tensor_pair.a.data[i]`
and either crashes (safe-release builds with a poison pattern) or
reads whatever the allocator reassigned the bytes to (release-fast).

### 2.2 The `keepAlive` workaround

We added a per-tape list of buffers the tape has "taken over":

```zig
// pre-PR-ε tape.zig
kept_alive: std.ArrayList([]f32),

pub fn keepAlive(self: *Tape, tensor: *Tensor) !void {
    if (tensor.owned) {
        try self.kept_alive.append(self.allocator, tensor.data);
        tensor.owned = false;   // stop the tensor from freeing on deinit
    }
}
```

The layer author's job became: for every intermediate tensor whose
lifetime ended before backward, call `tape.keepAlive(&t)` to transfer
ownership of its buffer to the tape. The forward code from §2.1
becomes:

```zig
pub fn forward(self: *Layer, input: Tensor, tape: ?*Tape) !Tensor {
    var h = try ops.matmul(allocator, input, self.weight, tape);
    if (tape) |t| try t.keepAlive(&h);              // <-- must remember
    defer h.deinit(allocator);                      // now a no-op
    var y = try ops.add(allocator, h, self.bias, tape);
    return y;
}
```

This worked. But it forced every layer author to think about
backward's buffer lifetime while writing forward code. Get it wrong
and the bug surfaces two function calls later in
`backwardMul`, with no obvious connection to the missing `keepAlive`.

### 2.3 How bad it got

The final pre-PR-ε audit showed:

| File | `keepAlive` call sites |
|------|---:|
| `src/autograd/gradcheck.zig` | 46 |
| `src/nn/attention.zig` | 10 |
| `src/nn/layernorm.zig` | 10 |
| `src/nn/linear.zig` | 7 |
| `src/nn/model.zig` | 6 |
| `src/nn/block.zig` | 5 |
| `src/nn/mlp.zig` | 2 |
| `src/lab/train.zig` | 2 |
| **Total** | **88** |

Eighty-eight manual bookkeeping calls spread across nine files. Eight
of them went into the training loop itself. Removing any one produces
a silent correctness bug that only manifests under AddressSanitizer
or with a specific allocator configuration.

### 2.4 The educational cost

`keepAlive` taught a *wrong* mental model: "in this autograd library,
layer authors are responsible for the lifetime of tape-internal
data." That is the opposite of what PyTorch does. A student who
learned the `keepAlive` pattern would then be surprised that
PyTorch's `ctx.save_for_backward(x)` makes no such demand.

A pedagogical library should teach the right thing, not a
fixable-but-wrong approximation. That was the ultimate motivation
for PR-ε.

---

## 3. The post-PR-ε model: tape owns its own copies

The fix is conceptually small: when `tape.record()` is called, walk
the `SavedData` payload, allocate fresh copies of every buffer it
references, and rewrite the snapshots to point at the copies.

```zig
// post-PR-ε tape.zig (excerpt)
pub fn record(self: *Tape, node: Node) LabError!NodeId {
    const id = self.next_id;
    self.next_id += 1;

    const owned_saved = try self.takeOwnershipOfSaved(node.saved);

    try self.nodes.append(self.allocator, Node{
        .id = id,
        .op = node.op,
        .parents = node.parents,
        .n_parents = node.n_parents,
        .saved = owned_saved,
    });

    return id;
}

fn takeOwnershipOfSaved(self: *Tape, saved: SavedData) LabError!SavedData {
    return switch (saved) {
        .nothing, .tensor_scalar, .reduce_info => saved,
        .tensor_ref => |t| SavedData{
            .tensor_ref = try self.cloneTensorData(t),
        },
        .tensor_pair => |p| SavedData{
            .tensor_pair = .{
                .a = try self.cloneTensorData(p.a),
                .b = try self.cloneTensorData(p.b),
            },
        },
        .ce_info => |info| SavedData{
            .ce_info = .{
                .logits = try self.cloneTensorData(info.logits),
                .targets = try self.cloneSlice(info.targets),
            },
        },
        .embedding_info => |info| SavedData{
            .embedding_info = .{
                .weight = try self.cloneTensorData(info.weight),
                .indices = try self.cloneSlice(info.indices),
            },
        },
    };
}

fn cloneTensorData(self: *Tape, t: Tensor) LabError!Tensor {
    const copy = self.allocator.dupe(f32, t.data) catch return error.OutOfMemory;
    self.kept_alive.append(self.allocator, copy) catch {
        self.allocator.free(copy);
        return error.OutOfMemory;
    };
    var c = t;
    c.data = copy;
    c.owned = false;
    c.storage = .{ .cpu = .{ .data = copy, .owned = false } };
    return c;
}
```

### 3.1 What changed

- `record()` no longer stores the caller's snapshots verbatim. It
  calls `takeOwnershipOfSaved`, which returns a new `SavedData` with
  tape-owned buffers.
- `cloneTensorData` allocates a copy, tracks the copy in
  `self.kept_alive` (the same list that existed before), and returns
  a Tensor snapshot whose `.data` points at the copy.
- The storage field of the returned snapshot also points at the
  copy, with `owned = false` — the tape owns it through
  `kept_alive`, not through the snapshot's `Storage.deinit` call.
- `cloneSlice` does the equivalent for the `[]const f32` data used
  by `ce_info` (targets) and `embedding_info` (indices).

### 3.2 What didn't change

- The `SavedData` union shape is identical. Every backward function
  reads it the same way it did before.
- The `Tensor` type is unchanged; we still store snapshots by value.
- The `Tape.kept_alive` list exists for the same purpose — it tracks
  buffers the tape is responsible for freeing.

The refactor is local to `record()` and two helper functions.
Backward.zig, the 22 different `OpKind` variants, and every op's
recording path are untouched.

### 3.3 Why this works

The forward code from §2.1 now:

```zig
pub fn forward(self: *Layer, input: Tensor, tape: ?*Tape) !Tensor {
    var h = try ops.matmul(allocator, input, self.weight, tape);
    defer h.deinit(allocator);                      // genuine free, fine
    var y = try ops.add(allocator, h, self.bias, tape);
    return y;
}
```

No `keepAlive`. `h` is freed at scope exit because it is truly
finished with. When `ops.add` called `tape.record(..., saved: {
tensor_pair: { a: h, b: self.bias }})`, the tape *copied* h's and
self.bias's data into its own buffers. The copy survives `h`'s
deinit. Backward reads the copy.

### 3.4 `keepAlive` became a no-op

We kept the old function signature for one transition PR:

```zig
pub fn keepAlive(self: *Tape, tensor: *Tensor) !void {
    _ = self;
    _ = tensor;
    // Intentionally empty.
}
```

This let the 86 call sites compile unchanged during the PR-ε
transition. After the refactor, we removed all of them in the same
PR. The function body remains empty today; a follow-up cleanup will
delete the function itself.

---

## 4. A worked memory trace: `y = gelu(x @ W)`

Let's trace exactly what ends up in the tape, byte by byte, for a
tiny example.

```zig
// Setup: x is (1, 2), W is (2, 1), so x@W is (1, 1), gelu((1,1)) is (1, 1).
var x = try Tensor.init(alloc, Shape.init2D(1, 2));  // data: [1.0, 2.0]
var W = try Tensor.init(alloc, Shape.init2D(2, 1));  // data: [3.0, 4.0]
x.requires_grad = true;
W.requires_grad = true;

var tape = Tape.init(alloc);
defer tape.deinit();

_ = try tape.trackLeaf(&x);   // node 0
_ = try tape.trackLeaf(&W);   // node 1

var h = try ops.matmul(alloc, x, W, &tape);  // node 2
defer h.deinit(alloc);

const y = try ops.unary.geluExact(alloc, h, &tape);  // node 3
// y lives on for backward; h.deinit fires at scope exit
```

### 4.1 After `trackLeaf(&x)` (node 0 recorded)

```
Heap (alloc'd by Tensor.init):
  A: [1.0, 2.0]         ← x.data.ptr

Tape:
  nodes:       [ { id=0, op=undefined, parents=(null,null), n_parents=0, saved=.nothing } ]
  kept_alive:  []
  leaf_map:    { 0 → &x }
```

`trackLeaf` stores no data — leaf nodes have `saved = .nothing`.

### 4.2 After `trackLeaf(&W)` (node 1 recorded)

```
Heap:
  A: [1.0, 2.0]         ← x.data.ptr
  B: [3.0, 4.0]         ← W.data.ptr

Tape:
  nodes:       [ leaf0, leaf1 ]
  kept_alive:  []
  leaf_map:    { 0 → &x, 1 → &W }
```

### 4.3 After `matmul(x, W)` (node 2 recorded)

`matmul` computes x @ W = [11.0] into a fresh output tensor `h`:

```
Heap:
  A: [1.0, 2.0]         ← x.data.ptr
  B: [3.0, 4.0]         ← W.data.ptr
  C: [11.0]             ← h.data.ptr (freshly allocated)
```

Then `matmul` calls `tape.record(..., saved: tensor_pair{ a: x, b: W })`.

Inside `record`, `takeOwnershipOfSaved` runs. Because the saved data
is a `.tensor_pair`, it calls `cloneTensorData(x)` and
`cloneTensorData(W)`:

```zig
// cloneTensorData(x)
const copy = alloc.dupe(f32, x.data);  // copy of A: [1.0, 2.0]
// → heap now has buffer D: [1.0, 2.0]
self.kept_alive.append(copy);
// → kept_alive now [D]
var c = x;                  // copy struct
c.data = copy;              // point at D
c.storage.cpu.data = copy;
c.storage.cpu.owned = false;
return c;

// cloneTensorData(W)
const copy = alloc.dupe(f32, W.data);  // copy of B: [3.0, 4.0]
// → heap now has buffer E: [3.0, 4.0]
self.kept_alive.append(copy);
// → kept_alive now [D, E]
... similar ...
return c;
```

After `record` completes:

```
Heap:
  A: [1.0, 2.0]         ← x.data.ptr (the user's x)
  B: [3.0, 4.0]         ← W.data.ptr (the user's W)
  C: [11.0]             ← h.data.ptr
  D: [1.0, 2.0]         ← tape-owned copy of x
  E: [3.0, 4.0]         ← tape-owned copy of W

Tape:
  nodes:
    [0] leaf,
    [1] leaf,
    [2] { op=matmul, parents=(0,1), n_parents=2,
          saved=.tensor_pair{ a: {data→D, ...}, b: {data→E, ...} } }
  kept_alive: [D, E]
```

Two new heap buffers exist. They duplicate the user's data. That is
the memory cost of the new model.

### 4.4 After `gelu(h)` (node 3 recorded)

`gelu` allocates a fresh output `y` with gelu of 11.0, then records
`saved = .tensor_ref = h`:

```
Heap:
  A [x], B [W], C [h], D [copy of x], E [copy of W], F [y=gelu(11)]
```

`takeOwnershipOfSaved` sees `.tensor_ref` and calls
`cloneTensorData(h)`:

```
Heap after clone:
  A [x], B [W], C [h], D [copy x], E [copy W], F [y], G [copy of h]

Tape:
  nodes:
    [0] leaf,
    [1] leaf,
    [2] matmul{saved: pair(→D, →E)},
    [3] gelu{saved: ref(→G)}
  kept_alive: [D, E, G]
```

### 4.5 After `h.deinit` fires

`h` goes out of scope. Its `deinit` frees buffer C:

```
Heap:
  A [x], B [W],    [freed], D [copy x], E [copy W], F [y], G [copy h]
  kept_alive: [D, E, G]
```

The tape's saved `gelu` node references buffer G, not C. Backward is
safe.

### 4.6 What pre-PR-ε would have done here

No copies, no `kept_alive` for the saved data. The tape's node 2
would reference buffers A and B *directly* (fine — they live for the
whole program). Node 3 would reference buffer C *directly*.

When `h.deinit` fires and C is freed, node 3's saved data slice
becomes a dangling pointer. Backward reads freed memory. `keepAlive`
existed to prevent this: the pre-ε `forward` would have had to call
`tape.keepAlive(&h)`, moving C into `kept_alive` so deinit did not
free it.

---

## 5. Memory accounting

"We allocate copies" sounds expensive. Let's quantify for the actual
model used in training.

### 5.1 Model size

From `TransformerConfig` at typical Stage 6 settings:

```
V = 2000  (vocab)        D = 32   (d_model)
T = 16    (seq_len)      F = 128  (d_ff)
B = 4     (batch)        1 block, 1 head
```

### 5.2 Per-step forward intermediates

Walking through the model (token_embed → pos_embed → add → block
(ln1 → attn → residual → ln2 → mlp → residual) → ln_f → lm_head):

| Step | Output shape | f32 count |
|---|---|---:|
| token_embed | (B, T, D) | 2 048 |
| pos_embed | (1, T, D) | 512 |
| add | (B, T, D) | 2 048 |
| ln1 composed intermediates (7 tensors) | (B, T, D) × 7 | 14 336 |
| attn Q | (B, T, D) | 2 048 |
| attn K | (B, T, D) | 2 048 |
| attn V | (B, T, D) | 2 048 |
| attn K^T | (B, D, T) | 2 048 |
| attn scores | (B, T, T) | 1 024 |
| attn mask 3D | (1, T, T) | 256 |
| attn weights | (B, T, T) | 1 024 |
| attn out | (B, T, D) | 2 048 |
| attn projection | (B, T, D) | 2 048 |
| residual 1 | (B, T, D) | 2 048 |
| ln2 composed intermediates (7 tensors) | (B, T, D) × 7 | 14 336 |
| mlp fc1 | (B, T, F) | 8 192 |
| mlp gelu | (B, T, F) | 8 192 |
| mlp fc2 | (B, T, D) | 2 048 |
| residual 2 | (B, T, D) | 2 048 |
| ln_f composed intermediates (7 tensors) | (B, T, D) × 7 | 14 336 |
| lm_head | (B, T, V) | 128 000 |
| cross-entropy | (scalar) | 1 |

Total intermediates ≈ **210 000 f32 ≈ 840 KB**.

### 5.3 What the tape saves

Not every intermediate is saved by backward — some ops (like `neg`)
don't need inputs at all. The ops that save their full input tensor
are: `mul`, `div`, `matmul`, `matmulBatch`, `softmax`, `exp`, `log`,
`sqrt`, `gelu`, `relu`, `log_softmax`, `embedding`, `cross_entropy`
(saves logits), and `reshape`/`transpose2d` (save a reference).

Rough estimate of bytes the tape copies per forward pass:

- matmul and matmulBatch save *both* inputs: ~4 × `(B, T, D)` +
  ~200 KB for lm_head's `(D, V)` pair = ~350 KB
- softmax, gelu, mul save one input of `(B, T, D)` or `(B, T, F)` or
  `(B, T, T)`: ~150 KB combined
- ln composed intermediates save many small tensors: ~60 KB combined
- cross_entropy saves logits `(B*T, V)`: 512 KB

Total copies ≈ **1.0 MB per step** for our 1-block 1-head model.

On a 16 GB machine this is negligible. For GPT-2-small (117M params,
batch 8, seq 1024), the same analysis would put it at roughly 200 MB
per step — still fine for most GPUs, and PyTorch pays the same cost
for its per-op `ctx.save_for_backward` allocations.

### 5.4 The tradeoff in one sentence

> We pay ~1 MB of extra memory per training step to eliminate 86 lines
> of distributed bookkeeping and reduce a whole class of use-after-
> free bugs to zero.

Good deal.

---

## 6. Alternative designs we considered

### 6.1 Refcounted storage (PyTorch's choice)

Give every `Storage` a reference count. Multiple tensors can point at
the same storage; the storage is freed when the last reference drops.
The tape holds refs to the storages it cares about. No copying.

**Pros:** zero memory overhead; exactly PyTorch's model.

**Cons:** Zig has no built-in Rc. Implementing one is ~50 LOC but
adds a type parameter to `Storage`, complicating every switch.
Reference counting also obscures ownership: "who is responsible for
freeing this?" is answered by "whoever drops the last ref", which
becomes hard to predict.

**Verdict:** right long-term choice, wrong for Stage 6.5. PR-ε favours
clarity over optimality.

### 6.2 Detach and clone (TensorFlow 1.x)

Every saved tensor is a `detach()` + `clone()` at record time. The
detached tensor is independent of the original graph. This is almost
what we ended up with, but `detach` in TF1 also involves breaking the
gradient link — which we don't need.

**Pros:** same buffer isolation as ours.

**Cons:** introduces a "detached" concept that doesn't exist
elsewhere in our system, adding a vocabulary item for no gain.

**Verdict:** equivalent to our solution with extra noise. Rejected.

### 6.3 Lazy materialisation

Don't save tensors at all; save the forward op signatures and replay
forward from scratch during backward. Aka "activation checkpointing"
or "gradient checkpointing".

**Pros:** near-zero saved-state memory; standard technique for huge
models.

**Cons:** drastically more compute; changes what backward means
(because recomputing can see different random draws, different
non-determinism). Requires every op to be deterministic given its
inputs.

**Verdict:** great for Stage 9 optimisation work. Wrong for a
pedagogical baseline where "backward does exactly what the math
says" must be obvious.

### 6.4 Per-op owned saved tensor type

Introduce `SavedTensor = struct { data: []f32, shape, strides,
offset, device }` as a distinct type from `Tensor`. Every backward
function would accept `SavedTensor` inputs, making the "this is
tape-owned" distinction visible in signatures.

**Pros:** strongest type-system guarantee that backward functions
don't accidentally hold onto caller buffers.

**Cons:** roughly doubles the autograd diff size. Every switch in
`backward.zig` would need to learn the new type. For 2× the churn,
we get a static guarantee that the dynamic invariant checks already
deliver.

**Verdict:** the "right" design if we were starting from scratch.
Doesn't justify the churn retrofitting an existing library.

---

## 7. The `keepAlive` deprecation path

PR-ε's final state:

```zig
/// DEPRECATED (PR-ε): `tape.record()` now copies saved buffers
/// automatically, so layer code no longer needs to call this.
/// It is kept as a no-op for one transition PR so old call sites
/// compile while they are removed; the function will be deleted
/// entirely once `nn/` and `gradcheck` are clean.
pub fn keepAlive(self: *Tape, tensor: *Tensor) !void {
    _ = self;
    _ = tensor;
    // Intentionally empty. See doc comment.
}
```

The PR also removed all 86 call sites from `src/nn/`, `src/lab/`, and
`src/autograd/gradcheck.zig`. A grep of the tree shows only the
function definition and one historical-context comment remain.

The function itself will be deleted in a follow-up cleanup PR. We
keep it for one PR so that if someone pulls an intermediate commit,
their code still compiles. This is the same pattern PyTorch uses for
deprecating APIs.

---

## 8. PyTorch parallels

The post-PR-ε behaviour is a direct match for PyTorch's
`torch.autograd.Function.save_for_backward`. A side-by-side:

### PyTorch

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        y = x @ w
        ctx.save_for_backward(x, w)   # takes refcounted ownership
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors      # survived any intermediate frees
        grad_x = grad_y @ w.t()
        grad_w = x.t() @ grad_y
        return grad_x, grad_w
```

### Ours

```zig
pub fn matmul(alloc: Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    // ... compute y ...
    if (tape) |t| if (a.requires_grad or b.requires_grad) {
        const node_id = try t.record(Node{
            .op       = .matmul,
            .parents  = .{ a.tape_node, b.tape_node },
            .n_parents = 2,
            .saved    = .{ .tensor_pair = .{ .a = a, .b = b } },
            //            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            //            takeOwnershipOfSaved(node.saved) copies
            //            the data. This is save_for_backward.
        });
        y.tape_node = node_id;
    }
    return y;
}
```

### Side-by-side semantics

| Aspect | PyTorch | Ours |
|---|---|---|
| Ownership transfer | Refcount increment in `save_for_backward` | Eager copy in `takeOwnershipOfSaved` |
| Memory cost | Near-zero (shared storage) | ~1 MB per step for our tiny model |
| Safety against forward-pass drops | Yes (refcount holds storage alive) | Yes (we have an independent copy) |
| Safety against in-place mutation | Partial (version counter detects it) | Yes by construction (caller can mutate original freely) |
| Layer author burden | Call `save_for_backward(x, w)` explicitly | Record with `saved = tensor_pair{...}`; tape copies automatically |

The `save_for_backward` vocabulary is worth knowing because it's what
most PyTorch tutorials and stack traces use. When you read
`ctx.save_for_backward(x)` in PyTorch code, mentally translate it to
our pattern: "the op includes `x` in `Node.saved`, and
`tape.record` makes sure the data outlives any intermediate frees."

---

## 9. The in-place mutation question

One subtlety worth flagging: in PyTorch, `x.mul_(2)` (in-place
multiply) interacts with autograd via a *version counter*. If backward
is called and the saved `x` has a different version than when it was
saved, PyTorch raises an error because the saved value is stale.

In our system, `tape.record` takes a copy at record time. The caller
can do whatever they want with the original `x` afterwards — including
in-place mutation — and backward still sees the value at the moment
of save.

```zig
var x = try Tensor.init(alloc, Shape.init1D(3));
x.data[0] = 1.0;

var y = try ops.mul(alloc, x, x, &tape);
// Tape now holds a copy of x at this point, with data[0] = 1.0.

x.data[0] = 999.0;  // caller mutates — not a problem for us.

try tape.backward(&y);
// Backward reads x.data[0] = 1.0 (from the saved copy), not 999.0.
```

PyTorch's version-counter approach is strictly more conservative but
also more confusing. In exchange for slightly higher memory cost, we
get simpler semantics: what goes into the tape is what backward sees,
always.

---

## 10. Common mistakes

### "I still see `keepAlive` in the code"

Three legitimate places remain:

1. `src/autograd/tape.zig:~382` — the deprecated no-op definition.
2. `src/autograd/tape.zig` — a doc comment describing the old
   pattern for historical reference.
3. `src/nn/linear.zig` — a comment in an older worked example.

Everywhere else should be gone. If you see a `tape.keepAlive(...)`
actually called, it compiles but does nothing, so there's no
correctness problem; but it's dead code and should be removed.

### "My new op crashes in backward with `index out of bounds`"

Your op is probably recording a `SavedData` variant you didn't wire
through `takeOwnershipOfSaved`. If you add a new variant to the
`SavedData` union, you must:

1. Extend `takeOwnershipOfSaved`'s switch to copy whatever buffers
   the variant references.
2. Use `cloneTensorData` for full tensors, `cloneSlice` for `[]const
   f32`, or let the variant pass through if it contains no buffers
   (like `tensor_scalar` or `reduce_info`).

The compiler will not catch this — the switch in
`takeOwnershipOfSaved` is *not* an exhaustive `switch(saved)`
because we pattern-match selectively. Adding a test that records the
new op in a scope that frees its inputs before backward is the
safest way to catch the bug.

### "My training run used 2× more memory after PR-ε"

Unlikely but possible if your training step holds many tensors alive
across many forward passes before the tape is destroyed. The
`kept_alive` list grows until `Tape.deinit`; if your loop uses one
tape per epoch instead of one per step, the copies accumulate.

Fix: create and destroy the tape per step:

```zig
for (0..num_steps) |_| {
    var tape = Tape.init(alloc);
    defer tape.deinit();
    // forward + backward + step
}
```

This is what `src/lab/train.zig` does.

### "My tape.zig build failed with `access of pointer to a union's payload through a copy`"

PR-δ's `keepAlive` fix had this problem. When you switch on
`tensor.storage`, capturing with `|*s|` gives you a pointer to a
*copy* of the union, not the original. Writing through that pointer
doesn't mutate the tensor. The fix is to access the field directly:

```zig
// WRONG (silently no-ops the mutation)
switch (tensor.storage) {
    .cpu => |*s| s.owned = false,
    .cuda => {},
}

// RIGHT
if (tensor.storage == .cpu) {
    tensor.storage.cpu.owned = false;
}
```

---

## 11. Exercises

### Exercise 1 — Add a `tanh` op with proper saved-data handling

`tanh(x)` has backward `∂tanh/∂x = 1 - tanh²(x)`. Backward needs
`tanh(x)` (which is `y`, the output) rather than `x` — slightly
unusual.

Add:

1. A `.tanh` variant to `OpKind` in `src/autograd/node.zig`.
2. A `pub fn tanh(alloc, t, tape)` in `src/tensor/ops/unary.zig`
   that records `saved: .tensor_ref = output_of_this_op`.
3. A `backwardTanh` in `src/autograd/backward.zig` that computes
   `grad_x = grad_y * (1 - y²)`.
4. A test that runs tanh, frees the input in an explicit scope, and
   asserts backward still produces the correct gradient.

Without PR-ε, the test would have required a `tape.keepAlive(&y)`
call in the forward. With PR-ε, it does not.

### Exercise 2 — Prove the memory bound by instrumenting

Add a counter to `cloneTensorData` that prints
"tape cloned N bytes for op X". Run
`zig build run-example -Dexample=04_overfit_one_batch` and sum the
counter across one training step. Compare to the §5.3 estimate.

### Exercise 3 — Catch a missing `takeOwnershipOfSaved` branch

Create a synthetic `SavedData` variant called `.tensor_triple` that
carries three tensors. Do *not* extend `takeOwnershipOfSaved` to
handle it. Write an op that records with this variant. Predict what
happens in backward. Run and verify.

Then fix `takeOwnershipOfSaved`. Note how small the change is — this
is the whole point of centralising ownership.

---

## 12. File reference

| File | What to read |
|---|---|
| `src/autograd/tape.zig` | `record`, `takeOwnershipOfSaved`, `cloneTensorData`, `cloneSlice`, deprecated `keepAlive` |
| `src/autograd/node.zig` | `SavedData` union — the shape of what the tape owns |
| `src/tensor/ops/*.zig` | Every op's recording block — note: no `keepAlive` anywhere |
| `src/nn/*.zig` | Layer forwards — note: no `keepAlive` anywhere |
| `src/autograd/gradcheck.zig` | Gradient check harness — note: 46 `keepAlive` calls removed |

---

## 13. Test commands

```bash
# Full test suite must pass
zig build test

# Confirm the training loss trajectory is unchanged
zig build run-example -Dexample=04_overfit_one_batch
# Final loss should match the pre-PR-ε value of 0.537453 exactly.

# Grep-audit that keepAlive is gone
Select-String -Path src/nn/*.zig, src/lab/*.zig -Pattern 'keepAlive'
# Should return no results except comments.
```

Next chapter: `07c_optimizer_state.md` — why AdamW state keyed by
pointer was a latent bug, and how `ParamId` fixes it.
