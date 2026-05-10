# 07c — Optimizer State and `ParamId`

This chapter explains a small but pernicious class of bug: optimizer
state (Adam's `m` and `v` moments; SGD's velocity) getting silently
reset because the key that identifies each parameter is not stable
across a parameter's full lifetime. PR-ζ fixes this by keying state
on a parameter *identity*, `ParamId`, rather than on a parameter
*pointer*.

It is the shortest of the Stage 6.5 hardening PRs — about 200 lines
changed — but it teaches a general principle that matters for any
system that associates per-object state with objects whose addresses
might move.

Read `docs/04_nn.md` (the optimizer primer) first.

---

## 1. What optimizer state is and why it exists

A gradient-descent optimizer that only ever sees fresh gradients
would behave badly in practice. Real optimizers remember something
about past gradients.

### 1.1 SGD with momentum

```
v = momentum * v + grad + weight_decay * param
param -= lr * v
```

`v` is per-parameter velocity. The optimizer needs a `v` tensor the
same shape as each parameter, persisted across calls to `step()`.
Without it, momentum has nothing to accumulate into.

### 1.2 AdamW

```
m = β₁ * m + (1 - β₁) * grad           (first moment, mean of grad)
v = β₂ * v + (1 - β₂) * grad²          (second moment, uncentered variance)
m̂ = m / (1 - β₁ᵗ)                      (bias-corrected)
v̂ = v / (1 - β₂ᵗ)                      (bias-corrected)
param -= lr * (m̂ / (√v̂ + ε) + wd * param)
```

`m` and `v` are per-parameter, same shape as the parameter, and they
accumulate across steps. The whole point of Adam is that the step
size adapts based on the historical second moment — if you lose `v`,
every parameter starts getting updates scaled as if the run just
began.

### 1.3 The key question

> How does the optimizer associate a parameter tensor with *its* `m`
> and `v`?

This is the question PR-ζ is about.

---

## 2. The pre-PR-ζ answer: pointer keys

Pre-PR-ζ, both optimizers keyed state on the parameter's data
pointer, coerced to a `usize`:

```zig
// src/optim/adamw.zig (pre-ζ)
state: std.AutoHashMap(usize, ParamState),

pub fn step(ctx: *anyopaque, params: []const *Tensor) anyerror!void {
    // ...
    for (params) |param| {
        const grad = param.grad orelse continue;
        const key = @intFromPtr(param.data.ptr);        // <-- HERE

        if (!self.state.contains(key)) {
            var m = try Tensor.init(self.allocator, param.shape);
            m.fill(0.0);
            var v = try Tensor.init(self.allocator, param.shape);
            v.fill(0.0);
            try self.state.put(key, .{ .m = m, .v = v });
        }
        // ... update m, v, param ...
    }
}
```

This works as long as:

1. The parameter's `.data.ptr` never changes for the entire training
   run.
2. No other tensor's `.data.ptr` ever collides with a parameter's
   (two `@intFromPtr` values happen to be equal because the
   allocator reused a freed buffer).

Both assumptions are true in Stage 6's CPU-only single-step-at-a-time
loop. They stop being true the moment you:

- Load a checkpoint, which `Tensor.init`'s fresh buffers
- Transfer a parameter to CUDA via `.toCuda()` (Stage 7)
- Transfer a parameter back to CPU via `.toCpu()` (Stage 7)
- Replace a parameter's backing buffer to grow it (not done today,
  but common in systems with dynamic vocab or adapter layers)
- Reuse a tensor buffer after freeing an earlier parameter

In every one of these cases, the parameter's `data.ptr` changes but
its *identity* as "the token-embedding weight matrix" does not.
Under the pointer-key model, the optimizer silently loses its state
and restarts with zero `m` and `v` — which looks like "the training
was fine, but then after the checkpoint reload it got weird".

---

## 3. A concrete disaster scenario

Consider this plausible workflow (all of it works with Stage 6
training code; only the middle step is new):

```zig
// Phase A: train for 1000 steps
var model = try TinyWordTransformer.init(alloc, cfg, &rng);
var adam = try AdamW.init(alloc, .{ .lr = 1e-3 });
for (0..1000) |_| {
    // forward, backward, adam.step, zero_grad
}
// adam.state now contains well-tuned m and v for every parameter.

// Phase B: save checkpoint and reload it
try model.save(io, "step-1000.ckpt");
var reload = try TinyWordTransformer.init(alloc, cfg, &rng2);
try reload.load(io, "step-1000.ckpt");
// reload's parameter buffers have DIFFERENT .data.ptr values
// than model's did. Loading overwrote the *contents* but the
// *Tensor objects* were freshly allocated by `Tensor.init`.

// Phase C: keep training from step 1000
for (0..1000) |_| {
    // forward on `reload`, backward, adam.step, zero_grad
}
```

Under the pointer-key model, Phase C's first call to `adam.step` sees
parameters with unrecognised keys (because the pointers are new). It
allocates fresh `m` and `v` filled with zeros for each one. The
first 50–100 steps after the reload proceed with essentially
unregularised updates because Adam's second-moment adaptation has no
history.

Training curves show a visible hiccup at the checkpoint. Often the
model temporarily gets worse. A savvy engineer spends hours looking
for a bug in the checkpoint format before realising the optimizer
state was lost.

### 3.1 PyTorch's solution

PyTorch stores optimizer state in a dict keyed by... also the
parameter, but parameters are Python objects whose `id(p)` is stable
across assignment. The real answer is that PyTorch serialises
`optimizer.state_dict()` alongside the model, and best-practice code
saves and loads both:

```python
torch.save({'model': model.state_dict(),
            'optim': optim.state_dict()}, 'ckpt.pt')
# ...
ckpt = torch.load('ckpt.pt')
model.load_state_dict(ckpt['model'])
optim.load_state_dict(ckpt['optim'])
```

The `state_dict` is keyed by parameter *name* (a stable string). Our
`ParamId` is the integer analogue.

---

## 4. The PR-ζ design

### 4.1 ParamId

```zig
// src/nn/module.zig
pub const ParamId = u32;

var next_param_id: std.atomic.Value(ParamId) = .init(1);

pub fn assignParamId(t: *Tensor) void {
    if (t.param_id == null) {
        t.param_id = next_param_id.fetchAdd(1, .monotonic);
    }
}
```

A `ParamId` is a globally unique 32-bit integer. Each parameter gets
one at layer-initialisation time. The ID is stored on the tensor
itself (`Tensor.param_id: ?u32`), so it travels with the parameter
across buffer replacements.

Why a global counter rather than per-model? Because parameters from
different models might coexist in the same process (tests do this;
multi-model training does this). A global counter guarantees no
collisions without requiring a model-level registry.

Why `?ParamId` (optional)? Because only *parameters* need IDs —
intermediate tensors (activations, gradients, saved copies) never
appear in the optimizer, so they stay `null`. Making the field
optional forces the optimizer to distinguish "this tensor isn't a
parameter" from "this is parameter #42".

Why `u32`? 4 billion unique parameter-creations per process is
enough for any plausible workload, and `u32` is smaller and faster
to hash than `u64`. If you somehow exceed the limit, the atomic add
wraps; a future PR could widen the counter or detect wrap-around.

### 4.2 `assignParamId`

The function is idempotent: calling it on a tensor that already has
an ID is a no-op. This matters for checkpoint loading — the loaded
parameter keeps its pre-load ID, so the optimizer state matches.

```zig
pub fn assignParamId(t: *Tensor) void {
    if (t.param_id == null) {
        t.param_id = next_param_id.fetchAdd(1, .monotonic);
    }
}
```

Atomicity is future-proofing for multi-threaded parameter creation.
Today all our nn layers initialise on one thread, so the atomic
barrier is free in practice.

### 4.3 Layer initialisation

Every layer that owns parameters calls `assignParamId` on each one
during `init`:

```zig
// src/nn/linear.zig
pub fn init(alloc, d_in, d_out, use_bias, rng) !Linear {
    const weight = try ops_create.randn(alloc, Shape.init2D(d_out, d_in), rng, 0.0, bound/3.0);
    var bias: ?Tensor = null;
    if (use_bias) {
        bias = try ops_create.zeros(alloc, Shape.init1D(d_out));
    }

    var layer = Linear{ .weight = weight, .bias = bias, /* ... */ };
    module.assignParamId(&layer.weight);
    if (layer.bias) |*b| module.assignParamId(b);
    return layer;
}
```

Same pattern in `Embedding` and `LayerNorm`. Higher-level layers
(`TransformerBlock`, `TinyWordTransformer`, `MLP`,
`CausalSelfAttention`) get IDs by composition — they create the
smaller layers which do the ID assignment.

### 4.4 Optimizer state, keyed by `ParamId`

```zig
// src/optim/adamw.zig (post-ζ)
state: std.AutoHashMap(ParamId, ParamState),

pub fn step(ctx: *anyopaque, params: []const *Tensor) anyerror!void {
    // ...
    for (params) |param| {
        const grad = param.grad orelse continue;
        const key = param.param_id orelse return error.InvalidArgument;
        //          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //          A tensor passed to the optimizer without an ID
        //          is a bug. Fail loudly.

        if (!self.state.contains(key)) { /* alloc m, v */ }
        // ... update ...
    }
}
```

Three changes from the pointer-key version:

1. The hashmap key type changed from `usize` to `ParamId` (u32).
2. The key derivation is `param.param_id orelse ...` instead of
   `@intFromPtr(param.data.ptr)`.
3. A parameter without an ID causes `error.InvalidArgument` rather
   than silently becoming a unique key.

The third point deserves emphasis. Under the pointer-key model, any
tensor could be an "optimizer parameter" — the optimizer would
happily track `m` and `v` for it. That's permissive but wrong: if a
user accidentally passes an intermediate tensor to the optimizer,
the optimizer wastes memory and updates random values. Under the
ID-key model, the optimizer refuses.

This is the "fail loudly" spirit that runs through all of Stage 6.5.

---

## 5. The checkpoint-reload test, walked step by step

The most important test added in PR-ζ:

```zig
test "AdamW state persists across buffer replacement (same ParamId)" {
    const alloc = std.testing.allocator;

    var adam = try AdamW.init(alloc, .{ .lr = 0.01 });
    defer adam.deinit(alloc);
    var opt = adam.optimizer();

    const Shape = @import("../tensor/shape.zig").Shape;
    const assignParamId = @import("../nn/module.zig").assignParamId;

    // ========== Phase 1: first parameter lifetime ==========
    var p1 = try Tensor.init(alloc, Shape.init1D(2));
    assignParamId(&p1);
    const id = p1.param_id.?;

    var g1 = try Tensor.init(alloc, Shape.init1D(2));
    g1.data[0] = 1.0;
    g1.data[1] = -0.5;
    p1.grad = &g1;
    try opt.step(&.{&p1});
    try std.testing.expect(adam.state.contains(id));
    // → adam.state now has m, v for id.

    p1.deinit(alloc);
    g1.deinit(alloc);
    // p1's data buffer is freed. The pointer that used to be
    // @intFromPtr(p1.data.ptr) is now dangling.

    // ========== Phase 2: checkpoint "reload" ==========
    var p2 = try Tensor.init(alloc, Shape.init1D(2));
    defer p2.deinit(alloc);
    p2.param_id = id;     // <-- explicitly preserve ID across reload
    // p2.data.ptr is different from p1.data.ptr. Pointer key would
    // miss. But p2.param_id == id, so ParamId key hits.

    var g2 = try Tensor.init(alloc, Shape.init1D(2));
    defer g2.deinit(alloc);
    g2.data[0] = 0.3;
    g2.data[1] = 0.1;
    p2.grad = &g2;

    const state_before = adam.state.count();
    try opt.step(&.{&p2});
    const state_after = adam.state.count();
    try std.testing.expectEqual(state_before, state_after);
    //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //   The invariant: the hashmap did NOT grow. p2 reused p1's
    //   state rather than creating fresh state. Pointer-keying
    //   would have grown the map from 1 to 2 entries.
}
```

Run it pre-PR-ζ (hypothetically): `state_after` would be 2, not 1.
Post-PR-ζ: `state_after` is 1.

This is the smallest test that proves the whole point of the
refactor in one assertion.

---

## 6. Why the ID is on the Tensor, not a side-table

Two representations were considered:

### 6.1 Store ID on the Tensor (chosen)

```zig
pub const Tensor = struct {
    // ...
    param_id: ?u32 = null,
    // ...
};
```

Every tensor carries 8 bytes of ID-or-null. Only parameter tensors
set it. The optimizer reads `param.param_id` directly.

### 6.2 Store ID in a side-table

```zig
pub const ParamRegistry = struct {
    ids: std.AutoHashMap(*Tensor, u32),   // keyed by tensor pointer
};
```

A registry maps tensor pointers to IDs. Each optimizer step looks up
the ID via the registry.

### 6.3 Trade-offs

| Concern | On-Tensor | Side-table |
|---|---|---|
| Memory per tensor | 8 bytes always | 0 for non-params, hashmap entry per param |
| Lookup cost | Field read | HashMap probe |
| Survives pointer changes | Yes | **No** — keys are the pointers we said are unstable |
| Needs a registry object | No | Yes — must be passed around or be global |

The side-table approach defeats the whole point: its lookups use the
same unstable pointers that motivated the fix. The on-Tensor approach
is strictly better for our problem.

### 6.4 The eight-byte overhead

Every `Tensor` now carries `param_id: ?u32` — an optional u32, which
Zig represents as 8 bytes (the u32 plus an alignment/sentinel). For
our 1-block 1-head model with ~100 000 intermediate tensors
allocated across training, that's ~800 KB of additional metadata
over the full run. Negligible for our model; not free, but the
safety win is worth it.

---

## 7. How IDs cascade through the model

Parameters are only created inside the three leaf layers that own
tensors: `Linear`, `Embedding`, `LayerNorm`. Higher-level layers
compose these:

```
TinyWordTransformer.init
├── Embedding (tok_embed)      ── assigns id for .weight
├── Embedding (pos_embed)      ── assigns id for .weight
├── TransformerBlock.init
│   ├── LayerNorm (ln1)        ── assigns ids for .gamma, .beta
│   ├── CausalSelfAttention.init
│   │   ├── Linear (w_q)       ── assigns ids for .weight (bias=false)
│   │   ├── Linear (w_k)       ── assigns ids
│   │   ├── Linear (w_v)       ── assigns ids
│   │   └── Linear (w_o)       ── assigns ids
│   ├── LayerNorm (ln2)
│   └── MLP.init
│       ├── Linear (fc1)
│       └── Linear (fc2)
├── LayerNorm (ln_f)
└── Linear (lm_head)
```

Every parameter has a unique ID by the time `TinyWordTransformer.init`
returns. The `parameters()` method walks the tree and yields pointers
to each parameter tensor in a consistent order. The optimizer iterates
this list and uses `param.param_id` to key its state.

### 7.1 What about bias?

`Linear` bias is optional (controlled by `use_bias`). The init code:

```zig
module.assignParamId(&layer.weight);
if (layer.bias) |*b| module.assignParamId(b);
```

The `if (layer.bias) |*b|` captures a pointer to the Tensor inside
the Optional. `assignParamId(b)` assigns the ID through that pointer
so the bias is tracked.

Without the pointer capture, `if (layer.bias) |b|` would copy the
Optional's payload and `assignParamId(&b)` would mutate the stack
copy. This is a recurring Zig gotcha; see AGENTS.md §11–14 for
related lessons.

---

## 8. PyTorch parallels

### 8.1 Parameter identity in PyTorch

```python
class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.empty(d_out))
```

`nn.Parameter` is a subclass of `Tensor` that the optimizer
recognises. The identity of the parameter is the Python object's
`id(p)` — stable across buffer replacement because Python object
identity doesn't depend on the memory address of the underlying data
storage.

When you `optim.load_state_dict(...)`, PyTorch matches entries by
position (parameter index) rather than by `id(p)`. Our `ParamId` is
functionally equivalent: a stable handle that survives data changes.

### 8.2 state_dict compatibility

PyTorch's `model.state_dict()` is keyed by *string names* like
`'layer1.weight'`. Our `model.collectNamedParams()` produces the
analogous (name, tensor) pairs. The two systems are isomorphic:

| PyTorch | Ours |
|---|---|
| `nn.Parameter.id` | `Tensor.param_id` |
| `state_dict['layer1.weight']` | Entry in `collectNamedParams()` |
| `optim.state_dict()[id]` | `AdamW.state[param_id]` |
| `optim.load_state_dict(d)` | (would be) load by name, look up tensor, copy into its param_id's state slot |

The optimizer-state save/load roundtrip is not yet implemented in our
library — we save only model weights today. If we add it in a future
PR, the format is a direct parallel of PyTorch's: serialise `m` and
`v` keyed by parameter *name* (using `collectNamedParams`),
deserialise by finding each name's tensor and writing into its
ParamId-keyed state entry.

### 8.3 The missing piece

We do *not* currently persist optimizer state across checkpoint
boundaries. The test in §5 only demonstrates that it *survives* a
buffer replacement within a single process; it does not survive
process exit. Adding full optimizer state persistence is a natural
Stage 8 or 9 feature.

The key property PR-ζ gives us: if you *do* add that persistence
later, you already have the stable identity needed to correctly
match saved state back to live parameters.

---

## 9. Things we did not do

### 9.1 Per-model ID counters

We use a single global counter. A per-model counter would let us
serialise optimizer state with per-model IDs that remain stable
across processes. But:

- It would introduce a `ParamRegistry` object that every layer needs
  access to during init, complicating the layer API.
- It interacts poorly with any future multi-model setup.
- The global counter already works for every use case we have today.

If future optimizer-state serialisation needs per-model IDs, we can
introduce them as a second layer on top of the global IDs (similar to
how PyTorch has both `id(p)` and `state_dict` names).

### 9.2 Parameter name stored on the tensor

Names would be nice for debugging. We debated storing a `?[]const u8`
name on every parameter tensor. The `collectNamedParams` convention
(a separate list of name-tensor pairs) was already established, and
adding another field to `Tensor` just for debugging felt like
over-reach. Names stay in a side-structure.

### 9.3 Forcing ID assignment at Tensor.init

We could have made `Tensor.init` always assign a `ParamId`. But then
every intermediate (activations, gradients, saved copies —
potentially millions over a training run) would consume IDs,
wasting half the 4-billion space in the first few epochs.

Only parameters need IDs. Intermediates do not. The current model
reflects that.

### 9.4 Protecting against ID re-use

If a parameter tensor is deinit'd and then a new tensor happens to
be assigned the same ID (not possible today because the counter is
monotonic), the optimizer would conflate their state. We do not
currently guard against this; the monotonic counter makes it
structurally impossible in one process, and cross-process scenarios
are not supported yet.

---

## 10. Common mistakes

### "My optimizer step panics with `error.InvalidArgument`"

A parameter without a `ParamId` was passed to the optimizer. Check:

- Is your layer's `init` calling `module.assignParamId(&layer.weight)`
  for every learnable tensor?
- If you built a layer outside the standard nn/, did you forget to
  assign IDs?
- Are you accidentally passing a non-parameter tensor (like a grad
  or intermediate) into the optimizer's params slice?

The `step` function refuses to invent an ID. This is by design —
better to crash than to silently track state for the wrong tensor.

### "I wrote a new layer and the optimizer ignores its params"

Check that:

1. `assignParamId` is called for each learnable tensor in `init`.
2. The layer's `parameters(&list)` method appends pointers to each
   learnable tensor.
3. The parent layer's `parameters` calls this one.

A common mistake is to add a new learnable tensor field (say, a
scaling factor) and forget step 2, so the optimizer never sees it.
The model still runs — the parameter just never gets updated.

### "I want to manually set a ParamId to a specific value"

Only do this when deliberately simulating a checkpoint reload (see
§5). The normal flow is: `assignParamId(&t)` once at layer init, and
let the global counter pick the number.

If you manually assign the same ID to two different tensors, the
optimizer will conflate their state and produce wrong updates. The
`assignParamId` function's idempotent check (`if (t.param_id ==
null)`) is what prevents accidental re-assignment after layer init.

### "My parallel test runs use overlapping ID ranges"

They do, intentionally. The global counter is process-wide. Each test
that creates a layer allocates fresh IDs; different tests use
different IDs. Since each test has its own `AdamW.init`, the state
hashmap is per-test and the IDs don't collide.

If you need deterministic IDs across tests (for snapshot testing),
call `module.resetParamIdCounterForTests()` at the start of the test.

---

## 11. Exercises

### Exercise 1 — Simulate a full checkpoint round-trip

Extend the §5 test to:

1. Train `model_a` for 10 steps with AdamW. Save state as
   `adam.state.iterator()`'s contents (you'll need to serialise `m`
   and `v` tensors — use the same format as model weights).
2. Create `model_b` with fresh weights but same config.
3. Load weights into `model_b` via `model.load`.
4. Reconstruct a new `AdamW` instance and populate its state by
   mapping saved `(name, m, v)` triples to `model_b`'s parameters
   via name → tensor → ParamId.
5. Take a step. Verify the parameter changes match what a
   continue-from-checkpoint run would produce.

This is the full PyTorch `optim.load_state_dict` workflow. Think
about what could go wrong: missing name, shape mismatch, extra name.

### Exercise 2 — Inspect the parameter tree

Add a debug method `pub fn dumpParams(self: *TinyWordTransformer)
void` that prints each parameter's name, `ParamId`, and shape. Run
it after `TinyWordTransformer.init` and confirm:

- Every learnable tensor has a non-null ID.
- No two tensors have the same ID.
- The ID counter at the end of init matches the total parameter
  count.

### Exercise 3 — A ParamId collision test

Write a test that:

1. Creates two linear layers `l1` and `l2`.
2. Manually sets `l2.weight.param_id = l1.weight.param_id` (breaking
   the invariant).
3. Passes both params to the optimizer and takes a step.
4. Asserts that the optimizer mis-updates one of them.

This shows why `assignParamId`'s idempotence check matters.

---

## 12. File reference

| File | What to read |
|---|---|
| `src/nn/module.zig` | `ParamId`, `assignParamId`, `resetParamIdCounterForTests`, `next_param_id` atomic |
| `src/tensor/tensor.zig` | `Tensor.param_id: ?u32` field declaration |
| `src/nn/linear.zig` | `Linear.init` — assigns IDs for weight and bias |
| `src/nn/embedding.zig` | `Embedding.init` — assigns ID for weight |
| `src/nn/layernorm.zig` | `LayerNorm.init` — assigns IDs for gamma and beta |
| `src/optim/adamw.zig` | State keyed by `ParamId`, `error.InvalidArgument` on missing ID |
| `src/optim/sgd.zig` | Same pattern for SGD velocity map |

---

## 13. Test commands

```bash
# Run optimizer tests
zig build test -- --test-filter "AdamW"
zig build test -- --test-filter "SGD"

# The persistence test specifically
zig build test -- --test-filter "persists across buffer"

# Full suite to catch regressions
zig build test
```

Next chapter: `07d_checkpoint_format.md` — the `ZTLC v2` binary
format, every field it validates, and why checkpoint strictness is
the single biggest protection against silently-wrong training runs.
