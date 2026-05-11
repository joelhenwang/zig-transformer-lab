# 10 — PyTorch Parallels

> The last chapter in the zig-transformer-lab book. Maps every
> abstraction in this library back to its PyTorch counterpart so that
> a Python/PyTorch user can read our Zig code as though it were a
> familiar DSL. Complementary to every other chapter — each section
> cross-references the relevant mechanics / concepts file.
>
> Pinned to **PyTorch 2.3** for any specific API or source-file
> references. Architectural comparisons are version-agnostic and hold
> across the 2.x series.

---

## 1. Intent

You're here because you learned ML in PyTorch and now you're reading a
Zig codebase. The vocabulary mostly maps but the conventions and
memory model differ. This chapter is a glossary: for each major
concept in the library, where is the analogue in PyTorch, what's the
same, what's different, and why the library's choice looks the way it
does.

The chapter follows the architectural layer order: `Tensor` → autograd
→ `nn.Module` → optimiser → backend → kernels → checkpoint. Each
section assumes you've read the previous one.

---

## 2. Tensor vs `torch.Tensor`

### 2.1 Common ground

At bytes level, a `zig-transformer-lab` Tensor and a `torch.Tensor`
are the same idea: a flat contiguous (or strided) buffer of scalar
values, plus metadata telling downstream code how to interpret it as
a multi-dim array.

Shared concepts:

- **Storage**: the underlying buffer of scalars. PyTorch calls this
  `torch.Storage` (or `torch.UntypedStorage` since 2.1); we call it
  `Storage` (a tagged union over CPU and CUDA variants).
- **Shape**: the per-axis size. PyTorch: `tensor.size()` → `torch.Size`.
  Ours: `tensor.shape` → `Shape` with `init1D`/`init2D`/... constructors.
- **Strides**: step in the flat buffer per unit move along each axis.
  PyTorch: `tensor.stride()`. Ours: `tensor.strides: [4]usize` derived
  from shape at init for contiguous tensors.
- **Device**: where the buffer lives. PyTorch: `tensor.device` →
  `torch.device("cpu")` / `torch.device("cuda:0")`. Ours:
  `tensor.device: Device` → `.cpu` or `.cuda`.
- **Dtype**: element type. PyTorch: `torch.float32`, `torch.int64`, etc.
  Ours: `f32` only (policy decision D7).

### 2.2 Key differences

| Aspect | PyTorch | This library |
|---|---|---|
| Memory management | Reference-counted `Storage`; buffers freed when refcount hits 0 | Explicit `deinit(alloc)` per tensor; `owned` flag tracks ownership |
| View vs copy | `tensor.view(*)` requires contiguity; `reshape(*)` copies if needed | `view` builds a new Tensor with the same storage; `reshape` requires contiguous; tracked variants record a reshape node |
| Dtype | Runtime-tagged, many dtypes | Compile-time f32 only |
| Device multiplexing | Dispatch at op-call time via keys | Dispatch via a switch on `tensor.device` |
| Autograd enable flag | `tensor.requires_grad = True` | Same field name; tape uses it to decide whether to track leaf nodes |

### 2.3 Side-by-side: same op, two languages

Element-wise add of two tensors.

**PyTorch**:

```python
import torch
a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)
c = a + b                       # forward
c.sum().backward()              # backward
print(a.grad)                   # tensor([[1., 1.], [1., 1.]])
```

**zig-transformer-lab**:

```zig
const std = @import("std");
const ztl = @import("zig_transformer_lab");

var a = try ztl.Tensor.init(alloc, ztl.shape.Shape.init2D(2, 2));
defer a.deinit(alloc);
a.data[0] = 1; a.data[1] = 2; a.data[2] = 3; a.data[3] = 4;
a.requires_grad = true;

var b = try ztl.Tensor.init(alloc, ztl.shape.Shape.init2D(2, 2));
defer b.deinit(alloc);
b.data[0] = 5; b.data[1] = 6; b.data[2] = 7; b.data[3] = 8;
b.requires_grad = true;

var tape = ztl.autograd.Tape.init(alloc);
defer tape.deinit();
_ = try tape.trackLeaf(&a);
_ = try tape.trackLeaf(&b);

var c = try ztl.ops.elementwise.add(alloc, a, b, &tape);
defer c.deinit(alloc);

var loss = try ztl.ops.reduce.sumAll(alloc, c, &tape);
defer loss.deinit(alloc);
try tape.backward(&loss);
// a.grad.data == [1,1,1,1]
```

The Zig version is louder because you see the allocator, the tape,
the defer chain, and the ownership flag. None of these are hidden in
PyTorch; they are just handled by the runtime. Reading the Zig
version teaches you what the runtime does.

### 2.4 See also

- `docs/02_tensors.md` — row-major conventions, stride derivation,
  broadcasting rules.
- `docs/02b_from_tensors_to_training.md` — conceptual bridge from
  tensor ops to training.
- `docs/02d_storage_and_views.md` — why the `Storage` union exists
  (Stage 6.5 / PR-δ).

---

## 3. Autograd: tape vs dynamic graph

### 3.1 The mental model

PyTorch's autograd builds an implicit graph as the forward pass
executes. Every tensor with `requires_grad = True` that participates
in an op becomes a node; each op becomes a `torch.autograd.Function`
node with a `.backward` method; `tensor.grad_fn` chains back to the
graph root.

Ours is shaped identically but explicit. We call our graph a **tape**.
`src/autograd/tape.zig`'s `Tape` is an append-only list of `Node`s
(`src/autograd/node.zig`). Each op checks if it has a tape and — if
so — appends a node recording its inputs, output, kind, and the
backward-only `SavedData` payload.

### 3.2 What's the same

- Both are dynamic graphs: built at each forward pass, from scratch,
  by the op functions themselves.
- Both support leaf nodes with `requires_grad` flags; gradients flow
  into leaves during `backward()`.
- Both free the graph after `backward()`: PyTorch frees automatically
  unless `retain_graph=True`; ours requires explicit `tape.deinit()`.
- Both implement the chain rule via per-op backward functions.

### 3.3 What's different

| Aspect | PyTorch | This library |
|---|---|---|
| Graph storage | Distributed across `torch.autograd.Function` objects per op | Centralised in `Tape.nodes` |
| Graph ownership | Managed by the C++ autograd engine | The `Tape` struct; caller calls `init` + `deinit` |
| Op registration | C++ `Function` subclasses with `forward` / `backward` | Zig `OpKind` enum + one `backwardX` function per kind |
| SavedData | Held by Function objects; tensors that need gradient kept via `save_for_backward` | Tagged union in `SavedData`; snapshots by value not by pointer |
| Multi-backward | Supported via `retain_graph=True` | Not supported; tape is single-use |
| Higher-order gradients | Supported via `create_graph=True` | Not supported |

Our restrictions (no retain, no higher-order) are pedagogical; they
keep the tape implementation small enough to read end to end.

### 3.4 Adding a new op, side by side

The end goal of learning autograd is "can I add a new op with a
gradient?". Here's the shape in each framework.

**PyTorch**:

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return my_forward_formula(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * my_derivative(x)
```

**zig-transformer-lab** (full recipe in `docs/08b_from_cuda_to_training.md`
Exercise 4):

```zig
// 1. Add variant to OpKind enum.
// src/autograd/node.zig:
pub const OpKind = enum(u8) {
    ...,
    abs,   // or whatever
};

// 2. Forward in ops/unary.zig:
pub fn abs(alloc, x, tape) !Tensor {
    var out = try Tensor.init(alloc, x.shape);
    for (0..x.data.len) |i| out.data[i] = @abs(x.data[i]);
    if (tape) |tp| {
        const node_id = try tp.record(.abs, &.{ x.tape_node }, ...);
        out.tape_node = node_id;
    }
    return out;
}

// 3. Backward in src/autograd/backward.zig:
fn backwardAbs(alloc, grad_out, saved) !ParentGrads {
    const x = saved.tensor_ref;
    // d/dx |x| = sign(x)
    ...
}
```

Two steps become three (forward, backward, OpKind registration), but
all three live close together — the enum is in `node.zig` next to the
`Node` struct; the backward is in `backward.zig` next to all the
others. No C++ runtime registration magic.

### 3.5 See also

- `docs/03_autograd.md` — tape mechanics.
- `docs/03b_from_autograd_to_training.md` — what autograd buys you
  conceptually.
- `docs/03c_saved_tensors.md` — the snapshot-by-value rule (Stage 6.5
  / PR-ε).

---

## 4. `nn.Module` protocol

### 4.1 PyTorch

A PyTorch `nn.Module` is any class inheriting from `torch.nn.Module`
that:

- Defines `forward(*args)` (the only mandatory method).
- Registers parameters as attributes of type `nn.Parameter` (or
  registers child modules that do).
- Implicitly exposes `parameters()`, `named_parameters()`,
  `state_dict()`, `load_state_dict()` via the base class's magic.

### 4.2 This library

`src/nn/module.zig` describes the protocol. Every layer (Linear,
Embedding, etc.) follows this shape:

```zig
pub const MyLayer = struct {
    weight: Tensor,
    // ... other fields ...
    allocator: Allocator,

    pub fn init(alloc, config, rng) !MyLayer { ... }
    pub fn forward(self: MyLayer, x: Tensor, tape: ?*Tape) !Tensor { ... }
    pub fn parameters(self: MyLayer, list: *std.ArrayList(*Tensor)) void {
        list.append(self.allocator, &self.weight) catch unreachable;
        // ... recurse into sub-layers if any ...
    }
    pub fn deinit(self: *MyLayer) void { ... }
};
```

### 4.3 Mapping table

| PyTorch | This library |
|---|---|
| `class MyLayer(nn.Module)` | `pub const MyLayer = struct { ... }` |
| `nn.Parameter(...)` | A `Tensor` field with `requires_grad = true` |
| `self.weight = nn.Parameter(...)` | `self.weight = try Tensor.init(alloc, shape)` |
| `def forward(self, x): ...` | `pub fn forward(self: MyLayer, x: Tensor, tape: ?*Tape) !Tensor` |
| `model.parameters()` | `model.parameters(&list)` (caller-allocated list) |
| `model.named_parameters()` | `model.collectNamedParams(&list)` (only `TinyWordTransformer` today) |
| `model.to("cuda")` | `model.moveToCuda(&ctx)` |
| `model.to("cpu")` | `model.moveToCpu()` |
| `model.state_dict()` | `model.save(io, path)` (writes ZTLC v3 to disk directly — we skip the in-memory state_dict intermediate) |
| `model.load_state_dict(...)` | `model.load(io, path)` |

### 4.4 Design differences

PyTorch's base class uses `__setattr__` overrides to auto-register
parameters. We don't: callers list their parameters explicitly in
`.parameters()`. Drawback: you can forget to list one; we catch this
via `collectNamedParams` test which asserts the expected count.
Advantage: explicit is readable.

PyTorch supports hooks (`register_forward_hook`,
`register_backward_hook`) for logging and debugging. We don't ship
these; if you need them, add a wrapper module.

### 4.5 See also

- `docs/04_nn.md` — per-layer mechanics.
- `docs/04b_from_nn_to_training.md` — conceptual bridge.
- `src/nn/README.md` — file-by-file tour.

---

## 5. Optimisers

### 5.1 Mapping

| PyTorch | This library |
|---|---|
| `torch.optim.SGD(params, lr=0.1, momentum=0.9)` | `SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.9 })` |
| `torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999))` | `AdamW.init(alloc, .{ .lr = 1e-3, .beta1 = 0.9, .beta2 = 0.999 })` |
| `optimizer.step()` | `opt.step(params.items)` |
| `optimizer.zero_grad()` | `opt.zeroGrad(params.items)` |
| `optimizer.state_dict()` | No checkpoint format yet (state is regenerated each run) |

### 5.2 Coupled vs decoupled weight decay

PyTorch's `Adam` and `AdamW` differ in whether weight decay is
applied inside the adaptive update (`Adam`) or as a separate scalar
multiply afterwards (`AdamW`). This is the "decoupled weight decay"
distinction from the Loshchilov & Hutter paper.

Our library ships only the decoupled variant (`AdamW`). `SGD`
implements **coupled** weight decay (applied via the gradient, which
is the traditional SGD recipe).

### 5.3 Numerical demo

Given a parameter `w = 1.0`, gradient `g = 0.5`, `lr = 0.01`, weight
decay `wd = 0.1`.

**Coupled (SGD with weight decay)**:

```
g_effective = g + wd * w = 0.5 + 0.1 * 1.0 = 0.6
w_new       = w - lr * g_effective = 1.0 - 0.01 * 0.6 = 0.994
```

**Decoupled (AdamW)**:

```
# Adam part, ignoring momentum/variance for simplicity:
update = lr * g = 0.01 * 0.5 = 0.005
w_after_step = w - update = 0.995
# Decoupled weight decay:
w_new = w_after_step * (1 - lr * wd) = 0.995 * 0.999 = 0.994005
```

Very close numerically in this toy example, but diverges when `g` is
adaptively scaled by the Adam moment estimates (because coupled would
scale the decay too, decoupled doesn't).

### 5.4 See also

- `docs/04_nn.md` §5 — full update math.
- `docs/07_cpu_training.md` §7.11.3 — the `beta2 = 0.95` vs `0.999`
  cautionary tale.
- `src/optim/README.md` — file-by-file.

---

## 6. Backend dispatcher

### 6.1 PyTorch's dispatcher

`torch.Tensor.add(other)` goes through ATen's dispatch mechanism: the
tensor's `dispatch_key_set` (combining device + dtype + layout +
autograd flags) picks a function from a registry, which routes to
the CPU / CUDA / etc. implementation. The registration happens at
C++ library load time via ATen's `TORCH_LIBRARY_IMPL` macros.

### 6.2 Our dispatcher

Every op in `src/tensor/ops/*.zig` starts with a switch on
`tensor.device`:

```zig
return switch (a.device) {
    .cpu => cpuAdd(...),
    .cuda => cudaAdd(...),
};
```

That's the whole dispatcher. Two cases, both visible in source.
Adding a new device means adding a case. There is no registry, no
macro magic, no C++ at compile time.

### 6.3 Mapping

| PyTorch | This library |
|---|---|
| `at::native::add_cpu` | `cpuAdd` in `src/tensor/ops/elementwise.zig` |
| `at::native::add_cuda` | `cudaAdd` in `src/tensor/ops/elementwise.zig` |
| `TORCH_LIBRARY_IMPL(aten, CPU, ...)` | No analogue — we just `switch`. |
| `DispatchKey::CUDA` | `Device.cuda` |
| `DispatchKey::AutogradCUDA` | Handled by the `tape != null` flag inside each op |

The price we pay: every new device (TPU, Metal, etc.) would require
touching every op to add a case. At our scale (≈15 ops) that's a
one-afternoon refactor, not a system-wide redesign. At PyTorch's
scale, the registry pays off.

### 6.4 See also

- `docs/08_backends_cuda.md` §12 (the add op walkthrough).
- `src/backend/README.md`.

---

## 7. cuBLAS wrapper

### 7.1 How PyTorch handles row-major

ATen tensors are row-major by default (like NumPy). cuBLAS is
column-major. PyTorch hides the conversion inside
`at::cuda::blas::gemm` which recomputes the operand layout on the
fly and invokes cuBLAS with the appropriate `op_A` / `op_B` flags
and swapped operands when beneficial.

### 7.2 Our version

We use the same trick (operand swap, `transA = N` and `transB = N`)
but expose it explicitly in `src/backend/cuda/gemm.zig`.

The derivation is in `docs/08_backends_cuda.md` §3-§4 with two fully
worked numerical examples in §14. The leading-dimension reference
card at §14.3 is worth printing.

### 7.3 Differences

| Aspect | PyTorch | This library |
|---|---|---|
| Where the trick lives | Deep inside ATen C++ | One `.zig` file (`gemm.zig`), <300 lines |
| Non-contig operands | Auto-copied to contiguous | Rejected with `error.InvalidLayout`; caller must `broadcastTo` |
| Transposed operands | Handled via `op_A` / `op_B` flags | Materialised as copies (a Stage 9+ optimisation would flip to the flag-based path) |
| Mixed precision (fp16, tf32) | `torch.amp` autocast | Not supported; f32 only |

### 7.4 See also

- `docs/08_backends_cuda.md` §3 — row-major derivation.
- `docs/08_backends_cuda.md` §14 — worked numerical examples.

---

## 8. Custom CUDA kernels

### 8.1 PyTorch's kernels

ATen ships fused CUDA kernels in
`aten/src/ATen/native/cuda/*.cu` (as of PyTorch 2.3). Every common op
has one; performance-critical ops have multiple variants. Kernel
registration via `REGISTER_CUDA_DISPATCH` macros; build system via
CMake + nvcc.

### 8.2 Our kernels

Eight hand-authored `.cu` files in `src/backend/cuda/kernels/` plus
cuBLAS for matmul. Compiled offline by `build.zig` via nvcc to
`zig-out/ptx/*.ptx`, loaded at runtime via `cuModuleLoadData`. See
`src/backend/cuda/README.md` for the catalog.

### 8.3 Architectural comparison: fused cross-entropy

PyTorch's `CrossEntropyLoss` (from `aten/src/ATen/native/cuda/Loss.cu`
and `LossNLL.cu` in version 2.3) does essentially:

1. Compute log-softmax of logits: write `log_probs` to a scratch
   buffer.
2. Gather `log_probs[batch, target]` for each row.
3. Reduce to scalar loss.

Our `src/backend/cuda/kernels/ce_loss.cu` has a single `ce_fused`
kernel that does all three steps plus the backward's `grad_logits`
calculation in one launch. Tradeoff:

- Pro: one kernel launch instead of three; no scratch `log_probs`
  buffer.
- Pro: loss *and* gradient both produced, saving the separate
  `backward` launch.
- Con: the kernel is more complex (holds more of the computation in
  shared memory); tuning is harder.

This is a conscious optimisation that fits our scale. PyTorch's
generality (arbitrary reduction choices, class weights, label
smoothing) would make a fused kernel messier; they chose the flexible
multi-launch form.

Both approaches are legitimate. The library is a study of a specific
small-scale design point.

### 8.4 No code is copied

Every `.cu` file in this repo is author-original. ATen is a design
reference we studied; the implementations derive from the math, not
the ATen source. See `docs/00_overview.md` §3 for the reference-license
policy.

### 8.5 See also

- `docs/08_backends_cuda.md` §8 — kernel catalog.
- `src/backend/cuda/kernels/*.cu` — the source files themselves.
- `src/backend/cuda/README.md` — one-line summary per kernel.

---

## 9. Checkpoint formats

### 9.1 PyTorch options

PyTorch has two dominant formats:

- **`torch.save` + `torch.load`** — pickles the whole Python object
  graph including the `state_dict`. Convenient, Python-specific.
- **safetensors** — a memory-mapped tensor-bag format. Safer (no
  arbitrary code execution at load time) and faster. Increasingly the
  default in the HF ecosystem.

### 9.2 Our format

ZTLC v3 — a tiny custom binary format defined in
`docs/07d_checkpoint_format.md` (Stage 6.5 / PR-η), extended in Stage 8
M6 to add `n_layer` / `n_head` / `dropout` header fields.

| Aspect | `torch.save` | safetensors | ZTLC v3 |
|---|---|---|---|
| Execution at load time | Yes (unpickles) | No (just mmap) | No (parses fixed header + raw bytes) |
| Portability | Requires PyTorch | Library in multiple langs | Zig only (format is trivial — could be reimplemented in ~50 Python lines) |
| Compactness | OK | Better (no pickle overhead) | Best (no metadata beyond header) |
| Tensor payload | Pickled arrays | Raw f32 bytes | Raw f32 bytes |
| Device info | In pickle | In header | Not stored — residency is a runtime property |

ZTLC is deliberately minimal. The format is documented once in
`docs/07d_checkpoint_format.md` and `src/nn/model.zig:333` in full.
No dependency on any external format library.

### 9.3 Cross-device checkpoints

Our Stage 8 M8-c change made `save` / `load` work from both CPU and
CUDA tensors (DtoH / HtoD via scratch buffers per param, no
moveToCpu round-trip). A checkpoint file is device-neutral — it
stores raw bytes, and the destination model decides residency.

### 9.4 See also

- `docs/07d_checkpoint_format.md` — full format spec.
- `docs/stage8_plan.md` Milestone 6 — the v2→v3 migration.

---

## 10. Bridging exercise: port a 10-line PyTorch snippet

Port this standard PyTorch training snippet to our library:

```python
import torch, torch.nn as nn

model = nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.tensor([[1., 2., 3., 4.]])
y = torch.tensor([0])

for step in range(5):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step {step}: loss={loss.item():.4f}")
```

### 10.1 Worked solution

```zig
const std = @import("std");
const ztl = @import("zig_transformer_lab");

pub fn main(init: std.process.Init) !void {
    const alloc = std.heap.page_allocator;

    // Build a Linear(4 -> 2). We use TinyWordTransformer's Linear
    // directly; in a real program you'd wrap this in a Module.
    var rng = ztl.rng.Rng.init(42);
    var linear = try ztl.nn.linear.Linear.init(
        alloc, 4, 2, true, &rng);
    defer linear.deinit();

    // Gather parameters (one Linear has weight and optional bias).
    var params: std.ArrayList(*ztl.Tensor) = .empty;
    defer params.deinit(alloc);
    try params.ensureTotalCapacity(alloc, 4);
    linear.parameters(&params);

    // Optimiser.
    var adam = try ztl.optim.adamw.AdamW.init(alloc, .{ .lr = 1e-3 });
    defer adam.deinit(alloc);
    const opt = adam.optimizer();

    // Input: (1, 4).
    var x = try ztl.Tensor.init(alloc, ztl.shape.Shape.init2D(1, 4));
    defer x.deinit(alloc);
    x.data[0] = 1; x.data[1] = 2; x.data[2] = 3; x.data[3] = 4;

    // Target: scalar class index 0 packed as (1,) f32.
    var y = try ztl.Tensor.init(alloc, ztl.shape.Shape.init1D(1));
    defer y.deinit(alloc);
    y.data[0] = 0;

    for (0..5) |step| {
        var tape = ztl.autograd.Tape.init(alloc);
        defer tape.deinit();

        for (params.items) |p| {
            p.requires_grad = true;
            _ = try tape.trackLeaf(p);
        }

        var logits = try linear.forward(x, &tape);
        defer logits.deinit(alloc);

        var loss = try ztl.ops.loss.crossEntropy(alloc, logits, y, &tape);
        defer loss.deinit(alloc);
        const loss_val = loss.data[0];

        try tape.backward(&loss);
        try opt.step(params.items);
        opt.zeroGrad(params.items);

        std.debug.print("step {d}: loss={d:.4}\n", .{ step, loss_val });
    }
}
```

### 10.2 What's different

- **Explicit tape lifecycle**: create, trackLeaf every param, deinit
  after step. PyTorch hides this behind `loss.backward()`.
- **Explicit allocators**: every tensor carries the allocator it was
  made with; `defer deinit(alloc)` is mandatory. PyTorch uses a
  garbage collector.
- **Defensive `requires_grad` set**: PyTorch sets this once at
  parameter creation; our tape resets it each tape-deinit (leaves are
  re-tracked fresh). Calling `p.requires_grad = true` inside the loop
  is the safe idiom.
- **No implicit graph**: the `tape` is passed explicitly. This is
  how the library stays introspectable — you can always see which
  ops recorded which nodes.

Line count: PyTorch ~10 lines, Zig ~40. Most of the extra lines are
allocator discipline and the tape lifecycle.

---

## 11. Common mistakes when switching mental models

Bugs that people coming from PyTorch keep making in the first week.

- **Assuming `tensor.data` works on CUDA.** It doesn't — `data` is
  the empty-stub CPU slice. Read via `tensor.toCpu(alloc).data[i]`.
  PyTorch's `.data` is always valid (on any device).
- **Forgetting `tape.deinit()`.** PyTorch frees the graph in
  `backward()`; we don't. Every step needs `defer tape.deinit()`.
- **Calling `.backward()` multiple times on the same loss.** Our
  tape is single-use. PyTorch allows this with `retain_graph=True`;
  we don't.
- **Expecting `tensor.grad` to persist across steps.** After
  `tape.deinit` our leaf tensors have `.grad = null`. Next tape
  creates fresh grad buffers. PyTorch accumulates `tensor.grad`
  across steps unless you call `.zero_grad()` — opposite convention.
- **Expecting `.view()` to track gradients.** Our `reshape()` returns
  a view but does NOT record a node. Use `reshapeTracked` inside
  any forward chain with a gradient. PyTorch's `.view()` always
  tracks through autograd.
- **Forgetting to collect parameters.** PyTorch auto-registers
  `nn.Parameter` attributes. We don't — you must call
  `model.parameters(&list)` explicitly. Forgetting means the
  optimiser never sees those tensors and they don't train.
- **Passing a stack-local allocator into a struct field.** Zig
  structs are moved by value; a `*const Allocator` stored as a struct
  field points at the old stack location after the return. Use
  heap-allocated or owned-by-caller allocators.
- **Assuming the default stream is asynchronous.** On CUDA we use the
  default stream in a blocking configuration (for simplicity); a
  `tensor.toCpu()` call implicitly synchronises. PyTorch uses
  multiple streams and `torch.cuda.synchronize()` is frequently
  required before host reads.

---

## 12. Further reading

- **Inside each chapter**: at the end of every `docs/0X_*.md` file
  there's a "Common mistakes" section and a set of exercises. Those
  are the canonical entry points to the mechanics of each subsystem.
- **PyTorch internals**: Edward Yang's "PyTorch internals" blog post
  remains the best single reference —
  <http://blog.ezyang.com/2019/05/pytorch-internals/>. Dated to 2019
  but still accurate in structure.
- **ATen dispatch**: the PyTorch official docs at
  <https://pytorch.org/docs/stable/torch.compiler_ir.html> and the
  source tree under `aten/src/ATen/`.
- **Why Zig for ML infrastructure**: `docs/01_zig_primer.md` §"When
  this matters for ML" section.
- **safetensors format**: <https://huggingface.co/docs/safetensors>
  — worth understanding as the modern counterpart to our ZTLC v3.

---

*End of Chapter 10 — PyTorch Parallels. End of the zig-transformer-lab
book.*
