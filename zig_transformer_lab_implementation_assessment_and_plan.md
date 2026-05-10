# zig-transformer-lab — Deep Implementation Assessment and Improvement Plan

**Prepared for:** the development agent implementing `joelhenwang/zig-transformer-lab`  
**Prepared on:** 2026-05-10  
**Artifact:** engineering assessment + implementation roadmap  
**Scope:** source-inspection review of the public repository and current project plan, with a focus on whether the CPU implementation is ready for CUDA-backed training.  
**Repository reviewed:** <https://github.com/joelhenwang/zig-transformer-lab>

---

## 0. How the implementation agent should use this document

This document is an engineering control document. Treat it as a set of review findings, gate criteria, and implementation instructions.

The current project goal is good: build a small, heavily commented Zig 0.16.0 library that teaches tensor libraries, autograd, training loops, and CUDA acceleration by implementing a tiny one-block, one-head transformer from scratch. That goal is worth preserving. The project should remain small, readable, explicit, and educational.

The immediate recommendation is blunt:

> **Do not start Stage 7 by writing CUDA kernels. Start by hardening the CPU tensor/autograd system and by creating a real backend/storage boundary.**

CUDA will not fix current ambiguity. It will make every ambiguity harder to debug. Device pointers, asynchronous execution, cuBLAS layout rules, kernel launch failures, stream semantics, and GPU memory lifetime will amplify problems around tensor ownership, view semantics, and autograd saved data.

The next implementation stage should be named something like:

```text
Stage 6.5 — CPU hardening and CUDA-readiness refactor
```

Only after Stage 6.5 passes should the agent begin Stage 7 CUDA implementation.

---

## 1. Sources and review limitations

### 1.1 Sources inspected

The assessment is based on public repository files and official reference documentation, especially:

- Repository README: <https://github.com/joelhenwang/zig-transformer-lab>
- Raw README: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/README.md>
- `AGENTS.md`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/AGENTS.md>
- `plan.md`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/plan.md>
- `build.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/build.zig>
- `src/tensor/tensor.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/tensor.zig>
- `src/tensor/shape.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/shape.zig>
- `src/tensor/ops/elementwise.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/ops/elementwise.zig>
- `src/tensor/ops/matmul.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/ops/matmul.zig>
- `src/tensor/ops/loss.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/ops/loss.zig>
- `src/autograd/tape.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/autograd/tape.zig>
- `src/autograd/node.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/autograd/node.zig>
- `src/lab/train.zig`: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/lab/train.zig>
- PyTorch autograd tutorial: <https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>
- PyTorch autograd notes: <https://docs.pytorch.org/docs/stable/notes/autograd.html>
- NVIDIA CUDA Driver API / module management: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>
- NVIDIA CUDA Programming Guide / Driver API: <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html>
- NVIDIA cuBLAS documentation: <https://docs.nvidia.com/cuda/cublas/>
- Zig 0.16.0 release notes: <https://ziglang.org/download/0.16.0/release-notes.html>

### 1.2 What was not done

I did not run the repository locally. I did not execute:

```bash
zig build test
zig build run-example -Dexample=06_train_shakespeare
zig build test -Dcuda=true
```

So this is a source-level review, not a verified local test report. Claims in the repo such as “215+ tests pass” are treated as project claims until the implementation agent re-runs the commands on the target machine and records exact outputs.

### 1.3 Review posture

This review is intentionally critical. The project is educational, so readability and correctness are not secondary concerns; they are the product. A small educational tensor/autograd project can tolerate slow CPU loops, simple data structures, and limited shape support. It should not tolerate unclear ownership, stale comments, silent shape errors, or device-memory hacks.

---

## 2. Executive assessment

### 2.1 Overall verdict

`zig-transformer-lab` is a promising educational project with a strong mission and a well-chosen scope. A tiny word-level transformer is the right size for teaching the internals of PyTorch-like systems. The project has already accumulated useful components: tensor metadata, broadcasting, matmul, reverse-mode tape autograd, neural-network layers, optimizers, tokenization/data, checkpointing, and a CPU training loop.

However, the current implementation should be treated as **CPU prototype quality**, not **CUDA-ready architecture**.

The biggest issue is not the absence of CUDA kernels. The biggest issue is that the current core abstractions are not strict enough:

- `Tensor.data: []f32` is a CPU memory slice and cannot represent CUDA device memory safely.
- The `Tensor.device` field exists, but the storage model has not actually separated CPU and CUDA memory.
- View semantics are partial: the project has shape/stride metadata, but several operations still behave as if tensors are simple contiguous flat arrays.
- Autograd lifetime depends on manual `tape.keepAlive()` discipline in high-level code.
- The code/document formatting appears line-compressed in raw files, which undermines the educational purpose.
- README/AGENTS/planning status is not fully synchronized: README advertises CUDA build/run commands, while `AGENTS.md` says Stage 7 CUDA is not started.

### 2.2 Rating by area

| Area | Assessment | Reason |
|---|---:|---|
| Project purpose | Strong | Small PyTorch-like educational lab is a good target. |
| Scope control | Strong | f32-only, one-block, one-head transformer, no third-party Zig dependencies. |
| CPU pedagogical value | Medium-high | Many comments explain intent, but formatting and stale notes reduce clarity. |
| Tensor abstraction | Medium | Shape/stride model exists, but storage/view semantics are incomplete. |
| Autograd abstraction | Medium-low | Correct direction, fragile lifetime model. |
| Operation correctness risk | Medium-high | Non-contiguous paths and index tensors need hardening. |
| Build/document hygiene | Low-medium | Raw files appear compressed into very long logical lines. |
| CUDA readiness | Low | Need storage/backend refactor before kernels. |
| Testing direction | Medium | Many tests are claimed, but more invariants/oracle/parity tests are needed. |
| Agent guidance | Medium | `AGENTS.md` captures lessons, but needs enforceable gates and CUDA-readiness rules. |

### 2.3 Highest-priority recommendation

Create a pre-CUDA hardening stage with this exit criterion:

```text
The CPU implementation has explicit tensor invariants, verified view semantics,
operation-owned autograd saved data, strict shape/index validation, exact docs,
and a backend/storage boundary that can represent CPU and CUDA without hacks.
```

Only then implement CUDA.

---

## 3. Current repository state

### 3.1 Stated project mission

The README describes the project as a small, heavily commented Zig 0.16.0 library for training a tiny one-block, one-head, word-level transformer from scratch. It is intended to teach PyTorch-like system internals: tensor libraries, autograd, training loops, and CUDA acceleration.

That mission is coherent. The project should stay focused on:

- one model family,
- one dtype (`f32`),
- one CPU backend,
- one CUDA backend,
- one educational training loop,
- one clear path from Python/PyTorch mental models to Zig/CUDA implementation.

Do not expand scope before CUDA parity works.

### 3.2 Stated progress

`AGENTS.md` says:

- Stage 1 scaffold: done.
- Stage 2 CPU tensor foundation: done.
- Stage 3 tape-based autograd: done.
- Stage 4 NN layers + optimizers: done.
- Stage 5 tokenizer + data pipeline: done.
- Stage 6 end-to-end CPU training: done.
- Stage 7 CUDA backend: not started.

The same file reports several important Stage 6 fixes:

- `backwardCrossEntropy` target rounding fix.
- untracked reshape bug fixed with `reshapeTracked()`.
- gradient clipping added.
- AdamW `beta2` changed from `0.95` to `0.999`.
- parameter pointer lifetime issues fixed by collecting params after the model is in final location.
- use-after-free on intermediates worked around with `tape.keepAlive()`.

These notes are valuable. They should be preserved, but they should also be converted into tests and hard rules, not left as historical commentary only.

### 3.3 Mismatch to correct immediately

The README already includes CUDA build/run commands:

```bash
zig build test -Dcuda=true
zig build run-example -Dexample=06_train_shakespeare -Dcuda=true
```

But `AGENTS.md` says Stage 7 CUDA is not started. That mismatch matters. A reader can reasonably infer that CUDA is available when it is not.

Fix the README now:

```md
## CUDA status

CUDA is planned but not implemented yet. The `-Dcuda=true` build option may exist
as scaffolding, but GPU kernels and CUDA training are not ready. See `AGENTS.md`
and `docs/08_backends_cuda.md` for the Stage 7 plan.
```

Then move the future CUDA commands into a clearly labeled section:

```md
## Planned CUDA commands after Stage 7
```

---

## 4. Architectural assessment

### 4.1 What the architecture is trying to be

The intended architecture is a minimal educational dynamic-graph tensor library:

```text
Tensor + Shape + Strides
        |
        v
Tensor ops: elementwise, reduce, matmul, softmax, loss
        |
        v
Tape-based autograd records forward operations
        |
        v
NN layers: Linear, Embedding, LayerNorm, Attention, MLP, Block, Model
        |
        v
Trainer + optimizer + dataset + generation
        |
        v
CUDA backend for selected tensor ops
```

That is the right structure.

### 4.2 What the architecture currently risks becoming

Without intervention, the architecture can become a fragile “flat slice plus flags” system:

```text
[]f32 + shape metadata + ad hoc tape nodes + manual keepAlive + CUDA hacks
```

That would be hard to teach and hard to debug. The main risk is adding CUDA by bolting `CUdeviceptr` behavior onto `Tensor.data` or by sprinkling `if (device == .cuda)` inside operations without a storage/dispatch model.

### 4.3 Correct architectural target

The target should be:

```text
Storage owns memory.
Tensor is a view over storage.
Ops dispatch by device.
Autograd records stable saved values.
CUDA backend is explicit and testable.
```

Concretely:

```text
Storage
  - CPU: []f32
  - CUDA: CUdeviceptr + len + context/device id

Tensor
  - storage reference
  - shape
  - strides
  - storage offset
  - dtype
  - requires_grad
  - grad pointer or grad slot
  - tape node id

Op
  - validates shape/device/layout
  - dispatches to CPU or CUDA backend
  - records backward node if tape is active

Tape node
  - owns saved values needed for backward
  - does not rely on high-level callers keeping random intermediates alive
```

---

## 5. Source hygiene and documentation assessment

### 5.1 Major problem: line-compressed source files

Several raw files appear compressed into very long logical lines. Examples from raw source inspection:

- `README.md` appears as 4 lines.
- `build.zig` appears as 11 lines.
- `src/tensor/tensor.zig` appears as 43 lines despite containing a large amount of code and tests.
- `src/tensor/ops/elementwise.zig` appears as 14 lines.
- `src/lab/train.zig` appears as 30 lines.

Even if this compiles, it is not acceptable for an educational repository.

A teaching repository should be readable in GitHub, readable in diffs, readable by humans who are new to Zig, and easy to annotate. Giant logical lines defeat that.

### 5.2 Required hygiene PR

First PR before any implementation work:

```bash
zig fmt build.zig src/**/*.zig tests/**/*.zig examples/**/*.zig
```

Also format markdown files so sections, lists, and code fences are readable.

Add a hygiene script that checks formatting, rejects extreme line lengths, and rejects scratchpad comments such as `Wait`, `I think`, `oops`, `temporary hack`, or similar self-dialog wording. The details can live in `tools/check_hygiene.py`, then be wired to:

```bash
zig build hygiene
```

### 5.3 Comments should teach invariants, not uncertainty

Some comments in the project appear to capture development-history thoughts: old bugs, “why this was fixed,” and self-corrective notes. Historical notes can be useful in docs, but production code comments should not sound uncertain.

Bad style:

```text
Wait, this needs to...
This was wrong before...
I think...
```

Good style:

```text
Invariant: a differentiable reshape must be represented as a tape node.
Reason: cross-entropy returns gradients in the reshaped logits shape, so backward
must restore the original 3D logits shape before propagating into the model.
```

For an educational project, comments should answer:

1. What invariant is being preserved?
2. What bug would happen if this were removed?
3. What PyTorch concept does this correspond to?
4. Who owns the memory?
5. What shape is expected at each point?

---

## 6. Tensor system assessment

### 6.1 What is good

The current `Tensor` concept has the right educational ingredients:

- shape metadata,
- strides,
- explicit dtype field,
- explicit device field,
- ownership flag,
- autograd metadata,
- `view`, `reshape`, `transpose2d`, `copyTo`, `fill`, and indexed access.

That is exactly the right surface for teaching “what a tensor really is.”

The use of explicit allocators is also correct for Zig. It makes memory ownership visible and teaches the discipline that Python/PyTorch usually hide.

### 6.2 Major problem: `[]f32` cannot represent CUDA memory

The current `Tensor` stores:

```zig
data: []f32,
device: Device,
```

This is a CPU-centric abstraction. A Zig slice is a host pointer plus length. CUDA device memory is not a normal host-accessible slice. The CUDA Driver API distinguishes `CUdeviceptr` for heap memory on the device, and it requires explicit context/device management before using device pointers.

The current comment in `tensor.zig` says Stage 7 may make `.cuda` tensors have `data` pointing to device memory. That direction should be rejected.

Do not do this:

```zig
// Bad idea
Tensor{ .data = some_fake_slice_from_CUdeviceptr, .device = .cuda }
```

It invites accidental host reads:

```zig
// This would compile-looking code but would be semantically wrong for CUDA.
const x = tensor.data[0];
```

### 6.3 Required storage abstraction

Introduce storage before CUDA kernels.

A possible design:

```zig
pub const Device = enum {
    cpu,
    cuda,
};

pub const CpuStorage = struct {
    data: []f32,
    owned: bool,
};

pub const CudaStorage = struct {
    ptr: CUdeviceptr,
    len: usize,          // number of f32 elements
    owned: bool,
    device_id: i32,
    context_id: usize,   // or pointer/id to the owning CudaContext
};

pub const Storage = union(Device) {
    cpu: CpuStorage,
    cuda: CudaStorage,
};

pub const Tensor = struct {
    storage: Storage,
    shape: Shape,
    strides: Strides,
    offset: usize,
    dtype: DType,
    requires_grad: bool,
    grad: ?*Tensor,
    tape_node: ?NodeId,
};
```

Benefits:

- CPU code cannot accidentally index CUDA memory.
- CUDA code cannot accidentally call `allocator.free()` on device memory.
- Views can be represented as storage + offset + shape + strides.
- Device transfer becomes explicit.
- Deinit dispatches by storage kind.

### 6.4 Add storage offset

The current view model appears to share a data slice and swap strides, but there is no explicit storage offset field. That is manageable for simple full-tensor transpose views, but it becomes limiting for slices and future layout operations.

Add:

```zig
offset: usize
```

Then logical indexing becomes:

```zig
physical_offset = tensor.offset + sum(indices[i] * tensor.strides.values[i])
```

This also makes invariant checking possible:

```text
offset + max_logical_offset(shape, strides) < storage.len
```

### 6.5 Add tensor invariant checks

Implement a debug invariant function and call it after every tensor construction and shape/layout transformation.

Pseudocode:

```zig
pub fn checkInvariants(self: Tensor) LabError!void {
    if (self.shape.rank == 0 or self.shape.rank > Shape.max_rank) return error.InvalidShape;

    for (0..self.shape.ndim()) |axis| {
        if (self.shape.dims[axis] == 0) return error.InvalidShape;
    }

    const n = totalElementsChecked(self.shape) catch return error.InvalidShape;
    if (n == 0) return error.InvalidShape;

    const storage_len = self.storage.len();
    const max_off = try maxLogicalOffset(self.shape, self.strides);
    if (self.offset + max_off >= storage_len) return error.InvalidShape;

    if (self.grad) |g| {
        if (!shapeEquals(g.shape, self.shape)) return error.ShapeMismatch;
        if (g.device() != self.device()) return error.DeviceMismatch;
    }
}
```

Add `LabError` entries as needed:

```zig
DeviceMismatch,
InvalidShape,
InvalidLayout,
InvalidIndex,
```

### 6.6 Fix non-contiguous copy behavior

`copyTo` is currently documented as supporting non-contiguous tensors, but the inspected implementation falls back to a flat `for (0..self.data.len)` copy in the non-contiguous branch. That is not a logical strided copy. For a transposed tensor, it can copy the wrong order.

Required fix:

```zig
pub fn copyTo(self: Tensor, dst: *Tensor) LabError!void {
    if (!shapeEquals(self.shape, dst.shape)) return error.ShapeMismatch;
    try requireSameDevice(self, dst.*);

    if (self.device() != .cpu or dst.device() != .cpu) {
        return error.NotImplemented; // until CUDA copy path exists
    }

    const n = totalElements(self.shape);
    for (0..n) |logical_i| {
        const src_off = logicalOffsetFromLinear(self.shape, self.strides, self.offset, logical_i);
        const dst_off = logicalOffsetFromLinear(dst.shape, dst.strides, dst.offset, logical_i);
        dst.cpuData()[dst_off] = self.cpuData()[src_off];
    }
}
```

Then add tests:

```zig
test "copyTo transposed view preserves logical order" { ... }
test "copyTo transposed source into contiguous dst" { ... }
test "copyTo contiguous source into transposed dst" { ... }
test "copyTo rejects device mismatch" { ... }
```

### 6.7 Fix scalar and in-place operations on non-contiguous tensors

Current scalar ops such as `addScalar`, `mulScalar`, `neg`, `fill`, and `addInPlace` appear to iterate over flat `data[i]`. That is correct only for contiguous tensors.

Pick and enforce one of these policies:

#### Policy A — full strided support

All simple elementwise ops use logical offset helpers.

```zig
for (0..totalElements(a.shape)) |i| {
    const in_off = a.logicalOffset(i);
    out.data[i] = a.data[in_off] + scalar;
}
```

#### Policy B — explicit contiguous-only support

Most ops reject non-contiguous inputs:

```zig
if (!a.isContiguous()) return error.NonContiguousUnsupported;
```

#### Recommended policy for this project

Use a hybrid policy:

```text
- Tensor can represent strided views.
- Copy, fill, elementwise, and reductions support strided CPU tensors.
- Matmul can support strided 2D reads on CPU, but CUDA matmul may require contiguous tensors initially.
- Heavy CUDA ops may require contiguous inputs until a later stage.
- Every op must document its layout policy.
```

This teaches views without overloading the first CUDA pass.

### 6.8 Shape zero-dimension policy

The project should explicitly disallow zero dimensions for now.

Zero-sized tensors are useful in mature tensor libraries, but they complicate every rule:

- `totalElements` becomes zero.
- reductions need identity behavior.
- softmax over an empty dimension is undefined/ambiguous.
- CUDA launch grids need special handling.
- autograd accumulation needs careful edge cases.

For a tiny transformer educational project, zero dimensions add no value. Reject them early.

### 6.9 `Shape.squeeze` should be strict

If `squeeze(shape, axis)` allows squeezing a dimension whose size is not 1, it should be changed. Squeezing a non-1 dimension hides bugs.

Correct policy:

```text
squeeze(axis) may only remove dimension axis when dims[axis] == 1.
squeezeAll() may only remove dimensions equal to 1.
```

### 6.10 Recommended tensor tests

Add these tests before CUDA:

```text
Tensor invariants:
- init rejects zero dim
- totalElements overflow rejects or traps in debug
- view shares storage but does not own
- transpose view has correct offset/strides
- reshape preserves total element count
- reshape non-contiguous returns NotImplemented or copies explicitly
- copyTo transposed source logical order
- fill transposed view mutates logical elements only
- addScalar transposed source matches expected logical output
- addInPlace transposed lhs + contiguous rhs works or explicitly rejects

Device/layout:
- CPU tensor exposes host slice
- CUDA tensor does not expose host slice
- device mismatch in binary ops rejects
- non-contiguous CUDA inputs reject until supported
```

---

## 7. Shape and broadcasting assessment

### 7.1 What is good

The project uses fixed-rank shape metadata with a small `max_rank`, which is good for this scope. General-purpose rank-N dynamic shape systems are unnecessary here.

Right-aligned broadcasting also matches the behavior students will recognize from NumPy and PyTorch.

### 7.2 Main risk

Broadcasting code must combine correctly with strides. Broadcasting an already-strided tensor is a common source of mistakes.

Example:

```text
A = transpose(original)  # non-contiguous
B = scalar/vector broadcast
out = A + B
```

This must use the input tensor’s strides, not assume flat row-major input.

### 7.3 Required broadcasting helpers

Create reusable helpers:

```zig
pub const Indexer = struct {
    shape: Shape,
    strides: Strides,
    offset: usize,

    pub fn offsetFromLinear(self: Indexer, linear: usize) usize { ... }
};

pub fn broadcastedInputOffset(
    out_linear: usize,
    out_shape: Shape,
    in_shape: Shape,
    in_strides: Strides,
    in_offset: usize,
) usize { ... }
```

All broadcasting ops must use these helpers.

### 7.4 Test matrix for broadcasting

```text
CPU forward:
- (2,3) + (2,3)
- (2,3) + (3,)
- (2,1) + (2,3)
- (1,3) + (2,3)
- (2,3,4) + (1,4)
- transposed (3,2) + (2,) if supported
- transposed (3,2) + scalar

Backward:
- grad through broadcast add reduces over broadcast axes
- grad through broadcast mul with shape reduction
- grad with rank mismatch
- grad with non-contiguous input if supported
```

---

## 8. Autograd assessment

### 8.1 What is good

The project uses a tape-based dynamic graph. That is the right educational choice.

PyTorch autograd records the operations that produced tensors and then performs reverse-mode differentiation during backward propagation. Your project is intentionally modeling that idea in small, readable form. This is the correct abstraction for teaching.

### 8.2 Main problem: manual `keepAlive()` is a footgun

The project currently has `tape.keepAlive()` and high-level code is expected to call it for intermediates that backward will need.

That is fragile.

It means a layer author must remember:

- which intermediate tensors are owned,
- which intermediates backward will access,
- which tensors are views,
- which buffers will be freed before backward,
- which saved `Tensor` snapshots contain borrowed data slices.

That is too much hidden responsibility for high-level model code.

In a PyTorch-like system, the operation that records the backward node should save what backward needs. Callers should not need to know.

### 8.3 Required direction: operation-owned saved data

Move toward this rule:

```text
If an operation's backward implementation needs a tensor/value/shape/scalar,
the operation's recording function must save it in the tape node.
```

High-level model code should not call `keepAlive()` except for rare explicitly documented ownership transfers.

Current undesirable pattern:

```zig
var logits = try model.forward(...);
try tape.keepAlive(&logits);
var flat = try reshapeTracked(...);
```

Preferred pattern:

```zig
var logits = try model.forward(...);
var flat = try ops.reshapeTracked(...);
// reshapeTracked internally records/saves what backward needs
```

### 8.4 Saved tensor design

Add an explicit `SavedTensor` concept.

For CPU-only stage:

```zig
pub const SavedTensor = struct {
    shape: Shape,
    strides: Strides,
    offset: usize,
    data: []f32,
    owns_copy: bool,
};
```

For CPU/CUDA stage:

```zig
pub const SavedTensor = struct {
    shape: Shape,
    strides: Strides,
    offset: usize,
    storage_ref: StorageRef,
    owns_storage: bool,
    device: Device,
};
```

Decide per op whether to save:

- a by-value shape/scalar only,
- a borrowed view with guaranteed storage lifetime,
- an owned copy,
- a compact saved auxiliary buffer.

For education, favor clarity over memory optimality. If copying a small tensor makes backward ownership obvious, copy it. Later optimize.

### 8.5 Tape graph invariants

Add tape invariants:

```text
- every recorded non-leaf node has n_parents matching the op kind
- every parent NodeId exists in this tape
- every requires_grad output with tape_node has a valid node id
- every leaf parameter is tracked fresh per tape
- every backward op validates incoming gradient shape
- every gradient accumulation validates target shape/device
- no saved tensor points to freed storage
```

### 8.6 Leaf node policy

The AGENTS file mentions a stale `tape_node` bug: parameters kept old node IDs after a tape was destroyed, so `trackLeaf()` had to always create a fresh node.

Turn this into a formal policy:

```text
Tensor.tape_node is tape-local metadata.
A tensor may retain a stale tape_node after the tape that produced it is destroyed.
Therefore, TrackLeaf must never trust existing tape_node values from a previous tape.
```

Better design:

```zig
pub const TapeId = u64;

pub const Tensor = struct {
    tape_ref: ?struct { tape_id: TapeId, node_id: NodeId },
};
```

Then stale IDs can be detected instead of silently ignored.

### 8.7 Double backward policy

Do not implement double backward yet. Explicitly document that gradients are not recorded as differentiable tensors.

Add tests:

```text
- calling backward twice without retain_graph returns error or recomputes only when explicitly allowed
- backward on non-scalar loss requires explicit seed gradient or returns InvalidArgument
- zeroGrad clears gradients but not parameter values
```

### 8.8 Required gradcheck infrastructure

Add finite-difference gradient checks for simple ops:

```text
- add
- sub
- mul
- div
- addScalar
- mulScalar
- neg
- mean
- sum
- sqrt
- gelu
- matmul
- softmax+crossEntropy
- layernorm small case
- embedding scatter-add
```

Gradcheck pseudocode:

```zig
fn gradcheck(fn_under_test, input_tensor, eps: f32, tol: f32) !void {
    // 1. compute analytic gradient via tape
    // 2. perturb each input element by +eps and -eps
    // 3. compute centered difference
    // 4. compare analytic vs numeric
}
```

For `f32`, use tolerances that acknowledge numerical noise:

```text
eps: 1e-3 or 1e-2 depending on op
tolerance: 1e-2 for unstable functions, tighter for linear ops
```

---

## 9. Tensor operation assessment

### 9.1 Elementwise ops

Strengths:

- Broadcasting support exists.
- Shape contracts are documented.
- Basic tests exist.
- Tape recording helpers exist.

Weaknesses:

- Scalar ops and in-place ops appear contiguous-only but not always documented as such.
- Broadcast helpers need to respect storage offsets after the storage refactor.
- Device mismatch checks need to be introduced before CUDA.
- Non-contiguous behavior must be enforced consistently.

Required changes:

```text
- All binary ops call requireSameDevice(a, b).
- All CPU elementwise ops use logical offset helpers.
- In-place ops reject overlapping aliases unless explicitly safe.
- In-place ops on tensors participating in autograd are either forbidden or tightly scoped.
```

Autograd in-place warning:

For a beginner educational system, forbid in-place mutation of tensors that require grad except optimizer updates on leaf parameters after backward.

Add:

```zig
if (a.requires_grad) return error.InPlaceOnGradTensorUnsupported;
```

where appropriate.

### 9.2 Matmul

Strengths:

- The naive `ikj` loop is good pedagogically.
- It uses strides for `a` and `b` reads in inspected code.
- The shape contract is clear.

Weaknesses:

- CUDA matmul will require careful row-major/cuBLAS mapping.
- Output is assumed contiguous, which is fine, but should be documented.
- Batched matmul needs separate CUDA mapping; do not assume the 2D wrapper generalizes automatically.

Required changes:

```text
- Add requireSameDevice.
- CPU matmul supports strided reads but produces contiguous output.
- CUDA matmul initially requires contiguous inputs or explicitly makes contiguous copies.
- Add tiny asymmetric tests for every row-major/cuBLAS wrapper.
```

Tiny tests:

```text
A(2,3) @ B(3,4)
A(1,3) @ B(3,1)
A(5,1) @ B(1,7)
A(2,2) @ I
I @ A(2,2)
Batched: (2,2,3) @ (2,3,4)
Grad A: dY @ B^T
Grad B: A^T @ dY
```

### 9.3 Softmax

Softmax is numerically sensitive. The CPU implementation should use max-subtraction:

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

Required tests:

```text
- rows sum to 1
- constant row gives uniform distribution
- large positive inputs do not overflow
- large negative inputs do not underflow to all zeros
- masked attention rows do not produce NaN when at least one token is unmasked
```

### 9.4 Cross-entropy

Current target handling uses `f32` target tensors and rounding to convert to class IDs. That is pragmatic because the project is f32-only, but it is not clean.

Class labels are not differentiable f32 tensors. They are integer indices.

Recommended options:

#### Option A — introduce `IndexTensor`

```zig
pub const IndexTensor = struct {
    data: []u32,
    shape: Shape,
    strides: Strides,
    owned: bool,
};
```

Use `IndexTensor` for:

- token IDs,
- embedding input IDs,
- cross-entropy targets.

#### Option B — keep f32 tensors but validate strictly

If avoiding another tensor type, then cross-entropy and embedding must validate:

```text
- target is finite
- target is not NaN
- target is not negative
- target is exactly integer-valued after round-trip
- target < class_count
- target tensor requires_grad == false
```

Do not accept fractional class IDs silently.

### 9.5 Embedding

Embedding backward is scatter-add into the embedding table. That is a core teaching opportunity.

Requirements:

```text
- input IDs must be validated as indices, not gradients
- repeated token IDs accumulate gradients
- out-of-range token ID returns InvalidIndex
- CUDA version must use atomic add or an equivalent reduction/scatter strategy
```

Add tests:

```text
- repeated token IDs produce summed gradient rows
- unused token rows get zero gradient
- invalid ID rejects
- embedding input cannot require grad
```

### 9.6 LayerNorm

LayerNorm is often a source of subtle gradient bugs. Since the project currently composes LayerNorm from primitive ops, that is educationally good.

Before CUDA:

```text
- keep CPU LayerNorm as composed ops for teaching
- add a fused CUDA LayerNorm later only after composed CPU/CUDA parity is established
- keep both code paths documented
```

Add tests:

```text
- output mean approximately 0 per row
- output variance approximately 1 per row when gamma=1 beta=0
- gradient compared to PyTorch small tensors
- epsilon behavior documented
```

---

## 10. Neural network and model assessment

### 10.1 What is good

The module set is appropriate:

- Linear,
- Embedding,
- LayerNorm,
- GELU,
- CausalSelfAttention,
- MLP,
- TransformerBlock,
- TinyWordTransformer,
- SGD,
- AdamW,
- Trainer.

Pre-norm causal self-attention is a good choice for training stability and simplicity.

### 10.2 Risk: model structs and pointer lifetime

`AGENTS.md` reports a dangling pointer bug where params were collected before the model was copied into the `Trainer` struct. That is a real Zig ownership design smell.

If a struct contains owned tensors and you take pointers to its fields, copying the struct can invalidate the relationship between pointers and owners.

Required policy:

```text
Do not casually copy initialized modules that own tensors.
```

Possible approaches:

1. **Explicit no-copy convention.** Add comments and avoid copies by API discipline.
2. **Heap-allocate modules.** Store the model in stable heap memory and pass pointers.
3. **Init-in-place pattern.** Initialize the final struct location directly.
4. **Parameter IDs instead of raw pointers.** Use stable identities and a parameter registry.

For this project, use a simple explicit design:

```zig
pub const TinyWordTransformer = struct {
    // owns tensors; do not copy after init

    pub fn initInto(self: *TinyWordTransformer, allocator: Allocator, config: Config, seed: u64) !void { ... }
    pub fn deinit(self: *TinyWordTransformer, allocator: Allocator) void { ... }
    pub fn collectParams(self: *TinyWordTransformer, out: *ArrayList(*Tensor)) !void { ... }
};
```

Then avoid returning initialized owning modules by value where field pointers are later retained.

### 10.3 Parameter count should be exact or removed

If `TinyWordTransformer.paramCount()` is only a rough estimate, remove it or rename it.

For a teaching project, inaccurate metadata is worse than missing metadata. Students will use parameter counts to reason about model memory and training cost.

Correct options:

```text
- exactParamCount()
- parameterBytes()
- activationEstimateBytes(batch, seq_len)
- remove count until implemented exactly
```

### 10.4 Model forward should minimize manual lifetime logic

The model forward currently appears to use `keepAlive()` for intermediates. After the autograd refactor, layer code should look boring:

```zig
var h = try self.ln1.forward(allocator, x, tape);
var attn = try self.attn.forward(allocator, h, tape);
var x2 = try ops.add(allocator, x, attn, tape);
var h2 = try self.ln2.forward(allocator, x2, tape);
var mlp = try self.mlp.forward(allocator, h2, tape);
return try ops.add(allocator, x2, mlp, tape);
```

It should not need to know which intermediate buffers backward will need.

---

## 11. Optimizer and training loop assessment

### 11.1 AdamW state keying

The AdamW implementation reportedly keys optimizer state by `@intFromPtr(param.data.ptr)`.

That works only while:

- parameter buffers never reallocate,
- parameters stay on CPU,
- checkpoint reload does not replace storage,
- no device transfer changes storage identity,
- no module copy invalidates assumptions.

CUDA will break or complicate this.

Required change:

```zig
pub const ParamId = u64;

pub const Parameter = struct {
    id: ParamId,
    name: []const u8,
    tensor: *Tensor,
};
```

Optimizer state should be keyed by `ParamId`, not by data pointer.

### 11.2 Optimizer state device policy

Decide now:

```text
For CPU training: optimizer state lives on CPU.
For CUDA training: optimizer state should live on CUDA for performance.
```

Early CUDA implementation can do:

```text
- forward/backward on CUDA
- copy gradients to CPU
- CPU AdamW step
- copy parameters back to CUDA
```

But this is educationally confusing and slow. If used as a temporary bring-up path, label it clearly:

```text
Stage 7.0 debug-only hybrid optimizer, not final CUDA training.
```

Preferred final Stage 7 design:

```text
- parameters on CUDA
- gradients on CUDA
- AdamW moments on CUDA
- AdamW update kernel on CUDA
```

### 11.3 Training-loop lifetime

Review every `defer` inside the training loop after formatting. In Zig, `defer` runs at scope exit, not necessarily at loop iteration end unless a scoped block is created.

Use explicit per-step blocks:

```zig
for (0..config.max_steps) |step| {
    {
        var batch = try dataset.nextBatch(...);
        defer batch.deinit(allocator);

        var tape = Tape.init(allocator);
        defer tape.deinit();

        // forward, loss, backward, step
    }
}
```

This is clearer for learners and safer for memory.

### 11.4 Gradient clipping

Gradient clipping is a good addition. Keep it.

But document:

```text
- global norm definition
- when it runs relative to AdamW moment updates
- what happens if grad norm is NaN or Inf
- whether parameters without grad are skipped
```

Add tests:

```text
- grad norm below threshold unchanged
- grad norm above threshold scaled
- zero gradients do not produce divide-by-zero
- NaN gradient triggers NumericalError
```

### 11.5 Generation

Generation is useful for demos, but it should not distract from Stage 7. Keep CPU generation stable; CUDA generation can initially transfer logits to CPU for sampling because sampling is not the training bottleneck.

Document that generation sampling is not part of the CUDA performance target.

---

## 12. Checkpointing assessment

### 12.1 Current risk

Checkpointing is easy to under-validate. A teaching project should fail loudly on checkpoint mismatch.

Required validations:

```text
- magic bytes
- version
- endian policy
- dtype
- number of parameters
- parameter names
- parameter shapes
- byte counts
- duplicate names
- missing expected names
- unexpected names
- partial read
- trailing data policy
```

### 12.2 Add checkpoint manifest

Write a small model manifest:

```text
model_type: TinyWordTransformer
version: 1
vocab_size
seq_len
d_model
hidden_dim
parameter_count
```

Then each parameter:

```text
name_len: u32
name: bytes
rank: u8
dims: [rank]u64
dtype: f32
data_len: u64
raw_data: f32[data_len]
```

### 12.3 CUDA checkpoint policy

Checkpoints should be device-neutral.

When saving CUDA model parameters:

```text
- copy parameter data to CPU
- write CPU f32 data
```

When loading into CUDA model:

```text
- read into CPU temporary
- copy to CUDA parameter storage
```

Do not serialize raw CUDA pointers.

---

## 13. Build system assessment

### 13.1 What is good

The build has the right intended options:

```text
-Dcuda=true
-Dcuda_arch=sm_89
-Dcuda_home=...
-Dexample=...
-Dseed=...
```

It also has the right high-level CUDA approach for this project: offline `nvcc -ptx`, then runtime Driver API module loading.

### 13.2 Problems to correct

The inspected `build.zig` appears compressed into 11 lines. Reformat it.

The `nvcc` invocation in `plan.md` uses:

```bash
--use_fast_math
```

Remove that from the correctness-first CUDA stage. Fast math can change numerical results and make CPU/GPU parity harder to reason about.

Recommended staged policy:

```text
Stage 7 correctness mode:
- nvcc -O2 or -O3
- no --use_fast_math
- deterministic tests where possible
- tight CPU/GPU tolerances

Stage 7 performance mode:
- optional -Dcuda_fast_math=true
- documented tolerance relaxation
- only after correctness mode passes
```

### 13.3 Add build steps

Add:

```bash
zig build hygiene
zig build test-cpu
zig build test-oracle
zig build test-cuda
zig build docs-check
```

Suggested behavior:

```text
hygiene: formatting + line length + scratchpad comment check
test-cpu: all CPU tests
test-oracle: runs Python oracle comparisons if numpy/torch exist
test-cuda: all CUDA tests when -Dcuda=true
docs-check: required docs exist and basic line-count/readability checks pass
```

### 13.4 Build should not imply CUDA is implemented

If `-Dcuda=true` is accepted before Stage 7 works, it should produce a clear `NotImplemented` or skip message, not a misleading partial compile.

---

## 14. CUDA-readiness assessment

### 14.1 Current CUDA readiness: not ready

The project is conceptually pointed in the right direction, but the implementation should not enter CUDA kernels yet.

Blockers:

```text
- Tensor storage is CPU-slice based.
- Device field is not enough.
- Autograd saved data relies on borrowed CPU slices/manual keepAlive.
- Non-contiguous view support is incomplete.
- Device mismatch behavior is undefined.
- cuBLAS row-major policy is not yet encoded/tested.
- README status is misleading.
```

### 14.2 CUDA wrapper design

Implement CUDA in layers:

```text
src/backend/cuda/bindings.zig
  raw dynamic symbols loaded with dlopen/dlsym

src/backend/cuda/result.zig
  CUresult/cublasStatus_t translation

src/backend/cuda/context.zig
  device selection, cuInit, context create/destroy, current context

src/backend/cuda/stream.zig
  stream create/destroy/synchronize

src/backend/cuda/mem.zig
  cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH, cuMemcpyDtoD

src/backend/cuda/module.zig
  load PTX, get function, unload module

src/backend/cuda/launch.zig
  launch kernel, check errors, synchronize in debug tests

src/backend/cuda/cublas.zig
  cublasCreate/destroy/setStream/sgemm wrappers
```

No tensor ops until these wrappers are tested.

### 14.3 CUDA context policy

The CUDA Driver API requires `cuInit()` before using Driver API functions, and work happens inside CUDA contexts attached to devices. Use one explicit context object.

Do not hide global context initialization deep inside tensor ops.

Recommended:

```zig
pub const CudaContext = struct {
    device: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
    cublas: cublasHandle_t,

    pub fn init(device_ordinal: i32) !CudaContext { ... }
    pub fn deinit(self: *CudaContext) void { ... }
};
```

Every CUDA tensor either stores a reference/id to this context or is used only while the context is current.

### 14.4 CUDA error handling

Every CUDA call must be wrapped:

```zig
try cuda.check(cuMemAlloc(...));
try cuda.check(cuMemcpyHtoD(...));
try cublas.check(cublasSgemm(...));
```

Add human-readable error names when possible:

```zig
pub fn check(result: CUresult) CudaError!void {
    if (result != CUDA_SUCCESS) {
        // optionally call cuGetErrorName/cuGetErrorString
        return error.CudaError;
    }
}
```

After every kernel launch in tests:

```zig
try cuda.check(cuLaunchKernel(...));
try cuda.check(cuCtxSynchronize());
```

Do not rely on later calls to surface asynchronous errors without explanation.

### 14.5 Device transfer policy

Forbid implicit CPU/GPU transfers.

Bad:

```zig
var y = try ops.add(allocator, cpu_tensor, cuda_tensor, tape); // silently copies
```

Good:

```zig
var x_gpu = try x_cpu.toCuda(ctx);
var y_gpu = try ops.add(allocator, x_gpu, z_gpu, tape);
var y_cpu = try y_gpu.toCpu(allocator);
```

If devices differ:

```zig
return error.DeviceMismatch;
```

### 14.6 cuBLAS row-major policy

The project stores tensors row-major. cuBLAS uses column-major storage conventions. This is one of the easiest places to create a silent correctness bug.

Do not write a cuBLAS wrapper from memory. Derive it and test it.

For row-major `C = A @ B`, with:

```text
A: M x K row-major
B: K x N row-major
C: M x N row-major
```

Memory can be viewed as column-major transposes:

```text
A_row memory == A^T_col memory  (K x M)
B_row memory == B^T_col memory  (N x K)
C_row memory == C^T_col memory  (N x M)
```

Since:

```text
C = A B
C^T = B^T A^T
```

The cuBLAS call should compute column-major `C_col_view = B_col_view * A_col_view` with dimensions corresponding to `N x M`.

But do not rely on this paragraph alone. Add tiny matrix tests.

### 14.7 CUDA implementation order

Use this exact order:

#### Step 1 — CUDA loader smoke test

```text
- dlopen libcuda.so.1
- resolve cuInit/cuDeviceGetCount/cuDeviceGet/cuCtxCreate/cuCtxDestroy
- print or test device count
```

#### Step 2 — context + memory roundtrip

```text
- allocate device buffer
- copy CPU -> GPU
- copy GPU -> CPU
- compare exact values
```

#### Step 3 — PTX module load + vector add kernel

```text
- compile one vector_add.cu to PTX
- load PTX via driver API
- get function
- launch kernel
- compare CPU result
```

#### Step 4 — CUDA tensor storage

```text
- Tensor.toCuda(ctx)
- Tensor.toCpu(allocator)
- Tensor.deinit dispatches to cuMemFree for owned cuda storage
- no host indexing for CUDA tensor
```

#### Step 5 — same-shape elementwise kernels

```text
add, sub, mul, div, neg, scalar ops
```

No broadcasting yet.

#### Step 6 — broadcasting elementwise kernels

Implement index mapping carefully. CPU is oracle.

#### Step 7 — reductions

```text
sum, mean, variance support used by layernorm
```

Use simple kernels first, not optimized reductions.

#### Step 8 — cuBLAS matmul

```text
2D matmul first
batched matmul second
backward matmul tests third
```

#### Step 9 — softmax and causal mask

```text
row-wise softmax
masked fill or causal mask kernel
attention score path
```

#### Step 10 — cross-entropy and embedding

```text
embedding gather forward
embedding scatter-add backward
cross-entropy forward/backward
```

#### Step 11 — AdamW update kernel

```text
param, grad, m, v update on CUDA
```

#### Step 12 — model parity

```text
same seed, same config, same batch
CPU forward vs CUDA forward
CPU gradients vs CUDA gradients
one optimizer step parity
```

#### Step 13 — training benchmark

Only now measure speed.

### 14.8 CUDA acceptance criteria

Before claiming CUDA training works:

```text
- all CPU tests pass
- all CUDA smoke tests pass
- CPU/CUDA tensor transfer roundtrip passes
- CPU/CUDA each op forward parity passes
- CPU/CUDA each differentiable op backward parity passes
- CPU/CUDA full model forward diff < 5e-5, or documented tolerance
- CPU/CUDA full model gradient diff < 1e-4, or documented tolerance
- one CPU and CUDA optimizer step produce comparable parameters
- full training run decreases loss on tiny corpus
- no reported allocator leaks
- no CUDA memory leaks under cuda-memcheck/compute-sanitizer where practical
```

---

## 15. Testing and oracle plan

### 15.1 Testing philosophy

The project should use layered tests:

```text
1. Pure Zig unit tests for tiny examples.
2. Finite-difference gradcheck for local gradients.
3. Python/PyTorch oracle tests for model-level behavior.
4. CPU/CUDA parity tests for every backend op.
5. End-to-end training smoke tests.
```

Do not substitute end-to-end loss decreases for op-level correctness. A model can appear to learn while some gradients are wrong.

### 15.2 Test organization

Suggested layout:

```text
tests/
  unit_all.zig
  tensor_invariants.zig
  autograd_gradcheck.zig
  oracle_cpu.zig
  integration_cpu_training.zig
  integration_cuda_smoke.zig
  integration_cuda_ops.zig
  integration_cuda_model.zig
```

### 15.3 CPU test commands

```bash
zig build hygiene
zig build test
zig build run-example -Dexample=01_tensor_playground
zig build run-example -Dexample=04_overfit_one_batch
zig build run-example -Dexample=06_train_shakespeare
```

### 15.4 Python oracle commands

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r tools/requirements.txt
python3 tools/oracle.py --case tensor_ops
python3 tools/oracle.py --case autograd_ops
python3 tools/oracle.py --case tiny_transformer_forward
python3 tools/oracle.py --case tiny_transformer_one_step
```

### 15.5 CUDA test commands

```bash
zig build test -Dcuda=true
zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true
compute-sanitizer zig-out/bin/08_cuda_vs_cpu
```

If compute-sanitizer is too slow for every run, use it for nightly/manual validation.

### 15.6 Required oracle cases

```text
Tensor ops:
- elementwise add/sub/mul/div with broadcasting
- sum/mean over axes
- matmul/batched matmul
- transpose/reshape logical behavior

Autograd:
- scalar expression
- broadcast gradient reduction
- matmul gradients
- layernorm gradients
- softmax/cross-entropy gradients
- embedding scatter-add gradients

Model:
- fixed config
- fixed seed
- fixed token IDs
- logits comparison
- loss comparison
- all parameter gradients comparison
- AdamW one-step comparison
```

### 15.7 Test tolerance guidance

Use different tolerances by operation:

```text
Exact or near exact:
- copy
- reshape
- transpose
- integer/token behavior
- simple add/sub/mul for tiny values

1e-6 to 1e-5:
- matmul small CPU/PyTorch
- elementwise CPU/PyTorch

1e-5 to 1e-4:
- softmax/cross-entropy
- layernorm
- CUDA forward/backward

Looser only with documented reason:
- fast math
- reduction order differences
- larger random tensors
```

---

## 16. Recommended implementation roadmap

### 16.1 Phase A — repository hygiene and status correction

Goal: make the project readable and honest.

Tasks:

```text
A1. Run zig fmt on all Zig files.
A2. Reformat Markdown files.
A3. Add hygiene checker.
A4. Remove scratchpad comments.
A5. Make README, AGENTS, plan, and docs agree on CUDA status.
A6. Add “reviewed source state” note to AGENTS.
A7. Re-run CPU tests and paste exact command outputs into session notes.
```

Exit criteria:

```text
- no huge logical lines in source
- README says CUDA is planned/not implemented
- zig build hygiene passes
- zig build test passes or failures are documented
```

### 16.2 Phase B — tensor invariants and view correctness

Goal: make the tensor model boring and exact.

Tasks:

```text
B1. Add checked shape constructors or validation.
B2. Reject zero dimensions.
B3. Add Tensor.offset.
B4. Add Storage abstraction for CPU only first.
B5. Add logical offset helpers.
B6. Fix copyTo non-contiguous logic.
B7. Decide layout policy per op.
B8. Update fill/scalar/in-place ops.
B9. Add view/transpose/broadcast tests.
```

Exit criteria:

```text
- all tensor invariant tests pass
- non-contiguous tests either pass or explicitly reject with clear errors
- no op silently assumes contiguity unless documented and tested
```

### 16.3 Phase C — autograd ownership hardening

Goal: remove high-level manual `keepAlive()` reliance.

Tasks:

```text
C1. Define SavedTensor/SavedValue model.
C2. Make op recorders save what backward needs.
C3. Add tape/node invariant checks.
C4. Add TapeId or equivalent stale-node detection.
C5. Remove keepAlive calls from model/training code where possible.
C6. Add tests for reshapeTracked, transposeTracked, matmul, embedding, CE.
C7. Add gradcheck harness.
```

Exit criteria:

```text
- model forward/training code does not manually keep ordinary intermediates alive
- gradcheck passes for core differentiable ops
- stale tape-node test passes
```

### 16.4 Phase D — parameter/checkpoint/training hardening

Goal: make training state robust before moving to GPU.

Tasks:

```text
D1. Introduce ParamId/Parameter registry.
D2. Key optimizer state by ParamId.
D3. Make owning module copy policy explicit.
D4. Validate checkpoint load strictly.
D5. Make paramCount exact or remove it.
D6. Add gradient clipping numerical tests.
D7. Scope per-step resources explicitly in training loop.
```

Exit criteria:

```text
- optimizer state survives storage replacement in tests
- checkpoint mismatch tests fail loudly
- trainer memory does not grow across steps
```

### 16.5 Phase E — backend seam, CPU only

Goal: introduce the device/backend architecture before CUDA.

Tasks:

```text
E1. Add DeviceMismatch error.
E2. Add Backend enum/interface.
E3. Route ops through CPU backend.
E4. Ensure all CPU tests still pass.
E5. Add CUDA backend stubs returning NotImplemented.
```

Exit criteria:

```text
- no functionality change for CPU
- op dispatch path is explicit
- CUDA stubs exist but do not pretend to work
```

### 16.6 Phase F — CUDA wrapper and storage

Goal: make CUDA memory safe before math kernels.

Tasks:

```text
F1. Dynamic loader for libcuda/libcublas.
F2. CudaContext init/deinit.
F3. CUDA memory allocation/free/copy.
F4. Tensor.toCuda/toCpu.
F5. CUDA storage deinit.
F6. Device mismatch tests.
```

Exit criteria:

```text
- CUDA memory roundtrip passes
- no device pointer stored as []f32
- CUDA tensor cannot be read through host data slice
```

### 16.7 Phase G — CUDA op parity

Goal: implement operations one by one with CPU parity.

Tasks:

```text
G1. same-shape elementwise
G2. scalar ops
G3. broadcasting elementwise
G4. reductions
G5. matmul via cuBLAS
G6. batched matmul
G7. softmax/mask
G8. embedding
G9. cross-entropy
G10. AdamW
```

Exit criteria:

```text
- every op has CPU/CUDA forward parity test
- every differentiable op has CPU/CUDA backward parity test
```

### 16.8 Phase H — model CUDA parity and training

Goal: full model parity before speed claims.

Tasks:

```text
H1. Move model params to CUDA.
H2. Run forward parity with fixed seed/batch.
H3. Run backward parity.
H4. Run one-step AdamW parity.
H5. Run tiny corpus training.
H6. Run Shakespeare training.
H7. Only then benchmark.
```

Exit criteria:

```text
- full model forward diff within target
- full model gradient diff within target
- loss decreases on CPU and CUDA
- CUDA benchmark reported with config and hardware
```

---

## 17. Detailed PR plan for the agent

### PR 1 — Formatting and source-of-truth docs

**Purpose:** make repository readable and status honest.

Changes:

```text
- Run zig fmt.
- Format README/AGENTS/docs markdown.
- Add tools/check_hygiene.py.
- Add zig build hygiene if feasible.
- Update README CUDA status.
```

Tests:

```bash
zig build hygiene
zig build test
```

### PR 2 — Tensor invariant system

Changes:

```text
- Add InvalidShape/InvalidLayout errors.
- Add checked totalElements.
- Add checkInvariants.
- Reject zero dimensions.
- Add debug calls after constructors/views.
```

Tests:

```text
- zero dim rejection
- overflow cases
- valid shape cases
```

### PR 3 — Storage and offset, CPU only

Changes:

```text
- Add Storage union with only CPU active for now.
- Add Tensor.offset.
- Replace direct data access helpers where needed.
- Keep public compatibility wrappers if necessary.
```

Tests:

```text
- view shares storage
- transpose uses offset 0 and swapped strides
- deinit only frees owner
```

### PR 4 — Logical indexing and non-contiguous correctness

Changes:

```text
- Add logicalOffsetFromLinear.
- Fix copyTo.
- Fix fill and scalar ops or reject non-contiguous explicitly.
- Update elementwise indexing.
```

Tests:

```text
- transposed copy
- transposed scalar op
- transposed fill
- broadcast with transposed input
```

### PR 5 — Operation layout policy table

Changes:

```text
- Add docs/tensor_layout_policy.md.
- Every op declares: contiguous required? strided CPU supported? CUDA support planned?
- Add runtime checks matching policy.
```

Tests:

```text
- rejected layout returns expected error
```

### PR 6 — SavedTensor refactor

Changes:

```text
- Add SavedTensor/SavedValue.
- Move saved-data ownership into op recording.
- Reduce or remove tape.keepAlive use.
```

Tests:

```text
- backward after intermediates leave local scope
- reshapeTracked shape restoration
- no use-after-free under allocator checks
```

### PR 7 — TapeId and autograd invariants

Changes:

```text
- Add TapeId to tape references or equivalent stale-node protection.
- Add debug graph validation.
- Add clearer leaf tracking policy.
```

Tests:

```text
- two training steps with fresh tapes do not reuse stale node ids
- params tracked fresh each step
```

### PR 8 — Gradcheck harness

Changes:

```text
- Add src/autograd/gradcheck.zig.
- Add tests for core ops.
```

Tests:

```bash
zig build test
```

### PR 9 — Parameter registry and optimizer state identity

Changes:

```text
- Add ParamId.
- collectNamedParams returns Parameter objects.
- AdamW/SGD key state by ParamId.
```

Tests:

```text
- optimizer state persists across steps
- parameter storage replacement does not lose state unless explicitly reset
```

### PR 10 — Checkpoint validation

Changes:

```text
- Strict checkpoint manifest.
- Fail on missing/unexpected/duplicate parameters.
- Validate dtype/shape/count.
```

Tests:

```text
- save/load roundtrip
- corrupted magic fails
- wrong shape fails
- missing param fails
- duplicate param fails
```

### PR 11 — Backend dispatch seam

Changes:

```text
- Add backend interface.
- Route CPU ops through CPU backend.
- Add CUDA stubs returning NotImplemented.
```

Tests:

```text
- all CPU tests still pass
- CUDA op call without implementation returns NotImplemented
```

### PR 12 — CUDA dynamic loader

Changes:

```text
- dlopen/dlsym libcuda and libcublas.
- Resolve minimal symbols.
- Add error wrappers.
```

Tests:

```bash
zig build test -Dcuda=true
```

### PR 13 — CUDA context/memory/storage

Changes:

```text
- CudaContext.
- cuMemAlloc/free/copy wrappers.
- Tensor.toCuda/toCpu.
```

Tests:

```text
- CPU -> CUDA -> CPU roundtrip
- CUDA deinit frees memory
- host data access on CUDA tensor is impossible or errors
```

### PR 14 — CUDA vector add kernel

Changes:

```text
- Add elementwise.cu with simple vector add.
- Build PTX.
- Load module and launch.
```

Tests:

```text
- vector add parity
- kernel launch error path
```

### PR 15+ — CUDA ops in dependency order

Add one op family per PR. Do not combine too much.

Order:

```text
elementwise -> reductions -> matmul -> batched matmul -> softmax/mask -> embedding -> loss -> optimizer -> full model
```

---

## 18. Agent operating rules

Add this section to `AGENTS.md`.

### 18.1 Non-negotiable rules

```text
1. Do not begin CUDA kernels until CPU hardening gates pass.
2. Never represent CUDA device memory as []f32.
3. Never silently copy tensors between CPU and CUDA.
4. Every op must document its shape, device, layout, ownership, and autograd policy.
5. Every differentiable op must have backward tests.
6. Every CUDA op must have a CPU oracle parity test.
7. Every saved tensor needed for backward must be owned or retained by the op/tape, not by caller superstition.
8. Every checkpoint load must validate names, shapes, dtype, and byte counts.
9. Every change to OpKind must update backward implementation, tests, and docs in the same patch.
10. No source file should contain scratchpad/self-dialog comments.
```

### 18.2 Before editing checklist

```text
- Read the relevant source file and its tests.
- Read the docs chapter for that subsystem.
- Identify every tensor's owner.
- Identify every view's base storage.
- Identify whether inputs can be non-contiguous.
- Identify whether the op participates in autograd.
- Identify whether this change affects CPU/CUDA dispatch.
```

### 18.3 Before committing checklist

```bash
zig fmt build.zig src/**/*.zig tests/**/*.zig examples/**/*.zig
zig build hygiene
zig build test
```

If CUDA was touched:

```bash
zig build test -Dcuda=true
zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true
```

If model/training was touched:

```bash
zig build run-example -Dexample=04_overfit_one_batch
zig build run-example -Dexample=06_train_shakespeare
```

If oracle comparisons were touched:

```bash
python3 tools/oracle.py --case tiny_transformer_one_step
```

### 18.4 Session note template

Use a template like:

```md
# Session note — YYYY-MM-DD

## Goal

## Files changed

## Invariants affected

## Tests run

Paste exact commands and exact output summaries.

## Known failures or skipped tests

## Follow-up required
```

### 18.5 Forbidden shortcuts

```text
- Do not mark a stage done without exact test output.
- Do not add CUDA code that only works for one hard-coded shape unless documented as a temporary test kernel.
- Do not skip CPU oracle tests because CUDA seems to work.
- Do not use fast math before correctness parity.
- Do not introduce third-party Zig dependencies.
- Do not broaden the model before CUDA parity.
- Do not hide memory copies inside ops.
- Do not store optimizer state by raw data pointer after the ParamId refactor.
```

---

## 19. Skills and guideline updates

The project has a `skills/modern-zig-0-16-tutor` directory. Add or update the following reference files so the implementation agent has durable rules.

### 19.1 Add `18-tensor-autograd-invariants.md`

Suggested content:

```md
# Tensor and autograd invariants

## Tensor invariants

- Shape rank is 1..4.
- Dimensions are nonzero.
- Total element count must not overflow usize.
- Tensor has storage, shape, strides, and offset.
- `offset + max_logical_offset(shape, strides)` must fit in storage length.
- A view never owns storage.
- Owned CPU storage is freed with the allocator that created it.
- Owned CUDA storage is freed with `cuMemFree` in the owning context.
- CUDA device memory is never represented as `[]f32`.
- A CPU tensor may expose a host slice; a CUDA tensor may not.
- `grad`, when present, has the same shape and device as the tensor.

## Autograd invariants

- Tensor tape node IDs are tape-local.
- Leaf params must be tracked fresh for each tape.
- Operation recorders save everything backward needs.
- High-level model code should not manually keep ordinary intermediates alive.
- Backward validates incoming gradient shape at every node.
- Gradient accumulation validates shape/device.
- In-place mutation of tensors requiring grad is forbidden unless explicitly documented.
```

### 19.2 Add `19-cuda-backend-checklist.md`

Suggested content:

```md
# CUDA backend checklist

Before writing kernels:

- `cuInit` works.
- device count is checked.
- context create/destroy works.
- stream create/destroy works.
- cuBLAS create/destroy works.
- device allocation/free works.
- CPU->CUDA and CUDA->CPU copies work.
- PTX module load works.
- kernel function lookup works.
- one vector-add kernel launches and synchronizes.

Rules:

- Every CUDA call is checked.
- Every kernel launch has a debug synchronization test.
- No implicit CPU/GPU transfer.
- No device pointer in `[]f32`.
- No `--use_fast_math` until correctness mode passes.
- CPU parity test required for every CUDA op.
```

### 19.3 Add `20-row-major-cublas-gemm.md`

Suggested content:

```md
# Row-major cuBLAS GEMM rules

Project tensors are row-major.
cuBLAS uses column-major storage conventions.

For row-major C = A @ B:

- A is M x K row-major.
- B is K x N row-major.
- C is M x N row-major.
- The same memory can be interpreted as column-major transposes.
- Compute C^T = B^T @ A^T in cuBLAS terms.

Required tests:

- non-square A/B/C.
- M, N, K all different.
- vector-like matrices.
- gradient wrt A.
- gradient wrt B.
- batched GEMM separately.

Never change GEMM flags without updating the derivation and tests.
```

### 19.4 Add `21-agent-review-gates.md`

Suggested content:

```md
# Agent review gates

A PR is not complete until:

- source is formatted
- hygiene checks pass
- relevant unit tests pass
- relevant integration tests pass
- docs are updated
- AGENTS notes are updated when policy changes
- exact test commands and outputs are recorded

Extra gates:

- Tensor/storage change: run all tensor, autograd, model smoke tests.
- Autograd change: run gradcheck and one training smoke test.
- Optimizer change: run overfit-one-batch and checkpoint tests.
- CUDA change: run CPU/CUDA parity tests.
```

### 19.5 Update existing skill rules

Add this to any general Zig tutor instructions:

```text
Prefer clear ownership over cleverness. When reviewing code, ask:
- Who owns this memory?
- Can this tensor be a view?
- Is this tensor contiguous?
- What device is this tensor on?
- Does backward need this value later?
- Is this pointer stable after the struct is moved?
```

---

## 20. What to remove, rewrite, or preserve

### 20.1 Remove or disable

```text
- Misleading README CUDA commands until CUDA works.
- Rough/inaccurate paramCount or any approximate metadata exposed as fact.
- Scratchpad comments.
- Manual keepAlive requirements in high-level model/training code.
- Any plan to store CUDA memory in []f32.
- --use_fast_math in correctness-first CUDA builds.
```

### 20.2 Rewrite

```text
- Tensor storage model.
- Non-contiguous copy/fill/scalar operation paths.
- Autograd saved data ownership.
- Optimizer state identity.
- Checkpoint validation.
- README/AGENTS/docs status synchronization.
```

### 20.3 Preserve

```text
- Tiny transformer scope.
- Zig 0.16.0 pin.
- f32-only rule.
- No third-party Zig dependencies.
- CPU as correctness oracle.
- PyTorch comparison tooling.
- Staged learning-oriented docs.
- Heavy comments, once reformatted and cleaned.
```

---

## 21. Risk register

| Risk | Severity | Probability | Mitigation |
|---|---:|---:|---|
| CUDA memory represented as `[]f32` | Critical | Medium | Add storage union before CUDA. |
| Non-contiguous views produce wrong values | High | High | Add logical indexing helpers and tests. |
| Autograd saved tensors point to freed memory | High | Medium-high | Operation-owned `SavedTensor`; remove manual keepAlive reliance. |
| cuBLAS GEMM silently transposed/wrong | High | Medium | Row-major derivation and tiny asymmetric tests. |
| README implies CUDA works | Medium | High | Status correction PR. |
| Optimizer state lost after storage/device change | High | Medium | ParamId registry. |
| Checkpoint loads wrong model silently | High | Medium | Strict manifest validation. |
| Agent claims tests pass without running them | High | Medium | Session notes require exact commands/output. |
| Fast math causes confusing parity failures | Medium | Medium | Disable until performance mode. |
| Scope creep into multi-head/multi-layer before CUDA | Medium | Medium | Keep locked decisions; defer Stage 8 generalization. |

---

## 22. Suggested updated `AGENTS.md` section

Add this near the top of `AGENTS.md`:

```md
## Current engineering gate

Stage 7 CUDA is blocked until Stage 6.5 CPU hardening is complete.

Stage 6.5 acceptance criteria:

- Source formatting/hygiene checks pass.
- README, AGENTS, plan, and docs agree that CUDA is not yet implemented.
- Tensor invariants are implemented and tested.
- Zero dimensions are rejected.
- Storage/offset model exists; CUDA memory is never represented as []f32.
- Non-contiguous view behavior is either correct or explicitly rejected per op.
- copyTo, fill, scalar ops, and elementwise ops obey the layout policy.
- Autograd saved data is operation-owned; high-level model code does not manually keep ordinary intermediates alive.
- Stale tape-node behavior is tested.
- Optimizer state is keyed by stable ParamId, not data pointer.
- Checkpoint loading validates model/parameter metadata strictly.
- `zig build test` passes and exact output is recorded.

Only after this gate may the agent start CUDA wrapper work.
```

Add this to the CUDA section:

```md
## CUDA implementation rules

- Implement CUDA wrapper/context/memory tests before tensor CUDA ops.
- Implement Tensor.toCuda/toCpu before math kernels.
- Implement one op family at a time.
- Every CUDA op requires CPU forward parity.
- Every differentiable CUDA op requires CPU backward parity.
- No implicit device transfers.
- No fast math until correctness mode passes.
- cuBLAS row-major GEMM must be tested with asymmetric matrices.
```

---

## 23. Appendix A — example tensor storage pseudocode

This is intentionally pseudocode. The agent must adapt it to the actual Zig 0.16.0 style used in the repo.

```zig
pub const StorageKind = enum { cpu, cuda };

pub const CpuStorage = struct {
    data: []f32,
    owned: bool,
};

pub const CudaStorage = struct {
    ptr: CUdeviceptr,
    len: usize,
    owned: bool,
    device_id: i32,
};

pub const Storage = union(StorageKind) {
    cpu: CpuStorage,
    cuda: CudaStorage,

    pub fn len(self: Storage) usize {
        return switch (self) {
            .cpu => |s| s.data.len,
            .cuda => |s| s.len,
        };
    }

    pub fn device(self: Storage) Device {
        return switch (self) {
            .cpu => .cpu,
            .cuda => .cuda,
        };
    }
};

pub const Tensor = struct {
    storage: Storage,
    shape: Shape,
    strides: Strides,
    offset: usize,
    dtype: DType,
    requires_grad: bool,
    grad: ?*Tensor,
    tape_node: ?NodeId,

    pub fn cpuData(self: Tensor) LabError![]f32 {
        return switch (self.storage) {
            .cpu => |s| s.data,
            .cuda => error.DeviceMismatch,
        };
    }

    pub fn logicalOffset(self: Tensor, linear: usize) usize {
        return self.offset + logicalOffsetFromLinear(self.shape, self.strides, linear);
    }
};
```

---

## 24. Appendix B — example backend dispatch pseudocode

```zig
pub fn add(allocator: Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    try requireSameDevice(a, b);
    const out = switch (a.device()) {
        .cpu => try cpu_ops.add(allocator, a, b),
        .cuda => try cuda_ops.add(allocator, a, b),
    };
    try autograd.maybeRecordAdd(tape, &out, a, b);
    return out;
}
```

The important rule is that the public op validates and records semantics, while the backend implementation performs device-specific math. Avoid duplicating autograd logic separately in CPU and CUDA code unless absolutely necessary.

---

## 25. Appendix C — example saved tensor pseudocode

```zig
pub const SavedTensor = struct {
    shape: Shape,
    strides: Strides,
    offset: usize,
    storage: SavedStorage,

    pub fn fromBorrowed(t: Tensor) SavedTensor { ... }
    pub fn fromOwnedCopy(allocator: Allocator, t: Tensor) !SavedTensor { ... }
    pub fn deinit(self: *SavedTensor, allocator: Allocator) void { ... }
};

pub const SavedValue = union(enum) {
    none,
    scalar: f32,
    shape: Shape,
    tensor: SavedTensor,
    tensor_pair: struct { a: SavedTensor, b: SavedTensor },
    reshape: struct { from: Shape, to: Shape },
};
```

Backward implementations should consume `SavedValue` and never reach into freed local tensors.

---

## 26. Appendix D — CUDA bring-up checklist

```text
[ ] `zig build test` passes CPU.
[ ] README says CUDA is planned until implemented.
[ ] Tensor storage refactor complete.
[ ] Device mismatch errors exist.
[ ] CUDA dynamic loader resolves symbols.
[ ] `cuInit` smoke test passes.
[ ] context create/destroy test passes.
[ ] stream create/destroy test passes.
[ ] cuBLAS create/destroy test passes.
[ ] device memory alloc/free test passes.
[ ] HtoD/DtoH roundtrip test passes.
[ ] PTX module load test passes.
[ ] vector add kernel test passes.
[ ] Tensor.toCuda/toCpu test passes.
[ ] elementwise CUDA parity passes.
[ ] reduction CUDA parity passes.
[ ] matmul CUDA parity passes.
[ ] matmul backward CUDA parity passes.
[ ] softmax CUDA parity passes.
[ ] cross-entropy CUDA parity passes.
[ ] embedding CUDA parity passes.
[ ] AdamW CUDA update parity passes.
[ ] full model forward parity passes.
[ ] full model backward parity passes.
[ ] one-step training parity passes.
[ ] full CUDA training decreases loss.
```

---

## 27. Appendix E — documentation plan

The project wants book-level docs. Keep that, but make docs enforceable.

Recommended docs after hardening:

```text
docs/00_overview.md
  mission, locked decisions, current status, reading path

docs/01_zig_primer.md
  Zig basics needed for this project only

docs/02_tensors.md
  storage, shape, strides, offset, views, ownership

docs/03_autograd.md
  tape, nodes, saved tensors, backward, grad accumulation

docs/04_nn.md
  layers and model architecture

docs/05_transformer_math.md
  attention, MLP, layernorm, logits, cross-entropy

docs/06_tokenizer_data.md
  word tokenizer, dataset windowing, batching

docs/07_cpu_training.md
  training loop, optimizer, checkpoint, generation

docs/08_backends_cuda.md
  backend boundary, CUDA context, memory, kernels, cuBLAS

docs/09_debugging.md
  shape bugs, memory bugs, autograd bugs, CUDA bugs

docs/10_pytorch_parallels.md
  mapping from this project to PyTorch concepts
```

Each doc should end with:

```text
Common mistakes
Exercises
Solutions
Test commands
```

---

## 28. Final directive to the implementation agent

The project’s mission is not to produce a fast transformer first. The mission is to produce a readable, correct, educational transformer system that can explain PyTorch-like internals.

So optimize for this order:

```text
1. honesty
2. readability
3. correctness
4. explicit ownership
5. testability
6. parity with PyTorch/CPU
7. speed
```

The right next action is not CUDA. The right next action is:

```text
Make the CPU tensor/autograd core impossible to misunderstand.
```

Then CUDA becomes a clean backend implementation problem instead of a debugging trap.

---

## 29. References

1. Repository: <https://github.com/joelhenwang/zig-transformer-lab>
2. README: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/README.md>
3. AGENTS.md: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/AGENTS.md>
4. plan.md: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/plan.md>
5. Tensor source: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/tensor.zig>
6. Shape source: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/shape.zig>
7. Elementwise source: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/ops/elementwise.zig>
8. Matmul source: <https://raw.githubusercontent.com/joelhenwang/zig-transformer-lab/main/src/tensor/ops/matmul.zig>
9. PyTorch autograd tutorial: <https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>
10. PyTorch autograd notes: <https://docs.pytorch.org/docs/stable/notes/autograd.html>
11. NVIDIA CUDA Driver API module management: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>
12. NVIDIA CUDA Programming Guide Driver API: <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html>
13. NVIDIA cuBLAS documentation: <https://docs.nvidia.com/cuda/cublas/>
14. Zig 0.16.0 release notes: <https://ziglang.org/download/0.16.0/release-notes.html>
