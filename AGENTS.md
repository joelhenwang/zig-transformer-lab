# Agent Brief — zig-transformer-lab

## One-sentence mission

Build a pedagogical Zig 0.16.0 library that trains a tiny 1-block 1-head word-level
transformer on CPU, then on CUDA, with extensive documentation and heavily commented
code that teaches how PyTorch-like systems work internally.

## Current engineering gate — Stage 6.5 (CPU hardening)

Stage 7 CUDA is **blocked** until Stage 6.5 ships. Rationale and full plan are
in `zig_transformer_lab_implementation_assessment_and_plan.md` (researcher
review) plus the accepted plan in session notes.

Stage 6.5 consists of seven PRs, executed in order:

| PR  | Scope                                                  | Status      |
|-----|--------------------------------------------------------|-------------|
| α   | Honesty pass + Windows-portable build                  | Done (f9c1d3b) |
| β   | Fix strided elementwise ops and `copyTo`               | Done (f9c1d3b) |
| γ   | Tensor invariants + `LabError` expansion               | Done (f9c1d3b) |
| δ   | Storage union + offset field (CPU-only backend seam)   | Done (f9c1d3b) |
| ε   | Operation-owned `SavedTensor`; remove `keepAlive`      | Done (f9c1d3b) |
| ζ   | Stable `ParamId`-based optimizer state                 | Done (f9c1d3b) |
| η   | Strict checkpoint validation                            | Done (f9c1d3b) |
| docs | Teaching chapters 02c, 02d, 03c, 07c, 07d             | Written (uncommitted) |

Exit criteria:

- README, AGENTS, docs agree that CUDA is not yet implemented. ✓
- Tensor invariants implemented and tested; zero dimensions rejected. ✓
- Storage/offset model exists; CUDA memory is never represented as `[]f32`. ✓
- Non-contiguous view behavior is correct or explicitly rejected per op. ✓
- Autograd saved data is operation-owned; no manual `keepAlive` in `src/nn/`. ✓
- Optimizer state keyed by stable `ParamId`, not by data pointer. ✓
- Checkpoint loading validates metadata strictly (magic, version, shape, dtype). ✓
- `zig build test` passes on Windows and Linux; exact output recorded. ✓ (259 tests on Windows)
- Teaching docs for each architectural PR exist under `docs/`. In progress.

Only after this gate passes may Stage 7 CUDA wrapper work begin.

## Hard rules

- Do not violate Locked Decisions D1 through D14 or policies P1 through P4.
  Ask the user before changing any decision. See `docs/00_overview.md` for the full table.
- No third-party Zig dependencies.
- Zig version pinned at 0.16.0 (exact).
- f32 only. No mixed precision.
- GPU kernels are CUDA C only. No pure-Zig GPU kernels.
- Kernels are compiled offline with nvcc to .ptx and loaded via cuModuleLoadData.
- Never copy code from LGPL sources. Architectural ideas only, attributed in file headers.

## Current progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 — Scaffold | **Done** | Commit `153095b` |
| 2 — CPU Tensor Foundation | **Done** | Commit `001c74e` — 7 files, 3082 insertions |
| 3 — Tape-based Autograd | **Done** | Commit `f8405e3` — 15 files, 5112 insertions |
| 4 — NN Layers + Optimizers | **Done** | Commit `b02801b` — 18 files, 3638 insertions |
| 5 — Tokenizer + Data Pipeline | **Done** | Commit `d286c8a` — 7 source files, 2 data files, 2 docs, 1 example |
| 6 — End-to-end CPU Training | **Done** | Commit `015da3c` — Trainer, generation, gradient clipping, bug fixes |
| 6.5 — CPU Hardening | **In progress** | PR-α underway |
| 7 — CUDA Backend | **Blocked on 6.5** | See Stage 7 section below |
| 8–9 | Not started | |

**Stage 3 committed:** `stage(3): tape-based autograd`
**Stage 4 committed:** `stage(4): nn layers and optimizers`
**Stage 5 committed:** `stage(5): word-level tokenizer and dataset`
**Stage 6 committed:** `stage(6): end-to-end cpu training`

### Stage 6 — What was implemented

### Stage 6 — New source files

1. `src/lab/train.zig` — Trainer struct (TrainConfig, train loop, gradient clipping, grad norm logging), generate() function (autoregressive top-k + temperature sampling), GenerateOpts
2. `examples/06_train_shakespeare.zig` — CPU training on Shakespeare with Trainer
3. `examples/07_generate.zig` — Load checkpoint, generate text with top-k/temperature settings
4. Wired `lab` module into `src/root.zig` with test block entry

### Stage 6 — Bug fixes

5. Fixed `backwardCrossEntropy` — missing `@round` on target index (line 666 of backward.zig). `@intFromFloat` truncates towards zero, so `2.9999` → 2 instead of 3. Added `@round` to match the forward pass.
6. Fixed untracked reshape bug in training loop — `logits_3d.reshape(...)` creates a VIEW sharing tape_node, causing CE gradient (B*T, V) to bypass reshape backward. Replaced with `ops_shape.reshapeTracked()` so gradient flows with correct shape.
7. Same untracked reshape fix applied to examples 04 and 05.
8. Added gradient clipping to Trainer — global L2 norm clipping with configurable `grad_clip_norm` (default 5.0). Prints grad norm alongside loss.
9. Changed default `beta2` from 0.95 to 0.999 (standard Adam value). beta2=0.95 caused training instability: the second moment estimate adapts too fast, making the effective learning rate oscillate when gradients change direction.
10. Fixed `model.zig` — NamedParam type for collectNamedParams (anonymous struct mismatch across scopes). Changed save/load to take `*TinyWordTransformer` (pointer self) so collectNamedParams can access mutable weight pointers.
11. Fixed dangling pointer bug in Trainer.init() — collecting params before model is copied into struct makes pointers to local variable's fields. Moved params collection and optimizer creation into train().

### Stage 6 — Documentation

12. `docs/07_cpu_training.md` — 492 lines: training loop trace, reshape bug, @round bug, gradient clipping, beta2 analysis, generation algorithm, checkpoint format, PyTorch equivalents

### Stage 6 — Acceptance criteria (all pass)

- `zig build test` — 215+ tests pass, 0 leaks
- Training is stable at lr=1e-3 with beta2=0.999 on both tiny.txt and Shakespeare

**Training dynamics discovered:**
- lr=1e-3 with beta2=0.999 is stable on tiny.txt (loss 6.1→4.2 over 500 steps)
- lr=3e-3 causes divergence after ~200 steps even with gradient clipping at norm=1.0
- Shakespeare at V=2000, D=32, T=16, B=4: loss 7.75→7.35 over 500 steps (stable but slow)
- Gradient norms are typically 1.5-3.5 on tiny.txt, 1.5-2.1 on Shakespeare

**What was completed in Stage 4 (committed):**

### New source files
1. `src/nn/module.zig` — TransformerConfig, Module convention, collectParamsSlice
2. `src/nn/linear.zig` — Linear: weight (D_out, D_in), Kaiming init, 3D→2D reshape, tape-tracked forward
3. `src/nn/embedding.zig` — Embedding: weight table, forward gathers rows, records `.embedding` OpKind
4. `src/nn/layernorm.zig` — LayerNorm: gamma/beta, composed from ~7 tape-tracked ops (mean, sub, mul, sqrt, div, add)
5. `src/nn/activations.zig` — GELU wrapper (stateless, wraps ops.unary.geluExact)
6. `src/nn/attention.zig` — CausalSelfAttention (1 head): 4 Linear sub-layers, causal mask, QKV→K^T→matmulBatch→scale→mask→softmax→matmulBatch→W_o
7. `src/nn/mlp.zig` — MLP: fc1→GELU→fc2
8. `src/nn/block.zig` — TransformerBlock: pre-norm residual (LN→Attn→+x→LN→MLP→+h)
9. `src/nn/model.zig` — TinyWordTransformer: tok_embed+pos_embed→block→ln_f→lm_head, save/load checkpoint format
10. `src/optim/optimizer.zig` — Optimizer vtable (ctx + step/zeroGrad/deinit function pointers)
11. `src/optim/sgd.zig` — SGD with momentum and coupled weight decay
12. `src/optim/adamw.zig` — AdamW with bias correction (β₁ᵗ, β₂ᵗ) and decoupled weight decay

### Modified files (Stage 3 → Stage 4)
13. `src/autograd/node.zig` — Added OpKind.sqrt (21→22 variants)
14. `src/autograd/backward.zig` — Added backwardSqrt, backwardEmbedding (scatter-add), fixed backwardMatmulBatch for 3D tensors
15. `src/autograd/tape.zig` — Added `kept_alive: ArrayList([]f32)`, `keepAlive(self, *Tensor)`, fixed `trackLeaf()` to always create fresh node (ignoring stale tape_node)
16. `src/tensor/ops/unary.zig` — Added `sqrt(alloc, t, tape)` with tape recording
17. `src/tensor/ops/shape_ops.zig` — reshapeTracked, transpose2dTracked, transposeInner2d (3D view)
18. `src/root.zig` — Wired nn.* and optim.* modules with re-exports and test blocks

### Examples
19. `examples/04_overfit_one_batch.zig` — Training loop: V=32, D=16, T=8, B=2, 50 steps with AdamW. Loss 3.83→3.09.

### Documentation (committed in Stage 4 and 5)
20. `docs/04_nn.md` — 858 lines
21. `docs/04b_from_nn_to_training.md` — 1072 lines: why each layer exists, optimizer math, complete training step trace, gradient flow, shape trace, keepAlive memory management, PyTorch equivalents, 8 common mistakes

### Key bugs fixed in Stage 4
- **trackLeaf() stale tape_node collision.** After destroying a tape, param.tape_node still held the old node ID. Next step's trackLeaf() short-circuited, returning a stale ID that collided with other nodes. Fix: trackLeaf() always creates a fresh node, ignoring existing tape_node.
- **Use-after-free on intermediate tensors.** NN forward methods create owned intermediates (transposes, reshapes, matmul outputs) that are defer-freed before backward runs. Tape's SavedData holds data slices into freed buffers. Fix: tape.keepAlive() transfers buffer ownership to tape's kept_alive list; sets tensor.owned=false so deinit is no-op.
- **backwardMatmulBatch on 3D tensors.** Was calling tp.b.transpose2d() on 3D tensors (crash). Fix: use ops_shape.transposeInner2d(tp.b) which swaps dims[1]/dims[2] of a 3D tensor.
- **Loss read-after-free.** Reading loss.data[0] after loss.deinit() in example 04. Fix: save const loss_val = loss.data[0] before deinit.

### Acceptance criteria (all pass)
- `zig build test` — 150+ tests pass, 0 leaks
- `zig build run-example -Dexample=04_overfit_one_batch` — Loss decreases from 3.83 to 3.09 over 50 steps

## Known dead code / pitfalls

- **`tests/unit_all.zig` is dead code.** It is NOT wired into `build.zig` and would
  fail to compile if used (wrong import names). The actual test aggregator is
  `src/root.zig` (its `test { _ = ...; }` block references each sub-module).
  Do NOT add tests to `unit_all.zig` — add them co-located in source files.

- **`plan.md` lists `src/core/allocator.zig`** in its repository layout, but this
  file was never created and is not imported anywhere. The plan's layout is stale
  on this point; `docs/00_overview.md` is accurate.

- **`root.zig` line 14** says "add them here and in the test block at the bottom."
  This is correct — only `src/root.zig` test blocks matter. The earlier stale
  reference to `tests/unit_all.zig` has been removed.

## Workflow

- Implement stages 1 through 9 in order. Do not interleave stages.
- Commit after each stage with `stage(N): <summary>`; paste acceptance outputs
  into the commit body.
- Docs chapter for a stage ships in the same commit as the stage's code.

## Style

- File header block: purpose, shape contract, math formulas, ownership, errors,
  TODOs, credits.
- Explicit allocators. No hidden globals. No `@panic` outside tests.
- `zig fmt` clean. Tests co-located via `test "..."` blocks.
- Every public function has a worked shape example in its doc comment.
- Pedagogical comments: explain *why*, not just *what*. ~30-40% of non-blank
  lines should be comments.

## Zig 0.16.0 patterns

Use modern 0.16 API exclusively:
- `pub fn main(init: std.process.Init) !void`
- `b.createModule(...)`, `root_module.addImport(...)`
- `var list: ArrayList(T) = .empty; list.append(gpa, v)`
- `@ptrFromInt`, `@intFromPtr`, `@intFromEnum` (not the old cast builtins)
- `callconv(.c)` not `callconv(.C)`
- Explicit re-exports, never `usingnamespace`
- `std.Io.Dir.cwd()` not `std.fs.cwd()`
- `const` for non-mutated locals (Zig 0.16.0 enforces this as an error, not a warning)

## This project's Shape API (MUST follow — tripped up 3 subagents)

The `Shape` struct in `src/tensor/shape.zig` uses rank-specific constructors:

```zig
const s1 = Shape.init1D(10);          // 1D: (10,)
const s2 = Shape.init2D(2, 3);       // 2D: (2, 3)
const s3 = Shape.init3D(2, 3, 4);    // 3D: (2, 3, 4)
const s4 = Shape.init4D(2, 3, 4, 5); // 4D: (2, 3, 4, 5)
```

**CRITICAL:** There is NO `Shape.init(&.{2, 3})` constructor. It does not exist.
If you write `Shape.init(...)` with a slice/array argument, compilation will fail.

The `rank` field is a `u2` storing `ndim - 1` (0-3 for 1D-4D). Use `shape.ndim()`
to get the dimension count (returns `rank + 1`). Never compare `rank` directly
to a dimension count without adding 1.

`Shape.equals` is a **free function**, not a method: `equals(a, b)` not `a.equals(b)`.

## This project's SavedData API (MUST follow — tripped up in Stage 3)

SavedData stores Tensor structs **by value** (snapshots), not by pointer:

```zig
// WRONG — dangling pointer to stack-local by-value parameter:
.saved = .{ .tensor_ref = @constCast(&tensor_param) }       // DANGLING!
.saved = .{ .tensor_pair = .{ .a = @constCast(&a), .b = @constCast(&b) } }  // DANGLING!

// CORRECT — snapshot by value, data slice shares original heap buffer:
.saved = .{ .tensor_ref = tensor_param }                    // snapshot
.saved = .{ .tensor_pair = .{ .a = a, .b = b } }          // snapshot
.saved = .{ .ce_info = .{ .logits = logits, .targets = targets.data } } // logits by value
```

**Why:** Ops take `Tensor` by value (Zig convention). `@constCast(&param)` creates
a pointer to a stack-local copy that's destroyed when the function returns. By
storing the whole struct, we capture the `data` slice (pointing to the original
heap buffer) along with shape/strides. The heap buffer is alive as long as the
caller's tensor outlives the tape — same contract as PyTorch's autograd.

## Zig 0.16.0 compilation gotchas

Real errors encountered during implementation. New entries added per stage:

1. **`std.fs.cwd()` does not exist in 0.16.0.** The `std.fs` module is mostly
   deprecated. Use `std.Io.Dir.cwd()` for filesystem access. In `build.zig`,
   avoid runtime filesystem operations entirely — use build-system commands
   and static file lists instead of directory iteration.

2. **`var` vs `const` is enforced as an error.** If a local variable is never
   mutated after initialization, Zig 0.16.0 emits a hard error (not a warning).
   Use `const` by default. Only use `var` when you actually mutate the variable.

3. **Unused function parameters are errors.** If a function parameter isn't
   used, either prefix with `_` or remove it. No silent warnings.

4. **`build.zig.zon` fingerprint must match.** The `fingerprint` field is
   verified by the compiler. If you create a new project, let the compiler
   tell you the correct value on first build — it prints the expected value.

5. **Unused build options must be suppressed.** If you declare `b.option(...)`
   but don't use the value, add `_ = variable;` to suppress the error.

6. **`addSystemCommand` takes `[]const []const u8`.** All arguments must be
   available at build-configuration time. `b.fmt()` returns `[]const u8` which
   works, but you can't do runtime string construction inside the arg array.

7. **`LazyPath` uses `.cwd_relative = "path"`.** When adding library/rpath
   entries, use `std.Build.LazyPath{ .cwd_relative = "/path" }`.

8. **`exe.builder.allocator` doesn't exist.** In 0.16.0, access the build
   allocator via `b.allocator` (inside `build` function scope), not through
   the compile step.

9. **Test discovery goes through module imports.** Zig's test runner only finds
   `test "..."` blocks in files transitively imported by the test root. Our
   `src/root.zig` re-exports everything and its own `test { }` block references
   each sub-module. Don't use relative file paths in test files — import
   through the module.

10. **`std.fmt.bufPrint` for string formatting.** `std.io.fixedBufferStream`
    may not exist or have a different API. `std.fmt.bufPrint(buf, fmt, args)`
    is the reliable way to format into a buffer.

11. **`std.Io.get()` does not exist in 0.16.0.** There is no global stdout
    writer. In examples, use `init.io.lockStderr(&buffer, null)` to get a
    locked stderr writer. The `Init` struct provides the `io: Io` field.

12. **`@constCast(&by_value_param)` creates dangling pointers.** When ops
    take `Tensor` by value, storing a pointer to the parameter in SavedData
    dangles after the function returns. Store the Tensor by value (snapshot)
    instead. The `data` slice in the snapshot shares the original heap buffer.
    Discovered in Stage 3 when `backwardMul` crashed with `ShapeMismatch` —
    the saved `tensor_pair` pointers pointed to freed stack frames.

13. **`lockStderr` buffer can be zero-length.** Use `var buf: [0]u8 = undefined`
    for unbuffered stderr writing in examples (matches example 01 pattern).
    A `[1024]u8` buffer works but may swallow output silently if the buffer
    isn't flushed before program exit.

14. **Gradient tensors for null-parent inputs leak without explicit cleanup.**
    When an op input has `tape_node = null` (doesn't require grad), the
    backward function still computes its gradient, but the accumulation loop
    skips it (null parent ID). Without a cleanup loop after accumulation,
    these heap-allocated gradient tensors leak. Fix: iterate parent_grads
    again and free any where `node.parents[pi]` is null.

15. **`fetchSwapRemove` → `fetchRemove` in Zig 0.16.0 HashMap API.** The
    method is `fetchRemove`, not `fetchSwapRemove`. The latter does not exist.

16. **`@floatFromInt(@as(u64, @bitCast(x)))` is WRONG for f32→f64 conversion.**
    Use `@floatCast(f64, x)` to widen f32 to f64. `@bitCast` reinterprets
    the bit pattern (producing garbage), while `@floatCast` converts the value.

17. **`trackLeaf()` must ignore stale `tape_node` values.** Each training step
    creates a fresh `Tape.init()`, so old `tape_node` IDs from destroyed tapes
    are invalid. Short-circuiting on existing `tape_node` caused ID collisions
    (e.g., 3D intermediate and 1D parameter both claiming tape_node=13).
    Fix: trackLeaf() always creates a fresh node.

18. **`keepAlive` required for every intermediate in nn forward().** Any tensor
    created by an op (reshapeTracked, transpose2dTracked, matmul, add, etc.)
    that is `defer`-freed in a forward method must have `tape.keepAlive(&tensor)`
    called before the defer. Without this, the tape's SavedData holds slices
    into freed buffers → use-after-free during backward.

19. **`transpose2d()` crashes on 3D tensors.** Use `transposeInner2d()` for 3D
    tensors, which swaps dims[1]/dims[2] instead of dims[0]/dims[1]. Discovered
    when `backwardMatmulBatch` called `tp.b.transpose2d()` on a (B,T,D) tensor.

20. **Read tensor data before deinit, not after.** `loss.data[0]` after
    `loss.deinit()` is use-after-free. Save `const val = loss.data[0]` first.
    Same pattern applies any time you need a scalar value from a tensor you're
    about to free.

21. **`std.Io.Dir.cwd().openFile(io, path, .{})` requires `io` parameter.**
    File I/O operations in 0.16.0 take a `std.Io` as the second argument.
    `createFile(io, path, .{})` same pattern. In example code, use `init.io`;
    in test code, create `var threaded = std.Io.Threaded.init(allocator, .{});`
    and use `threaded.io()`.

22. **`file.writer(io, &buf)` / `file.reader(io, &buf)` require buffer.**
    The writer/reader are now buffered by default. You must provide a buffer.
    Use `var buf: [4096]u8 = undefined; var writer = file.writer(io, &buf);`
    Call `writer.flush()` before the writer goes out of scope.

23. **`writer.interface.print(...)` — print is on the interface.** File.Writer
    has an `interface` field of type `Io.Writer`. Print, writeAll, writeInt,
    etc. are on `writer.interface`, not on `writer` directly.

24. **`reader.interface.takeDelimiter('\n')` replaces `readUntilDelimiterOrEof`.**
    The old method doesn't exist on `Io.Reader`. Use `takeDelimiter` which
    returns `?[]u8` (null at EOF, slice valid until next read).

25. **`ArrayList(T).init(allocator)` does NOT exist in 0.16.0.** Use
    `var list: ArrayList(T) = .empty;` and then `list.append(allocator, item)`.
    `list.deinit(allocator)` takes the allocator parameter.

26. **Anonymous struct types in ArrayList don't match across scope boundaries.**
    `ArrayList(struct { word: []const u8, count: u32 })` creates a different
    type in each scope. Use `const MyStruct = struct { ... }; var list:
    ArrayList(MyStruct) = .empty;` to get a consistent type.

27. **`file.stat(io)` replaces `file.metadata()`.** The stat method now
    requires the `io` parameter. The returned `Stat` struct has `.size` as
    a direct field (not a method).

28. **`cwd.readFileAlloc(io, path, allocator, .limited(N))` — simplest file read.**
    Reads the entire file into a heap-allocated buffer with a size limit.
    Much simpler than openFile + reader + readAll.

29. **`@round` is required before `@intFromFloat` on target indices.**
    `@intFromFloat` truncates towards zero: `@intFromFloat(2.9999)` = 2, not 3.
    In `backwardCrossEntropy`, this caused the one-hot gradient to point at
    the wrong class. Fix: `@intFromFloat(@round(value))` to match the
    forward pass which already used `@round`.

30. **Untracked `reshape()` creates silent gradient shape mismatch.**
    `logits_3d.reshape(Shape.init2D(B*T, V))` returns a VIEW that shares
    `tape_node` with the 3D original. The CE backward stores a gradient of
    shape `(B*T, V)` under this node ID, but the matmul backward expects
    `(B, T, V)`. This "works" accidentally because both shapes are
    row-major contiguous, but it's a latent bug. Fix: always use
    `ops_shape.reshapeTracked()` in the training loop.

31. **`collectNamedParams` needs pointer self (`*TinyWordTransformer`).**
    With `self: TinyWordTransformer` (by value), `&self.weight` creates a
    pointer to a stack-local copy that dangles after the function returns.
    Fix: `self: *TinyWordTransformer` so `&self.weight` points to the
    actual model field. Also requires `save()` and `load()` to take `*self`.

32. **Don't collect params in `init()` if model is copied into struct.**
    `model.parameters(&params)` on a local `model` variable collects
    pointers to the local's fields. After `return Trainer{ .model = model }`,
    those pointers dangle. Fix: collect params in `train()` after the
    model is in its final memory location.

33. **Don't store AdamW in struct — HashMap internal pointers corrupt on copy.**
    `var adam = AdamW.init(alloc, ...)` creates a local with a HashMap
    containing self-referential pointers. Copying into a struct field
    via `return Trainer{ .opt = adam }` corrupts these pointers.
    Fix: create AdamW locally in `train()` so its HashMap is valid for
    the entire training run.

34. **Gradient clipping is essential for training stability.**
    Without clipping, lr=3e-3 diverges even on tiny.txt. With clipping
    at `grad_clip_norm=5.0` (our default), lr=1e-3 is stable on both
    tiny.txt and Shakespeare. Gradient norms are typically 1.5-3.5.

35. **`beta2=0.95` causes training instability.** The second moment
    estimate adapts 50x faster than standard (beta2=0.999), making
    the effective learning rate oscillate when gradient directions
    change. Changed default from 0.95 to 0.999.

36. **`denom_floor=1e-2` needed for all softmax-related grad checks.**
    Near-zero gradients (< 0.01) cause inflated relative errors
    (often > 1.0) from finite-difference noise, even when the
    backward implementation is correct. Using `1e-8` as the denom
    floor makes relative error meaningless for these cases. Fix:
    use `denom_floor=1e-2` and pair with `max_abs_diff` check.

37. **Combined assertion: `max_rel_err < 0.05 OR max_abs_diff < 1e-2`.**
    High relative error with tiny absolute error is finite-difference
    noise on near-zero gradients, not a backward bug. The correct
    pattern for grad check assertions is: pass if EITHER relative
    error is small (real gradient, precise check) OR absolute error
    is tiny (near-zero gradient, noise-dominated). This pattern is
    used in the full model, TransformerBlock, CausalSelfAttention,
    and pipeline grad check tests.

38. **`sumAll(softmax(x))` is constant — gradient is exactly 0.**
    Each row of softmax sums to 1, so `sumAll(softmax(x)) = B*T`
    regardless of x. The gradient through this path is exactly 0,
    making it a degenerate case for grad checks. The "softmaxed"
    pipeline stage must be skipped in assertions. A non-trivial
    test requires `loss = sumAll(softmax(x) * target)` instead.

39. **Linear bias gradient for 3D input is correct.**
    `dL/db[j] = sum over (b,t) of dL/dy[b,t,j]` because bias
    broadcasts across batch and time dimensions. Verified with
    finite-difference grad check: max_rel_err=0.000002.

## CUDA sacred spots

- Row-major to column-major wrapping in `src/backend/cuda/gemm.zig`. Dedicated tests.
- Bounds checks in every kernel.
- Offline .ptx only (no NVRTC).
- Bindings module opens libraries with dlopen at runtime.
- Link `libc` + `dl`, never `libcuda` directly.
- In `build.zig`, use a static `kernel_names` list for nvcc compilation —
  don't iterate the filesystem at build time (avoids `std.fs.cwd()` issues).

## Zig 0.16.0 skill reference

The `skills/modern-zig-0-16-tutor/` directory contains a comprehensive Zig 0.16.0
reference library. **Read `skills/modern-zig-0-16-tutor/SKILL.md` every session** — it
has the canonical stale-pattern table (28 rows), compiler-error quick lookup (22 entries),
and 13 project-specific gotchas discovered during Stages 1-2.

When you hit a Zig compilation error or API question, load reference files by topic
(max 3 per turn to manage token budget):

| Task | Load |
|------|------|
| Build system errors | `references/08-build-system-0-16.md` |
| I/O (stdout, writer, formatting) | `references/07-io-0-16.md` |
| Allocators, arena, leaks | `references/05-memory-allocators.md` |
| ArrayList / HashMap | `references/06-containers-0-16.md` |
| C/CUDA interop | `references/09-c-interop-0-16.md` + `references/17-zig-cuda-interop-notes.md` |
| Testing, debugging | `references/10-testing-debugging.md` |
| Code review | `references/15-code-review-checklist.md` |
| ML/tensor design | `references/16-zig-for-ml-runtime-projects.md` |
| Tensor2D skeleton | `templates/tensor2d-skeleton-0-16.zig` |
| Numerical code testing | `recipes/numerical-code.md` |
| Stale pattern scanning | `scripts/grep_stale_patterns.py <dir>` |

Also available: 19 reference files, 12 recipes, 7 templates, 4 validation scripts.
See `skills/modern-zig-0-16-tutor/README.md` for the full directory map.

## Stage 7: Next steps

Stage 7 implements the CUDA backend. Per `plan.md`
and `docs/00_overview.md`:

### What to implement
1. `src/backend/backend.zig` — vtable
2. `src/backend/cpu_naive/dispatch.zig` — wraps Stage 2 ops
3. `src/backend/cuda/{bindings,context,mem,module,gemm,dispatch}.zig`
4. `src/backend/cuda/kernels/*.cu` — offline .ptx kernels
5. `docs/08_backends_cuda.md`

### Key design decisions already locked
- **D9:** cuBLAS for GEMM, custom kernels for elementwise/softmax/etc
- **D10:** .ptx loaded via cuModuleLoadData (no NVRTC)
- **D11:** dlopen for CUDA libraries at runtime

### Recommended implementation order (sub-stages)

Implement Stage 7 in this order to maintain a compilable/testable codebase
at every step:

#### 7.A — Bindings (`src/backend/cuda/bindings.zig`)
- Dynamically load `libcuda.so.1`, `libcudart.so`, `libcublas.so` via dlopen/dlsym
- Resolve only the symbols we need (see plan.md §7.A for the full list)
- All wrappers convert non-success codes into `error.CudaError`
- Debug builds log numeric code + symbol name; store `cuGetErrorString` /
  `cublasGetStatusString` message for `debug.lastCudaError()` retrieval
- **Test:** dlopen succeeds on a CUDA-capable machine; symbol resolution
  returns non-null function pointers

#### 7.B — Context (`src/backend/cuda/context.zig`)
- `CudaContext` struct: device, context, stream, cuBLAS handle, ptx_modules
  HashMap, allocator
- `init(alloc, device_id)`: cuInit → cuDeviceGet → cuCtxCreate_v2 →
  cuStreamCreate → cublasCreate_v2 → cublasSetStream_v2 → load .ptx
- `deinit()`: destroy in reverse order (cublas → stream → context)
- **Test:** init/deinit cycle on GPU 0 without errors

#### 7.C — Memory (`src/backend/cuda/mem.zig`)
- `DeviceBuffer` RAII over `cuMemAlloc_v2` / `cuMemFree_v2`
- `DeviceBuffer.from_host(ctx, slice_f32) !Self` — cuMemcpyHtoD_v2
- `self.to_host(alloc) ![]f32` — cuMemcpyDtoH_v2
- `self.deinit()` — cuMemFree_v2
- `Tensor.to_cuda(ctx)` and `Tensor.to_cpu(alloc)` preserve shape/strides
- **Test:** round-trip host→device→host preserves data exactly

#### 7.D — Backend vtable + CPU naive dispatch
- `src/backend/backend.zig`: Backend struct + VTable (matmul, add, softmax,
  layernorm, gelu, embedding_fwd/bwd, causal_mask, ce_loss, adamw_step,
  to_device, to_host)
- `src/backend/cpu_naive/dispatch.zig`: wraps existing Stage 2 CPU ops into
  vtable function signatures
- **Test:** CPU backend produces same results as direct op calls

#### 7.E — Row-major GEMM wrapper (`src/backend/cuda/gemm.zig`)
- **This is the single most error-prone spot in the codebase.**
- Presents row-major `C = alpha * A @ B + beta * C` API
- Internally calls cuBLAS (column-major) with swapped operands
- Batched matmul uses `cublasSgemmStridedBatched` for `(B,M,K)@(B,K,N)`
- **Test:** multiply two known `(2,3)@(3,4)` matrices; compare vs CPU
  within `1e-5`

#### 7.F — PTX module loading (`src/backend/cuda/module.zig`)
- Load .ptx files from `zig-out/ptx/` via `cuModuleLoadData`
- Cache by file stem in `CudaContext.ptx_modules`
- `getFunction(ctx, module_name, kernel_name) !CUfunction`
- **build.zig:** `nvcc -arch=sm_89 -ptx` step for each `.cu` file; output
  to `zig-out/ptx/`. Use static `kernel_names` list (no filesystem iteration)

#### 7.G — CUDA kernels (`src/backend/cuda/kernels/*.cu`)
Minimum set (each with forward + backward where applicable):
- `elementwise.cu` — add, sub, mul, div, scalar ops, in-place residual add
- `softmax.cu` — row-wise with max-subtract trick; block per row, shared
  memory for max + sum
- `layernorm.cu` — online mean/variance per row (Welford), gamma/beta affine
- `gelu.cu` — exact forward and backward
- `embedding.cu` — forward gather; backward `atomicAdd` scatter
- `causal_mask.cu` — adds -infinity above diagonal into scores
- `ce_loss.cu` — fused `log_softmax + NLL + grad` w.r.t. logits
- `adamw.cu` — per-parameter step with bias correction

Every kernel: starts with `if (idx >= n) return;` bounds check, never
out-of-bounds reads/writes, has matching unit test with tiny fixed inputs
and CPU reference.

#### 7.H — CUDA dispatch (`src/backend/cuda/dispatch.zig`)
- Implements `Backend.VTable` for CUDA path
- Each dispatch function: packs params → `cuLaunchKernel` →
  `cuStreamSynchronize` (for now; async later if needed)
- Autograd calls `backend.vtable.matmul(...)` instead of `ops.matmul(...)`
  directly; CPU and CUDA provide identical semantics

#### 7.I — Cross-validation (`examples/08_cuda_vs_cpu.zig`)
- Full-model forward on CPU and CUDA on identical random batch: max abs
  diff < `5e-5`
- One training step on both: gradient max abs diff < `1e-4`, parameter diff
  after `optim.step` < `2e-4`

### Stage 7 acceptance criteria
- Every kernel has a unit test
- `08_cuda_vs_cpu.zig` passes tolerances above
- `06_train_shakespeare.zig -Dcuda=true` is at least 30x faster than CPU
  and produces a similar loss trajectory (final loss within 10% of CPU run
  at matched steps)
- `docs/08_backends_cuda.md` committed alongside code

### Stage 7 docs outline (`docs/08_backends_cuda.md`)
- Why matmul dominates transformer compute (FLOP-count derivation)
- Why cuBLAS
- What cuBLAS hides (algorithm selection, tensor cores; not used here since
  f32, but mentioned)
- How PyTorch dispatches high-level ops to cuBLAS/cuDNN (ATen dispatcher)
- Row-major/column-major derivation with diagrams
- PTX loading lifecycle and why we precompile offline
- Kernel-by-kernel walkthrough

## When stuck

- Ask the user with a crisp options-style question (max 4 options).
- Never guess at hardware, environment, or decisions.
- Check the skill's compiler-error quick lookup table in `SKILL.md`.
- See `SESSION_GUIDE.md` for full project state and continuation instructions.

## Full plan

See `plan.md` at the repository root and `docs/00_overview.md`.
