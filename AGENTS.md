# Agent Brief — zig-transformer-lab

## One-sentence mission

Build a pedagogical Zig 0.16.0 library that trains a tiny 1-block 1-head word-level
transformer on CPU, then on CUDA, with extensive documentation and heavily commented
code that teaches how PyTorch-like systems work internally.

## Current engineering gate — Stage 9 (documentation finalization)

Stages 1–8 have all shipped (tags `stage-7-complete`, `stage-8-complete`).
Stage 8 closed on 2026-05-11 with 306 CPU + 83 CUDA + 15 oracle tests
passing on the RTX 4060 Ti remote, compute-sanitizer clean on the
2-block/2-head/D=64 acceptance sweep, and `docs/09_debugging.md`
shipped at 600 lines.

Stage 9 is scoped per `plan.md` §873 as a docs-finalisation pass:
no new code. Planned deliverables:

1. `docs/10_pytorch_parallels.md` — mapping every Zig abstraction
   back to its PyTorch counterpart for readers coming from that
   framework.
2. Pad shorter chapters: `docs/07_cpu_training.md` (492 → 500+
   lines) and `docs/08_backends_cuda.md` (410 → 500+ lines) to
   match the ≥500 line bar the rest of the series meets.
3. Per-subfolder READMEs under `src/` (each module directory gets
   a short orientation note).
4. Inline-comment polish across the hot paths.

> **Stage 9 playbook:** not yet drafted. A fresh session should
> read `plan.md` §873 and sketch a `docs/stage9_plan.md` before
> beginning work, matching the Stage 7 / 8 planning cadence.

Optional deferred perf follow-up (noted at Stage 8 close):
- 2/2/64 wall-clock on CUDA is 2.61× the 1/1/32 baseline (over
  the originally-scoped 2× budget). A pure-GPU gradient clip
  (sumSq reduce + mulScalar) would reclaim ~5 % by eliminating
  the DtoH/HtoD scratch path in `Trainer.train`. Tracked under
  Stage 9 perf if it comes up.

Historical context:
- Stage 6.5 (CPU hardening) passed in commit `f9c1d3b` on 2026-05.
- Stage 7 (CUDA backend) landed in commits `07bd274`..`584160b`
  over 2026-05-10..11, with 267 CPU + 73 CUDA tests passing on
  RTX 4060 Ti and a measured 30.59× speedup at the Shakespeare
  config. Documented in `docs/stage7_plan.md` + `docs/stage7_endgame_plan.md`.
- Stage 8 (debugging + N-block) landed in commits
  `5c93fe2`..`f4362e3` over 2026-05-11. Documented in
  `docs/stage8_plan.md` and `docs/stage8_handoff.md`.

## PyTorch oracle (post-6.5, CPU safety net before Stage 7)

A PyTorch reference-implementation harness ships alongside Stage 6.5
as a CPU safety net. It is **not** a Stage 7 dependency — Stage 7 can
start without it — but we chose to add it before CUDA so we have a
known-correct CPU baseline to compare against.

Components:

- `tools/oracle.py` — generates `.ztlt` binary fixtures from PyTorch
- `src/testing/oracle.zig` — Zig loader + `expectClose` comparator
- `tests/integration_oracle.zig` — one test per case
- `tests/fixtures/*` — checked-in binary fixtures (~7 KB total)
- `docs/oracle.md` — workflow reference

Build step: `zig build test-oracle` (separate from `zig build test`
so a fresh clone without regenerated fixtures still has a green
default suite).

Current cases (14): `add_2d`, `add_broadcast_2d_1d`, `mul_broadcast`,
`matmul_2d`, `softmax_3d_last_axis`, `cross_entropy_3d`, `gelu_2d`,
`layernorm_3d`, `embedding_3d`, `matmul_batch_3d`, `log_softmax_3d`,
`sum_axis_3d`, `mean_axis_3d`, `full_model_forward`.

The `full_model_forward` case is the headline integration test: it
loads 15 per-parameter `.ztlt` files into a fresh `TinyWordTransformer`
and asserts that the forward logits match PyTorch within `5e-4`
absolute. All 14 pass on Windows. See `docs/oracle.md` for how to
add new cases.

## Remote RTX workflow (Stage 7 and beyond)

The RTX 4060 Ti box at `joelwang-rtx@192.168.1.197` is the CUDA
development target. Two bash helpers at the repo root wrap the SSH
plumbing so the Windows host (where OpenCode runs) can drive work on
the remote Linux machine without manual SSH sessions.

### Scripts

- `run_remote_example.sh` — run any command on the remote, with the
  PyTorch venv active and CWD at the repo root. Callers pass a
  single quoted string as the argument.
- `sync_remote_example.sh` — rsync the working tree to the remote.
  Excludes `.git/`, build artefacts, venv, and pyc caches. Uses
  `--delete` to mirror (files deleted locally are removed from the
  remote too).

Both scripts embed the remote user@host and repo path. Update them
if the target changes.

### When to use which

**Committed-work workflow (primary, matches our git habit):**

```
git push                                             # from Windows
bash run_remote_example.sh "git pull --ff-only"      # pull on remote
bash run_remote_example.sh "zig build test -Dcuda=true"
```

**Uncommitted iteration workflow (for rapid debugging):**

```
bash sync_remote_example.sh                           # rsync .
bash run_remote_example.sh "zig build test-oracle"
```

After landing a fix via the iteration path, commit locally and push
through the primary path so remote history stays in sync with origin.

### Pre-Stage-7 smoke test

One invocation verifies every toolchain component we need:

```
bash run_remote_example.sh "git pull --ff-only 2>&1 | tail -3 && zig version && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv && nvcc --version | tail -1 && which compute-sanitizer"
```

Confirmed state on first run (`joelwang-rtx-MS-7C56`):

| Component | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| CPU | AMD Ryzen 9 5900XT (32 threads) |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Driver | 595.58.03 |
| CUDA Toolkit | 13.2 (`/usr/local/cuda-13.2/bin/nvcc`) |
| compute-sanitizer | `/usr/local/cuda-13.2/bin/compute-sanitizer` |
| Zig | 0.16.0 (exact) |
| `zig build test` | 263/263 pass (matches Windows count) |
| `zig build test-oracle` | 14/14 pass — fixtures generated on Windows match on Linux within tolerance |

### CUDA 13 note

The remote has CUDA Toolkit 13.2, not the 12.x the original plan
assumed. 13.2 is a superset API-wise for what we need (dlopen the
Driver API + cuBLAS, compile `.ptx` with `nvcc -ptx -arch=sm_89`).
Monitor for:

- Any deprecated symbol removals at dlopen time — we resolve
  specific `cu*` / `cublas*` entry points and a missing one surfaces
  as `error.CudaError` from `bindings.zig`.
- Any `.ptx` format change across CUDA major versions — our kernels
  target `sm_89` (Ada Lovelace, matches the RTX 4060 Ti) so this is
  unlikely to matter.

### Common pitfalls

- **Line endings.** `.sh` files committed with CRLF fail on Linux
  (`/bin/bash^M: bad interpreter`). If you edit `.sh` files on
  Windows and they stop working on remote, check with
  `file ./run_remote_example.sh` — it should say LF.
- **Bash `$@` inside ssh double-quotes.** The wrapper passes `$@`
  through to the remote shell via a double-quoted heredoc.
  Callers should pass a single quoted string (e.g. `bash
  run_remote_example.sh "zig build test"`); multi-word unquoted
  arguments get re-joined by bash's word-splitting.
- **SSH banner noise.** The remote has a fancy login banner
  (neofetch-style system info) that prints on every SSH connect.
  It clutters stdout but does not affect command exit codes.
  Filter with `| Select-Object -Last N` if noise is a problem.
- **SSH multiplexing not configured.** Each command pays ~1s of
  TCP setup cost. Fine for dozens of commands; if a session does
  hundreds we'd add `ControlMaster auto` to `~/.ssh/config`.
- **Long-running commands (>120s).** OpenCode's bash tool has a
  default 2-minute timeout. For `compute-sanitizer` on a full
  training step or for long training runs, pass an explicit higher
  timeout or split the command.

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
| 6.5 — CPU Hardening | **Done** | Commits `f9c1d3b`, `28e73e1`, `97b0aaa`, `3331801` (refactor + docs + oracle + oracle expansion) |
| 7-setup — Remote RTX workflow | **Done** | Commit `1e3b540` — SSH scripts, `.gitattributes`, smoke test confirmed |
| 7 — CUDA Backend | **Done** | PRs α–ξ landed (commits `07bd274`–`584160b`). 267 CPU + 73 CUDA tests pass on RTX 4060 Ti, compute-sanitizer memory-clean. Measured speedup at Shakespeare config: **30.59×** (CPU 143.7 ms/step, CUDA 4.7 ms/step). See `docs/stage7_plan.md` + `docs/stage7_endgame_plan.md`. |
| 8 — Debugging + N-block | **Done** | All 8 milestones landed (commits `5c93fe2`..`f4362e3`, 2026-05-11). 306 CPU + 83 CUDA + 15 oracle tests pass on RTX 4060 Ti. `src/debug/` utilities, multi-block `TinyWordTransformer`, multi-head attention, ZTLC v3 checkpoint, Trainer CUDA support (route A), `examples/10_train_deep.zig` acceptance example, and `docs/09_debugging.md` (600 lines) all shipped. M8-f acceptance: 2/2/64 Shakespeare run compute-sanitizer clean (0 errors, 0 leaks); wall-clock 12.28 ms/step = 2.61x the 1/1/32 baseline (over 2x budget, deferred to Stage 9 perf follow-up). See `docs/stage8_plan.md`. |
| 9 | Not started | |

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

## When stuck

- Ask the user with a crisp options-style question (max 4 options).
- Never guess at hardware, environment, or decisions.
- Check the skill's compiler-error quick lookup table in `SKILL.md`.
- See `SESSION_GUIDE.md` for full project state and continuation instructions.

## Full plan

See `plan.md` at the repository root and `docs/00_overview.md`.
