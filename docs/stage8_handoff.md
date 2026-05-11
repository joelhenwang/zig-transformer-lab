# Stage 8 Handoff — Continuation Plan for a Fresh Session

> **Purpose.** Everything a fresh OpenCode session needs to resume Stage 8
> (debugging discipline + N-block refactor) at Milestone 7 without
> re-deriving context. Read end-to-end before touching code.
>
> **Status on 2026-05-11:** Milestones 1-6 landed on `main`, verified on
> the RTX 4060 Ti remote. Milestones 7 (`docs/09_debugging.md`) and 8
> (acceptance sweep on 2-block / 2-head / D=64) remain.

---

## 0 — Recommended first-actions for the new session

Do these in order before writing any code.

1. **Confirm session mode.** You are in interactive plan-first mode by
   default. When the user says "start" or "go", switch to build mode.
   Default bash timeout is 120 s; use `timeout` parameter for long
   commands (remote tests, sanitizer runs).
2. **Read these files in full**:
   - `AGENTS.md` — project contract, hard rules, Zig 0.16 gotchas.
   - `docs/stage8_plan.md` — the full milestone plan (this doc is the
     execution log; `stage8_plan.md` is the spec).
   - `docs/stage8_handoff.md` — this file.
   - `docs/stage7_endgame_plan.md` — the Stage 7 final plan, which is
     the style reference for how Stage 8 is being executed.
3. **Confirm the local toolchain is green:**
   ```powershell
   zig build test                   # expect 306 passed (Windows)
   zig build test -Dcuda=true       # expect exit 0 (CUDA tests skip)
   zig build test-oracle            # expect All 15 tests passed
   ```
4. **Confirm the remote is green** (Linux RTX 4060 Ti box):
   ```bash
   bash run_remote_example.sh 'cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && git pull --ff-only && zig build test -Dcuda=true 2>&1 > /dev/null; for b in $(ls -t .zig-cache/o/*/test | head -2); do echo "=== $b ==="; $b 2>&1 | tail -2; done'
   ```
   Expected: 306 CPU + 79 CUDA tests pass.
5. **Check the timer conventions.** The remote wrapper embeds a neofetch
   banner; only `| Select-Object -Last N` output is meaningful. The
   bash tool's 120-s default is fine for most commands; use 300000-
   600000 ms for `zig build test`, 900000 ms for remote compilation +
   test runs.

---

## 1 — Locked decisions (user-confirmed, do NOT revisit)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Stage order | Stage 8 first, then Stage 9 | Stage 9 docs richer after multi-head code ships |
| 2 | Multi-head parity | Oracle fixture written first | Enforces parity from PR 1; prevents silent drift |
| 3 | `src/debug/` API | Opt-in function calls (not macros) | Matches plan.md §838; macros can layer later |
| 4 | Checkpoint format | ZTLC v3 with v2 backward compat | Preserves existing (hypothetical) v2 files; clean upgrade path |
| 5 | Autonomous-run cadence | Per-commit for M1-M3, M6-M8; autonomous for M4-M5 | Multi-head refactor benefits most from unbroken context |
| 6 | Debug prints in tests | Removed in Phase 1 housekeeping | Noise-free CI output |
| 7 | WIP commits in history | Left intact (not rewritten) | `stage-7-complete` tag already on origin |

**Style precedent:** every milestone mirrors the `stage7_endgame_plan.md`
cadence — one focused commit per milestone (except M1 which was 4
sub-commits by design), commit body explaining rationale + gotchas +
verification, push + remote-verify before moving on.

---

## 2 — What has shipped (M1-M6)

All commits verified on RTX 4060 Ti. Head of `main` at time of this
writing: `8f57498`.

### Milestone 1 — `src/debug/` utilities (4 commits, +15 CPU + 4 CUDA tests)

- `5c93fe2 stage(8m1-shape): assertShape + assertRank + assertDim`
- `149b2bd stage(8m1-finite): assertFinite + hasNaN + hasInf (CPU + CUDA)`
- `280472a stage(8m1-compare): device-aware compare with CompareReport`
- `1fb972e stage(8m1-dump): tensor dump/load via .ztlt format`

Files added: `src/debug/{shape,finite,compare,dump}.zig` + wiring in
`src/root.zig` (new `debug` namespace). Every helper is device-aware
(CUDA path DtoH-copies when needed). Failure-path messages print to
stderr via `std.debug.print` (same pattern `backward.zig` uses).

### Milestone 2 — `TransformerConfig` extension (1 commit, +2 CPU tests)

- `7e964bf stage(8m2): TransformerConfig n_layer/n_head/dropout`

Added `n_layer: u8 = 1`, `n_head: u8 = 1`, `dropout: f32 = 0.0`.
Defaults preserve every Stage 2-7 test bit-for-bit. `dropout` field is
reserved — not implemented in Stage 8. Flipped `docs/00_overview.md`
decision D5.

### Milestone 3 — Multi-block model (1 commit, +6 CPU tests)

- `ab794c0 stage(8m3): TinyWordTransformer multi-block`

Replaced `block: TransformerBlock` with `blocks: []TransformerBlock`.
`init` allocates `cfg.n_layer` blocks; `forward` loops; `parameters`
iterates; `deinit` frees blocks + slice. `collectNamedParams` now
emits `blocks[<i>].*` names via `std.fmt.allocPrint`, with a
`freeBlockNames` helper for cleanup. Load path handles v2 by
rewriting `block.*` -> `blocks[0].*` when `self.cfg.n_layer == 1`.

Call sites updated: `src/autograd/gradcheck.zig` (`.block.attn` ->
`.blocks[0].attn`), `tests/integration_cuda.zig` (struct literals).

### Milestone 4 — Multi-head attention (1 commit, +4 CPU tests)

- `9b814d9 stage(8m4): multi-head CausalSelfAttention + transposeAxes12_4d op`

Added new rank-4 axis-1/2 transpose primitive:

- `src/tensor/ops/shape_ops.zig`: `transposeAxes12_4d` (view) and
  `transposeAxes12_4dTracked` (tracked, materialized).
- `src/autograd/node.zig`: new `OpKind.transpose_axes12_4d`
  (exhaustiveness test updated to 24 variants).
- `src/autograd/backward.zig`: `backwardTransposeAxes12_4d` (same
  permutation applied twice; self-inverse).

`CausalSelfAttention` refactored: `init` signature is now
`(alloc, d_model, n_head, max_seq_len, use_bias, rng)`. Rejects
`n_head == 0` or `d_model % n_head != 0` with `InvalidArgument`.
Forward uses `splitHeads` / `mergeHeads` helpers that compose the
3-op reshape+permute+reshape chain. Scale is `1/sqrt(d_head)` not
`1/sqrt(d_model)`.

`src/nn/block.zig` passes `cfg.n_head` through.

### Milestone 5 — Oracle fixture + parity test (2 commits, +1 oracle test)

- `8a7b971 stage(8m5): multihead_attention_3d oracle fixture generator + test`
- `7aef7e6 stage(8m5): multihead_attention_3d fixture files (20 .ztlt + manifest)`

New oracle case in `tools/oracle.py`: `case_multihead_attention_3d`
(B=2, T=3, D=8, H=2, bias=true). PyTorch reference composes the same
reshape -> permute -> matmul chain as Zig (no `scaled_dot_product_attention`
to avoid flash-attention numerical differences). Writes 9 inputs +
output + 9 gradients = 20 `.ztlt` files.

Parallel test in `tests/integration_oracle.zig`: loads all 9 inputs,
overwrites attention weights with oracle bytes, runs forward + backward
via `sumAll`, compares within `rel=1e-4 abs=1e-4` fwd and `rel=1e-3
abs=5e-4` bwd (playbook bands).

Also fixed the existing `full_model_forward` oracle test to rewrite
`blocks[0].*` -> `block.*` when constructing fixture paths (fixtures
use old names).

### Milestone 6 — ZTLC v3 checkpoint format (2 commits, +3 CPU tests)

- `2382bd8 stage(8m6): ZTLC v3 checkpoint format (backward-compatible with v2)`
- `8f57498 fix(8m6): rewrite v2 compat test as synthetic in-memory v2 file`

v3 header adds 12 bytes after the `bias+pad`:
`n_layer: u32`, `n_head: u32`, `dropout: f32`. `save()` writes v3.
`load()` accepts v2 and v3, gated on `version == 2 or 3`. For v2,
defaults `n_layer=1, n_head=1, dropout=0.0` and activates the
`block.*` -> `blocks[0].*` name rewrite (gated by `!is_v3 and
self.cfg.n_layer == 1`).

New tests:
- `Checkpoint v3 save/load round-trip preserves n_layer/n_head/dropout`
- `Checkpoint v3 load rejects n_layer mismatch`
- `Checkpoint v2 backward compat — single-block load via name rewrite`
  (builds a synthetic v2 file by post-processing a v3 save).

### Cumulative test count after M1-M6

| Metric | Before Stage 8 | After M6 |
|---|---|---|
| CPU tests | 267 | 306 |
| CUDA tests | 73 | 79 |
| Oracle tests | 14 | 15 |

---

## 3 — What remains (M7, M8)

### Milestone 7 — `docs/09_debugging.md`

**Goal.** ~500-line teaching chapter that documents the `src/debug/`
utilities shipped in M1, the shape-assert / NaN-detection / gradcheck
workflow, and the common CUDA-kernel bug catalog from
`plan.md` §854.

**Structure (from `docs/stage8_plan.md` §8 "Milestone 7" section):**

1. **Shape-assert driven development.** Worked example: a wrong
   `transpose2d` produces a `ShapeMismatch` three calls downstream;
   walk the reader through how `debug.assertShape` pins the error on
   its first occurrence. Use a realistic Q/K/V projection mistake.
2. **NaN/Inf detection workflow.** When it happens, where to look
   first: softmax overflow (missing max-subtract), `log(0)` from a
   probability collapse, large `lr` causing weight explosion.
   Demonstrate `debug.assertFinite` as a trip-wire in a training
   loop.
3. **Gradient checking a new op.** Using `src/autograd/gradcheck.zig`
   + finite-difference reference. Walk through what `gradCheckTwoSided`
   does internally. Include the `denom_floor=1e-2` rule for
   near-zero grads.
4. **CPU vs GPU output comparison.** `debug.compare` + `debug.dump`.
   Demo: dump a CPU tensor, dump the CUDA equivalent, load both in
   Python for side-by-side inspection. Reference the
   `tools/oracle.py:read_tensor_ztlt` routine if / when added (not
   in the repo yet — surface as an exercise for the reader).
5. **Overfit-one-batch smoke test.** Revisit `examples/04`, explain
   why loss going to zero means the pipeline is correct end-to-end.
6. **compute-sanitizer walkthrough.** Include a deliberately broken
   kernel in a sidebar file `examples/sidebar_broken_kernel.cu` (NOT
   wired into `build.zig`'s `kernel_names`). Show the exact expected
   `compute-sanitizer --tool=memcheck` output for an illegal access,
   a missed sync, a host/device pointer mixup.
7. **Nsight Compute beginner pass on `softmax_last`.** Launch
   command: `ncu --target-processes all --set full ./.zig-cache/o/.../test`.
   Inspect warp occupancy (should be 8 warps/block × 256 threads);
   memory throughput (pointer-chased through shared memory).
8. **Common CUDA errors catalog** (six classes, per `plan.md` §854):
   1. Illegal memory access — dropped bounds check
   2. Wrong grid/block — off-by-one on `T`
   3. Non-coalesced access — wrong stride in `layernorm_rowwise`
   4. Missed `cudaStreamSynchronize` — reading host data before copy completes
   5. Host/device pointer mixup
   6. Row-major vs column-major in cuBLAS

**Acceptance:**

- `>= 500` lines in the chapter (`zig build docs` line-count check).
- Every referenced tool introduced with a one-line context + runnable
  example.
- All file references live (paths exist in the repo).

**Commit subject.** `stage(8m7): docs/09_debugging.md`

**Estimated effort.** 3 hours of writing + cross-referencing.

### Milestone 8 — Acceptance sweep

**Goal.** Run the 2-block, 2-head, D=64 Shakespeare-config model
end-to-end under `compute-sanitizer`; confirm wall-clock is <2× the
1/1/32 baseline.

**Steps:**

1. **Build option plumbing.** `build.zig` currently accepts
   `-Dcuda=true`, `-Dcuda_arch`, `-Dcuda_home`, `-Dexample=...`,
   `-Dseed=...`. Add `-Dn_layer=1`, `-Dn_head=1`, `-Dd_model=32`
   as build options that propagate to `examples/06_train_shakespeare`.
   OR: a simpler approach — add a second example
   `examples/10_train_deep.zig` hardcoded to 2/2/64 config. Pick
   whichever is faster; I recommend option B for a one-shot
   acceptance test.
2. **Run on remote:**
   ```bash
   bash run_remote_example.sh "cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && zig build run-example -Dexample=10_train_deep -Dcuda=true 2>&1 | tail -20"
   ```
3. **compute-sanitizer run** (scope: one training step, not 100, to
   stay inside the 120-s bash timeout unless we extend to 600000):
   ```bash
   bash run_remote_example.sh "cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && /usr/local/cuda-13.2/bin/compute-sanitizer --tool=memcheck --leak-check=full ./.zig-cache/o/<bin>/10_train_deep 2>&1 | tail -10"
   ```
4. **Wall-clock comparison:** capture steady-state ms/step for 2/2/64
   vs 1/1/32. Acceptance: `t_2/2/64 < 2 * t_1/1/32`. If we miss
   this, document the actual ratio and mark as Stage 9 perf
   follow-up (matches the Stage 7 M6 pattern).
5. **Final loss sanity check:** same order of magnitude as the
   1/1/32 baseline's final loss at matched step count.

**Commit subject.** `stage(8m8): 2-block 2-head D=64 acceptance sweep`

**Estimated effort.** 1-2 hours (code is tiny; runtime dominated by
the training + sanitizer invocations).

### Close-out commit

After M7 + M8 land:

1. Update `docs/stage8_plan.md` exit criteria checkboxes.
2. Update `AGENTS.md` progress table Stage 8 row to **Done**.
3. Update `SESSION_GUIDE.md` §3 Stage 8 row to COMPLETE with test
   counts + commit range.
4. Optional: `git tag stage-8-complete`.
5. Optional: sketch a `docs/stage9_plan.md` or defer to a fresh
   planning session.

**Commit subject.** `stage(8): debugging + N-block refactor complete`

---

## 4 — Known landmines and how to avoid them

These are the specific gotchas encountered during M1-M6 that a fresh
session is likely to trip over.

### 4.1 `block.*` vs `blocks[0].*` naming

- **Where it matters:** `collectNamedParams`, checkpoint save/load,
  `tests/integration_oracle.zig` `full_model_forward` fixture paths.
- **Rule:** Post-M3 code emits `blocks[<i>].*`. v2 checkpoints and
  the `full_model_forward` oracle fixture use `block.*`. Rewrite at
  the load/lookup boundary.
- **Symptom when broken:** `error.IoError` from checkpoint load,
  `error.IoError` from fixture path open.

### 4.2 `freeBlockNames` is required after every `collectNamedParams`

- **Where it matters:** Any caller of `collectNamedParams`. The
  `blocks[<i>]` prefix strings are `std.fmt.allocPrint`-allocated
  and leak without cleanup.
- **Rule:**
  ```zig
  var params: std.ArrayList(NamedParam) = .empty;
  defer params.deinit(alloc);
  try model.collectNamedParams(&params);
  defer model.freeBlockNames(&params);  // <-- required
  ```
- **Symptom when broken:** DebugAllocator leak report on test exit.

### 4.3 `i0..i127` are primitive types in Zig 0.16

- **Where it matters:** `shape_ops.zig` `transposeAxes12_4dTracked`
  used `i0, i1, i2, i3` as loop variables and Zig's primitive-type
  lookup table treated `i0` as the zero-bit integer type.
- **Rule:** Use semantic names (`ib, it, ih, id`) or suffix with
  letters (`ii0`, `ii1`). Do not use bare `i<digit>` as a
  variable name.
- **Symptom when broken:** `error: name shadows primitive 'i0'`.

### 4.4 `ShapeMismatch` vs `InvalidArgument`

- `ShapeMismatch`: shapes don't match a contract (e.g. `a.shape !=
  b.shape` in an op that requires equality; checkpoint rank
  mismatch).
- `InvalidArgument`: structural invariant violated (e.g. `n_head ==
  0`, `d_model % n_head != 0`, axis out of range).
- Loading a v2 file with wrong `n_layer` returns `ShapeMismatch` at
  the cfg-mismatch check.

### 4.5 `std.ArrayList(u8).writer(alloc).writeInt` does not exist

- **Where it matters:** The synthetic v2 checkpoint builder in
  `Checkpoint v2 backward compat` test.
- **Rule:** Write primitives via `std.mem.writeInt` into a local
  `[4]u8` buffer, then `appendSlice` the buffer into the ArrayList.
  Or use `try std.io.writer(alloc, ...)` patterns — but the simpler
  path is the in-place byte write.

### 4.6 cuBLAS / matmulBatch rejects non-contiguous transpose views

- **Where it matters:** `backward.zig` `backwardMatmulBatch`. Any
  new attention variant that feeds a transpose view into
  `matmulBatch` on CUDA will fail with `error.InvalidLayout`.
- **Rule:** Materialize the transpose view via
  `cuda_dispatch.broadcastTo(view, view.shape)` before calling
  `matmulBatch`. Existing precedent in `backwardMatmul` and
  `backwardMatmulBatch`.

### 4.7 `tape.deinit` nulls leaf.grad pointers

- **Where it matters:** Training loops that rebuild the tape every
  step.
- **Rule:** After `tape.deinit`, `param.grad == null` (the tape
  owned those tensors). The next forward + backward + tape will
  re-populate them.
- **Landed fix:** `584160b fix(autograd): tape.deinit nulls leaf.grad pointers`
  (last commit of Stage 7).

### 4.8 Windows CUDA test runner "failed command" is cosmetic

- When `zig build test -Dcuda=true` on Windows shows a failed
  command + "warning: unable to open library directory" — exit
  code is still 0 for the build. The cosmetic message is the
  Zig runner's report of a zero-test-output scenario for the CUDA
  tests (all SkipZigTest on Windows). Check `$LASTEXITCODE` not
  the output text.

---

## 5 — Key files and their current state

### `src/debug/` (new in M1)

```
src/debug/
  shape.zig     assertShape, assertRank, assertDim
  finite.zig    assertFinite, hasNaN, hasInf (CPU + CUDA)
  compare.zig   compare(alloc, a, b, opts) -> CompareReport
  dump.zig      dump(alloc, io, path, t) + load(alloc, io, path)
```

Re-exported as `lab.debug.{shape, finite, compare, dump}` via
`src/root.zig`.

### `src/nn/model.zig` (heavily rewritten in M3, M6)

- `blocks: []TransformerBlock` replaces `block: TransformerBlock`.
- `collectNamedParams` emits `blocks[<i>].*`, leaks without
  `freeBlockNames`.
- `save()` writes ZTLC v3.
- `load()` accepts v2 and v3.
- `moveToCuda` / `moveToCpu` iterate over all blocks via `parameters()`.

### `src/nn/attention.zig` (heavily rewritten in M4)

- `n_head` field on the struct.
- `init(alloc, d_model, n_head, max_seq_len, use_bias, rng)`.
- `forward` uses `splitHeads` / `mergeHeads` helpers.
- Scale is `1/sqrt(d_head)` not `1/sqrt(d_model)`.

### `src/tensor/ops/shape_ops.zig` (new op in M4)

- `transposeAxes12_4d(t)` view + `transposeAxes12_4dTracked`
  materialized.
- CPU path: 4-nested-loop stride-aware copy.
- CUDA path: view + `broadcastTo`.

### `tools/oracle.py` (new case in M5)

- `case_multihead_attention_3d` + CASES entry.
- B=2, T=3, D=8, H=2. bias=True.

### `tests/integration_oracle.zig` (M5 + M3 compat)

- New test: `oracle multihead_attention_3d: forward and backward parity`.
- `full_model_forward` test rewrites `blocks[0].*` -> `block.*` at
  fixture-path construction time.

### `tests/integration_cuda.zig` (M3, M4 touches)

- `CausalSelfAttention` struct literals include `.n_head = 1` /
  `.n_head = cfg.n_head`.

### `AGENTS.md`

- §"Current engineering gate" flipped to Stage 8.
- §"Stage 7: Next steps" removed entirely (superseded by
  `docs/stage7_plan.md` + `docs/stage7_endgame_plan.md`).
- Progress table Stage 7 row → Done.
- Progress table Stage 8 row → In progress (will flip to Done at
  M8 close-out).

### `SESSION_GUIDE.md`

- §3 Stage 7 row → COMPLETE.
- §3 Stage 8 row → NOT STARTED (flip to IN PROGRESS / COMPLETE
  when M7 and M8 finish).
- §7 pointer: "If Stage 8, read `docs/stage8_plan.md`".

---

## 6 — Test counts and test-set layout

Current (at HEAD `8f57498`):

| Suite | Count | Notes |
|---|---|---|
| `zig build test` (CPU) | 306 | Includes 15 debug helper tests and 6 checkpoint tests from Stage 8 |
| `zig build test -Dcuda=true` (CUDA) | 79 | `SkipZigTest` on Windows |
| `zig build test-oracle` (PyTorch parity) | 15 | Includes `multihead_attention_3d` |

M7 adds no tests (docs only). M8 will add 1 or 2 tests (training
loop smoke + post-sanitizer pass); expect ~308 CPU / 79-81 CUDA /
15 oracle at close-out.

---

## 7 — Remote workflow cheat sheet

```bash
# Pull + verify a commit landed
bash run_remote_example.sh 'cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && git pull --ff-only && zig build test -Dcuda=true 2>&1 > /dev/null; for b in $(ls -t .zig-cache/o/*/test | head -2); do echo "=== $b ==="; $b 2>&1 | tail -2; done'

# Run a specific example
bash run_remote_example.sh 'cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true 2>&1 | tail -20'

# Regenerate a single oracle fixture (PyTorch venv is at tools/.venv/)
bash run_remote_example.sh 'cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && tools/.venv/bin/python tools/oracle.py generate --case <case_name> 2>&1 | tail -5'

# compute-sanitizer a test binary
bash run_remote_example.sh 'cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && /usr/local/cuda-13.2/bin/compute-sanitizer --tool=memcheck ./.zig-cache/o/<bin>/test 2>&1 > /tmp/sani.log; tail -5 /tmp/sani.log'

# Commit fixture files remotely (the remote has its own git identity set)
bash run_remote_example.sh 'cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab && git add tests/fixtures/<case>/ tests/fixtures/manifest.json && git -c user.email="oracle@remote" -c user.name="oracle-remote" commit -m "stage(XmY): <case> fixture files" && git push 2>&1 | tail -3'
```

The remote's git is authenticated for push via a token. The
`run_remote_example.sh` wrapper preserves the SSH neofetch banner;
use `| Select-Object -Last N` (PowerShell) to trim.

---

## 8 — Commit style

Every commit body follows this template:

```
<header line>

<one-paragraph rationale: why this commit exists>

<section: what changed structurally>

<section: tests added / test count>

<section: Windows verification>

<section: files touched>
```

See any `stage(8m*)` commit body for a canonical example. Keep the
header line under ~72 chars; use `stage(<milestone-id>): <description>`
for milestone commits and `fix(<scope>): <description>` for follow-on
bug fixes within a milestone.

---

## 9 — If things go sideways

### The remote build fails after a push

1. Check the test binary name — Zig test artifact paths include a
   hash that changes with source. Use `ls -t .zig-cache/o/*/test`
   to find the newest.
2. If the error is a Zig compile error on remote but not Windows,
   usually one of:
   - Pattern used a Windows-specific path (fix with forward
     slashes).
   - Zig 0.16 std lib version mismatch (rare — same version on
     both).
   - The test references CUDA types in a `SkipZigTest` block but
     the block didn't activate (check `if (comptime builtin.os.tag
     != .linux)`).

### A test passes on Windows but fails on remote

- Most common: missing CUDA-path coverage. The test runs CPU-only
  on Windows but its CUDA branch trips a bug on Linux.
- Use `bash run_remote_example.sh "zig build test -Dcuda=true 2>&1
  | grep -A10 '<test name>'"` to get the full failure + stack.

### A commit accidentally broke the existing CPU test suite

- Run `git bisect` against `main` (test counts are deterministic).
  Last-known-good as of this doc: `8f57498` with 306 CPU tests.

### A fixture file seems wrong

- Regenerate via the Python oracle (requires PyTorch; only runs on
  remote). Fixtures are committed; do NOT regenerate locally on
  Windows — the generator requires `tools/.venv/` which is gitignored.

---

## 10 — Stage 8 is complete when

- [ ] M7: `docs/09_debugging.md` >= 500 lines.
- [ ] M8: 2-block / 2-head / D=64 Shakespeare run on CUDA,
  `compute-sanitizer` memcheck clean, wall-clock < 2× baseline.
- [ ] AGENTS.md progress table Stage 8 row → Done with commit
  range.
- [ ] SESSION_GUIDE.md §3 reflects Stage 8 completion.
- [ ] docs/stage8_plan.md exit-criteria checkboxes all ticked.
- [ ] Optional git tag `stage-8-complete` pushed to origin.
- [ ] CPU test count stable at >= 306 with no regressions.

Then Stage 9 (documentation finalization per `plan.md` §873) is the
natural next project — it's the last stage and is strictly a docs
pass (no new code).

---

*End of Stage 8 handoff document. Head of `main` at time of writing: `8f57498`. 306 CPU + 79 CUDA + 15 oracle tests pass on both Windows and RTX 4060 Ti remote.*
