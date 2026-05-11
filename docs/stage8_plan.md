# Stage 8 Plan — Debugging Discipline + N-block / N-head Refactor

> **Scope.** Generalise the transformer from the hard-coded `n_layer=1,
> n_head=1` shape of Stages 2–7 to configurable `N`, ship the
> `src/debug/` utilities that would have made Stage 7 debugging faster,
> and document both with a new `docs/09_debugging.md` chapter. Eight
> milestones, ~15–20 hours across ~4–5 sessions.

Companion reading: `plan.md` §828 (Stage 8 scope), `docs/00_overview.md`
decision D5 (hard-coded 1/1 was a Stage 2–7 constraint), `AGENTS.md`
§"Current engineering gate". Style reference:
`docs/stage7_endgame_plan.md` — the format of milestone cards, commit
cadence, and exit criteria mirrors that document.

---

## Locked decisions

User-confirmed before drafting this plan:

| # | Decision | Choice |
|---|---|---|
| 1 | Stage order | Stage 8 first, then Stage 9 (Stage 9 benefits from real multi-head code) |
| 2 | Multi-head parity strategy | **Oracle fixture first** — write `multihead_attention_3d` in `tools/oracle.py` before the multi-head refactor lands |
| 3 | `src/debug/` API shape | **Opt-in function calls** (matches `plan.md` §838); compile-time macros layered later if needed |
| 4 | Checkpoint compatibility | **ZTLC v3 with v2 backward compatibility** — existing `shakespeare_ckpt.bin` keeps working, read as `n_layer=1, n_head=1, dropout=0.0` |
| 5 | Autonomous-run cadence | **Per-commit** for Milestones 1–3 and 6–8; **one autonomous run** for Milestones 4–5 (multi-head refactor benefits from unbroken context) |

---

## Milestone 1 — `src/debug/` utilities

**Goal.** Ship the four utilities described in `plan.md` §838 as a
standalone, opt-in debugging API. Each function is self-contained; the
existing Stage 1–7 code is not touched.

**Files to add.**

| Path | Purpose | Approx LOC |
|---|---|---|
| `src/debug/shape.zig` | `assertShape(t: Tensor, expected: Shape)` with detailed error message (both shapes, rank, element count). Additional `assertRank(t, expected_rank)`, `assertDim(t, axis, expected)` variants. | ~140 |
| `src/debug/finite.zig` | `assertFinite(t: Tensor)` — scans for NaN/Inf and reports the first offending flat index plus its value. CUDA path does a DtoH round-trip (tolerable cost in a debug helper). `hasNaN(t)` / `hasInf(t)` boolean variants for inline checks. | ~130 |
| `src/debug/compare.zig` | `compare(a: Tensor, b: Tensor, opts: CompareOpts) !void` — device-aware diff helper wrapping `src/testing/oracle.zig`'s `maxAbsDiff`/`maxRelErr` but returning a structured `CompareReport { max_abs, max_rel, idx_of_worst }` for use outside the oracle test harness. | ~110 |
| `src/debug/dump.zig` | `dump(t: Tensor, path: []const u8, io: std.Io)` — writes a `.ztlt` tensor file that `tools/oracle.py` can already read via its `read_tensor` routine. CUDA tensors are DtoH-copied first. Symmetrical `load(path, io, alloc)` returns a new CPU tensor. | ~150 |
| `src/debug/root.zig` (umbrella) | Re-exports the 4 modules so callers do `const debug = @import("zig_transformer_lab").debug;` and then `debug.assertShape(...)`. | ~30 |
| `src/root.zig` | Add `pub const debug = @import("debug/root.zig");` + corresponding `_ = debug;` in the test block. | +2 lines |
| `tests/integration_debug.zig` (new) | ~15 unit tests exercising every public entry point on CPU and CUDA (CUDA tests `SkipZigTest` on non-Linux). | ~300 |

**Gotchas.**

- `finite.zig` CUDA variant needs to handle CUDA tensors without a PTX
  module. Simplest: issue a DtoH copy of the entire tensor, then scan
  on the host. A dedicated reduction kernel (flag-per-block) would be
  faster but over-engineered for a debug helper.
- `dump.zig` must reuse the `ZTLT` format already defined in
  `src/testing/oracle.zig:53-65` (magic `"ZTLT"`, version 1, rank, dims,
  n_elements, f32 payload) so the existing Python oracle can read the
  dumps. Rank >= 5 is not supported by the format; error loudly.
- `compare.zig` should NOT depend on `tests/` — `src/testing/oracle.zig`
  is the right dependency seam (it's in `src/`).
- API bikeshedding: `assertShape(t, Shape.init2D(3, 4))` is the intended
  call site. Keep Shape-literal construction ergonomic.

**Tests.**

| Test | Purpose |
|---|---|
| `assertShape: passes on matching shape` | Smoke |
| `assertShape: fails with rich error on mismatch` | Error-path |
| `assertFinite: passes on clean tensor` | Smoke |
| `assertFinite: detects NaN at index N` | Error-path |
| `assertFinite: detects Inf at index N` | Error-path |
| `assertFinite: CUDA tensor with NaN injected` | CUDA |
| `compare: identical tensors report zero diff` | Smoke |
| `compare: near-zero tensors report abs-only diff` | Numerical |
| `compare: CUDA vs CPU tensor` | Device |
| `dump/load round-trip preserves shape and values` | Smoke |
| `dump CUDA tensor writes the device bytes` | CUDA |
| `assertRank / assertDim helpers` | Smoke |

**Acceptance.**

- 15 new tests pass on both Windows (CUDA tests skip) and Linux remote.
- `zig build test` remains at 267 CPU tests + 15 debug tests = 282 CPU
  tests; no regressions.
- compute-sanitizer clean on the CUDA debug tests.

**Commit plan.** Four commits:

1. `stage(8m1-shape): assertShape + assertRank + assertDim`
2. `stage(8m1-finite): assertFinite CPU+CUDA + hasNaN/hasInf`
3. `stage(8m1-compare): device-aware compare with CompareReport`
4. `stage(8m1-dump): dump/load ZTLT tensor round-trip`

Each <100 LOC touched outside the new file; easy to review.

**Estimated effort.** 4–5 hours.

---

## Milestone 2 — `TransformerConfig` extension

**Goal.** Add `n_layer`, `n_head`, `dropout` fields with defaults that
preserve every Stage 1–7 test bit-for-bit.

**Files.**

| Path | Change |
|---|---|
| `src/nn/module.zig` | Add three fields to `TransformerConfig`: `n_layer: u8 = 1`, `n_head: u8 = 1`, `dropout: f32 = 0.0`. Update `test "TransformerConfig defaults are pedagogically small"` to include the new fields. |
| `docs/00_overview.md` | Decision D5: flip from "Generalization to configurable `n_layer`, `n_head` is Stage 8" to a reference to the shipped feature. |
| `tests/` (various) | No changes expected — defaults preserve behavior. |

**Gotchas.**

- `dropout` field is present but NOT consumed this stage. Implementing
  dropout requires random mask generation + mask-aware backward; that's
  out of Stage 8 scope. Document the field as "reserved for Stage 9+".
- The `n_layer: u8` type caps us at 255 blocks; acceptable for a
  pedagogical project. `n_head: u8` caps at 255 heads with the same
  rationale. A future `u32` widening is additive.

**Acceptance.**

- All existing tests pass unchanged (defaults preserve behavior).
- One new `TransformerConfig defaults include Stage 8 fields` test
  covering the three new fields.
- docs/00_overview.md D5 updated.

**Commit plan.** One commit: `stage(8m2): TransformerConfig n_layer/n_head/dropout`.

**Estimated effort.** 0.5–1 hour.

---

## Milestone 3 — `TinyWordTransformer` multi-block

**Goal.** Replace `block: TransformerBlock` with `blocks: []TransformerBlock`.
Forward loops; parameters iterate; save/load handle variable param count.

**Files.**

| Path | Change |
|---|---|
| `src/nn/model.zig` | `block` field → `blocks: []TransformerBlock`. `init` allocates `cfg.n_layer` blocks. `forward` loops through each. `parameters` iterates. `collectNamedParams` prefixes with `blocks[<i>].` for each block. `save`/`load` already use named parameters so the variable count is automatic. `deinit` loops. |
| `src/nn/model.zig` (ALSO) | `moveToCuda`/`moveToCpu` already walk `parameters` — no change needed. |
| `examples/06_train_shakespeare.zig` | Uses `TransformerConfig{}` → still `n_layer=1` by default. No change required; optional overload to demo `n_layer=2`. |
| `tests/integration_cuda.zig` | Add a `cuda TinyWordTransformer 2-block forward parity` test. Oracle is the CPU model's output. |

**Gotchas.**

- Naming: `blocks[0].attn.w_q.weight` needs to be stable across saves.
  Use `std.fmt.allocPrint` with `"blocks[{}].attn.w_q.weight"` and
  free the strings after `collectNamedParams` completes — OR keep them
  alive until `save` returns. The second is simpler; match the
  existing pattern.
- Backward-compatibility: Stage 7 checkpoints name parameters as
  `block.attn.w_q.weight` (no index). The ZTLC v2 loader in Milestone 6
  must rewrite these to `blocks[0].attn.w_q.weight` on load.
- Memory: each extra block adds ~4 MB at D=32 (D×D × 4 Linears × 4
  bytes × 2 for weight+grad). At `n_layer=6` we're at ~24 MB per step
  — still tiny, fits in cache.

**Acceptance.**

- All existing tests pass.
- 2-block forward parity test passes CPU (and CUDA with existing
  routing — no per-op changes needed since block composition is
  already op-routed).
- Save/load round-trip for a 2-block model recovers identical
  parameters.

**Commit plan.** One commit: `stage(8m3): TinyWordTransformer multi-block`.

**Estimated effort.** 2–3 hours.

---

## Milestone 4 — Multi-head attention

**Goal.** Generalise `CausalSelfAttention` from 1 head to `n_head ≥ 1`.
`n_head=1` must be bit-identical to the Stage 7 code path (or within
reshape-induced rounding of 1 ULP).

**Files.**

| Path | Change |
|---|---|
| `src/nn/attention.zig` | Take `n_head` from the `CausalSelfAttention` struct (plumbed through from config). Forward: `Q, K, V` are `(B, T, D)` from the Linear projections. Reshape each to `(B, T, n_head, d_head)` where `d_head = D / n_head`, transpose middle axes to `(B, n_head, T, d_head)`, flatten to `(B·n_head, T, d_head)` for `matmulBatch`. Same trick on the output projection path. `init` asserts `D % n_head == 0`. |
| `src/tensor/ops/shape_ops.zig` | Potentially a `transpose_middle2d` tracked op for `(B, T, H, D) → (B, H, T, D)` — or compose from existing reshape + transposeInner2d by reinterpreting as `(B·T, H·D)` etc. Need to pick one; prefer composition if it works cleanly with existing tracked ops. |
| `src/autograd/backward.zig` | Add `backwardTransposeMiddle2d` if we add a new tracked op. Otherwise no changes. |
| `src/autograd/node.zig` | Add `.transpose_middle2d` OpKind if adding the op. Update exhaustive switch + test count. |

**Gotchas.**

- Our `Shape` max rank is 4. `(B, T, n_head, d_head)` is rank 4 —
  fits, but the attention pipeline reshapes to rank 3 for
  `matmulBatch`. Check we never produce a rank-5 shape.
- `transpose_middle2d` ((B,T,H,D) → (B,H,T,D)) is NOT the same as
  `transposeInner2d`. Decide: (a) new tracked op with its own
  backward, or (b) compose from reshape + transposeInner2d of the
  (B·T, H, D) view, or (c) `(B,H,T,D)` via 3-way permutation.
  Option (a) is most honest; options (b)/(c) are fiddly.
- CUDA: `matmulBatch` on `(B·n_head, T, d_head)` uses
  `cublasSgemmStridedBatched` — already supported. No new kernel.
- Causal mask for multi-head: the mask is shape `(T, T)` broadcast
  to `(B·n_head, T, T)`. The existing `reshapeTracked` +
  `add` broadcast flow handles this — same mask for every head.

**Tests.**

| Test | Purpose |
|---|---|
| `CausalSelfAttention n_head=1: bit-identical to pre-Stage-8` | Regression |
| `CausalSelfAttention n_head=2, D=64: forward matches composed CPU reference` | Smoke |
| `CausalSelfAttention n_head=4, D=64: forward matches` | Smoke |
| `D % n_head != 0 returns InvalidArgument` | Error-path |
| `cuda CausalSelfAttention n_head=2, D=64: CUDA vs CPU parity` | CUDA |

**Acceptance.**

- `n_head=1` pre-change vs post-change forward output matches to
  within 1 ULP per element.
- `n_head=2`, `D=64`, `T=16`, `B=4` forward matches a
  hand-composed multi-head reference on CPU.
- CUDA version same within playbook tolerance (rel=1e-3, abs=5e-4).

**Commit plan.** Two commits:

1. `stage(8m4-cpu): multi-head CausalSelfAttention (CPU path)`
2. `stage(8m4-cuda): multi-head attention CUDA parity`

**Estimated effort.** 4–5 hours.

---

## Milestone 5 — Multi-head oracle fixture + parity

**Goal.** PyTorch-referenced parity for the multi-head attention path.

**Files.**

| Path | Change |
|---|---|
| `tools/oracle.py` | Add `case_multihead_attention_3d(dir)`. Inputs: `a`, `w_q`, `w_k`, `w_v`, `w_o` (+ biases if `bias=True`), `n_head`. Runs `torch.nn.functional.scaled_dot_product_attention` with causal mask equivalent OR assembles via explicit matmul + softmax. Writes input_0..input_4, output, grad_input_0..4. |
| `tools/oracle.py` (manifest) | Register the new case in the dispatch table. |
| `tests/fixtures/multihead_attention_3d/` | Generated via `python tools/oracle.py generate` — checked into git (pattern: all existing fixtures are checked in). |
| `tests/integration_oracle.zig` | Add `oracle multihead_attention_3d` test. |
| `tests/integration_cuda.zig` | Add `cuda oracle multihead_attention_3d` test. |

**Gotchas.**

- PyTorch's `scaled_dot_product_attention` has an `is_causal=True`
  flag; use that for the reference. Set `attn_mask=None`.
- PyTorch may use a fused flash-attention path that differs in
  numerical details from the naive `matmul → softmax → matmul`
  composition. Use the explicit composition for consistency with our
  Zig implementation.
- Dropout: always set `dropout_p=0.0` in the oracle (dropout is not
  implemented on our side).

**Acceptance.**

- Oracle CPU test passes at rel=1e-4, abs=5e-5 (looser than matmul
  oracle; softmax compounds drift).
- Oracle CUDA test passes at rel=1e-3, abs=5e-4 (playbook band).

**Commit plan.** One commit: `stage(8m5): multihead_attention_3d oracle fixture + parity tests`.

**Estimated effort.** 2 hours.

---

## Milestone 6 — ZTLC v3 checkpoint format

**Goal.** Persist `n_layer`, `n_head`, `dropout` in checkpoints without
breaking existing `shakespeare_ckpt.bin`.

**Files.**

| Path | Change |
|---|---|
| `src/nn/model.zig` | `save` writes `version = 3` and three new header fields after the existing ones: `n_layer: u32`, `n_head: u32`, `dropout: f32` (16 bytes, no padding needed since each aligns). `load` branches on `version`: for `version == 2`, implicitly treats `n_layer=1, n_head=1, dropout=0.0` and renames incoming parameter strings from `block.*` to `blocks[0].*` so they match the new naming. For `version == 3`, reads the new fields directly. Reject other versions. |
| `docs/07d_checkpoint_format.md` | Document v3 header, the v2-to-v3 migration semantics, and the `block.*` → `blocks[0].*` rename. |
| `tests/` | Two new tests: `ZTLC v3 round-trip preserves n_layer/n_head/dropout` and `ZTLC v2 loads into v3-shaped model with n_layer=1`. |
| `shakespeare_ckpt.bin` | NO change. The v2 loader in the new code reads it identically to before. |

**Gotchas.**

- Parameter renaming on v2 load: do this at the name-resolution layer
  (lookup table from v2 names to v3 names), not by rewriting the
  file. Simpler and keeps v2 files intact on disk.
- End-trailer `"END."` magic already exists in the format; preserve
  it in v3.

**Acceptance.**

- `shakespeare_ckpt.bin` (v2, from Stage 6) loads into a freshly-built
  Stage-8 model with `n_layer=1`, produces bit-identical logits on
  the same input as the Stage-6 generate script.
- A freshly-saved v3 checkpoint (2-block model) loads back to
  bit-identical parameters.

**Commit plan.** One commit: `stage(8m6): ZTLC v3 with v2 backward compatibility`.

**Estimated effort.** 2 hours.

---

## Milestone 7 — `docs/09_debugging.md`

**Goal.** ~500-line teaching chapter documenting the `src/debug/`
utilities, the shape-assert / NaN-detection / gradcheck workflow, and
the common CUDA-kernel bugs catalog from `plan.md` §854.

**Sections (plan.md §845 breakdown).**

1. **Shape-assert driven development.** Example: a wrong `transpose2d`
   produces a shape mismatch three functions downstream; walk the
   reader through reading the chain.
2. **NaN/Inf detection workflow.** When it happens, where to look first
   (softmax overflow, `log(0)`, large `lr`), how to use
   `debug.assertFinite` as a trip-wire.
3. **Gradient checking a new op.** Using `autograd/gradcheck.zig` +
   the finite-difference reference.
4. **CPU vs GPU output comparison.** Using `debug.compare` +
   `debug.dump` for Python-side inspection.
5. **Overfit-one-batch smoke test.** Revisit `examples/04`.
6. **`compute-sanitizer --tool=memcheck` walkthrough.** Sidebar file
   `examples/sidebar_broken_kernel.cu` (NOT compiled by default; shown
   only in the doc), with the exact output the reader should expect.
7. **Nsight Compute beginner pass on `softmax_last`.** Launch
   command, inspect warp occupancy and memory throughput.
8. **Common CUDA errors catalog.** Six classes with minimal repros:
   illegal memory access (dropped bounds check), wrong grid/block
   (off-by-one on T), non-coalesced access (wrong stride in
   layernorm_rowwise), missed `cudaStreamSynchronize`, host/device
   pointer mixup, row-major vs column-major in cuBLAS.

**Acceptance.**

- Chapter is ≥500 lines.
- `zig build docs` shows the new chapter in the line-count table.
- Every referenced tool (`debug.*`, `compute-sanitizer`, Nsight) is
  introduced with a one-line context + a runnable example.

**Commit plan.** One commit: `stage(8m7): docs/09_debugging.md`.

**Estimated effort.** 3 hours.

---

## Milestone 8 — Acceptance sweep

**Goal.** Prove the 2-block, 2-head, D=64 model trains end-to-end under
`compute-sanitizer`, with wall-clock <2× the 1/1/32 baseline.

**Steps.**

1. Modify `examples/06_train_shakespeare.zig` to accept a
   `-Dn_layer` / `-Dn_head` / `-Dd_model` build option override.
   Default remains 1/1/32 so Stage 6–7 behavior is preserved.
2. Run `zig build run-example -Dexample=06_train_shakespeare
   -Dcuda=true -Dn_layer=2 -Dn_head=2 -Dd_model=64` on the remote RTX
   box.
3. Under `compute-sanitizer --tool=memcheck`: confirm 0 memory errors.
4. Compare wall-clock per step against a reference 1/1/32 run on the
   same input: require `t_2/2/64 < 2 × t_1/1/32`.
5. Inspect final loss — should be in the same order of magnitude as
   the 1/1/32 baseline's final loss; sanity check only.

**Acceptance.**

- `compute-sanitizer` memcheck clean (excluding the one documented
  `expectError` case).
- Wall-clock budget met.
- No test regressions on CPU or CUDA.

**Commit plan.** One commit: `stage(8m8): 2-block 2-head D=64 acceptance sweep`.

**Estimated effort.** 1–2 hours.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Multi-head reshape chain has an off-by-one / wrong-axis bug that only fires at `n_head ≥ 2` | High | The multi-head oracle fixture (Milestone 5, landed before Milestone 4's second commit) catches this before it propagates. |
| `n_head=1` drifts from pre-Stage-8 output due to the new reshape paths | Medium | Dedicated regression test in Milestone 4: `n_head=1` must be within 1 ULP of the pre-change Stage 7 output on a seeded input. |
| ZTLC v2 → v3 param renaming breaks existing `shakespeare_ckpt.bin` | Medium | Add a dedicated `v2 compat` test that loads the real checkpoint and confirms forward-pass bit-parity. |
| Debug utility API creep (feature-requests during code review) | Low | Ship Milestone 1 as a focused 4-commit PR; anything beyond the 4 listed functions is Stage 9+ scope. |
| 2-block, 2-head run exceeds `2×` wall-clock budget | Medium | If so: defer optimization to Stage 9 perf follow-up; document the actual ratio in Milestone 8. |
| Dropout field tempts implementation now | Low | The plan explicitly scopes dropout out. Stick to the plan. |
| CUDA compute-sanitizer OOM on 2-block 2-head | Low | D=64 is still tiny (total model ~200 KB parameters). If it happens, shrink `d_ff` to 128. |

---

## Session-by-session execution

| Session | Milestones | Target end-state |
|---|---|---|
| 1 | M1 (debug utilities) | 4 commits, `src/debug/*` in place, 282 CPU tests |
| 2 | M2 + M3 | 2 commits, `TransformerConfig` extended, `TinyWordTransformer` multi-block, all existing tests green |
| 3 | M4 (multi-head attention) | 2 commits, 1 autonomous run; `n_head=2` parity test green |
| 4 | M5 + M6 | 2 commits; oracle fixture, ZTLC v3 |
| 5 | M7 + M8 | 2 commits; `docs/09_debugging.md`, acceptance sweep, Stage 8 tag |

---

## Exit criteria — Stage 8 is Done when

- [x] M1: 4 commits; `src/debug/{shape,finite,compare,dump}.zig` + umbrella; 15 new tests pass.
      (commits `5c93fe2`, `149b2bd`, `280472a`, `1fb972e`, 2026-05-11)
- [x] M2: `TransformerConfig` has `n_layer`, `n_head`, `dropout` with defaults; all existing tests pass.
      (commit `7e964bf`, 2026-05-11)
- [x] M3: `TinyWordTransformer` holds `blocks: []TransformerBlock`; 2-block forward parity test green.
      (commit `ab794c0`, 2026-05-11)
- [x] M4: Multi-head attention; `n_head=1` regression within 1 ULP; `n_head=2/4` forward parity.
      (commit `9b814d9`, 2026-05-11)
- [x] M5: `multihead_attention_3d` oracle fixture + CPU + CUDA parity tests.
      (commits `8a7b971` + `7aef7e6`, 2026-05-11)
- [x] M6: ZTLC v3 round-trip test; v2 compatibility test loads synthetic v2 file with name rewrite.
      (commits `2382bd8` + `8f57498`, 2026-05-11)
- [ ] M7: `docs/09_debugging.md` ≥ 500 lines.
- [ ] M8: 2-block, 2-head, D=64 Shakespeare run on CUDA, `compute-sanitizer` clean, wall-clock <2× baseline.
- [ ] AGENTS.md progress table updated: Stage 8 → **Done** with commit range.
- [ ] SESSION_GUIDE.md §3 reflects Stage 8 completion.
- [ ] Optional git tag `stage-8-complete`.
- [ ] CPU test count stable at ≥ 282 (with the 15 debug tests added); CUDA tests ≥ 78 (with multi-head + oracle + 2-block tests added).
      **Actual: 306 CPU + 79 CUDA + 15 oracle tests pass on remote RTX 4060 Ti at HEAD `8f57498`.**

### Session 1 landed (2026-05-11)

Six milestones landed end-to-end in one session. Companion document:
`docs/stage8_handoff.md` captures every commit, landmine, and
remaining-work item for the fresh session that will execute M7 + M8.

| Commit | Milestone | CPU test delta | Remote verified |
|---|---|---|---|
| `5c93fe2` | M1-a | +9 | yes |
| `149b2bd` | M1-b | +6 | yes (CUDA paths too) |
| `280472a` | M1-c | +6 | yes |
| `1fb972e` | M1-d | +3 | yes |
| `7e964bf` | M2 | +2 | yes |
| `ab794c0` | M3 | +6 | yes |
| `9b814d9` | M4 | +4 | yes |
| `8a7b971` + `7aef7e6` | M5 | +1 oracle | yes (PyTorch fixture generated on remote) |
| `2382bd8` + `8f57498` | M6 | +3 | yes |

Test counts moved from 267 CPU / 73 CUDA / 14 oracle (Stage 7
complete) to **306 CPU / 79 CUDA / 15 oracle** (M1-M6 complete).

---

*End of Stage 8 plan.*
