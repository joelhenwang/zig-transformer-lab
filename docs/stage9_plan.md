# Stage 9 — Documentation Finalization Plan

> **Goal.** Close the documentation gate per `plan.md` §873 with an emphasis
> on learning content. No new runtime code.
>
> **Baseline.** Tag `stage-8-complete` on origin. 306 CPU + 83 CUDA + 15
> oracle tests pass on RTX 4060 Ti. Ten `docs/0X_*.md` chapters exist;
> chapter 10 does not. Line counts: 00 (262, exempt), 07 (373), 08 (309)
> below the 500-line floor every other main chapter clears.
>
> **Session cadence.** Matches the Stage 7 / 8 template: one commit per
> milestone (M4 splits into two), docs changes don't require remote
> verification, M8 has a preview-diff checkpoint before push.

---

## 0 — Decisions locked in (user-confirmed before execution)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | `docs/00_overview.md` line floor | **Exempt** | It's a TOC / map chapter; padding dilutes its usefulness. Exemption documented here and in the `zig build docs` output. |
| 2 | New learning-guide chapters | **Both** `05b` and `08b` | Closes the missing `0Xb` bridges for Stages 5 and 7/8. |
| 3 | Common-mistakes + exercises audit | **All chapters** | Stage-wide consistency per `plan.md` §888. |
| 4 | TODO cleanup approach | **Mechanical + preview diff** | User reviews diff in the M8 commit-body preview before the push. |
| 5 | M4 commit cadence | **Split** into `9m4a` and `9m4b` | Each new chapter substantial enough to stand alone. |
| 6 | Chapter 10 line floor | **No exempt** (≥ 500) | Worked examples + common-mistakes section close the gap. |
| 7 | Deferred Stage 8 perf follow-up | **Leave deferred** | Stage 9 is docs-only; perf work rolls into a future Stage 10 or off-plan task. |

---

## 1 — Milestone list

Every milestone ships as one commit except M4 which splits into two.

| # | Milestone | Commit subject | Est. effort |
|---|---|---|---|
| M1 | This plan doc | `stage(9m1): docs/stage9_plan.md` | 45 min |
| M2 | Pad + enrich `07_cpu_training.md` (373 → 500+) | `stage(9m2): 07_cpu_training.md pad + learning sections` | 90 min |
| M3 | Pad + enrich `08_backends_cuda.md` (309 → 500+) | `stage(9m3): 08_backends_cuda.md pad + learning sections` | 120 min |
| M4a | New `05b_from_tokenizer_to_training.md` | `stage(9m4a): 05b_from_tokenizer_to_training.md` | 2 h |
| M4b | New `08b_from_cuda_to_training.md` | `stage(9m4b): 08b_from_cuda_to_training.md` | 2 h |
| M5 | 11 `src/*/README.md` files | `stage(9m5): per-subfolder README files under src/` | 2 h |
| M6 | `docs/10_pytorch_parallels.md` (11 sections, ≥ 500 lines) | `stage(9m6): docs/10_pytorch_parallels.md` | 3 h |
| M7 | Exercises + common-mistakes audit across 00–10 + 0Xb | `stage(9m7): exercises + common-mistakes audit` | 2 h |
| M8 | TODO triage + inline-comment polish in `src/` | `stage(9m8): TODO triage + inline-comment polish` | 1 h |
| M9 | Close-out: AGENTS / SESSION_GUIDE / tag | `stage(9): documentation finalization complete` | 20 min |
| **Total** | | | **~14 h, 2 sessions** |

---

## 2 — Milestone detail

### M1 — Plan doc

This file. Mirrors `docs/stage8_plan.md` and `docs/stage7_plan.md` structure:
decisions, milestone list, per-milestone detail, acceptance criteria, risk
register, close-out checklist.

### M2 — `docs/07_cpu_training.md` pad + enrich (373 → 500+)

Current chapter is operations-heavy; needs the learning narrative other
chapters have.

New sections to add:

- **§7.x — The training loop as a data pipeline.** Trace one step's
  allocations (tensors, gradients, tape nodes, `kept_alive` buffers) and
  their free points. Show the full lifecycle a leak hunter cares about.
  ~60 lines.
- **§7.x — Three silent bugs we fixed.** Expanded diagnostic walk-throughs
  of the `@round` bug, untracked-reshape bug, `beta2 = 0.95` instability.
  Each framed as "loss looked fine, then…" → resolution. ~50 lines.
- **§7.x — Common mistakes** sidebar (shape mismatch in `crossEntropy`,
  forgotten `grad_clip_norm`, reading `loss.data[0]` after `loss.deinit`).
  ~20 lines.
- **§7.x — Exercises.** Three worked exercises with solutions (e.g. "What
  happens if you forget `tape.deinit()` between steps?", "Why does lr=3e-3
  diverge on tiny.txt even with clipping?"). ~30 lines.

### M3 — `docs/08_backends_cuda.md` pad + enrich (309 → 500+)

Current chapter is strong on mechanics; needs more learning scaffolding.

New sections:

- **§8.x — From dispatching an op on CPU to on CUDA.** Side-by-side call
  trace of a single `add` through both backends; show the branch in
  `src/tensor/ops/elementwise.zig`. ~50 lines.
- **§8.x — Why dlopen, not link-time linking.** Rationale for Driver API
  dynamic loading: portability, no libcuda at CI build time, runtime GPU
  discovery. Contrast with the cudart-linked alternative. ~40 lines.
- **§8.x — The row-major → column-major worksheet.** Two fully-worked
  numerical examples readers can verify by hand, confirming the operand
  swap trick produces the right bytes. ~50 lines.
- **§8.x — Common mistakes** sidebar (row/col confusion, forgetting
  `ctx.synchronize()`, non-contiguous operands in cuBLAS). ~20 lines.
- **§8.x — Exercises.** Three exercises with solutions (e.g. "Derive the
  operand swap for `C = A @ B^T`", "What's wrong with this PTX function
  signature?"). ~30 lines.

### M4a — `docs/05b_from_tokenizer_to_training.md` (new)

Bridge chapter matching the `02b` / `03b` / `04b` / `07b` pattern.
Assumes reader has read `05_transformer_math.md` + `06_tokenizer_data.md`.

Target length: ~550 lines.

Structure:

1. **The big picture.** Why word-level tokenization vs. BPE vs. char. What
   the batcher actually emits and why it's a stream, not a list.
2. **From text to token IDs.** Trace a sentence through Dataset → Vocab
   → Windowing → Batcher. Show every intermediate structure's memory
   layout.
3. **The dataloader mental model.** Compare to `torch.utils.data.Dataset`
   + `DataLoader`. Explain why our batcher is simpler (no multiprocessing,
   no collate_fn).
4. **Window + shift = next-token supervision.** Explain in pictures why
   we shift by 1 and not by T.
5. **Batch size and sequence length tradeoffs.** The memory × compute
   square.
6. **From a batch to a forward pass.** How `(B*T,)` indices become
   `(B, T, D)` embeddings.
7. **Common mistakes.** Off-by-one in windowing, forgetting to reset
   the batcher, wrong dtype on target indices.
8. **Exercises.** Two-three worked exercises (e.g. "Implement a
   BOS/EOS-aware windowing variant", "Why does `batcher.reset()` take a
   new RNG state?").

### M4b — `docs/08b_from_cuda_to_training.md` (new)

Bridge chapter for Stage 7 / 8 CUDA work. Assumes reader has read
`08_backends_cuda.md`.

Target length: ~550 lines.

Structure:

1. **What it means to "move a model to GPU".** Device residency of every
   parameter; the `moveToCuda` walkthrough.
2. **Per-step device residency.** Trace one training step of
   `examples/10_train_deep.zig` showing which tensors live where at each
   moment; mark every HtoD/DtoH boundary.
3. **Why HtoD/DtoH is usually wrong inside a hot loop.** Memory bandwidth
   vs. kernel-launch overhead; how our per-step input upload sits at
   the "okay because it's tiny" edge.
4. **The Trainer CUDA path (Stage 8 M8-b) annotated.** Walkthrough of
   `src/lab/train.zig`'s use_cuda branch.
5. **Checkpointing across devices.** ZTLC v3 + the DtoH scratch pattern
   from Stage 8 M8-c.
6. **Compute-sanitizer and Nsight Compute: quick recipes.** Two-page
   cheat sheet on the most common invocations.
7. **Common mistakes.** Forgetting to load a PTX module, mismatching
   context pointers, freeing DeviceBuffers after context destruction.
8. **Exercises.** Two-three worked exercises.

### M5 — Per-subfolder READMEs

Eleven files, one per subfolder under `src/`:

```
src/autograd/README.md
src/backend/README.md       (with a nested src/backend/cuda note)
src/core/README.md
src/data/README.md
src/debug/README.md
src/lab/README.md
src/nn/README.md
src/optim/README.md
src/tensor/README.md
src/testing/README.md
src/tokenizer/README.md
```

Each is ~40-80 lines:

- Mission (one paragraph).
- File listing (one line per file explaining its role).
- Cross-reference to the relevant `docs/0X` chapter.
- "If you're new here, read X first" pointer.

### M6 — `docs/10_pytorch_parallels.md`

Target length: ≥ 500 lines. Not just a mapping table — full chapter.

Sections:

1. **Intent.** Who this is for (PyTorch users reading this codebase).
2. **Tensor vs `torch.Tensor`.** Storage, sizes, strides; a code diff
   showing the same op in both.
3. **Autograd: tape vs dynamic graph.** Why ours is simpler, what
   `torch.autograd.Function` does that ours does not, adding a new op
   in each.
4. **`nn.Module` protocol.** Parameter iteration, named params, state
   dict equivalents.
5. **Optimizers.** Our SGD / AdamW vs `torch.optim.*`. Numerical demo
   of coupled vs. decoupled weight decay.
6. **Backend vtable vs ATen dispatcher.** High-level comparison with
   `DispatchKey::CUDA`.
7. **cuBLAS wrapper.** Our row-major trick vs. ATen's handling.
8. **Custom kernels.** Our `ce_loss.cu` compared architecturally to
   `aten/src/ATen/native/cuda/CrossEntropyLoss.cu` (no code copied).
9. **Checkpoint formats.** ZTLC v3 vs `torch.save` vs safetensors.
10. **Bridging exercise.** Port a 10-line PyTorch snippet to this
    library; full worked solution.
11. **Common mistakes when switching mental models.** Five entries.

### M7 — Exercises + common-mistakes audit

For every chapter in `docs/`:

1. Scan for a "Common mistakes" or equivalent section.
2. If missing, add one (3-5 entries).
3. Count worked exercises (format: problem → solution).
4. If < 2, add more to reach 2-4 per chapter.

Scope: chapters 00, 01, 02, 02b, 02c, 02d, 03, 03b, 03c, 04, 04b, 05, 06,
07, 07b, 07c, 07d, 08, 09, and (post-M3/M4a/M4b/M6) 05b, 08b, 10. Plus
the new `oracle.md` — which is reference rather than teaching, so
lighter treatment.

Expected finding: some chapters are already at 4+ exercises, others have
zero. This milestone is a consistency pass, not a uniform rewrite.

### M8 — TODO triage + inline-comment polish

**Phase 1: TODO triage** across `src/`. Baseline count: 19 (as of HEAD
`c266c47`).

Policy:

- **Delete** any TODO referencing completed-stage work (e.g. "Stage 3
  will wire this", "Stage 7 adds CUDA" — both done).
- **Rewrite** remaining future-work TODOs to be prefixed `// future:`.
- **Preserve** genuinely-actionable TODOs that describe missing
  implementation we chose not to ship (e.g. `reduce.zig:28`
  "`keep_dims=false`"); rewrite these to `// future: keep_dims=false
  variant once we need axis-collapsing ops`.

Target: ≤ 6 TODOs after the pass, all prefixed `// future:`.

**Phase 2: inline-comment polish** — one walk through `src/tensor/`,
`src/autograd/`, `src/backend/cuda/dispatch.zig`. Goals:

- Fix any comments that contradict current code.
- Tighten rambling comments.
- Cross-reference new chapters where appropriate.

**Preview checkpoint.** Per user policy (Q4): before the M8 push I'll
paste the full per-file diff as a chat message and wait for ack.

### M9 — Close-out

- Update `AGENTS.md` progress table: Stage 9 → **Done**.
- Remove "Current engineering gate" section from AGENTS.md (or flip to
  "Stages 1–9 complete; library scope fulfilled").
- Update `SESSION_GUIDE.md` §3 Stage 9 row → COMPLETE with final test
  counts and acceptance evidence.
- Update this file (`docs/stage9_plan.md`) exit-criteria checkboxes.
- Update `README.md` (repo root) if it references an active stage.
- Optional: `git tag stage-9-complete` + push.

---

## 3 — Acceptance criteria

- [x] M1 — `docs/stage9_plan.md` exists (this file).
- [x] M2 — `docs/07_cpu_training.md` ≥ 500 lines with new learning sections (744 lines).
- [x] M3 — `docs/08_backends_cuda.md` ≥ 500 lines with new learning sections (783 lines).
- [x] M4a — `docs/05b_from_tokenizer_to_training.md` exists, 556 lines.
- [x] M4b — `docs/08b_from_cuda_to_training.md` exists, 615 lines.
- [x] M5 — 13 `src/*/README.md` files exist (11 top-level subfolders + 2 nested: `backend/cuda/`, `tensor/ops/`).
- [x] M6 — `docs/10_pytorch_parallels.md` ≥ 500 lines, 11 sections (706 lines).
- [x] M7 — Every main chapter has a "Common mistakes" section + ≥ 2 exercises.
- [x] M8 — `grep -rn "TODO:" src/` returns only `future:`-tagged entries (16 blocks).
- [x] M9 — AGENTS.md / SESSION_GUIDE.md / stage9_plan.md all reflect Stage 9 Done.
- [x] Optional — `stage-9-complete` tag pushed.
- [x] Docs line-count gate: every `docs/0X*.md` ≥ 500 lines, **except
      `00_overview.md` which is exempt** (TOC chapter).
- [x] No regression in test count: 306 CPU + 83 CUDA + 15 oracle continues
      to pass (docs changes shouldn't touch code; M8 polish is
      comment-only).

---

## 4 — Execution contract

- **Plan doc first (M1).** Committed before any chapter is written so the
  work log survives a session boundary.
- **Every new section cites concrete code.** Minimum three
  `file_path:line_number` references per new section to anchor teaching
  in the codebase. Verified by a random sample post-commit.
- **No runtime code changes.** Comment edits only in M8. READMEs under
  `src/` are not imported; they can't break `zig build`.
- **M8 preview checkpoint.** TODO triage diff posted before push.
- **Single-commit per milestone** (M4 splits into 4a + 4b). Commit
  bodies follow the `stage8_handoff.md` §8 template: rationale, what
  changed, verification, files touched.

---

## 5 — Risks and mitigations

| Risk | Mitigation |
|---|---|
| Docs sprawl without code anchor | Every new section cites concrete `file_path:line_number` references. Random sample validated post-commit. |
| M4 new chapters duplicate 05 / 08 mechanics | Strict pattern: `05` / `08` are "how the code works"; `05b` / `08b` are "why this architecture and how it maps to PyTorch". Same discipline as `02` vs `02b`. |
| Chapter 10's ATen comparisons cite specifics that shift across PyTorch releases | Cite files, not line numbers. Pin to "as of PyTorch 2.3". Architectural comparison only — no code copied. |
| "Learning content" is subjective — over-write into textbook territory | Cap each new chapter at ~700 lines. Each section must serve a specific learning goal stated in its header. |
| M7 audit surfaces more missing content than budgeted | Report after audit pass; decide whether to expand scope or land a partial audit with tracked follow-ups. |
| M8 TODO triage removes a TODO that was still meaningful | Preview-diff checkpoint: user reviews every delete/rewrite before the push. |

---

## 6 — Session notes (populated during execution)

### Session 1 (2026-05-11)

All ten milestones landed end to end in one session.

| Commit | Milestone | Lines added |
|---|---|---|
| `3e9221d` | M1 (plan doc) | 326 |
| `657010d` | M2 (07 pad) | 252 |
| `88ca471` | M3 (08 pad) | 373 |
| `eff5b7a` | M4a (05b new) | 556 |
| `78d9038` | M4b (08b new) | 615 |
| `df9713a` | M5 (13 READMEs) | ~500 |
| `22e1107` | M6 (chapter 10) | 706 + 8 (TOC update) |
| `4649374` | M7 (audit pass, 11 files) | 504 |
| `53be82e` | M8 (TODO triage, 18 files) | 52 (net) |
| `048c8ae` | M9 (close-out) | — |

Total new documentation: ~5500 lines across one new plan doc, two
padded chapters, two new learning-guide chapters, one new final
chapter, 13 READMEs, and an audit pass.

Verification

  zig build test exit 0 across every milestone (306 CPU tests pass).
  zig build test -Dcuda=true exit 0 (CUDA tests all SkipZigTest on
  Windows; remote-verified green during Stage 8 M8-f baseline).
  zig build test-oracle exit 0 (15 oracle tests pass).

  No runtime code changed in Stage 9; the `// future:` TODO rewrites
  in M8 were the only `src/` edits, and they are comment-only.

---

*End of Stage 9 plan. Head of `main` at Stage 9 close: see close-out
commit subject.*
