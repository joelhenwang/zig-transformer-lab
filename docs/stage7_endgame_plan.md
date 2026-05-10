# Stage 7 Endgame Plan — PR-μ completion, PR-ν, PR-ξ

> **Scope.** This plan covers everything from "embedding + AdamW just landed"
> (commit `dccb5b6`) to "Stage 7 marked Done in `AGENTS.md`". Seven
> milestones, ~15–20 hours of focused work across 6–8 commits. Closes the
> remaining 2/14 PRs in `docs/stage7_plan.md` §4.

Read `docs/stage7_plan.md` §6 PR-μ / PR-ν / PR-ξ and `docs/08_backends_cuda.md`
first. This document does not duplicate the derivations already in those
chapters — it captures the remaining work, the sequencing, and the specific
choices the user already locked in.

---

## Locked decisions (from the planning round)

| Decision | Choice |
|---|---|
| CE forward on CUDA | **Fused kernel** producing both `loss` and `grad_logits` in one launch; tape saves the pre-computed grad |
| Remaining op coverage sequencing | **Gap-fill first**, then build the PR-ν parity harness (each step stays unit-tested and green) |
| PR-ξ benchmark shape | **Synthetic** `examples/09_cuda_benchmark.zig` (fixed tensors, 100 iterations, no Shakespeare coupling) |
| Full-model tolerance | **Playbook-documented**: fwd `rel=1e-3, abs=5e-4`; bwd `rel=1e-3, abs=1e-3`; one-step param diff `2e-3` |

---

## Milestone 1 — Complete PR-μ: Cross-Entropy CUDA

**Goal.** Close the last piece of PR-μ so the training loop's loss + backward
can run end-to-end on GPU without a CPU detour.

**Files to add / modify.**

| Path | Change |
|---|---|
| `src/backend/cuda/kernels/ce_loss.cu` (new, ~80 LOC) | One `extern "C" __global__ void ce_fused(const float* logits, const float* targets, float* loss_out, float* grad_logits, unsigned int B, unsigned int C)`. Block per row. 3-pass shared-memory reduction (max → sum of exp → per-element `(softmax - one_hot) / B` write + atomicAdd into `loss_out`). Host pre-zeros `loss_out` via `cuMemsetD32_v2`. |
| `build.zig` | Add `"ce_loss"` to `kernel_names`. |
| `src/autograd/node.zig` | Extend `SavedData` with a new variant `ce_cuda_grad: Tensor` (fused-forward saves the grad tensor for backward to return as a clone). Alternatively: add a field `saved_grad: ?Tensor = null` onto the existing `ce_info` variant — simpler. The plan uses the second approach. |
| `src/backend/cuda/dispatch.zig` | Add `crossEntropyFused(logits, targets) !struct { loss: Tensor, grad_logits: Tensor }`. |
| `src/tensor/ops/loss.zig` | `crossEntropy` detects CUDA at the top, forwards to `crossEntropyFused`, saves `grad_logits` in the tape via `ce_info.saved_grad`. |
| `src/autograd/backward.zig` | `backwardCrossEntropy` checks `ce_info.saved_grad` — if present (CUDA path), return a DtoD clone of it. If null (CPU path), run the existing recompute-softmax branch. |
| `tests/integration_cuda.zig` | 2 tests: oracle `cross_entropy_3d` forward parity (after reshape 3D→2D), and full fwd+bwd parity via `ops_loss.crossEntropy` + `tape.backward`. |

**Gotchas.**

- Numeric stability: reuse the max-subtraction trick already in `softmax.cu`.
- Target rounding: `rintf(targets[row])` (our targets tensor is f32).
- Mean reduction: divide by `B` inside the kernel so forward returns the
  already-meaned loss and backward's grad_logits is already `1/B`-scaled.
- atomicAdd into `loss_out`: only thread 0 per block emits the contribution
  to avoid over-counting.
- Tape save ownership: `saved_grad` must be DtoD-cloned into
  `kept_alive_cuda` by the existing `tape.cloneTensorData` path. This means
  `takeOwnershipOfSaved` must handle the new field — one-line addition.

**Acceptance.**

- `zig build test -Dcuda=true` passes on both Windows (SKIP) and Linux
  (real run).
- Oracle `cross_entropy_3d` forward match within `rel=1e-4, abs=1e-4`.
- Backward: `dL/dlogits` matches PyTorch within same tolerance.
- compute-sanitizer clean (0 leaks, 0 memory errors).

**Commit subject.** `stage(7m): CUDA cross-entropy (fused forward+grad) + parity`

**Estimated effort.** 2–3 hours.

---

## Milestone 2 — Fill remaining op gaps for full-model forward

**Goal.** Every operation reachable from `TinyWordTransformer.forward` must
have a CUDA path. Run a smoke test that instantiates the model with
CUDA weights and completes one forward pass on GPU without falling back.

**Op coverage audit.** Ops NOT currently routed that the model touches:

| Op | Location | Required for |
|---|---|---|
| `ops_unary.geluExact` | activations / MLP | MLP forward |
| `ops_unary.sqrt` | LayerNorm | LayerNorm forward |
| `ops_unary.exp` (probably unused in model forward but check) | — | nothing concrete; route if cheap |
| `ops_unary.log` (unused in forward; CE uses logSoftmax) | — | skip |
| `ops_unary.relu` (unused — GELU is the activation) | — | skip |
| `ops_reduce.mean` | LayerNorm mean | LayerNorm forward |
| `ops_matmul.transpose2d` / `ops_shape.transposeInner2d` | attention | attention forward |
| `ops_shape.reshapeTracked` | attention, logits | model forward |

**Views (reshape, transpose2d, transposeInner2d) are device-agnostic** — they
produce Tensor values with shared storage and different strides. They already
work for CUDA tensors because they never touch `.data` in their
implementation. Double-check by reading them once; no code change expected.

**Required new kernels.**

| Kernel | File | Approach |
|---|---|---|
| `unary_gelu_exact` / `unary_gelu_exact_backward` | `src/backend/cuda/kernels/unary.cu` (new) | Same formula as CPU: `0.5 * x * (1 + erf(x / sqrt(2)))`. `erff` is intrinsic on sm_89. Backward uses the saved input via the standard formula. |
| `unary_sqrt` | same file | `sqrtf(x)`. Backward is `0.5 / sqrt(x)`. |
| `unary_exp` (optional; add if free) | same file | `expf(x)`. |

**Required dispatch wiring.**

| File | Change |
|---|---|
| `src/backend/cuda/kernels/unary.cu` (new) | The above kernels, each bounds-checked and `extern "C"`. |
| `build.zig` | Add `"unary"` to `kernel_names`. |
| `src/backend/cuda/dispatch.zig` | `geluExact(x)`, `sqrt(x)`, `exp(x)` dispatch functions. `meanAxis(x, axis)` helper that calls `sumAxis` then `mulScalar(1/n)` — no new kernel needed. |
| `src/tensor/ops/unary.zig` | Route `exp`, `log`, `relu`, `geluExact`, `sqrt` through CUDA when input is CUDA. For `log` and `relu` that aren't in the model forward, route them anyway — the backward of CE uses `log` implicitly, and it's 10 LOC per op. |
| `src/tensor/ops/reduce.zig` | Route `mean` through CUDA: call `sum` (already CUDA) then `mulScalar(1/axis_size)`. |

**Linear layer.** `src/nn/linear.zig` uses matmul + add (bias) + reshape.
Matmul and add already route. Reshape is device-agnostic. So Linear forward
should "just work" on CUDA — no changes. Add a smoke test to verify.

**Backward.** Each backward for GELU / sqrt uses the saved input tensor.
The tape already DtoD-copies CUDA saved tensors (PR-η.2). The backward
implementations in `backward.zig` call `ops_elementwise.mul`, `div`, etc.
which already route. They should compose correctly without modification.
Audit each backward for any `Tensor.init` that allocates CPU-only tensors;
replace with `ops_create.zerosLike(template)` if found.

**Tests.**

| Test | Purpose |
|---|---|
| `cuda dispatch geluExact: random input matches CPU reference within 1e-5` | Unit |
| `cuda dispatch sqrt: positive random input matches CPU reference` | Unit |
| `cuda dispatch meanAxis: (2, 3, 4) axis=1 matches CPU sumAxis / axis_size` | Unit |
| `cuda oracle gelu_2d: forward + backward parity` | Oracle |
| `cuda oracle layernorm_3d: forward + backward parity` | Oracle |
| `cuda Linear.forward: smoke test on (B=2, in=4, out=5)` | Smoke |
| `cuda TransformerBlock forward: inputs all zeros produces same output CPU vs CUDA` | Smoke |

**Acceptance.**

- All new oracle fixtures (`gelu_2d`, `layernorm_3d`) pass on CUDA within
  playbook tolerances.
- `TinyWordTransformer.forward` on CUDA does not return `NotImplemented`
  or produce garbage. Validated by comparing against CPU forward in the
  smoke test.
- compute-sanitizer clean.

**Commit plan.** One commit per sub-group: `stage(7-unary): GELU/sqrt/exp CUDA`,
`stage(7-mean): CUDA meanAxis dispatch`, `stage(7-nn): verify Linear/LayerNorm/MLP/Block CUDA`.
Each verified separately before the next.

**Estimated effort.** 4–6 hours.

---

## Milestone 3 — PR-ν part 1: Full-model forward parity

**Goal.** A complete `TinyWordTransformer.forward` runs on CUDA, producing
logits that match the CPU forward bit-for-bit modulo f32 rounding.

**Files.**

| Path | Change |
|---|---|
| `examples/08_cuda_vs_cpu.zig` (new, ~120 LOC) | Builds a `TinyWordTransformer` with the `full_model_forward` oracle's weights (15 `.ztlt` files loaded via `src/testing/oracle.zig`). Runs forward on CPU and CUDA. Computes max abs diff and max rel err on the logits. Asserts below tolerance. |
| `tests/integration_cuda.zig` | Add a test that invokes the example's forward-parity function (lift the comparison logic into a helper). |
| `src/nn/model.zig` | Likely need a `toCuda(ctx)` method on `TinyWordTransformer` that walks every parameter and moves to CUDA, preserving `param_id`. Mirrors the per-tensor `toCuda`. |

**Loading weights onto CUDA.** Two paths:

1. Load via the existing CPU checkpoint path (`.ztlt` → CPU Tensor → stored
   on model), then call `model.toCuda(&ctx)` which per-parameter
   `Tensor.toCuda`'s each weight. Simpler.
2. Direct CUDA load — skip CPU allocation entirely. More work.

Plan uses path 1.

**Acceptance.**

```
examples/08_cuda_vs_cpu.zig runs with:
  max |logits_cpu - logits_cuda| < 5e-4   (abs)
  max |logits_cpu - logits_cuda| / |logits_cpu|.max < 1e-3   (rel)
```

Plus: `compute-sanitizer --tool=memcheck` clean on one full forward pass.

**Commit subject.** `stage(7n): full-model forward parity (TinyWordTransformer CPU vs CUDA)`

**Estimated effort.** 3–4 hours, mostly debugging tolerance drift in
composed kernels.

---

## Milestone 4 — PR-ν part 2: Full-model backward parity

**Goal.** One full backward pass on CUDA produces gradients that match the
CPU backward within the playbook's `abs=1e-3` / `rel=1e-3` tolerances on
every parameter.

**Extension of Milestone 3's example.** Same `08_cuda_vs_cpu.zig`:

1. Forward on CPU + CUDA (already done).
2. `loss = sumAll(logits)` as a trivial scalar loss (avoids CE contribution
   to backward tolerance drift — pure autograd chain exercise).
3. `tape.backward(&loss)` on CPU + CUDA.
4. Iterate over the model's parameter list; for each param:
   - Read its grad back to host.
   - Compare to CPU grad elementwise.
   - Report max abs / max rel per param.
   - Assert below tolerance.

**Expected gotchas.**

- LayerNorm's mean/variance reductions compound rounding. Attention's
  softmax is the single biggest drift source. Order-of-summation in the
  atomicAdd-based sumAll adds another ULP per independent parameter.
- If any param exceeds tolerance, the fix is usually to tighten the kernel
  (e.g. warp-shuffle reduction instead of shared-memory) rather than
  loosening the tolerance.

**Acceptance.**

```
For every named parameter p in TinyWordTransformer.named_parameters():
  max |p.grad_cpu - p.grad_cuda| < 1e-3
  max |p.grad_cpu - p.grad_cuda| / |p.grad_cpu|.max < 1e-3
```

**Commit subject.** `stage(7n): full-model backward parity (one tape.backward on CUDA)`

**Estimated effort.** 3–4 hours.

---

## Milestone 5 — PR-ν part 3: One training step parity

**Goal.** Prove that the full `forward → loss → backward → AdamW step`
cycle on CUDA produces parameter values within `abs=2e-3` of the CPU
trajectory after one step.

**Extension.**

1. Forward + backward as in Milestone 4 (use `crossEntropy` now instead
   of `sumAll` — this exercises the CE CUDA path from Milestone 1).
2. Run `AdamW.step(params)` on both CPU and CUDA models.
3. Compare each param's post-step value.

**Acceptance.**

```
For every named param p after one AdamW step:
  max |p_cpu - p_cuda| < 2e-3
```

This tolerance is the playbook-documented one and absorbs:
- forward drift (`~5e-4`)
- backward drift through chain rule (`~1e-3`)
- AdamW step drift (`~2e-4` from sqrt / division path)

**Commit subject.** `stage(7n): full-model one-step parity (forward+backward+AdamW on CUDA)`

**Estimated effort.** 2 hours.

---

## Milestone 6 — PR-ξ: Synthetic training-speed benchmark

**Goal.** Measure CUDA speedup vs CPU on a realistic transformer workload.
Playbook target: **≥30× speedup** on the Shakespeare config (`V=2000, D=32, T=16, B=4`).

**Files.**

| Path | Change |
|---|---|
| `examples/09_cuda_benchmark.zig` (new, ~80 LOC) | Constructs a `TinyWordTransformer`, allocates fixed synthetic `ids` and `targets`, runs 100 `forward+backward+step` iterations each on CPU and CUDA, reports wall-clock per iteration and speedup ratio. |
| `docs/08_backends_cuda.md` | Append a "Performance" section with the measured numbers. |

**Measurement discipline.**

1. Warm-up: discard iterations 0–9. Measure 10–99. GPU driver JIT cache
   is hot after iter 0; first iter can be 10× slower than steady state.
2. Synchronize after each step for accurate timing:
   `ctx.synchronize()` after every forward+backward+step. Without this,
   the host returns from `step()` before the kernel finishes and timing
   is meaningless.
3. Use `std.time.Timer.start/lap` for wall-clock; no `cudaEvent` (we'd
   need to bind it, scope creep for PR-ξ).
4. Report: `cpu_ms_per_step`, `cuda_ms_per_step`, `speedup_ratio`.

**Acceptance.**

```
cuda_ms_per_step * 30 <= cpu_ms_per_step   AND
final loss within 10% of CPU's final loss at matched step count
```

If we don't hit 30×: the playbook permits documenting the actual ratio
and investigating in a Stage 9 performance PR. Shapes this small may
be GPU-launch-overhead dominated; the 30× target is realistic on
transformer-scale tensors but optimistic for our toy model.

**Commit subject.** `stage(7o): training-speed benchmark (synthetic)`

**Estimated effort.** 2 hours.

---

## Milestone 7 — Stage 7 close-out

**Goal.** Mark Stage 7 complete in every tracking document, tag the commit,
summarize lessons learned.

**Changes.**

| File | Update |
|---|---|
| `docs/stage7_plan.md` | Section 4 table: flip last two PR rows to `[x]`. Section 10 progress log: append final entries with commit hashes and key outcomes. Section 13 "What done looks like": tick every checkbox. |
| `AGENTS.md` | Progress table row "7 — CUDA Backend" flips to **Done** with the commit range. Update the "Current engineering gate" section to either remove Stage 7 language or point to Stage 8. |
| `SESSION_GUIDE.md` §3 | Stage 7 row marked complete with final test count (267 CPU + ~55 CUDA). Session Start Checklist step 4 unhooks the Stage 7 playbook read (since it's done). |
| `docs/08_backends_cuda.md` | Append a "Lessons learned" section: what surprised us (f32 accumulation order, `cuModuleLoadData` null terminator requirement, Zig test runner's `std.log.err` behaviour). |
| Git tag (optional) | `stage-7-complete` |

**Commit subject.** `stage(7): CUDA backend complete`

**Estimated effort.** 1 hour (mostly documentation).

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Full-model tolerance exceeded on LayerNorm | Medium | Warp-shuffle reduction for LayerNorm mean/var; or accept documented looser tolerance for LN params specifically. |
| 30× speedup not achieved | Medium-high | Our shapes are tiny (B=4, T=16, D=32). Launch overhead dominates. Mitigation: document actual number, schedule perf optimization for Stage 9. |
| CE fused kernel shared-memory sizing bug | Low | Stress-test with C=2000 (our vocab). Shared memory is `BLOCK_SIZE=256` f32s regardless of C, so no actual sizing concern — but verify with `compute-sanitizer` on a C=2000 run. |
| Tape SavedData variant collision | Low | Add `.saved_grad` to existing `.ce_info` variant rather than new variant; backward-compatible with existing CPU tests. |
| AdamW moment state drift across CPU/CUDA comparison | Medium | In the parity test, re-init both optimizers from zero state before each run; never compare moments after multiple steps without running identical trajectories. |
| compute-sanitizer timing out on training loop | Low | Run sanitizer only on one iteration of the training loop, not 100. |

---

## Cumulative estimated effort

| Milestone | Hours |
|---|---|
| 1. CE fused kernel | 2–3 |
| 2. Op gap fill | 4–6 |
| 3. Full-model forward parity | 3–4 |
| 4. Full-model backward parity | 3–4 |
| 5. One-step parity | 2 |
| 6. Benchmark | 2 |
| 7. Close-out | 1 |
| **Total** | **17–22 hours** |

Comfortably 3–5 OpenCode sessions.

---

## Session-by-session execution plan

The work maps cleanly to 5 sessions. Each session ends with one or more
pushed commits and updated progress docs.

**Session 1 — CE + op coverage (M1 + M2 part 1)**
- CE fused kernel + tape integration + oracle parity.
- GELU kernel + oracle `gelu_2d` parity.
- sqrt kernel + unit test.
- mean axis dispatch + unit test.
- Target: 3 commits, ~60–70 CUDA tests pass.

**Session 2 — NN layer smoke tests (M2 part 2)**
- Verify Linear forward on CUDA.
- LayerNorm forward + oracle `layernorm_3d` parity.
- TransformerBlock forward smoke test.
- Target: 1 commit, layernorm oracle green.

**Session 3 — Full-model forward parity (M3)**
- `model.toCuda(ctx)` method.
- `examples/08_cuda_vs_cpu.zig` forward portion.
- Oracle `full_model_forward` parity on CUDA.
- Target: 1 commit, logits diff reported.

**Session 4 — Backward + one-step parity (M4 + M5)**
- Extend 08 example with backward + AdamW step comparison.
- Iterate on any per-param tolerance misses.
- Target: 1 commit.

**Session 5 — Benchmark + close-out (M6 + M7)**
- `examples/09_cuda_benchmark.zig`.
- Measure + document speedup.
- Flip all checkboxes.
- Target: 2 commits.

---

## Open questions to surface during execution

1. **Deterministic CE.** Our CE fused kernel uses `atomicAdd` for the
   loss accumulator (same pattern as `sumAll`). Non-deterministic by a
   few ULPs. Acceptable for training (doesn't affect gradient scale in
   any meaningful way) but not for bit-exact reproduction. Decide
   at M1 whether to add a `-Ddeterministic=true` variant later.
2. **LayerNorm variance precision.** `var = mean((x - mean(x))²)` on
   CUDA may drift more than on CPU due to reduction order. Consider
   Welford's online algorithm if we see >2× expected tolerance.
3. **Multi-GPU in benchmark.** Playbook hard-codes device 0. If the
   remote ever gets a second GPU, PR-ξ should not silently use GPU 0;
   already guaranteed by our Stage-7 scope freeze on device 0.
4. **Stage 8 handoff.** After M7, the natural next project is Stage 8
   (N-block transformer). Should we draft a Stage 8 playbook at Stage 7
   close-out, or let the next session start fresh? Playbook §14 suggests
   drafting. Ask the user at the M7 boundary.

---

## Exit criteria — Stage 7 is Done when

- [ ] M1: CE fused kernel + oracle `cross_entropy_3d` parity green.
- [ ] M2: every op the model uses has a CUDA route + at least one
  direct unit test. Confirmed by a grep for `NotImplemented` and
  `error.DeviceMismatch` in the forward path.
- [ ] M3: `examples/08_cuda_vs_cpu.zig` reports forward diff < 5e-4 abs.
- [ ] M4: per-parameter backward diff < 1e-3 abs across the model.
- [ ] M5: post-one-step param diff < 2e-3 abs across the model.
- [ ] M6: `examples/09_cuda_benchmark.zig` reports speedup (actual
  number whatever it is; 30× is aspirational, documenting the real
  number satisfies "done").
- [ ] M7: AGENTS.md, SESSION_GUIDE.md §3, docs/stage7_plan.md §4+§10+§13
  all show Stage 7 complete. Optional git tag `stage-7-complete`.
- [ ] All compute-sanitizer runs clean (0 leaks, 0 memory-access errors).
- [ ] CPU test count unchanged at 267/267 across every commit (no CPU
  regressions during the CUDA work).

---

*End of Stage 7 endgame plan.*
