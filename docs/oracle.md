# PyTorch Oracle — Reference

A short, practical reference for the PyTorch oracle added after
Stage 6.5. For the conceptual "why parity testing matters", see
the existing chapters `docs/02c_tensor_invariants.md` and
`docs/03_autograd.md`.

---

## What it is

A pair of tools that compare our CPU op implementations against
PyTorch, element by element, on deterministic inputs:

1. `tools/oracle.py` — Python script that runs a battery of ops
   with PyTorch and writes the inputs, outputs, and gradients to
   binary `.ztlt` files under `tests/fixtures/`.
2. `tests/integration_oracle.zig` — Zig test suite that loads the
   fixtures, runs our implementation on the same inputs, and
   asserts the results match within tolerance.

The two are wired together via `zig build test-oracle`.

---

## Quick start

```bash
# 1. Generate fixtures (once, or whenever you add/change a case).
python tools/oracle.py generate

# 2. Run the parity tests.
zig build test-oracle
```

If either step fails, the error tells you exactly where:

- Python errors → your oracle case's PyTorch code is wrong.
- `error.IoError` at fixture load → you forgot to run the Python
  generator, or the `.ztlt` file is corrupted.
- `error.NumericalError` from `oracle.expectClose` → your Zig
  implementation disagrees with PyTorch beyond tolerance. The error
  message includes both the measured `max_abs_diff` and
  `max_rel_err` so you can tell how far off you are.

---

## The fixture format (`.ztlt`)

Each tensor is serialised as one binary file:

```
offset   size   field
  0       4     magic "ZTLT"
  4       4     u32 version = 1
  8       1     u8 rank (1..4)
  9       3     u8[3] _pad (zeros)
 12      16     u32[4] dims (unused = 0)
 28       4     u32 n_elements
 32    n*4      f32[n_elements] little-endian
```

Always f32, always row-major, always little-endian. The format is
documented in detail inside `tools/oracle.py` and handled by
`src/testing/oracle.zig:loadTensor`.

Per case, the generator produces a directory:

```
tests/fixtures/<case_name>/
    meta.json
    input_0.ztlt    input_1.ztlt    ...
    output.ztlt
    grad_input_0.ztlt    grad_input_1.ztlt    ...
```

A `manifest.json` at the root lists every case with its tolerances,
making it easy to inspect without running Python.

---

## Current cases

| Case | Op | Shapes | Notes |
|---|---|---|---|
| `add_2d` | add | (2,3) + (2,3) | Plain elementwise, no broadcast |
| `add_broadcast_2d_1d` | add | (2,3) + (3,) | Backward must reduce `b`'s grad from (2,3) to (3,) |
| `mul_broadcast` | mul | (2,3) * (2,1) | Exercises saved-tensor_pair copy in PR-ε |
| `matmul_2d` | matmul | (4,5) @ (5,3) | Asymmetric MxK@KxN catches transpose/row-col bugs |
| `softmax_3d_last_axis` | softmax | (2,3,4) over dim=-1 | Row-wise max-subtraction stability |
| `cross_entropy_3d` | cross_entropy | logits (2,3,5), targets (2,3) | Flattened to (6,5)/(6,); mean reduction |
| `gelu_2d` | gelu_exact | (3,4) | Our `geluExact` matches PyTorch's `gelu(approximate='none')` |
| `layernorm_3d` | layernorm | (2,3,4), D=4 | End-to-end composed LayerNorm, all three param grads |
| `embedding_3d` | embedding | weight (6,4), ids (2,3) | Scatter-add backward with deliberate id repeats |
| `matmul_batch_3d` | matmul_batch | (2,3,4) @ (2,4,5) | Batched matmul, the shape used inside attention |
| `log_softmax_3d` | log_softmax | (2,3,4) over dim=-1 | Simpler gradient than softmax |
| `sum_axis_3d` | sum | (2,3,4) sum(axis=1) | Reduction with keepdim |
| `mean_axis_3d` | mean | (2,3,4) mean(axis=-1) | Reduction with keepdim |
| `full_model_forward` | full_model | V=8, D=4, T=4, F=8 | End-to-end TinyWordTransformer logits parity |

Tolerances per case are in `tests/fixtures/<case>/meta.json`.

The `full_model_forward` case is the integration test: PyTorch
generates one full set of model weights, saves each of the 15 named
parameters as a `.ztlt` file, runs a forward pass, and saves the
logits. The Zig test loads every parameter into a fresh
`TinyWordTransformer`, runs our forward, and asserts the logits
match within `5e-4` absolute tolerance. This catches any subtle
composition bug (wrong residual order, swapped Q/K transpose,
off-by-one in the causal mask) that the individual op tests would
miss.

---

## Tolerance philosophy

Every op has `forward_tol` and `backward_tol` values:

- **1e-5 absolute, 1e-4 relative** for pure algebraic ops (add, mul,
  matmul). This is near the f32 precision limit after a handful of
  rounding steps.
- **5e-5 absolute, 1e-4 relative** for non-linear ops (softmax,
  gelu, layernorm). Slightly looser because transcendental
  functions amplify rounding.
- **1e-4 absolute, 1e-4 relative** for cross-entropy. The
  log-sum-exp + NLL path accumulates more error.

`oracle.expectClose` passes if `max_rel_err < rel_tol OR max_abs_diff
< abs_tol`. This OR composition (rather than AND) is the pragmatic
choice: relative error is meaningless near zero, absolute error is
meaningless for large values. Either condition being satisfied
means the answer is "close enough" in at least one sensible metric.

This matches the pattern in `src/autograd/gradcheck.zig` (see
AGENTS.md gotcha #37 for the full reasoning).

---

## Adding a new case

1. Write a `case_<name>(dir)` function in `tools/oracle.py` that:
   - Creates input tensors with `make_randn(shape, seed=...)` and
     `requires_grad=True` for anything you want gradients for.
   - Runs the forward op.
   - Calls `.sum().backward()` to get gradients.
   - Writes inputs, output, and grad tensors via `write_tensor`.
   - Returns a dict of extra metadata for `meta.json`.
2. Add a `CaseSpec` entry to the `CASES` list with sensible tols.
3. Run `python tools/oracle.py generate --case <name>` to regenerate
   only your case (or `generate` without `--case` to regenerate all).
4. Add a matching test to `tests/integration_oracle.zig` that:
   - Loads the inputs, expected outputs, and expected grads.
   - Builds a tape, runs our op, calls `backwardThroughSum`.
   - Calls `oracle.expectClose` on forward and each grad.
5. Run `zig build test-oracle` to verify.

The tight coupling between Python case and Zig test is intentional:
both ends read the same fixture files, so there's no place for a
mismatch to hide. If you forget step 4, the Python case generates
but is never checked — a harmless silent failure. If you forget
step 1, the Zig test fails at `error.IoError` during fixture load.

---

## When to regenerate

- You changed a case's shape, seed, or tolerance.
- You added a new case.
- PyTorch got a minor version bump and you want to re-pin.

The fixture files are checked into git (they're small, ~7 KB total
today) so CI and fresh clones get the exact bytes the tests were
written against. If regeneration produces a non-empty diff, commit
the new bytes — that's expected after genuine changes.

If regeneration produces a non-empty diff without you changing
anything, it means PyTorch's numeric result drifted slightly between
versions. Usually harmless; sometimes surprising. Investigate before
committing.

---

## Common mistakes

### `error.IoError` on every case after a pull

Someone updated `tools/oracle.py` without regenerating fixtures.
Run `python tools/oracle.py generate`.

### Forward passes, backward fails with `error.NumericalError`

Your forward op is correct (or within tolerance) but the gradient
is wrong. Likely culprits:

- `saved = ...` in your op's `tape.record(...)` is missing a
  tensor backward needs.
- The backward function in `src/autograd/backward.zig` has a sign
  error or shape bug.
- A broadcast's backward isn't reducing the gradient to the input's
  shape (common with `add_broadcast`).

Print `max_abs_diff` and `max_rel_err` from the failure message —
they tell you roughly how far off. Off by a factor of 2 often means
a `1/2` factor missing; off by a sign means a negation bug.

### `error.IoError: magic mismatch`

The fixture file was produced by a different format version, or
the file is truncated. Regenerate: `python tools/oracle.py generate
--case <name>`.

### Tolerances feel arbitrary

They are empirical: we pick tight tolerances that pass on known-
good implementations and loose enough that legitimate f32 rounding
doesn't produce spurious failures. If you add a new op whose output
is sensitive to reduction order (e.g., large sums), you may need a
looser tolerance — document the reason in `meta.json`.

---

## File reference

| File | Role |
|---|---|
| `tools/oracle.py` | PyTorch fixture generator. The single source of truth for "what should the answer be?" |
| `tools/requirements.txt` | `numpy`, `torch` |
| `tests/fixtures/<case>/` | Binary fixtures and JSON metadata per case |
| `tests/fixtures/manifest.json` | Case list with tolerances |
| `src/testing/oracle.zig` | Zig loader, `loadTensor`, `expectClose`, `maxAbsDiff`, `maxRelErr` |
| `tests/integration_oracle.zig` | Parity tests; one test per case |
| `build.zig` | `test-oracle` step wiring |

---

## Relationship to Stage 7

When Stage 7 adds the CUDA backend, every CUDA op gets the same
treatment: a parity test comparing its output against the CPU
reference, which is in turn verified against the PyTorch oracle.
The chain is:

```
PyTorch (f64 or f32 reference) →(tolerance)→ Our CPU op
                                                    ↓
                                                (tolerance)
                                                    ↓
                                              Our CUDA op
```

The oracle parity tests lock down the CPU path. Stage 7's CPU/CUDA
parity tests lock down the CUDA path against CPU. Together, they
ensure CUDA output never drifts from PyTorch's reference by more
than the sum of the two tolerance budgets.

This is why the oracle was written before Stage 7 rather than
alongside it.
