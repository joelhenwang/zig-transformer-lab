# `src/tensor/`

The core `Tensor` type, shape arithmetic, and every op that takes
tensors in and produces tensors out. This is the lowest layer in the
library that anything downstream (autograd, nn, optim) depends on.

## Files

- `tensor.zig` — the `Tensor` struct plus the `Storage` union
  (`.cpu = CpuStorage` vs `.cuda = DeviceBuffer`). Covers init,
  deinit, view creation, device transfers (`toCuda`, `toCpu`), and
  the `checkInvariants` self-check.
- `shape.zig` — the `Shape` struct (rank + dims) with rank-specific
  constructors (`init1D`, `init2D`, `init3D`, `init4D`), the
  `totalElements` helper, and `equals`. Row-major strides derived
  here.
- `print.zig` — tensor pretty-printer used for debugging. Handles
  rank 1-4, abbreviated display for large tensors.
- `ops/` — one file per op family. See `src/tensor/ops/README.md`.

## If you're new here

Read `docs/02_tensors.md` (mechanics) then
`docs/02b_from_tensors_to_training.md` (concepts). The invariants
chapter `docs/02c_tensor_invariants.md` and storage chapter
`docs/02d_storage_and_views.md` are shorter deep-dives on specific
design decisions.

## Cross-references

- Row-major conventions: `docs/02_tensors.md` §2
- Storage/View split (Stage 6.5 / PR-δ): `docs/02d_storage_and_views.md`
- The five invariants every Tensor must satisfy:
  `docs/02c_tensor_invariants.md`
- Project-specific Shape API (tripped up 3 subagents!):
  `AGENTS.md` §"This project's Shape API"
