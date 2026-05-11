# `src/tensor/ops/`

Every tensor op the library supports. Organised by op family (one
file per family). Each op follows the same shape: device-switch on
top, per-device implementation below, tape-record at the end.

## Files

- `elementwise.zig` — `add`, `sub`, `mul`, `div`, `neg`, `addScalar`,
  `mulScalar`. Same-shape and broadcast-aware. CUDA routing detects
  broadcast vs fast-path at op entry.
- `shape_ops.zig` — `reshapeTracked`, `transpose2dTracked`,
  `transposeInner2dTracked`, `transposeAxes12_4dTracked`. Tape-tracked
  variants so backward can reinterpret gradients correctly.
- `reduce.zig` — `sum`, `sumAll`, `sumAxis`, `sumToShape`, `mean`,
  `meanAxis`. Last of these composed from `sumAxis + mulScalar` rather
  than a dedicated kernel.
- `matmul.zig` — `matmul` (2D) and `matmulBatch` (3D, broadcast over
  leading batch dim). Routes to cuBLAS on CUDA.
- `softmax.zig` — `softmax` and `logSoftmax`. Both use the max-subtract
  trick for numerical stability.
- `unary.zig` — `neg`, `exp`, `log`, `geluExact`, `sqrt`. Element-wise
  non-linear ops. Backward formulas in `src/autograd/backward.zig`.
- `loss.zig` — `crossEntropy`. Fused forward-plus-backward-logits
  kernel on CUDA (`ce_fused` in `src/backend/cuda/kernels/ce_loss.cu`).
- `create.zig` — `zerosLike`, `onesLike`. Device-aware factory
  helpers.

## If you're new here

Every op here follows the three-step pattern from
`docs/08_backends_cuda.md` §12: validate shapes, switch on device,
record on tape. Read that section if you're adding a new op.

## Cross-references

- Adding a new op (step-by-step): `docs/08b_from_cuda_to_training.md`
  Exercise 4
- Backward implementations: `src/autograd/backward.zig` (one function
  per OpKind)
- The fused CE kernel: `docs/08_backends_cuda.md` §8 (kernel catalog)
- Why `reshapeTracked` instead of `reshape`: `docs/07_cpu_training.md`
  §7.11.2
