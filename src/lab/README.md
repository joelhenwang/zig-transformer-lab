# `src/lab/`

High-level training orchestration. The `Trainer` struct in this folder
is what examples actually call; it ties together everything from
`src/data/`, `src/nn/`, `src/optim/`, `src/autograd/`, and (when
enabled) `src/backend/cuda/`.

## Files

- `train.zig` — `Trainer`, `TrainConfig`, `TrainResult`, plus the
  `generate` function for autoregressive text sampling. The Trainer
  supports both CPU and CUDA; Stage 8 M8-b added the `use_cuda` flag
  that flips the whole pipeline onto the GPU.

## If you're new here

Read `docs/07_cpu_training.md` (how the trainer works on CPU) and
then `docs/08b_from_cuda_to_training.md` §4 (the CUDA branch
annotated). The `examples/06_train_shakespeare.zig` and
`examples/10_train_deep.zig` files are the canonical callers.

## Cross-references

- CPU training walkthrough: `docs/07_cpu_training.md` §7.1
- CUDA training walkthrough: `docs/08b_from_cuda_to_training.md` §2
- Generation algorithm: `docs/07_cpu_training.md` §7.5
- Bug catalog (three silent bugs we fixed during Stage 6):
  `docs/07_cpu_training.md` §7.11
