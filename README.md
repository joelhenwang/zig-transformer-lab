# zig-transformer-lab

A small, heavily commented, pedagogical Zig 0.16.0 library for training a tiny
one-block, one-head, word-level transformer from scratch. Designed to teach
the internals of PyTorch-like systems — tensor libraries, autograd, training
loops, and CUDA acceleration.

Every line of code is written to teach. Comments explain *why* something works,
not just *what* it does. If you can read this codebase top to bottom and explain
every number that moves through the system, the project has succeeded.

## Hardware and software requirements

- Linux
- Zig 0.16.0 (exact)
- NVIDIA GPU with compute capability >= 8.0 recommended (developed on RTX 4060 Ti, sm_89)
- CUDA Toolkit >= 12 for GPU builds
- Python 3 with numpy and torch for the oracle (optional, in `tools/`)

## Quickstart

```bash
git clone <url>
cd zig-transformer-lab
zig build test
zig build run-example -Dexample=01_tensor_playground
```

## CUDA build

```bash
zig build test -Dcuda=true
zig build run-example -Dexample=06_train_shakespeare -Dcuda=true
```

## Reading order

See `docs/00_overview.md`. Start with `docs/01_zig_primer.md` if you are new to Zig.

## License

MIT. See `LICENSE`.
