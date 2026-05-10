# zig-transformer-lab

A small, heavily commented, pedagogical Zig 0.16.0 library for training a tiny
one-block, one-head, word-level transformer from scratch. Designed to teach
the internals of PyTorch-like systems — tensor libraries, autograd, training
loops, and (eventually) CUDA acceleration.

Every line of code is written to teach. Comments explain *why* something works,
not just *what* it does. If you can read this codebase top to bottom and explain
every number that moves through the system, the project has succeeded.

## Current status

| Stage | Status |
|-------|--------|
| 1 — Scaffold | Done |
| 2 — CPU tensor foundation | Done |
| 3 — Tape-based autograd | Done |
| 4 — NN layers + optimizers | Done |
| 5 — Tokenizer + data pipeline | Done |
| 6 — End-to-end CPU training | Done |
| **6.5 — CPU hardening + backend seam** | **In progress** |
| 7 — CUDA backend | Blocked on 6.5; not started |
| 8–9 | Not started |

CPU training works today. CUDA is **not** implemented yet — the build flag
`-Dcuda=true` wires up plumbing (nvcc step, dlopen linker paths) but the
`kernel_names` list is empty and `src/backend/` does not exist. See
`AGENTS.md` for the full plan.

## Hardware and software requirements

### For CPU development (current)
- Windows or Linux
- Zig 0.16.0 (exact)
- Python 3 with numpy and torch for the oracle (optional, in `tools/`)

### For future CUDA work (Stage 7+)
- Linux (the Windows CUDA port is not yet wired up)
- NVIDIA GPU with compute capability ≥ 8.0 (developed on RTX 4060 Ti, sm_89)
- CUDA Toolkit ≥ 12

## Quickstart

```bash
git clone <url>
cd zig-transformer-lab
zig build test
zig build run-example -Dexample=01_tensor_playground
zig build run-example -Dexample=06_train_shakespeare
```

## Planned CUDA commands (after Stage 7 lands)

These commands are **not functional yet**. They are listed here so you know
the intended shape of the interface once CUDA is wired up.

```bash
# Stage 7 (planned, not implemented):
zig build test -Dcuda=true
zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true
zig build run-example -Dexample=06_train_shakespeare -Dcuda=true
```

Attempting any of these today will fail: `tests/integration_cuda.zig` does
not exist and the kernel list in `build.zig` is empty.

## Reading order

See `docs/00_overview.md`. Start with `docs/01_zig_primer.md` if you are
new to Zig.

## License

MIT. See `LICENSE`.
