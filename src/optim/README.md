# `src/optim/`

Optimisers: algorithms that consume a parameter's `.grad` and update
its `.data`. Equivalent to `torch.optim.*`.

## Files

- `optimizer.zig` — `Optimizer` vtable: a struct with `ctx: *anyopaque`
  plus `step`, `zeroGrad`, `deinit` function pointers. Call sites
  type-erase so `Trainer` can be optimiser-agnostic.
- `sgd.zig` — `SGD` with optional momentum and coupled weight decay.
  The minimal baseline; useful for sanity-check training runs.
- `adamw.zig` — `AdamW` with bias correction and decoupled weight
  decay (the standard "decoupled" weight-decay variant from the
  AdamW paper). This is what every transformer in the wild uses.

## If you're new here

Read `docs/04_nn.md` §5 (optimiser mechanics) and
`docs/04b_from_nn_to_training.md` §3 (the optimiser math). The
`beta2 = 0.95` vs `beta2 = 0.999` discussion in
`docs/07_cpu_training.md` §7.11.3 is a cautionary tale about default
values.

## Cross-references

- AdamW update math: `docs/04_nn.md` §5.3
- ParamId-keyed state (Stage 6.5 / PR-ζ):
  `docs/07c_optimizer_state.md`
- CUDA adamw_step kernel: `src/backend/cuda/kernels/adamw.cu`
- Gradient clipping (pre-optimiser step): `docs/07_cpu_training.md`
  §7.4
