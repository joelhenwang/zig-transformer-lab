# Performance: arch-phase-3 gradient-clip benchmark

## Summary

The `arch-phase-3-complete` refactor replaced per-step HtoD/DtoH
gradient scanning with pure-GPU `sumOfSquaresAll` + `scaleInPlace`
ops. Measured improvement: **10.3%** wall-clock reduction at the
2/2/64 Shakespeare config.

## Results

| Metric | Baseline (26257ee) | Current (arch-phase-3) | Delta |
|--------|-------------------|----------------------|-------|
| Per-step wall-clock | 9.821 ms | 8.808 ms | -1.013 ms (-10.3%) |
| Total (200 steps) | 1964.1 ms | 1761.6 ms | -202.5 ms |
| Final loss | 5.4253 | 5.4253 | identical |
| Final grad_norm | 1.2831 | 1.2829 | within f32 noise |

## Configuration

- **Hardware**: RTX 4060 Ti 16 GB, AMD Ryzen 9 5900XT
- **Model**: 2 layers, 2 heads, D=64 (2/2/64)
- **Dataset**: Shakespeare (V=2000, T=16, B=4)
- **Steps**: 200
- **Build**: `-Doptimize=ReleaseFast -Dcuda=true`
- **Seed**: deterministic (default 1337)
- **Measurement**: single run per config, no warm-up (cold-cache)

## What changed

Before (`26257ee`): gradient clipping in `train.zig` performed:
- Per parameter: `allocator.alloc(host_buf)` + `storage.cuda.copyToHost`
  + host-side sum-of-squares loop
- If clipping fires: `copyToHost` + scale loop + `copyFromHost`
- At 2/2/64 config: ~20 parameters × 2 round-trips = ~40 PCIe transfers/step
- Total per-step DtoH: ~40 KB (parameter data scanned on host)

After (`arch-phase-3`): gradient clipping calls:
- `sumOfSquaresAll(g)` per param: CUDA `mul(t,t)` + `sumAll` + 4-byte DtoH
- `scaleInPlace(g, coeff)` if clipping fires: CUDA `mulScalar` kernel
- Total per-step DtoH: ~80 bytes (one f32 scalar per param)

The 10.3% improvement exceeds the original 5% estimate because the old
code's `allocator.alloc + free` per parameter per step adds host-side
overhead beyond just the PCIe transfers. The pure-GPU path eliminates
both the transfers AND the per-step allocations.

## Numerical parity

Final loss is bit-identical (5.4253) across both runs. Gradient norms
differ by < 0.0002 (f32 reassociation noise in the CUDA reduction
kernel vs. host-side sequential accumulation). Training trajectory is
numerically equivalent.

## Caveats

- Single run (no statistical averaging). GPU frequency was not pinned.
- Cold-cache first run. A warm run might show slightly different results.
- The 2/2/64 config is small enough to be launch-bound; larger models
  would see a proportionally smaller improvement from this optimization.
