# `src/core/`

Foundational types and utilities used across the whole library. Nothing
in this folder depends on anything outside `std`.

## Files

- `device.zig` — the `Device` enum: `.cpu` or `.cuda`. Tiny file, one
  type. Every `Tensor` has a `device` field of this type.
- `dtype.zig` — the `Dtype` enum and helpers. We use `f32`
  exclusively (policy decision — no mixed precision); this file
  exists for future-proofing and documentation.
- `errors.zig` — the `LabError` enum. All public functions in the
  library return `LabError!T`. Centralised so refactors see one place
  when an error category needs to expand.
- `rng.zig` — a small deterministic PRNG. `Rng.init(seed) -> Rng`;
  `rng.random().intRangeLessThan(usize, 0, N)` for index draws,
  `rng.random().float(f32)` for weight init. Seedable for
  reproducible training runs.

## If you're new here

Nothing fancy — these are leaf modules you'll see imported throughout.
`errors.zig` is the most useful one to skim: `LabError` enumerates
every failure category in the whole library.

## Cross-references

- Device routing: `docs/08_backends_cuda.md` §9
- Dtype decision (f32 only, no mixed precision): `docs/00_overview.md`
  decisions D7
- Rng reproducibility in training: `docs/07_cpu_training.md` §7.4
