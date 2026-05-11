# `src/debug/`

Tools for turning silent tensor bugs into loud ones as close to the
origin as possible. Opt-in function calls, not macros; you choose
where to place a trip-wire.

## Files

- `shape.zig` — `assertShape`, `assertRank`, `assertDim`. Panic with
  a rich stderr message when a tensor's shape doesn't match expected.
  Use at layer boundaries and right before ops that would silently
  accept a wrong shape.
- `finite.zig` — `assertFinite`, `hasNaN`, `hasInf`. Device-aware
  (DtoH-copies CUDA tensors for the scan). Use as a trip-wire when
  you suspect numerical blow-up.
- `compare.zig` — `compare(alloc, a, b, opts)` → `CompareReport`.
  Elementwise comparison with `max_abs_diff`, `max_rel_err`,
  `worst_idx`, pass/fail against tolerances. Handles any combination
  of CPU/CUDA tensors.
- `dump.zig` — `dump(alloc, io, path, t)` and `load(alloc, io, path)`.
  Writes a tensor to disk in the `.ztlt` binary format used by the
  PyTorch oracle. Cross-device (CUDA tensors DtoH-copy during dump),
  cross-language (`tools/oracle.py` + `numpy.frombuffer` can read the
  same file).

## If you're new here

Read `docs/09_debugging.md` — the whole chapter is about using this
folder. Each helper has a dedicated section there with worked
examples.

## Cross-references

- Real-use example (catching a multi-head reshape bug):
  `docs/09_debugging.md` §1
- The `.ztlt` format: `docs/09_debugging.md` §4.1 (definition) and
  `tools/oracle.py` (Python-side encoder/decoder)
- Why assertions over macros: `docs/stage8_handoff.md` §1 decision #3
