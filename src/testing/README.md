# `src/testing/`

Shared test utilities and helpers. Code here supports `zig build test`
and `zig build test-oracle`; none of it is used at runtime.

## Files

- `oracle.zig` — the `expectClose` comparator used by the PyTorch
  oracle integration tests. Reads `.ztlt` fixture files from
  `tests/fixtures/`, runs our CPU / CUDA implementation against the
  PyTorch reference, asserts agreement within tolerance bands.

## If you're new here

Read `docs/oracle.md` for the oracle workflow: generating fixtures
from `tools/oracle.py`, loading them in Zig, the tolerance-band
policy (`rel_tol`, `abs_tol`, `denom_floor`). Fixtures are checked
into `tests/fixtures/` and regenerated on demand.

## Cross-references

- Oracle workflow: `docs/oracle.md`
- Tolerance-band decisions: `docs/stage7_endgame_plan.md` §"Tolerance
  bands"
- Adding a new oracle case (PyTorch side): `tools/oracle.py`
- Gradient-check vs oracle parity (two different tools):
  `docs/09_debugging.md` §3
