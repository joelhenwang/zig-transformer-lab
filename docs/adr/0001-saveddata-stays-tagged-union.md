# ADR-0001: SavedData stays as a tagged union

## Status

Accepted (2026-05-12)

## Context

During an architecture review (post-Stage 9), we identified that the
`SavedData` tagged union in `src/autograd/node.zig` is a "protocol
without static enforcement" — the mapping from `OpKind` to the correct
`SavedData` variant is a convention documented in comments. A mismatch
produces a runtime error during backward, not a compile-time error
during forward/record.

The proposed alternative: replace the tagged union with per-op
`*anyopaque` pointers. Each op would define its own saved-data struct,
allocated at record time and cast back in the backward function. The
open union disappears; mismatches become structurally impossible.

## Decision

Keep the tagged union. Do not replace SavedData with opaque pointers.

## Rationale

1. **Pedagogical load-bearing role.** `docs/03_autograd.md` explicitly
   teaches the seven saved-data shapes as a single inspectable enum.
   Students see all variants in one place and can trace exactly what
   each op stores. Replacing with opaque pointers removes this teaching
   surface — they'd need to hunt through individual op files to
   understand what's saved, and the cast-from-anyopaque pattern is an
   advanced Zig idiom that obscures rather than teaches.

2. **The op set is frozen.** The project's scoped mission is complete
   (Stages 1-9 shipped). No new ops are planned. The growth pressure
   that would make the union unwieldy does not exist.

3. **Runtime check cost is negligible.** The switch inside each backward
   case is a single branch on an enum tag. In a library that already
   does O(N) gradient accumulation, one branch per op is unmeasurable.

4. **Debugging value.** The tagged union is inspectable in a debugger —
   you can see `.ce_info.logits.shape` directly. Opaque pointers
   require manual casting to inspect, which hurts the pedagogical
   debugging story in `docs/09_debugging.md`.

## Consequences

- New ops (if any are ever added) must add a variant to `SavedData` and
  update the convention comment in `node.zig` lines 161-174.
- Future architecture reviews should not re-propose opaque saved data
  without explicit reconsideration of the pedagogical mission.
- The runtime-check failure mode (wrong variant -> `error.InvalidArgument`
  at backward time) remains the error surface. This is acceptable because
  (a) the op set is frozen and (b) the existing test suite exercises every
  op's backward path.
