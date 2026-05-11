# `src/autograd/`

Tape-based reverse-mode automatic differentiation. The autograd engine
is what turns the forward pass of a computation into a backward pass
that fills in `param.grad` for every trainable tensor.

## Mission

Record each op's inputs, outputs, and kind as the forward pass runs;
replay the graph in reverse during `backward()` to compute gradients.
This is the zig-transformer-lab equivalent of `torch.autograd` —
dynamic, tape-based, device-aware.

## Files

- `tape.zig` — the `Tape` struct. Records forward-pass ops as nodes,
  owns the backward-only `SavedData` snapshots and the accumulated
  gradients. One tape per training step; explicit `deinit`.
- `node.zig` — the `Node` struct (one per op) and the `OpKind` enum
  (every op that has a backward). Also defines `SavedData`, the
  tagged union that captures exactly what each op's backward will
  need.
- `backward.zig` — one `backwardX` function per `OpKind`. Each
  consumes `grad_out` from the downstream op and produces
  `grad_input` tensors for each input. Device-aware (routes to CUDA
  kernels when gradients live on GPU).
- `gradcheck.zig` — finite-difference gradient checker. Used in tests
  to verify each op's `backwardX` matches numerical differentiation
  of the forward.

## If you're new here

Read `docs/03_autograd.md` (mechanics) first, then
`docs/03b_from_autograd_to_training.md` (conceptual bridge). The
`SavedData` pattern is explained in detail in
`docs/03c_saved_tensors.md` — which also documents why we snapshot by
value instead of holding pointers.

## Cross-references

- Tape lifetime in a training loop: `docs/07_cpu_training.md` §7.10
- Backward on CUDA: `docs/08_backends_cuda.md` §12
- Adding a new op (CPU + CUDA + backward + gradcheck):
  `docs/08b_from_cuda_to_training.md` Exercise 4
