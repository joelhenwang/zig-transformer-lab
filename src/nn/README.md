# `src/nn/`

Neural-network building blocks. Every `.zig` file is one layer or
one top-level primitive. Conceptually equivalent to `torch.nn`.

## Files

- `module.zig` — `TransformerConfig` and the `Module` protocol
  (init / forward / parameters / deinit). Defines `NamedParam`
  which the save/load code uses.
- `linear.zig` — `Linear`: `y = x @ W^T + b`. Supports 2D and 3D
  input tensors, optional bias, Kaiming init.
- `embedding.zig` — `Embedding`: gathers rows from a `(V, D)` weight
  matrix by integer index. Forward is a `.embedding` OpKind
  dispatched through the autograd.
- `layernorm.zig` — `LayerNorm`: mean/variance normalisation along
  the last axis with learned `gamma` and `beta`. Composed from ~7
  tape-tracked ops (mean, sub, mul, sqrt, div, add).
- `activations.zig` — `GELU` wrapper (stateless, wraps
  `ops.unary.geluExact`).
- `attention.zig` — `CausalSelfAttention`: multi-head self-attention
  with causal mask. Stage 8 M4 added multi-head support via the
  `(B, T, D) → (B, T, H, d) → (B, H, T, d) → (B·H, T, d)` reshape
  pipeline.
- `mlp.zig` — `MLP`: `fc1 → GELU → fc2`. Standard transformer
  feed-forward block.
- `block.zig` — `TransformerBlock`: pre-norm residual (LN → Attn →
  `+x` → LN → MLP → `+h`). One block per `n_layer`.
- `model.zig` — `TinyWordTransformer`: the end-to-end model.
  `tok_embed + pos_embed → blocks → ln_f → lm_head`. Handles device
  moves (`moveToCuda`, `moveToCpu`), checkpoint save/load (ZTLC v3,
  backward-compatible with v2), and `collectNamedParams` for
  serialisation.

## If you're new here

Read `docs/04_nn.md` (mechanics) then
`docs/04b_from_nn_to_training.md` (concepts). For transformer-specific
math, `docs/05_transformer_math.md` has the full shape trace.

## Cross-references

- PyTorch `nn.Module` parallels: `docs/04b_from_nn_to_training.md`
- Multi-head refactor (Stage 8 M4): `docs/stage8_plan.md` Milestone 4
- Checkpoint format: `docs/07d_checkpoint_format.md`
- Init strategies (Kaiming, Xavier): `docs/04_nn.md` §4.2
