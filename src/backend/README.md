# `src/backend/`

Hardware backend implementations. Today: CPU (inlined in
`src/tensor/ops/*.zig`) and CUDA (this folder's `cuda/` subdirectory).
Adding a new backend — say, Metal or ROCm — would slot in here as
another subdirectory.

## Layout

- `cuda/` — NVIDIA CUDA backend. Stage 7 lands the loader, memory
  primitives, and GEMM; Stage 8 adds the debug utilities and multi-head
  support. See `src/backend/cuda/README.md` for the subdirectory
  contents.

## Why a `backend/` subfolder exists

Before Stage 7 there was no `backend/` folder; CPU code lived next to
the op dispatcher. Stage 7 needed CUDA kernel bindings, PTX loader,
and cuBLAS wrapper, none of which belong in the device-agnostic tensor
layer. Pulling them under `backend/cuda/` keeps the tensor layer
device-agnostic (ops just call a dispatch function; the dispatch
function decides whether to run CPU code or call into a backend).

## If you're new here

Read `docs/08_backends_cuda.md` for the CUDA backend architecture and
the row-major → column-major trick. The bridge chapter
`docs/08b_from_cuda_to_training.md` covers what it means to "move a
model to GPU".

## Cross-references

- CUDA design decisions: `docs/00_overview.md` §6 (locked decisions
  D1-D14 table)
- The PTX lifecycle: `docs/08_backends_cuda.md` §7
- CUDA debug workflow: `docs/09_debugging.md` §§6-8
