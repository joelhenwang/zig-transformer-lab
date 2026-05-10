#!/usr/bin/env python3
"""
tools/oracle.py — PyTorch reference implementations for parity testing.

This is the single source of truth for "what SHOULD our CPU ops produce?".
It runs PyTorch on deterministic inputs (fixed seeds) and writes the
inputs, outputs, and gradients to binary fixture files that the Zig test
harness loads and compares against.

File layout produced under `tests/fixtures/`:

    tests/fixtures/
        manifest.json                # list of cases with tolerances
        add_2d/
            meta.json                # op name, shapes, tolerances
            input_0.ztlt             # left operand
            input_1.ztlt             # right operand
            output.ztlt              # forward result
            grad_input_0.ztlt        # ∂loss / ∂a
            grad_input_1.ztlt        # ∂loss / ∂b
        mul_broadcast/
            ...
        matmul_2d/
            ...
        softmax_3d/
            ...
        cross_entropy_3d/
            ...

Fixture binary format (".ztlt" = Zig Transformer Lab Tensor):

    offset  size    field
    ------  ------  -------------------------------------------
      0      4      magic = b"ZTLT"
      4      4      u32 version = 1
      8      1      u8 rank (1..4)
      9      3      u8[3] _pad (zeros)
     12     16      u32[4] dims (unused dims = 0)
     28      4      u32 n_elements (= product of first `rank` dims)
     32   n*4       f32[n_elements] data, row-major, little-endian

All integers little-endian. Floats IEEE-754 little-endian.

Usage:
    python tools/oracle.py generate          # write all fixtures
    python tools/oracle.py generate --case add_2d   # single case
    python tools/oracle.py list              # print cases and paths

The script is deterministic: the same seed + shape + op produces the
same fixture bytes on every platform that has little-endian x86 / arm64
and IEEE-754 f32. Regeneration should produce a byte-identical diff of
zero.

A design note worth reading before extending this script: the oracle
is a TEACHING tool as well as a test tool. Every case should have a
clear name, a short docstring, and tolerances chosen with the
"why this tolerance?" explained. Do not add cases without this
documentation — the goal is that a reader of this script learns
roughly as much as a reader of the corresponding docs/ chapter.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"

# --------------------------------------------------------------------------
# ZTLT binary writer
# --------------------------------------------------------------------------

ZTLT_MAGIC = b"ZTLT"
ZTLT_VERSION = 1
ZTLT_MAX_RANK = 4


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write a torch tensor to a .ztlt file. tensor must be f32 and CPU."""
    if tensor.dtype != torch.float32:
        raise ValueError(f"oracle only emits f32; got {tensor.dtype}")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    rank = tensor.ndim
    if rank < 1 or rank > ZTLT_MAX_RANK:
        raise ValueError(f"rank {rank} out of [1, {ZTLT_MAX_RANK}]")

    dims = list(tensor.shape) + [0] * (ZTLT_MAX_RANK - rank)
    n_elements = int(tensor.numel())
    header = struct.pack(
        "<4sIBBBBIIIII",
        ZTLT_MAGIC,
        ZTLT_VERSION,
        rank,
        0,
        0,
        0,          # 3 pad bytes
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        n_elements,
    )
    # 4 + 4 + 4 (rank+pad) + 16 (dims) + 4 (n_elements) = 32 bytes
    assert len(header) == 32, f"header size {len(header)} != 32"

    data = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    if data.size != n_elements:
        raise ValueError("numpy element count disagrees with tensor")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


# --------------------------------------------------------------------------
# Deterministic random
# --------------------------------------------------------------------------


def make_randn(shape: tuple[int, ...], seed: int, *, requires_grad: bool = False) -> torch.Tensor:
    """Deterministic f32 normal noise, fixed seed, row-major."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    t = torch.empty(shape, dtype=torch.float32).normal_(generator=gen)
    if requires_grad:
        t.requires_grad_(True)
    return t


def make_randint(
    shape: tuple[int, ...], low: int, high: int, seed: int
) -> torch.Tensor:
    """Deterministic int64 tensor in [low, high)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randint(low, high, shape, generator=gen)


# --------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------


@dataclass
class CaseSpec:
    name: str
    op: str
    forward_tol: float
    backward_tol: float
    description: str
    # Lazy builder — runs PyTorch and returns (inputs, output, gradients).
    build: Callable[[Path], dict] = field(repr=False)


def case_add_2d(dir: Path) -> dict:
    """Plain elementwise add, two (2, 3) tensors, no broadcasting."""
    a = make_randn((2, 3), seed=101, requires_grad=True)
    b = make_randn((2, 3), seed=102, requires_grad=True)
    y = a + b
    # For backward we need a scalar loss. Use sumAll so that
    # grad_output = ones_like(y) — this lets us recover dy/da, dy/db
    # directly from the ops' saved data.
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "input_1.ztlt", b.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    write_tensor(dir / "grad_input_1.ztlt", b.grad)
    return {
        "shapes": [list(a.shape), list(b.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
    }


def case_add_broadcast_2d_1d(dir: Path) -> dict:
    """Broadcast add: (2, 3) + (3,) = (2, 3). Checks shape reduction in backward."""
    a = make_randn((2, 3), seed=201, requires_grad=True)
    b = make_randn((3,), seed=202, requires_grad=True)
    y = a + b   # b broadcasts across the batch axis
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "input_1.ztlt", b.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    write_tensor(dir / "grad_input_1.ztlt", b.grad)
    return {
        "shapes": [list(a.shape), list(b.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
    }


def case_mul_broadcast(dir: Path) -> dict:
    """Broadcast mul: (2, 3) * (2, 1) = (2, 3). Exercises saved-data for both sides."""
    a = make_randn((2, 3), seed=301, requires_grad=True)
    b = make_randn((2, 1), seed=302, requires_grad=True)
    y = a * b
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "input_1.ztlt", b.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    write_tensor(dir / "grad_input_1.ztlt", b.grad)
    return {
        "shapes": [list(a.shape), list(b.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
    }


def case_matmul_2d(dir: Path) -> dict:
    """Matmul (M, K) @ (K, N). M=4, K=5, N=3 — asymmetric to catch transpose bugs."""
    a = make_randn((4, 5), seed=401, requires_grad=True)
    b = make_randn((5, 3), seed=402, requires_grad=True)
    y = a @ b  # (4, 3)
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "input_1.ztlt", b.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    write_tensor(dir / "grad_input_1.ztlt", b.grad)
    return {
        "shapes": [list(a.shape), list(b.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
    }


def case_softmax_3d_last_axis(dir: Path) -> dict:
    """Softmax over last axis of a (2, 3, 4) tensor. Our softmax op is row-wise on the last dim."""
    a = make_randn((2, 3, 4), seed=501, requires_grad=True)
    y = torch.softmax(a, dim=-1)
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    return {
        "shapes": [list(a.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
        "softmax_axis": -1,
    }


def case_cross_entropy_3d(dir: Path) -> dict:
    """
    Cross-entropy used as in the trainer: logits (B, T, V), targets (B, T) of ints.

    PyTorch's F.cross_entropy expects logits laid out (N, V) and integer
    targets (N,). Our implementation flattens (B, T, V) -> (B*T, V) and
    (B, T) -> (B*T,) before computing. We mirror that here and write out
    the flattened shapes to keep the fixture 2D.
    """
    B, T, V = 2, 3, 5
    logits = make_randn((B, T, V), seed=601, requires_grad=True)
    targets_i = make_randint((B, T), low=0, high=V, seed=602)

    # Flatten to (B*T, V) and (B*T,)
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets_i.reshape(B * T)

    # Default reduction is 'mean'; match what our trainer uses.
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")
    loss.backward()

    # Write logits as (B, T, V) so Zig sees the same rank the trainer uses.
    write_tensor(dir / "input_0.ztlt", logits.detach())
    # Targets as f32 tensor (our CE API takes f32 targets with @round).
    targets_f = targets_i.to(dtype=torch.float32)
    write_tensor(dir / "input_1.ztlt", targets_f)
    # Scalar loss as a (1,) tensor for easy comparison.
    write_tensor(dir / "output.ztlt", loss.detach().reshape(1))
    # Grad wrt logits in (B, T, V).
    write_tensor(dir / "grad_input_0.ztlt", logits.grad)
    return {
        "shapes": [[B, T, V], [B, T]],
        "output_shape": [1],
        "loss": "direct",
        "vocab_size": V,
        "reduction": "mean",
    }


def case_gelu_2d(dir: Path) -> dict:
    """Exact (erf-based) GELU on a (3, 4) tensor. Our op is geluExact."""
    a = make_randn((3, 4), seed=701, requires_grad=True)
    # torch.nn.functional.gelu with approximate='none' is the erf-based exact form.
    y = torch.nn.functional.gelu(a, approximate="none")
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    return {
        "shapes": [list(a.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
        "variant": "exact (erf-based)",
    }


def case_layernorm_3d(dir: Path) -> dict:
    """LayerNorm over the last dim of a (2, 3, 4) tensor, gamma/beta learnable."""
    D = 4
    a = make_randn((2, 3, D), seed=801, requires_grad=True)
    gamma = make_randn((D,), seed=802, requires_grad=True)
    beta = make_randn((D,), seed=803, requires_grad=True)
    ln = torch.nn.functional.layer_norm(
        a, normalized_shape=(D,), weight=gamma, bias=beta, eps=1e-5
    )
    ln.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "input_1.ztlt", gamma.detach())
    write_tensor(dir / "input_2.ztlt", beta.detach())
    write_tensor(dir / "output.ztlt", ln.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    write_tensor(dir / "grad_input_1.ztlt", gamma.grad)
    write_tensor(dir / "grad_input_2.ztlt", beta.grad)
    return {
        "shapes": [list(a.shape), list(gamma.shape), list(beta.shape)],
        "output_shape": list(ln.shape),
        "loss": "sumAll",
        "eps": 1e-5,
    }


def case_embedding_3d(dir: Path) -> dict:
    """
    Embedding lookup + scatter-add backward.

    Setup: weight (V, D) = (6, 4) float, ids (B, T) = (2, 3) int.
    Some ids repeat — this is the critical stress test for the backward
    pass. When the same vocab row is looked up by two positions, their
    gradients must ACCUMULATE at that row, not overwrite each other.
    A naive scatter (not scatter-add) backward is silently wrong for
    any non-unique id.
    """
    V, D = 6, 4
    weight = make_randn((V, D), seed=901, requires_grad=True)
    # Hand-pick ids with deliberate repeats.
    ids = torch.tensor([[0, 2, 2], [5, 0, 3]], dtype=torch.long)  # (2, 3)
    # torch.nn.functional.embedding returns (B, T, D).
    y = torch.nn.functional.embedding(ids, weight)
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", weight.detach())
    # Store ids as a float tensor (our embedding API takes f32 ids
    # with @round internally, matching cross-entropy target handling).
    ids_f = ids.to(dtype=torch.float32)
    write_tensor(dir / "input_1.ztlt", ids_f)
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", weight.grad)
    return {
        "shapes": [list(weight.shape), list(ids.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
        "vocab_size": V,
        "d_model": D,
        "repeats_present": True,
    }


def case_matmul_batch_3d(dir: Path) -> dict:
    """
    Batched matmul used inside attention: (B, M, K) @ (B, K, N).

    Chosen shapes: B=2, M=3, K=4, N=5. Asymmetric M != N to catch
    row/column-major bugs in the stride computation of the batched
    path. This is the shape of Q @ K^T after K is transposed in
    `CausalSelfAttention.forward`, so getting it right is non-optional
    for transformer training.
    """
    a = make_randn((2, 3, 4), seed=1001, requires_grad=True)
    b = make_randn((2, 4, 5), seed=1002, requires_grad=True)
    y = a @ b  # torch handles (B, M, K) @ (B, K, N) as batched matmul
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "input_1.ztlt", b.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    write_tensor(dir / "grad_input_1.ztlt", b.grad)
    return {
        "shapes": [list(a.shape), list(b.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
    }


def case_log_softmax_3d(dir: Path) -> dict:
    """
    log_softmax over last dim of a (2, 3, 4) tensor.

    log_softmax has a simpler gradient than softmax:
        d log_softmax(x)_i / d x_j = delta_ij - softmax(x)_j
    rather than softmax's product rule. Testing both pins down whether
    our unary.log composition with softmax.softmax agrees with the
    dedicated log_softmax kernel.
    """
    a = make_randn((2, 3, 4), seed=1101, requires_grad=True)
    y = torch.log_softmax(a, dim=-1)
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    return {
        "shapes": [list(a.shape)],
        "output_shape": list(y.shape),
        "loss": "sumAll",
    }


def case_sum_axis_3d(dir: Path) -> dict:
    """Reduction: sum over axis=1 of a (2, 3, 4) tensor. Keepdim=true (matches our API)."""
    a = make_randn((2, 3, 4), seed=1201, requires_grad=True)
    # keepdim=True because our `reduce.sum` preserves rank and sets the
    # reduced dim to 1 (matching how ln.forward uses it).
    y = a.sum(dim=1, keepdim=True)
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    return {
        "shapes": [list(a.shape)],
        "output_shape": list(y.shape),
        "axis": 1,
        "keepdim": True,
        "loss": "sumAll",
    }


def case_mean_axis_3d(dir: Path) -> dict:
    """Reduction: mean over the last axis of a (2, 3, 4) tensor. Keepdim=true."""
    a = make_randn((2, 3, 4), seed=1301, requires_grad=True)
    y = a.mean(dim=-1, keepdim=True)
    y.sum().backward()
    write_tensor(dir / "input_0.ztlt", a.detach())
    write_tensor(dir / "output.ztlt", y.detach())
    write_tensor(dir / "grad_input_0.ztlt", a.grad)
    return {
        "shapes": [list(a.shape)],
        "output_shape": list(y.shape),
        "axis": -1,
        "keepdim": True,
        "loss": "sumAll",
    }


def case_full_model_forward(dir: Path) -> dict:
    """
    The headline end-to-end case: full TinyWordTransformer forward
    parity against a PyTorch re-implementation of the same
    architecture.

    Strategy: build a faithful PyTorch model that uses the SAME
    shapes, SAME initialisation scale (we override it with
    oracle-generated values), and SAME pre-norm 1-block 1-head
    causal attention layout as our Zig `TinyWordTransformer`. We
    generate a random weight set in PyTorch, save every weight to a
    .ztlt file (the Zig test loads them into its model's
    parameters), run forward, save the logits.

    This means the Zig test does NOT need Rng parity with PyTorch —
    it overwrites every parameter with the oracle's bytes before
    running forward, giving us a deterministic "same weights →
    same logits" check.

    Config (tiny): V=8, D=4, T=4, F=8, B=2, bias=true.
    """
    V, D, T, F, B = 8, 4, 4, 8, 2
    torch.manual_seed(1400)

    # Params, all requires_grad=True for completeness (we only test
    # forward output here; backward would require a much larger
    # fixture and is not the goal of this case).
    tok_embed_w = torch.randn(V, D)
    pos_embed_w = torch.randn(T, D)
    ln1_gamma = torch.randn(D)
    ln1_beta = torch.randn(D)
    w_q = torch.randn(D, D)
    w_k = torch.randn(D, D)
    w_v = torch.randn(D, D)
    w_o = torch.randn(D, D)
    ln2_gamma = torch.randn(D)
    ln2_beta = torch.randn(D)
    mlp_fc1 = torch.randn(F, D)
    mlp_fc2 = torch.randn(D, F)
    ln_f_gamma = torch.randn(D)
    ln_f_beta = torch.randn(D)
    lm_head_w = torch.randn(V, D)
    # Biases for the four Linear layers that use bias in our model.
    # Our architecture uses bias=False for attention w_q/w_k/w_v/w_o,
    # bias=False for mlp.fc1/fc2, and bias=False for lm_head.
    # Bias=True is only used in LayerNorms (which already have beta)
    # so actually there are no Linear biases to save here.
    # => no extra bias params to generate.

    # Inputs: token ids (B, T).
    ids = torch.tensor([[0, 3, 5, 2], [7, 1, 4, 6]], dtype=torch.long)  # shape (2, 4)

    # -------- Forward (manual, tracing our own Zig model) --------
    # tok_embed + pos_embed
    tok = torch.nn.functional.embedding(ids, tok_embed_w)     # (B, T, D)
    pos_ids = torch.arange(T).unsqueeze(0)                    # (1, T)
    pos = torch.nn.functional.embedding(pos_ids, pos_embed_w) # (1, T, D)
    x = tok + pos                                             # (B, T, D)

    # ln1
    x_norm1 = torch.nn.functional.layer_norm(x, (D,), weight=ln1_gamma, bias=ln1_beta, eps=1e-5)

    # Single-head causal self-attention
    q = x_norm1 @ w_q.t()  # (B, T, D); our Linear is y = x @ W^T
    k = x_norm1 @ w_k.t()
    v = x_norm1 @ w_v.t()
    scale = 1.0 / (D ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale               # (B, T, T)
    causal = torch.tril(torch.ones(T, T)).unsqueeze(0)       # (1, T, T) — 1 where attend, 0 else
    scores = scores.masked_fill(causal == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn_out = attn @ v                                       # (B, T, D)
    attn_proj = attn_out @ w_o.t()                           # (B, T, D)
    x = x + attn_proj                                         # residual 1

    # ln2 + MLP
    x_norm2 = torch.nn.functional.layer_norm(x, (D,), weight=ln2_gamma, bias=ln2_beta, eps=1e-5)
    h1 = x_norm2 @ mlp_fc1.t()                                # (B, T, F)
    h1 = torch.nn.functional.gelu(h1, approximate="none")
    h2 = h1 @ mlp_fc2.t()                                     # (B, T, D)
    x = x + h2                                                # residual 2

    # Final LN + head
    x = torch.nn.functional.layer_norm(x, (D,), weight=ln_f_gamma, bias=ln_f_beta, eps=1e-5)
    logits = x @ lm_head_w.t()                                # (B, T, V)

    # -------- Write all parameters AND logits --------
    # Naming follows TinyWordTransformer.collectNamedParams so the Zig
    # test can build the parameter list and zip it with the fixture
    # directory in a loop.
    param_pairs = [
        ("tok_embed.weight",        tok_embed_w),
        ("pos_embed.weight",        pos_embed_w),
        ("block.ln1.gamma",         ln1_gamma),
        ("block.ln1.beta",          ln1_beta),
        ("block.attn.w_q.weight",   w_q),
        ("block.attn.w_k.weight",   w_k),
        ("block.attn.w_v.weight",   w_v),
        ("block.attn.w_o.weight",   w_o),
        ("block.ln2.gamma",         ln2_gamma),
        ("block.ln2.beta",          ln2_beta),
        ("block.mlp.fc1.weight",    mlp_fc1),
        ("block.mlp.fc2.weight",    mlp_fc2),
        ("ln_f.gamma",              ln_f_gamma),
        ("ln_f.beta",               ln_f_beta),
        ("lm_head.weight",          lm_head_w),
    ]
    for name, tensor in param_pairs:
        safe_name = name.replace("/", "_")  # not actually needed today
        write_tensor(dir / f"param__{safe_name}.ztlt", tensor)

    # Inputs: ids as f32 for our API.
    write_tensor(dir / "input_0.ztlt", ids.to(dtype=torch.float32))
    # Output: logits (B, T, V).
    write_tensor(dir / "output.ztlt", logits)

    return {
        "shapes": [list(ids.shape)],
        "output_shape": list(logits.shape),
        "loss": "forward_only",
        "config": {
            "vocab_size": V,
            "d_model": D,
            "max_seq_len": T,
            "d_ff": F,
            "bias": True,
        },
        "batch_size": B,
        "param_names": [p[0] for p in param_pairs],
    }


# Registry of cases. Order matters for reproducibility — new cases go at
# the end. Tolerances are chosen per-op:
#   - Pure algebraic ops (add, mul, matmul): 1e-5 absolute, 1e-4 relative.
#     Matches f32 precision limits after a handful of ops.
#   - Non-linear ops (softmax, gelu, layernorm): 5e-5 absolute, 1e-4
#     relative. Slightly looser because transcendental functions amplify
#     rounding error.
#   - Cross-entropy: 1e-4 absolute. The log-sum-exp + NLL path
#     accumulates more rounding; still well within what a real training
#     run tolerates.
CASES: list[CaseSpec] = [
    CaseSpec(
        name="add_2d",
        op="add",
        forward_tol=1e-5,
        backward_tol=1e-5,
        description="(2,3) + (2,3) elementwise, no broadcast",
        build=case_add_2d,
    ),
    CaseSpec(
        name="add_broadcast_2d_1d",
        op="add",
        forward_tol=1e-5,
        backward_tol=1e-5,
        description="(2,3) + (3,) broadcast; exercises backward shape reduction",
        build=case_add_broadcast_2d_1d,
    ),
    CaseSpec(
        name="mul_broadcast",
        op="mul",
        forward_tol=1e-5,
        backward_tol=1e-5,
        description="(2,3) * (2,1) broadcast; exercises saved-tensor_pair copy",
        build=case_mul_broadcast,
    ),
    CaseSpec(
        name="matmul_2d",
        op="matmul",
        forward_tol=1e-4,
        backward_tol=1e-4,
        description="(4,5) @ (5,3) asymmetric — catches row/col-major bugs",
        build=case_matmul_2d,
    ),
    CaseSpec(
        name="softmax_3d_last_axis",
        op="softmax",
        forward_tol=5e-5,
        backward_tol=1e-4,
        description="softmax over last dim of (2,3,4); max-subtraction stability",
        build=case_softmax_3d_last_axis,
    ),
    CaseSpec(
        name="cross_entropy_3d",
        op="cross_entropy",
        forward_tol=1e-4,
        backward_tol=1e-4,
        description="CE over (2,3,5) logits with mean reduction",
        build=case_cross_entropy_3d,
    ),
    CaseSpec(
        name="gelu_2d",
        op="gelu_exact",
        forward_tol=5e-5,
        backward_tol=1e-4,
        description="exact (erf-based) GELU on (3,4); our geluExact",
        build=case_gelu_2d,
    ),
    CaseSpec(
        name="layernorm_3d",
        op="layernorm",
        forward_tol=5e-5,
        backward_tol=2e-4,
        description="LayerNorm over last dim of (2,3,4); learnable gamma, beta",
        build=case_layernorm_3d,
    ),
    CaseSpec(
        name="embedding_3d",
        op="embedding",
        forward_tol=1e-5,
        backward_tol=1e-5,
        description="embedding lookup (V=6, D=4) with repeated ids to stress scatter-add backward",
        build=case_embedding_3d,
    ),
    CaseSpec(
        name="matmul_batch_3d",
        op="matmul_batch",
        forward_tol=1e-4,
        backward_tol=1e-4,
        description="(B, M, K) @ (B, K, N) batched matmul — the Q@K^T and A@V shape in attention",
        build=case_matmul_batch_3d,
    ),
    CaseSpec(
        name="log_softmax_3d",
        op="log_softmax",
        forward_tol=5e-5,
        backward_tol=1e-4,
        description="log_softmax over last dim of (2,3,4); simpler gradient than softmax",
        build=case_log_softmax_3d,
    ),
    CaseSpec(
        name="sum_axis_3d",
        op="sum",
        forward_tol=1e-5,
        backward_tol=1e-5,
        description="sum over axis=1 of (2,3,4) with keepdim=True",
        build=case_sum_axis_3d,
    ),
    CaseSpec(
        name="mean_axis_3d",
        op="mean",
        forward_tol=1e-5,
        backward_tol=1e-5,
        description="mean over last axis of (2,3,4) with keepdim=True",
        build=case_mean_axis_3d,
    ),
    CaseSpec(
        name="full_model_forward",
        op="full_model",
        forward_tol=5e-4,
        backward_tol=0.0,  # forward-only case
        description="end-to-end TinyWordTransformer forward (V=8, D=4, T=4, F=8); weights loaded from oracle",
        build=case_full_model_forward,
    ),
]


# --------------------------------------------------------------------------
# Generation driver
# --------------------------------------------------------------------------


def generate_case(spec: CaseSpec) -> dict:
    dir = FIXTURE_ROOT / spec.name
    # Wipe any old files so the case is self-consistent.
    if dir.exists():
        for f in dir.iterdir():
            if f.is_file():
                f.unlink()
    dir.mkdir(parents=True, exist_ok=True)

    extra = spec.build(dir)
    meta = {
        "name": spec.name,
        "op": spec.op,
        "description": spec.description,
        "forward_tol": spec.forward_tol,
        "backward_tol": spec.backward_tol,
        **extra,
    }
    with open(dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta


def generate_all(only: list[str] | None = None) -> None:
    manifest = {"cases": []}
    for spec in CASES:
        if only and spec.name not in only:
            continue
        print(f"[oracle] generating {spec.name} ({spec.op}) ...")
        meta = generate_case(spec)
        manifest["cases"].append(
            {
                "name": spec.name,
                "op": spec.op,
                "forward_tol": spec.forward_tol,
                "backward_tol": spec.backward_tol,
            }
        )
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[oracle] wrote {len(manifest['cases'])} case(s) under {FIXTURE_ROOT}")
    print(f"[oracle] manifest: {MANIFEST_PATH}")


def list_cases() -> None:
    for spec in CASES:
        print(f"  {spec.name:30s}  op={spec.op:16s}  {spec.description}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference fixtures for Zig parity tests.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Write all (or selected) fixtures.")
    gen.add_argument(
        "--case",
        action="append",
        default=None,
        help="Generate only the named case. Repeat to select multiple.",
    )

    sub.add_parser("list", help="List defined cases.")

    args = parser.parse_args()
    if args.cmd == "generate":
        generate_all(only=args.case)
    elif args.cmd == "list":
        list_cases()


if __name__ == "__main__":
    main()
