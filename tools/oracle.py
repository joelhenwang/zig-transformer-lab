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
