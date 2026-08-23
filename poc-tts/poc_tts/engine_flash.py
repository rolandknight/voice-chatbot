"""Chatterbox Flash engine wrapper.

Owns model lifecycle and every hardware decision. Never imports FastAPI --
that boundary is what lets the server tests run with this module mocked.
"""

from __future__ import annotations

import torch

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class UnsupportedDtypeError(Exception):
    """Raised when a dtype is explicitly requested that the device cannot run."""


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Resolve 'auto' | 'cuda' | 'cpu' against what the machine actually has."""
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA requested but not available on this machine")
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"invalid device {requested!r} (expected auto, cuda, or cpu)")
    return requested


def resolve_dtype(requested: str, device: str, bf16_supported: bool) -> torch.dtype:
    """Resolve the compute dtype.

    chatterbox-flash's from_pretrained defaults to bfloat16 unconditionally.
    On sm_75 (Turing, e.g. RTX 2060) torch.cuda.is_bf16_supported() is False,
    and taking that default gives emulated speeds or a hard failure. Auto
    therefore steps down to float16 on CUDA rather than trusting the library.
    """
    if requested == "auto":
        if device != "cuda":
            return torch.float32
        return torch.bfloat16 if bf16_supported else torch.float16

    if requested not in _DTYPES:
        raise ValueError(
            f"invalid dtype {requested!r} "
            "(expected auto, bfloat16, float16, or float32)"
        )
    if requested == "bfloat16" and device == "cuda" and not bf16_supported:
        raise UnsupportedDtypeError(
            "bfloat16 was requested but this GPU does not support it "
            "(sm_75 and older). Use dtype: auto or dtype: float16."
        )
    return _DTYPES[requested]


def resolve_backend(requested: str, flashinfer_available: bool) -> str:
    """Resolve the inference backend.

    'auto' picks flashinfer when importable, else torch SDPA. mlx is never
    selected automatically -- upstream marks it experimental and it must be
    named explicitly.
    """
    if requested == "auto":
        return "flashinfer" if flashinfer_available else "torch"
    if requested == "flashinfer" and not flashinfer_available:
        raise ValueError("flashinfer requested but not installed")
    if requested not in ("flashinfer", "torch", "mlx"):
        raise ValueError(
            f"invalid backend {requested!r} "
            "(expected auto, flashinfer, torch, or mlx)"
        )
    return requested
