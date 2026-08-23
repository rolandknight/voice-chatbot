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


import re
from pathlib import Path

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of roughly chunk_size characters.

    Splits on sentence boundaries and packs whole sentences together up to
    the target size. A single sentence longer than chunk_size is emitted
    whole rather than cut mid-word -- Flash handles long blocks better than
    it handles a severed clause.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def discover_voices(paths: list[Path]) -> list[str]:
    """List available reference .wav filenames across the search paths.

    Missing directories are skipped rather than raising: the vendor clone
    under vendor/chatterbox-tts-server/ is gitignored and may be absent.
    Names are de-duplicated, keeping first-path-wins ordering.
    """
    seen: dict[str, None] = {}
    for path in paths:
        if not path.is_dir():
            continue
        for wav in sorted(path.glob("*.wav")):
            seen.setdefault(wav.name, None)
    return sorted(seen)


def resolve_voice_path(name: str, paths: list[Path]) -> Path:
    """Resolve a reference filename to a concrete path, first match wins."""
    for path in paths:
        candidate = path / name
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"reference voice {name!r} not found. Searched: {searched}")
