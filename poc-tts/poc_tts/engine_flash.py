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


def resolve_backend(
    requested: str, flashinfer_available: bool, device: str = "cuda"
) -> str:
    """Resolve the inference backend.

    'auto' picks flashinfer only on CUDA -- flashinfer is a CUDA-only kernel
    library, so selecting it for a CPU engine would fail at generation time.
    Otherwise 'auto' falls back to torch SDPA. mlx is never selected
    automatically; upstream marks it experimental and it must be named.
    """
    if requested == "auto":
        if device == "cuda" and flashinfer_available:
            return "flashinfer"
        return "torch"
    if requested not in ("flashinfer", "torch", "mlx"):
        raise ValueError(
            f"invalid backend {requested!r} "
            "(expected auto, flashinfer, torch, or mlx)"
        )
    if requested == "flashinfer":
        if not flashinfer_available:
            raise ValueError("flashinfer requested but not installed")
        if device != "cuda":
            raise ValueError("flashinfer requires a CUDA device")
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


import importlib.util
import logging

import numpy as np
from chatterbox_flash import ChatterboxFlashTTS

logger = logging.getLogger(__name__)


class OutOfMemoryError(Exception):
    """Raised when generation runs out of VRAM, with actionable detail."""


def _flashinfer_available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def _vram_report() -> str:
    """Describe VRAM pressure. A bare 'CUDA out of memory' wastes the reader."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return (
        f"VRAM {free / 2**30:.2f} GB free of {total / 2**30:.2f} GB total; "
        f"this process holds {allocated / 2**30:.2f} GB. Flash needs roughly "
        f"2.0 GB of weights plus about 0.8 GB of working set at float16. "
        f"Check `nvidia-smi` for other processes holding the card."
    )


class FlashEngine:
    """Owns the Chatterbox Flash model for the process lifetime."""

    def __init__(self, engine_cfg: dict, generation_cfg: dict, voice_paths: list[Path]):
        self._engine_cfg = engine_cfg
        self._generation_cfg = generation_cfg
        self._voice_paths = voice_paths
        self._model = None

        self.device = resolve_device(
            engine_cfg.get("device", "auto"), torch.cuda.is_available()
        )
        bf16 = torch.cuda.is_bf16_supported() if self.device == "cuda" else False
        self.dtype = resolve_dtype(engine_cfg.get("dtype", "auto"), self.device, bf16)
        self.backend = resolve_backend(
            engine_cfg.get("backend", "auto"), _flashinfer_available(), self.device
        )
        self.drf_block_size = int(engine_cfg.get("drf_block_size", 16))
        self.sr = 24000

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model once. First call downloads ~3.2 GB."""
        logger.info(
            "loading Chatterbox Flash: device=%s dtype=%s backend=%s block=%d",
            self.device, self.dtype, self.backend, self.drf_block_size,
        )
        if self.backend == "torch":
            logger.info(
                "flashinfer not in use -- running the portable SDPA path, not "
                "the CUDA-graph path the published RTF figures come from."
            )
        self._model = ChatterboxFlashTTS.from_pretrained(
            device=self.device,
            dtype=self.dtype,
            drf_block_size=self.drf_block_size,
        )
        self.sr = getattr(self._model, "sr", 24000)

    def model_info(self) -> dict:
        """Exactly the keys ui/script.js updateModelUI reads.

        type == "flash" makes the existing UI keep exaggeration and CFG
        visible (Flash has both) and force English-only (Flash is English-
        only by construction). No UI branch is needed for either.
        """
        return {
            "loaded": self.loaded,
            "type": "flash",
            "class_name": "ChatterboxFlashTTS",
            "device": self.device,
            "sample_rate": self.sr if self.loaded else None,
            "supports_paralinguistic_tags": False,
            "available_paralinguistic_tags": [],
            "supports_multilingual": False,
            "supported_languages": {"en": "English"},
            "dtype": str(self.dtype).replace("torch.", ""),
            "backend": self.backend,
            "drf_block_size": self.drf_block_size,
        }

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        temperature: float | None = None,
        exaggeration: float | None = None,
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        n_cfm_timesteps: int | None = None,
        chunk_size: int = 120,
        split_text: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text with a reference voice. Returns (mono float32, sr)."""
        if not self.loaded:
            raise RuntimeError("model is not loaded -- call load() first")

        gen = self._generation_cfg
        prompt = str(resolve_voice_path(voice, self._voice_paths))
        chunks = chunk_text(text, chunk_size) if split_text else [t for t in [text.strip()] if t]
        if not chunks:
            raise ValueError("text is empty")

        pieces: list[np.ndarray] = []
        try:
            for chunk in chunks:
                wav = self._model.generate(
                    chunk,
                    audio_prompt_path=prompt,
                    temperature=temperature if temperature is not None else gen["temperature"],
                    exaggeration=exaggeration if exaggeration is not None else gen["exaggeration"],
                    cfg_scale=cfg_scale if cfg_scale is not None else gen["cfg_scale"],
                    num_steps=num_steps if num_steps is not None else gen["num_steps"],
                    n_cfm_timesteps=(
                        n_cfm_timesteps if n_cfm_timesteps is not None
                        else gen["n_cfm_timesteps"]
                    ),
                    backend=self.backend,
                )
                pieces.append(wav.detach().float().cpu().numpy().reshape(-1))
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError(
                f"ran out of VRAM during generation. {_vram_report()}"
            ) from exc

        return np.concatenate(pieces), self.sr
