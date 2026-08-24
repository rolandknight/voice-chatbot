"""Chatterbox Flash engine wrapper.

Owns model lifecycle and every hardware decision. Never imports FastAPI --
that boundary is what lets the server tests run with this module mocked.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from chatterbox_flash import ChatterboxFlashTTS

logger = logging.getLogger(__name__)


# --- device / dtype / backend resolution ------------------------------------

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
    On sm_75 (Turing, e.g. RTX 2060) the bare torch.cuda.is_bf16_supported()
    call returns True, because it counts emulation -- taking that default
    gives emulated (slow) bf16, not a clean failure, which is exactly what
    would make the bug silent. See _bf16_supported() below, which calls
    is_bf16_supported(including_emulation=False) to get the real hardware
    boundary. Auto therefore steps down to float16 on CUDA when native bf16
    isn't available, rather than trusting the library's unconditional
    default or the emulation-inclusive query.
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


# --- text chunking ------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _pack(units: list[str], chunk_size: int) -> list[str]:
    """Pack whole units together up to chunk_size; a unit longer than
    chunk_size is emitted on its own rather than cut mid-word."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= chunk_size:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int, split_on_clauses: bool = True) -> list[str]:
    """Split text into chunks of roughly chunk_size characters.

    Sentences are the unit: each generate() call is an independent draw with
    its own prosody and trailing silence, so anything smaller than a clause
    sounds like a list being read. Whole sentences are packed up to
    chunk_size. A sentence longer than chunk_size is split on clause
    punctuation (, ; :) when split_on_clauses is set -- the cheapest way to
    bring time-to-first-audio down on long sentences -- and otherwise
    emitted whole.
    """
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if split_on_clauses and len(sentence) > chunk_size:
            units.extend(c.strip() for c in _CLAUSE_END.split(sentence) if c.strip())
        else:
            units.append(sentence)
    return _pack(units, chunk_size)


# --- voice discovery ------------------------------------------------------------

_VOICE_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")


def discover_voices(paths: list[Path]) -> list[str]:
    """List available reference audio filenames across the search paths.

    Missing directories are skipped rather than raising: the vendor clone
    under vendor/chatterbox-tts-server/ is gitignored and may be absent, and
    the repo-root voices/ directory -- the source of truth in that case --
    ships only .mp3 files. Flash loads reference clips via librosa, which
    reads wav/mp3/flac/ogg alike, so all four are recognised here. Names are
    de-duplicated and the result is alphabetically sorted -- path order buys
    nothing observable here. (Contrast resolve_voice_path below, which
    genuinely is first-path-wins: it returns the first match it finds while
    walking paths in order.)
    """
    seen: dict[str, None] = {}
    for path in paths:
        if not path.is_dir():
            continue
        for ext in _VOICE_EXTENSIONS:
            for audio in sorted(path.glob(f"*{ext}")):
                seen.setdefault(audio.name, None)
    return sorted(seen)


def resolve_voice_path(name: str, paths: list[Path]) -> Path:
    """Resolve a reference filename to a concrete path, first match wins."""
    for path in paths:
        candidate = path / name
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"reference voice {name!r} not found. Searched: {searched}")


# --- hardware capability probes ------------------------------------------------

class OutOfMemoryError(Exception):
    """Raised when load or generation runs out of VRAM, with actionable detail."""


def _flashinfer_available() -> bool:
    """Whether flashinfer can actually run, not merely import.

    flashinfer JIT-compiles its kernels on first use and needs nvcc. Without a
    CUDA toolkit the package imports cleanly and then fails deep inside the
    first generate(), so importability alone is not enough.

    The capability >= 8 requirement is empirical, and the reason is NOT that
    Turing cannot compile flashinfer. On an RTX 2060 with CUDA 12.4 the JIT
    static-asserts twelve times ("install boost_math then recompile to support
    fp16 reduction") because chatterbox_flash REQUESTS fp16 QK reduction for
    any fp16/bf16 dtype (engines/flashinfer.py), flashinfer's own default for
    that flag is False, and the stock wheel is not built to support it. Turing
    only trips it because it has no native bf16, so the dtype guard correctly
    steps down to fp16.

    Disabling that request DOES make flashinfer compile and run on sm_75 --
    and produces WRONG OUTPUT. Measured over a 36-row sweep: the medium
    benchmark sentence yielded 14.88s of audio in 9 of 12 runs against a
    torch-SDPA median of 5.30s, and an energy profile showed the extra ~9s is
    continuous speech-level signal, not trailing silence. It hallucinates past
    the stop condition. Do not re-enable it by patching that flag.

    Ampere and newer pass bf16, never take the fp16 path, and run flashinfer
    normally -- so capability >= 8 is the right practical gate even though the
    original reasoning behind it was wrong.
    """
    if importlib.util.find_spec("flashinfer") is None:
        return False
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        return False
    if shutil.which("nvcc"):
        return True
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and (Path(root) / "bin" / "nvcc").exists():
            return True
    return (Path("/usr/local/cuda") / "bin" / "nvcc").exists()


def _bf16_supported() -> bool:
    """Whether the GPU supports bfloat16 NATIVELY.

    torch.cuda.is_bf16_supported() defaults to including_emulation=True and
    returns True on sm_75, where bf16 is emulated and slow -- which silently
    defeated this guard. Older torch has no such kwarg, so fall back to the
    real hardware boundary: native bf16 starts at compute capability 8.0.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


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


# --- FlashEngine ----------------------------------------------------------------

class FlashEngine:
    """Owns the Chatterbox Flash model for the process lifetime."""

    def __init__(self, engine_cfg: dict, generation_cfg: dict, voice_paths: list[Path]):
        self._generation_cfg = generation_cfg
        self._voice_paths = voice_paths
        self._model = None

        self.device = resolve_device(
            engine_cfg.get("device", "auto"), torch.cuda.is_available()
        )
        bf16 = _bf16_supported() if self.device == "cuda" else False
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
        try:
            self._model = ChatterboxFlashTTS.from_pretrained(
                device=self.device,
                dtype=self.dtype,
                drf_block_size=self.drf_block_size,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError(
                f"ran out of VRAM loading the model. {_vram_report()}"
            ) from exc
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

    def synthesize_stream(
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
        split_on_clauses: bool = True,
        cancel: threading.Event | None = None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (chunk_text, mono float32 pcm) per sentence chunk, in order.

        Validation (voice, text) happens before the first yield so callers
        can fail fast. Cancellation is checked between chunks: a chunk
        already inside generate() finishes (~1 s tuned) and is discarded by
        the caller. generate() itself cannot be interrupted.
        """
        if not self.loaded:
            raise RuntimeError("model is not loaded -- call load() first")
        gen = self._generation_cfg
        prompt = str(resolve_voice_path(voice, self._voice_paths))
        if split_text:
            chunks = chunk_text(text, chunk_size, split_on_clauses=split_on_clauses)
        else:
            chunks = [t for t in [text.strip()] if t]
        if not chunks:
            raise ValueError("text is empty")

        for chunk in chunks:
            if cancel is not None and cancel.is_set():
                return
            try:
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
            except torch.cuda.OutOfMemoryError as exc:
                raise OutOfMemoryError(
                    f"ran out of VRAM during generation. {_vram_report()}"
                ) from exc
            yield chunk, wav.detach().float().cpu().numpy().reshape(-1)

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
        """Whole-utterance synthesis: synthesize_stream concatenated."""
        pieces = [
            pcm for _, pcm in self.synthesize_stream(
                text, voice,
                temperature=temperature, exaggeration=exaggeration,
                cfg_scale=cfg_scale, num_steps=num_steps,
                n_cfm_timesteps=n_cfm_timesteps, chunk_size=chunk_size,
                split_text=split_text,
            )
        ]
        return np.concatenate(pieces), self.sr
