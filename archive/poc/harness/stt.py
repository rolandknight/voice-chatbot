"""faster-whisper transcription of captured bot audio (content assertions).

tiny.en on CPU int8 — plenty for short TTS confirmations. Model is loaded
lazily on first use and cached for the whole pytest run.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel("tiny.en", device="cpu", compute_type="int8")


def transcribe(pcm: bytes, rate: int = 16000) -> str:
    """s16 mono PCM -> lowercase transcript text."""
    from .audio import pcm_to_float, resample

    if rate != 16000:
        pcm = resample(pcm, rate, 16000)
    samples = pcm_to_float(pcm)
    if len(samples) == 0:
        return ""
    segments, _info = _model().transcribe(samples, language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip().lower()
