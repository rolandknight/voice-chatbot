"""Audio helpers: WAV I/O, silence, resampling, RMS speech detection.

All PCM is little-endian s16 mono bytes unless stated otherwise.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np
from scipy.signal import resample_poly

SPEECH_RMS_THRESHOLD = 500  # int16 RMS; conservative "not silence" floor


def load_wav(path: str | Path) -> tuple[bytes, int]:
    """Read a mono s16 WAV -> (pcm bytes, sample rate)."""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        return w.readframes(w.getnframes()), w.getframerate()


def save_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    """Write mono s16 PCM to a WAV file."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def silence(ms: float, rate: int) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)


def duration_s(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Polyphase-resample s16 mono PCM between rates."""
    if src_rate == dst_rate:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16)
    from math import gcd

    g = gcd(src_rate, dst_rate)
    y = resample_poly(x.astype(np.float64), dst_rate // g, src_rate // g)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def frame_rms(pcm: bytes, rate: int, frame_ms: int = 20) -> np.ndarray:
    """Per-frame int16 RMS values."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n = int(rate * frame_ms / 1000)
    if n == 0 or len(x) == 0:
        return np.array([])
    frames = x[: len(x) - len(x) % n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    return np.sqrt((frames**2).mean(axis=1))


def first_audio_ts(
    pcm: bytes, rate: int, threshold: float = SPEECH_RMS_THRESHOLD, frame_ms: int = 20
) -> Optional[float]:
    """Offset (seconds) of the first frame whose RMS exceeds threshold, or None."""
    rms = frame_rms(pcm, rate, frame_ms)
    hits = np.nonzero(rms > threshold)[0]
    return float(hits[0] * frame_ms / 1000) if len(hits) else None


def last_audio_ts(
    pcm: bytes, rate: int, threshold: float = SPEECH_RMS_THRESHOLD, frame_ms: int = 20
) -> Optional[float]:
    """Offset (seconds) of the end of the last frame whose RMS exceeds threshold."""
    rms = frame_rms(pcm, rate, frame_ms)
    hits = np.nonzero(rms > threshold)[0]
    return float((hits[-1] + 1) * frame_ms / 1000) if len(hits) else None


def has_speech(pcm: bytes, rate: int, threshold: float = SPEECH_RMS_THRESHOLD) -> bool:
    return first_audio_ts(pcm, rate, threshold) is not None


def median_f0(
    pcm: bytes,
    rate: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_ms: int = 40,
    hop_ms: int = 20,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_corr: float = 0.5,
) -> Optional[float]:
    """Median fundamental frequency (Hz) over voiced frames, via autocorrelation.

    Coarse by design — used to tell speakers apart (e.g. cloned male voice
    vs Kokoro af_heart), not for pitch tracking.
    """
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n = int(rate * frame_ms / 1000)
    hop = int(rate * hop_ms / 1000)
    lag_min = max(1, int(rate / fmax))
    lag_max = min(int(rate / fmin), n - 1)
    f0s = []
    for start in range(0, len(x) - n, hop):
        frame = x[start : start + n]
        if np.sqrt((frame**2).mean()) < rms_threshold:
            continue  # not voiced enough
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[n - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
        if ac[lag] < min_corr:
            continue
        f0s.append(rate / lag)
    return float(np.median(f0s)) if f0s else None


async def collect_speech(
    read: Callable[[float], Awaitable[Optional[bytes]]],
    rate: int = 16000,
    timeout: float = 30.0,
    trailing_silence_s: float = 1.5,
    threshold: float = SPEECH_RMS_THRESHOLD,
) -> bytes:
    """Capture one utterance from a bot-audio read function.

    `read(timeout)` must return new PCM bytes when available, None on
    timeout, and b"" once the stream has truly ended (connection closed) —
    see FlowCatAdapter.read_bot_audio. Waits up to `timeout` for speech to
    start (RMS over threshold), then returns once `trailing_silence_s`
    elapses with no further speech (silent frames or no frames at all).
    Repeatable: breaking out here never tears down the underlying stream,
    so sequential utterances (greeting, then reply) segment cleanly.
    """
    buf = bytearray()
    deadline = time.monotonic() + timeout
    speech_seen = False
    last_speech = 0.0
    while True:
        now = time.monotonic()
        if speech_seen and now - last_speech > trailing_silence_s:
            break
        wait = (last_speech + trailing_silence_s - now) if speech_seen else (deadline - now)
        if wait <= 0:
            if not speech_seen:
                raise TimeoutError(f"no bot speech within {timeout}s")
            break
        chunk = await read(min(wait, 0.25))
        if chunk is None:
            continue  # read timed out; loop re-checks the deadlines
        if chunk == b"":
            if not speech_seen:
                raise TimeoutError("audio stream ended before bot speech")
            break
        buf.extend(chunk)
        if has_speech(chunk, rate, threshold):
            speech_seen = True
            last_speech = time.monotonic()
    return bytes(buf)
