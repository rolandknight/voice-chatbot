"""PCM helpers shared by the WebRTC track and the chunked-PCM endpoint.

Everything here is numpy-only: no aiortc, no torch.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16, clipped. Same scaling as poc-tts's WAV path."""
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


class FrameSlicer:
    """Re-frame arbitrary-length int16 PCM into exact FRAME_SAMPLES frames.

    aiortc stamps every RTP packet cut from one AudioFrame with the same
    timestamp, so frames larger than 20 ms lose all but one packet
    (docs/web-rtc.md). Exact 480-sample frames are therefore a hard rule,
    and this is the one place that rule is enforced.
    """

    def __init__(self) -> None:
        self._carry = np.zeros(0, dtype=np.int16)

    def push(self, pcm_int16: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._carry, np.asarray(pcm_int16, dtype=np.int16)])
        n_full = len(buf) // FRAME_SAMPLES
        frames = [buf[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n_full)]
        self._carry = buf[n_full * FRAME_SAMPLES:]
        return frames

    def flush(self) -> list[np.ndarray]:
        if len(self._carry) == 0:
            return []
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        frame[:len(self._carry)] = self._carry
        self._carry = np.zeros(0, dtype=np.int16)
        return [frame]

    def clear(self) -> None:
        self._carry = np.zeros(0, dtype=np.int16)
