"""Outbound WebRTC audio track fed from a queue of PCM chunks.

Owns pacing and framing; knows nothing about Realtime events or the engine.
"""

from __future__ import annotations

import asyncio
import fractions
import time
from collections import deque

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from poc_tts_streaming.audio import FRAME_SAMPLES, SAMPLE_RATE, FrameSlicer, silence_frame, to_int16

_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class PcmQueueTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, *, paced: bool = True) -> None:
        super().__init__()
        self._paced = paced
        self._slicer = FrameSlicer()
        self._queue: deque[np.ndarray] = deque()
        self._pts = 0
        self._start: float | None = None
        self._waiters: list[asyncio.Future] = []

    # ---- producer side (event loop thread) --------------------------------

    def push(self, pcm_float32: np.ndarray) -> None:
        self._queue.extend(self._slicer.push(to_int16(pcm_float32)))

    def flush(self) -> None:
        self._queue.extend(self._slicer.flush())

    def clear(self) -> None:
        self._queue.clear()
        self._slicer.clear()
        self._resolve_waiters()

    @property
    def queued_frames(self) -> int:
        return len(self._queue)

    async def drained(self) -> None:
        if not self._queue:
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        await fut

    def _resolve_waiters(self) -> None:
        for fut in self._waiters:
            if not fut.done():
                fut.set_result(None)
        self._waiters.clear()

    # ---- consumer side (aiortc RTP sender) ---------------------------------

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._paced:
            if self._start is None:
                self._start = time.monotonic()
            wait = self._start + self._pts / SAMPLE_RATE - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

        if self._queue:
            samples = self._queue.popleft()
            if not self._queue:
                self._resolve_waiters()
        else:
            samples = silence_frame()

        frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = _TIME_BASE
        self._pts += FRAME_SAMPLES
        return frame
