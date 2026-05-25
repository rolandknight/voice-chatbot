"""BotSpeakingTracker — re-exported here so SFX skills can import it without
pulling in the legacy babel_skills module. Definition moved verbatim from the
old babel_skills.py so behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
from collections import deque

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotSpeakingTracker(FrameProcessor):
    """Watches the bot's TTS lifecycle and lets callers wait for the next
    silence boundary.

    Maintains a depth counter because Pipecat can emit a Started/Stopped pair
    per sentence chunk within one logical turn — we only fire pending events
    when the bot truly returns to idle.
    """

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self._pending: deque[asyncio.Event] = deque()

    def snapshot_next_silence(self) -> asyncio.Event:
        evt = asyncio.Event()
        self._pending.append(evt)
        return evt

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._depth += 1
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                while self._pending:
                    self._pending.popleft().set()
        await self.push_frame(frame, direction)
