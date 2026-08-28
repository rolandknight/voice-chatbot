"""SkillFilterProcessor — swaps the LLM's tool set per turn.

Sits between the user context aggregator and the LLM. On every LLMContextFrame
flowing downstream, reads the most recent user message, asks the registry for
the relevant subset, and calls `context.set_tools(...)` before letting the
frame through to the LLM. The mutation is in-memory and synchronous; we never
await or do I/O on the latency-critical path.
"""

from __future__ import annotations

import time

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from skills._loader import SkillRegistry


class SkillFilterProcessor(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        registry: SkillRegistry,
        k: int = 15,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._context = context
        self._registry = registry
        self._k = k
        self._debug = debug

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            t0 = time.perf_counter()
            transcript = _latest_user_text(self._context)
            schemas = self._registry.filter_for_turn(transcript, self._k)
            self._context.set_tools(ToolsSchema(standard_tools=schemas))
            if self._debug:
                elapsed_us = (time.perf_counter() - t0) * 1e6
                logger.debug(
                    f"SkillFilter: {len(schemas)} tools in {elapsed_us:.0f}µs "
                    f"({[s.name for s in schemas]})"
                )
        await self.push_frame(frame, direction)


def _latest_user_text(context: LLMContext) -> str:
    for msg in reversed(context.messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for piece in content:
                if isinstance(piece, dict):
                    txt = piece.get("text") or piece.get("content") or ""
                    if isinstance(txt, str):
                        parts.append(txt)
                elif isinstance(piece, str):
                    parts.append(piece)
            return " ".join(parts).strip()
    return ""
