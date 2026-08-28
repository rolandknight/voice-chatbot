from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams, LLMService

from skills._context import SkillContext


def _format_duration(minutes: float) -> str:
    if minutes < 1:
        secs = int(round(minutes * 60))
        return f"{secs} seconds"
    if abs(minutes - round(minutes)) < 0.05:
        m = int(round(minutes))
        return f"{m} minute" if m == 1 else f"{m} minutes"
    return f"{minutes:g} minutes"


async def _fire(llm: LLMService, delay: float, fire_label: str) -> None:
    try:
        await asyncio.sleep(delay)
        message = (
            f"Your {fire_label} timer is up." if fire_label
            else "Your timer is up."
        )
        await llm.push_frame(TTSSpeakFrame(text=message), FrameDirection.DOWNSTREAM)
        logger.info(f"Timer fired: {message}")
    except asyncio.CancelledError:
        logger.info("Timer cancelled")
        raise
    except Exception as e:
        logger.warning(f"Timer fire failed: {e}")


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    minutes_raw = params.arguments.get("minutes")
    label = (params.arguments.get("label") or "").strip()
    try:
        minutes = float(minutes_raw)
    except (TypeError, ValueError):
        await params.result_callback("I couldn't understand the timer duration.")
        return
    if minutes <= 0:
        await params.result_callback(
            "The timer duration needs to be greater than zero."
        )
        return

    seconds = minutes * 60.0
    pretty = _format_duration(minutes)
    tail = f" for {label}" if label else ""

    asyncio.create_task(_fire(params.llm, seconds, label))
    await params.result_callback(f"Timer set for {pretty}{tail}.")
