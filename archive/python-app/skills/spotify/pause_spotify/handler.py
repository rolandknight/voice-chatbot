from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    try:
        ok = await asyncio.to_thread(ctx.spotify_player.pause)
    except Exception as e:
        logger.warning(f"Spotify pause failed: {e}")
        await params.result_callback("Spotify isn't responding.")
        return
    await params.result_callback("Paused." if ok else "I couldn't pause Spotify.")
