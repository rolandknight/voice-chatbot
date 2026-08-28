from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    try:
        text = await asyncio.to_thread(ctx.spotify_player.now_playing)
    except Exception as e:
        logger.warning(f"Spotify now_playing failed: {e}")
        await params.result_callback("Spotify isn't responding.")
        return
    if not text:
        await params.result_callback("Nothing's playing on Spotify.")
        return
    await params.result_callback(f"This is {text}.")
