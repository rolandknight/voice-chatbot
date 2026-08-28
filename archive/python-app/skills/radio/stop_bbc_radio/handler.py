from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    radio_was_playing = ctx.radio_player.is_playing()
    spotify_was_playing = (
        ctx.spotify_player is not None and ctx.spotify_player.is_playing()
    )
    if radio_was_playing:
        try:
            await asyncio.to_thread(ctx.radio_player.stop)
        except Exception as e:
            logger.warning(f"Radio stop failed: {e}")
    if spotify_was_playing:
        try:
            await asyncio.to_thread(ctx.spotify_player.stop)
        except Exception as e:
            logger.warning(f"Spotify stop (via radio stop) failed: {e}")
    if not radio_was_playing and not spotify_was_playing:
        await params.result_callback("Nothing's playing.")
        return
    await params.result_callback("Stopped.")
