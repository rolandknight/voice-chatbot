from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    spotify_was_playing = ctx.spotify_player.is_playing()
    radio_was_playing = (
        ctx.radio_player is not None and ctx.radio_player.is_playing()
    )
    if spotify_was_playing:
        if ctx.spotify_injector is not None:
            ctx.spotify_player.clear_pcm_sink(ctx.spotify_injector.feed)
        try:
            await asyncio.to_thread(ctx.spotify_player.stop)
        except Exception as e:
            logger.warning(f"Spotify stop failed: {e}")
    if radio_was_playing:
        try:
            await asyncio.to_thread(ctx.radio_player.stop)
        except Exception as e:
            logger.warning(f"Radio stop (via spotify stop) failed: {e}")
    if not spotify_was_playing and not radio_was_playing:
        await params.result_callback("Nothing's playing.")
        return
    await params.result_callback("Stopped.")
