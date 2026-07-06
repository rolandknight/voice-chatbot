from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    name = (params.arguments.get("name") or "").strip()
    if ctx.radio_player is not None and ctx.radio_player.is_playing():
        try:
            await asyncio.to_thread(ctx.radio_player.stop)
        except Exception as e:
            logger.debug(f"cross-stop radio failed: {e}")
    # Route audio into this session's pipeline output (see play_spotify).
    if ctx.spotify_injector is not None:
        ctx.spotify_injector.start()
        ctx.spotify_player.set_pcm_sink(ctx.spotify_injector.feed)
    try:
        _, spoken = await asyncio.to_thread(ctx.spotify_player.play_playlist, name)
    except Exception as e:
        logger.warning(f"Spotify play_playlist failed: {e}")
        await params.result_callback("Spotify isn't responding.")
        return
    await params.result_callback(spoken)
