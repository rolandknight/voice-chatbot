from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    query = (params.arguments.get("query") or "").strip()
    kind = (params.arguments.get("kind") or "any").strip().lower() or "any"
    if ctx.radio_player is not None and ctx.radio_player.is_playing():
        try:
            await asyncio.to_thread(ctx.radio_player.stop)
        except Exception as e:
            logger.debug(f"cross-stop radio failed: {e}")
    # Route this session's Spotify audio into its pipeline output (WebRTC peer /
    # local Jabra) rather than a local speaker. `start()` must run on the event
    # loop; `set_pcm_sink` is thread-safe. ensure_audio_sink (inside
    # search_and_play) then starts librespot + the reader feeding this injector.
    if ctx.spotify_injector is not None:
        ctx.spotify_injector.start()
        ctx.spotify_player.set_pcm_sink(ctx.spotify_injector.feed)
    try:
        _ok, spoken = await asyncio.to_thread(
            ctx.spotify_player.search_and_play, query, kind
        )
    except Exception as e:
        logger.warning(f"Spotify search_and_play failed: {e}")
        await params.result_callback("Spotify isn't responding.")
        return
    await params.result_callback(spoken)
