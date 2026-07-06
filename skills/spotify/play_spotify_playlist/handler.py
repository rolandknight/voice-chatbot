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
    # Playback happens natively on the client's librespot endpoint (see
    # play_spotify); we only issue the Web API command.
    try:
        _, spoken = await asyncio.to_thread(ctx.spotify_player.play_playlist, name)
    except Exception as e:
        logger.warning(f"Spotify play_playlist failed: {e}")
        await params.result_callback("Spotify isn't responding.")
        return
    await params.result_callback(spoken)
