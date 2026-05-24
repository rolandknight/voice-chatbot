from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    direction = (params.arguments.get("direction") or "next").strip().lower()
    if direction not in {"next", "previous"}:
        direction = "next"
    try:
        if direction == "next":
            ok = await asyncio.to_thread(ctx.spotify_player.skip_next)
            done = "Skipped."
        else:
            ok = await asyncio.to_thread(ctx.spotify_player.skip_previous)
            done = "Went back."
    except Exception as e:
        logger.warning(f"Spotify skip {direction} failed: {e}")
        await params.result_callback("Spotify isn't responding.")
        return
    await params.result_callback(done if ok else "I couldn't skip the track.")
