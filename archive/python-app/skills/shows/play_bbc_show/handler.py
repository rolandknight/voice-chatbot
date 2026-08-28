from __future__ import annotations

import asyncio
import time

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from bbc_shows import resolve_show_episode
from radio import Station
from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    show = (params.arguments.get("show") or "").strip()
    date = (params.arguments.get("date") or "").strip() or None
    query = (params.arguments.get("query") or "").strip() or None
    if not show:
        await params.result_callback("Tell me which BBC show to play.")
        return
    try:
        episode = await resolve_show_episode(show, date=date, query=query)
    except Exception as e:
        logger.warning(f"Show resolve failed for {show!r}: {e}")
        await params.result_callback(
            f"I couldn't find {show} on BBC Sounds right now."
        )
        return
    if episode is None:
        if date:
            await params.result_callback(
                f"I couldn't find an episode of {show} for that date."
            )
        elif query:
            await params.result_callback(
                f"I couldn't find a {show} episode about {query}."
            )
        else:
            await params.result_callback(
                f"I couldn't find {show} on BBC Sounds."
            )
        return
    playable = Station(
        key=f"show_{int(time.time())}",
        display=episode.display,
        url=episode.url,
        aliases=(),
    )
    if ctx.spotify_player is not None:
        try:
            await asyncio.to_thread(ctx.spotify_player.stop)
        except Exception as e:
            logger.debug(f"cross-stop Spotify failed: {e}")
    try:
        await asyncio.to_thread(ctx.radio_player.start, playable)
    except Exception as e:
        logger.warning(f"Show start failed: {e}")
        await params.result_callback(f"I couldn't play {episode.display}.")
        return
    await params.result_callback(f"Playing {episode.display}.")
