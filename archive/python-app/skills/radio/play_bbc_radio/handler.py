from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from radio import lookup as lookup_station, match_station
from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    raw = (params.arguments.get("station") or "").strip()
    if not raw:
        await params.result_callback("Tell me which BBC station to play.")
        return
    station = lookup_station(raw) or match_station(raw)
    if station is None:
        await params.result_callback(f"I don't have a BBC station called {raw}.")
        return
    if ctx.spotify_player is not None:
        try:
            await asyncio.to_thread(ctx.spotify_player.stop)
        except Exception as e:
            logger.debug(f"cross-stop Spotify failed: {e}")
    try:
        await asyncio.to_thread(ctx.radio_player.start, station)
    except Exception as e:
        logger.warning(f"Radio start failed: {e}")
        await params.result_callback(f"I couldn't start {station.display}.")
        return
    await params.result_callback(f"Playing {station.display}.")
