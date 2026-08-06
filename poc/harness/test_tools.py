"""Tool-selection tests T3/T4 (poc/CONTRACT.md): indirect phrasing + media.

Same pattern as smoke: greeting consumed by the session fixture, then
fixture in -> stub call log + spoken reply out. Stub call args are the
primary signal; audio content is secondary (plan §8 self-reference risk).
"""

from __future__ import annotations

import pytest

TOOL_TIMEOUT_S = 60.0
REPLY_TIMEOUT_S = 60.0


@pytest.mark.tools
async def test_t3_music_indirect(session, stubs):
    """T3: "Put some music on." -> play_spotify with some query."""
    await session.send_fixture("t3_music.wav")
    call = await session.wait_for_tool("play_spotify", timeout=TOOL_TIMEOUT_S)
    assert isinstance(call["args"].get("query"), str)
    await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)


@pytest.mark.tools
async def test_t3_news_indirect(session, stubs):
    """T3: "I'd like to listen to the news." -> play_bbc_radio."""
    await session.send_fixture("t3_news.wav")
    call = await session.wait_for_tool("play_bbc_radio", timeout=TOOL_TIMEOUT_S)
    assert isinstance(call["args"].get("station"), str)
    await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)


@pytest.mark.tools
async def test_t4_bbc_round_trip(session, stubs):
    """T4: play BBC Radio 4, then stop — call sequence + args."""
    await session.send_fixture("t4_bbc.wav")
    call = await session.wait_for_tool("play_bbc_radio", timeout=TOOL_TIMEOUT_S)
    station = call["args"]["station"].lower()
    assert "4" in station or "four" in station, call["args"]
    await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)

    await session.send_fixture("t4_stop.wav")
    await session.wait_for_tool("stop_bbc_radio", timeout=TOOL_TIMEOUT_S)
    await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)


@pytest.mark.tools
async def test_t4_spotify_track(session, stubs):
    """T4: "Play Purple Rain by Prince." -> play_spotify with the track in query."""
    await session.send_fixture("t4_spotify.wav")
    call = await session.wait_for_tool("play_spotify", timeout=TOOL_TIMEOUT_S)
    assert "purple rain" in call["args"]["query"].lower(), call["args"]
    await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)
