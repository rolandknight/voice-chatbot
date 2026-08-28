"""T13 server-side wake over WebRTC (Listen mode, plan §6 / Phase 1a).

Requires the stack up with POC_WAKE_MODEL pointing at hey_babel.onnx:
a WakeGate swallows audio until "hey babel" fires (threshold 0.5), then a
15s session window opens with 0.5s pre-roll. Run: pytest harness -m wake.
The negative asserts — wake-less speech produces NO turn, and the window
re-arms after expiry — are the point of the test.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from . import audio, stt
from .conftest import TurnRunner
from .test_duplex import _drain, _wait_speech_start
from .test_smoke import TIMEISH

NO_TURN_WINDOW_S = 15.0
SESSION_EXPIRY_WAIT_S = 20.0  # > the server's 15s wake session window
TURN_TIMEOUT_S = 60.0


async def assert_no_turn(sess: TurnRunner, window_s: float = NO_TURN_WINDOW_S) -> None:
    """No bot speech and no tool calls for `window_s` after the send."""
    read = sess.adapter.read_bot_audio
    end = time.monotonic() + window_s
    while time.monotonic() < end:
        chunk = await read(0.25)
        assert not (chunk and chunk != b"" and audio.has_speech(chunk, 16000)), (
            "unexpected bot speech — wake gate did not swallow the turn"
        )
    calls = sess.stubs.get_calls()
    assert calls == [], f"unexpected tool calls through the wake gate: {calls}"


@pytest.mark.wake
async def test_t13_server_side_wake(session, stubs):
    """T13: wake gating correct both ways + window re-arms after expiry."""
    probes = session.adapter.probes

    # (a) no wake word -> audio swallowed, no turn.
    stubs.clear()
    await session.send_fixture("t1_time.wav")
    await assert_no_turn(session)

    # (b) wake-prefixed command -> turn fires, tool called, time spoken.
    stubs.clear()
    tx = await session.send_fixture("t13_wake.wav")
    call_task = asyncio.create_task(session.wait_for_tool("get_current_time", timeout=TURN_TIMEOUT_S))
    try:
        t_first, chunk = await _wait_speech_start(session.adapter.read_bot_audio, TURN_TIMEOUT_S)
    except BaseException:
        call_task.cancel()
        raise
    call = await call_task
    assert call["ts"] > tx["push_ts"]
    pcm = chunk + await audio.collect_speech(session.adapter.read_bot_audio, timeout=TURN_TIMEOUT_S)
    text = stt.transcribe(pcm)
    assert TIMEISH.search(text), f"wake-turn reply not time-ish: {text!r}"
    probes["wake_to_first_audio"] = t_first - tx["speech_onset_ts"]  # from "hey babel" onset
    probes["wake_e2e"] = t_first - tx["speech_end_ts"]  # from end of the command

    # (c) session window (15s) expires -> gate re-arms, wake-less speech ignored.
    await _drain(session.adapter.read_bot_audio, SESSION_EXPIRY_WAIT_S)
    stubs.clear()
    await session.send_fixture("t1_time.wav")
    await assert_no_turn(session)

    print(
        f"\nT13 probes: wake_onset->first_audio={probes['wake_to_first_audio']:.2f}s "
        f"speech_end->first_audio={probes['wake_e2e']:.2f}s reply={text!r}"
    )
