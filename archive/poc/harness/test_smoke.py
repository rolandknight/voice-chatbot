"""Smoke tests T1/T2 (poc/CONTRACT.md): basic turn + direct tool call.

Latency probes are asserted as *recorded* only — smoke never gates on
values (cloud LLM TTFT misses local targets by design, plan §5).
"""

from __future__ import annotations

import re

import pytest

from . import stt

TOOL_TIMEOUT_S = 60.0  # generous: whisper.cpp batch ~4 s windows + cloud TTFT
REPLY_TIMEOUT_S = 60.0

TIMEISH = re.compile(r"\d|o'?clock|noon|midnight|a\.?m|p\.?m|half past|quarter")


def assert_probes(session) -> None:
    """Latency probes recorded (values judged in T10, not smoke)."""
    probes = session.adapter.probes
    assert "connected" in probes
    assert "last_audio_sent" in probes
    assert "first_bot_frame" in probes


@pytest.mark.smoke
async def test_t1_time(session, stubs):
    """T1: "What time is it?" -> get_current_time called + time-ish spoken reply."""
    await session.send_fixture("t1_time.wav")
    call = await session.wait_for_tool("get_current_time", timeout=TOOL_TIMEOUT_S)
    assert call["args"] == {}
    pcm = await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)
    text = stt.transcribe(pcm)
    assert TIMEISH.search(text), f"reply not time-ish: {text!r}"
    assert_probes(session)


@pytest.mark.smoke
async def test_t2_timer(session, stubs):
    """T2: "Set a timer for five minutes." -> set_timer(minutes=5) + confirmation."""
    await session.send_fixture("t2_timer.wav")
    call = await session.wait_for_tool("set_timer", timeout=TOOL_TIMEOUT_S)
    assert float(call["args"]["minutes"]) == 5.0, call["args"]
    pcm = await session.await_bot_speech(timeout=REPLY_TIMEOUT_S)
    text = stt.transcribe(pcm)
    assert any(w in text for w in ("timer", "five", "5", "minute")), (
        f"reply doesn't confirm the timer: {text!r}"
    )
    assert_probes(session)
