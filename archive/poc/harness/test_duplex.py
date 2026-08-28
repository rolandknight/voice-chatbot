"""T5 barge-in regression (plan §6) for FlowCat's merged duplex builder.

The original cascaded path was half-duplex. The PoC is now pinned to the
PR #61 merge and this test protects its interruption behavior.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import pytest

from . import audio, stt
from .test_smoke import TIMEISH

TOOL_TIMEOUT_S = 60.0
REPLY_TIMEOUT_S = 60.0
BARGE_IN_STOP_BUDGET_MS = 1000.0
SILENCE_HOLD_S = 1.5  # covers the >=1.0s "stays silent" requirement
STOP_WAIT_S = 10.0  # how long the bot may keep talking before we call it a fail
COUNTING_RUNNING_S = 1.5  # let the counting reply run before interrupting

_WORDS = (
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty"
).split()
COUNTING = set(_WORDS) | {str(i) for i in range(1, 31)}


async def _wait_speech_start(read, timeout: float) -> tuple[float, bytes]:
    """Block until a bot chunk with speech RMS arrives; (monotonic ts, chunk)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = await read(0.25)
        if chunk == b"":
            raise AssertionError("bot audio stream ended while waiting for speech")
        if chunk and audio.has_speech(chunk, 16000):
            return time.monotonic(), chunk
    raise TimeoutError(f"no bot speech within {timeout}s")


async def _drain(read, seconds: float) -> None:
    """Consume (and discard) bot audio for `seconds`."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        await read(min(0.25, max(0.01, end - time.monotonic())))


async def _speech_stop_ts(read, give_up_ts: float, hold_s: float) -> Optional[float]:
    """When bot speech stops (start of a >=hold_s silent span), or None.

    None means speech was still arriving past `give_up_ts` — barge-in failed.
    """
    last_speech = time.monotonic()
    while True:
        if time.monotonic() - last_speech >= hold_s:
            return last_speech
        if last_speech > give_up_ts:
            return None
        chunk = await read(0.25)
        if chunk and chunk != b"" and audio.has_speech(chunk, 16000):
            last_speech = time.monotonic()


@pytest.mark.duplex
async def test_t5_barge_in(session, stubs):
    """T5: interrupt a long counting reply; bot stops fast, new turn answered."""
    read = session.adapter.read_bot_audio
    probes = session.adapter.probes

    # (b) trigger a long reply; wait for its speech to START, not finish.
    long_tx = await session.send_fixture("t5_long.wav")
    t_reply_start, _ = await _wait_speech_start(read, timeout=REPLY_TIMEOUT_S)
    probes["reply_start_latency"] = t_reply_start - long_tx["end_ts"]

    # (c) barge in while the bot is still counting.
    await _drain(read, COUNTING_RUNNING_S)
    int_tx = await session.send_fixture("t5_interrupt.wav")
    t_onset = int_tx["speech_onset_ts"]  # first non-silence sample on the wire

    # (d) bot audio must stop within the budget and stay silent.
    stop_ts = await _speech_stop_ts(read, give_up_ts=t_onset + STOP_WAIT_S, hold_s=SILENCE_HOLD_S)
    assert stop_ts is not None, (
        f"barge-in failed: bot still speaking {STOP_WAIT_S:.0f}s after interrupt onset"
    )
    stop_ms = max(0.0, (stop_ts - t_onset) * 1000)
    probes["bargein_stop_ms"] = stop_ms
    assert stop_ms <= BARGE_IN_STOP_BUDGET_MS, (
        f"bot audio stopped {stop_ms:.0f}ms after interrupt onset "
        f"(budget {BARGE_IN_STOP_BUDGET_MS:.0f}ms)"
    )

    # (e) new turn: tool call strictly after the interrupt, spoken time answer.
    call = await session.wait_for_tool("get_current_time", timeout=TOOL_TIMEOUT_S)
    assert call["ts"] > t_onset, (
        f"get_current_time at monotonic ts {call['ts']:.3f} predates interrupt onset {t_onset:.3f}"
    )
    t_second, first_chunk = await _wait_speech_start(read, timeout=REPLY_TIMEOUT_S)
    probes["second_reply_latency"] = t_second - t_onset
    pcm = first_chunk + await audio.collect_speech(read, timeout=REPLY_TIMEOUT_S)
    text = stt.transcribe(pcm)
    assert TIMEISH.search(text), f"second reply not time-ish: {text!r}"
    tokens = re.findall(r"[a-z']+|\d+", text)
    counting_run = any(
        all(t in COUNTING for t in tokens[i : i + 3]) for i in range(len(tokens) - 2)
    )
    assert not counting_run, f"counting resumed in the second reply: {text!r}"

    # (f) probes, printed for the report.
    print(
        f"\nT5 probes: reply_start_latency={probes['reply_start_latency']:.2f}s "
        f"bargein_stop_ms={probes['bargein_stop_ms']:.0f} "
        f"second_reply_latency={probes['second_reply_latency']:.2f}s"
    )
