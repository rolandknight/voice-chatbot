"""Phase 1 matrix tests T6-T12 (plan §6) against the full-duplex server.

Run order matters (see poc/reports/flowcat-cloud.md): T8 last — its
recall probe pollutes context. Restart the stack between T7 and T9.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from . import audio, stt, results
from .conftest import TurnRunner
from .test_duplex import _drain, _wait_speech_start
from .test_smoke import TIMEISH

POC_DIR = Path(__file__).resolve().parent.parent
TURN_TIMEOUT_S = 60.0


def _pct(vals: list[float], q: float) -> float:
    return float(np.percentile(vals, q))


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    except OSError:
        return 0


def _flowcat_rss_mb() -> Optional[float]:
    try:
        pid = (POC_DIR / "logs" / "flowcat.pid").read_text().strip()
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


async def timed_turn(
    sess: TurnRunner, fixture: str, tool: str, timeout: float = TURN_TIMEOUT_S, clear: bool = True
) -> dict[str, Any]:
    """One full turn with per-segment timings (monotonic clock shared with stubs)."""
    if clear:
        sess.stubs.clear()
    read = sess.adapter.read_bot_audio
    tx = await sess.send_fixture(fixture)
    call_task = asyncio.create_task(sess.wait_for_tool(tool, timeout=timeout))
    try:
        t_first, chunk = await _wait_speech_start(read, timeout)
    except BaseException:
        call_task.cancel()
        raise
    call = await call_task
    pcm = chunk + await audio.collect_speech(read, timeout=timeout)
    return {
        "e2e": t_first - tx["speech_end_ts"],  # speech-end -> first bot audio
        "stt_llm": call["ts"] - tx["speech_end_ts"],  # speech-end -> tool call
        "tool_to_audio": t_first - call["ts"],  # tool call -> first audio
        "call": call,
        "pcm": pcm,
    }


@pytest.mark.concurrency
async def test_t6_concurrent_sessions(session_factory, stubs):
    """T6: two interleaved sessions, different tools, zero cross-talk."""
    a = await session_factory()
    b = await session_factory()
    stubs.clear()
    ra, rb = await asyncio.gather(
        timed_turn(a, "t1_time.wav", "get_current_time", clear=False),
        timed_turn(b, "t2_timer.wav", "set_timer", clear=False),
    )
    text_a, text_b = stt.transcribe(ra["pcm"]), stt.transcribe(rb["pcm"])
    assert TIMEISH.search(text_a), f"A's reply not time-ish: {text_a!r}"
    assert any(w in text_b for w in ("timer", "five", "5", "minute")), (
        f"B's reply not timer-ish: {text_b!r}"
    )
    assert float(rb["call"]["args"]["minutes"]) == 5.0
    stubs.clear()
    ra2, rb2 = await asyncio.gather(
        timed_turn(a, "t4_bbc.wav", "play_bbc_radio", clear=False),
        timed_turn(b, "t3_music.wav", "play_spotify", clear=False),
    )
    assert isinstance(ra2["call"]["args"].get("station"), str)
    assert isinstance(rb2["call"]["args"].get("query"), str)
    # Transcript cross-talk: each client's user transcripts contain only its
    # own utterances (vacuously true if the server emits no transcript events).
    ev_a = [e.text.lower() for e in a.adapter.drain_events() if e.kind == "transcript-user" and e.text]
    ev_b = [e.text.lower() for e in b.adapter.drain_events() if e.kind == "transcript-user" and e.text]
    for t in ev_a:
        assert "timer" not in t and "music" not in t, f"B's speech in A's transcript: {t!r}"
    for t in ev_b:
        assert "time is it" not in t and "radio" not in t, f"A's speech in B's transcript: {t!r}"
    await a.adapter.close(graceful=True)
    await b.adapter.close(graceful=True)
    print(f"\nT6: A e2e={ra['e2e']:.2f}/{ra2['e2e']:.2f}s B e2e={rb['e2e']:.2f}/{rb2['e2e']:.2f}s "
          f"user-transcript events: A={len(ev_a)} B={len(ev_b)}")


@pytest.mark.teardown
async def test_t7_abrupt_teardown(session_factory, stubs):
    """T7: kill the peer mid-reply; server stays healthy, no log spin."""
    log = POC_DIR / "logs" / "flowcat.log"
    victim = await session_factory()
    stubs.clear()
    await victim.send_fixture("t5_long.wav")
    await _wait_speech_start(victim.adapter.read_bot_audio, TURN_TIMEOUT_S)
    await _drain(victim.adapter.read_bot_audio, 1.0)  # well inside the reply
    lines_before = _count_lines(log)
    kill_ts = time.monotonic()
    await victim.adapter.close(graceful=False)  # no bye: ICE/UDP dropped mid-flow
    fresh = await session_factory()  # probes["connected"] set before greeting wait
    connect_latency = fresh.adapter.probes["connected"] - kill_ts
    assert connect_latency <= 2.0, f"reconnect after abrupt kill took {connect_latency:.2f}s"
    await asyncio.sleep(max(0.0, 5.0 - (time.monotonic() - kill_ts)))
    window = time.monotonic() - kill_ts
    rate = (_count_lines(log) - lines_before) / window
    assert rate < 200, f"flowcat.log growing at {rate:.0f} lines/s after abrupt kill"
    r = await timed_turn(fresh, "t1_time.wav", "get_current_time")
    assert TIMEISH.search(stt.transcribe(r["pcm"]))
    print(f"\nT7: reconnect={connect_latency:.2f}s log_rate={rate:.1f} lines/s "
          f"({window:.1f}s window), post-kill T1 e2e={r['e2e']:.2f}s")


@pytest.mark.lifecycle
@pytest.mark.xfail(
    strict=False,
    reason="cascaded builder hard-disables idle timeout; Babel CONV-5 context wipe "
    "not implemented — recorded finding",
)
async def test_t8_idle_context_wipe(session, stubs):
    """T8: after 20s idle the context should be wiped (it is NOT today)."""
    await timed_turn(session, "t1_time.wav", "get_current_time")
    await _drain(session.adapter.read_bot_audio, 20.0)
    stubs.clear()
    await session.send_fixture("t8_recall.wav")
    pcm = await session.await_bot_speech(timeout=TURN_TIMEOUT_S)
    text = stt.transcribe(pcm)
    print(f"\nT8 recall reply after 20s idle: {text!r}")
    assert "time" not in text, f"context NOT wiped after idle: bot recalled {text!r}"


@pytest.mark.inflight
async def test_t9_inflight_tool_latency(session, stubs):
    """T9: 12s tool latency > idle timeout; reply still delivered, session lives."""
    stubs.set_latency("get_weather", 12.0)
    try:
        await session.send_fixture("t9_weather.wav")
        call = await session.wait_for_tool("get_weather", timeout=TURN_TIMEOUT_S)
        read = session.adapter.read_bot_audio
        t_first, chunk = await _wait_speech_start(read, TURN_TIMEOUT_S)
        pcm = chunk + await audio.collect_speech(read, timeout=TURN_TIMEOUT_S)
        held = t_first - call["ts"]
        assert held >= 12.0, f"reply {held:.1f}s after tool call — injected latency not honored?"
        text = stt.transcribe(pcm)
        assert any(w in text for w in ("18", "eighteen", "cloud", "degree")), (
            f"weather reply lost the canned forecast: {text!r}"
        )
    finally:
        stubs.set_latency("get_weather", 0)
    r = await timed_turn(session, "t1_time.wav", "get_current_time")
    assert TIMEISH.search(stt.transcribe(r["pcm"]))
    print(f"\nT9: tool held {held:.1f}s, reply spoken ({text!r}); follow-up T1 e2e={r['e2e']:.2f}s")


async def _drain_stray_reply(sess: TurnRunner) -> None:
    """After a failed turn, eat any half-delivered reply so timings stay clean."""
    with contextlib.suppress(Exception):
        await audio.collect_speech(sess.adapter.read_bot_audio, timeout=5.0)


@pytest.mark.latency
async def test_t10_latency_bench(session, stubs):
    """T10: 20 warm T1-class turns; record p50/p95, no hard latency gate (cloud LLM).

    STT-flaked turns (whisper mishears, LLM never calls the tool) are recorded
    and skipped rather than aborting the bench; >=15/20 must succeed.
    """
    turns = []
    flaked: list[tuple[int, str, str]] = []
    for i in range(20):
        fixture, tool = (
            ("t1_time.wav", "get_current_time") if i % 2 == 0 else ("t10_date.wav", "get_current_date")
        )
        try:
            r = await timed_turn(session, fixture, tool)
        except (TimeoutError, AssertionError) as e:
            flaked.append((i + 1, tool, str(e)))
            print(f"turn {i + 1:2d} {tool:17s} FLAKED: {e}")
            await _drain_stray_reply(session)
            continue
        turns.append(r)
        print(f"turn {i + 1:2d} {tool:17s} e2e={r['e2e']:5.2f}s stt+llm={r['stt_llm']:5.2f}s "
              f"tool->audio={r['tool_to_audio']:5.2f}s")
        await asyncio.sleep(0.5)
    assert len(turns) >= 15, f"only {len(turns)}/20 turns completed: {flaked}"
    print(f"T10 summary ({len(turns)}/20 turns ok, {len(flaked)} flaked):")
    for key, label in (("e2e", "speech-end->first-audio"), ("stt_llm", "speech-end->tool-call"),
                       ("tool_to_audio", "tool-call->first-audio")):
        vals = [t[key] for t in turns]
        print(f"  {label:24s} p50={_pct(vals, 50):5.2f}s p95={_pct(vals, 95):5.2f}s")
    results.record("t10_latency_bench", {
        "turns_ok": len(turns), "flaked": len(flaked),
        **{f"{k}_{p}": _pct([t[k] for t in turns], p)
           for k in ("e2e", "stt_llm", "tool_to_audio") for p in (50, 95)},
    })


@pytest.mark.failures
async def test_t11_tool_failure_surfaced(session, stubs):
    """T11: tool returns 500; failure is spoken, turn doesn't die silently."""
    stubs.set_fail("get_weather", 500)
    try:
        await session.send_fixture("t9_weather.wav")
        await session.wait_for_tool("get_weather", timeout=TURN_TIMEOUT_S)
        pcm = await session.await_bot_speech(timeout=TURN_TIMEOUT_S)
        text = stt.transcribe(pcm)
        print(f"\nT11 failure reply: {text!r}")
        failure_words = ("error", "problem", "sorry", "unable", "couldn't", "couldn",
                         "can't", "cannot", "trouble", "fail", "wrong", "issue")
        assert any(w in text for w in failure_words), (
            f"no failure acknowledgement in reply: {text!r}"
        )
    finally:
        stubs.set_fail("get_weather", None)
    r = await timed_turn(session, "t1_time.wav", "get_current_time")
    assert TIMEISH.search(stt.transcribe(r["pcm"]))
    print(f"T11: follow-up T1 e2e={r['e2e']:.2f}s — session survived the tool failure")


@pytest.mark.soak
async def test_t12_soak_30_turns(session, stubs):
    """T12: 30 alternating turns, one session; stable latency, bounded RSS."""
    rss_before = _flowcat_rss_mb()
    e2es: list[float] = []
    failures: list[tuple[int, str]] = []
    for i in range(30):
        fixture, tool = (
            ("t1_time.wav", "get_current_time") if i % 2 == 0 else ("t2_timer.wav", "set_timer")
        )
        try:
            r = await timed_turn(session, fixture, tool)
            e2es.append(r["e2e"])
            print(f"soak {i + 1:2d}/30 {tool:17s} e2e={r['e2e']:5.2f}s")
        except (TimeoutError, AssertionError) as e:
            failures.append((i + 1, str(e)))
            print(f"soak {i + 1:2d}/30 {tool} FAILED: {e}")
            await _drain_stray_reply(session)
        await asyncio.sleep(0.5)
    rss_after = _flowcat_rss_mb()
    assert not failures, f"{len(failures)}/30 turns failed: {failures}"
    p95_first, p95_last = _pct(e2es[:10], 95), _pct(e2es[-10:], 95)
    rss_note = (f"rss {rss_before:.0f}->{rss_after:.0f}MB"
                if rss_before and rss_after else "rss unavailable")
    print(f"T12: p95 first10={p95_first:.2f}s last10={p95_last:.2f}s {rss_note}")
    results.record("t12_soak", {
        "turns": len(e2es), "e2e_p50": _pct(e2es, 50), "p95_first10": p95_first,
        "p95_last10": p95_last, "rss_before_mb": rss_before, "rss_after_mb": rss_after,
    })
    assert p95_last <= p95_first * 1.5, (
        f"latency degraded over soak: p95 {p95_first:.2f}s -> {p95_last:.2f}s"
    )
    if p95_last < p95_first * 0.5:
        print("T12 note: last-10 p95 more than 50% below first-10 (warm-up effect, not gated)")
    if rss_before and rss_after:
        assert rss_after - rss_before < 200, f"flowcat RSS grew {rss_after - rss_before:.0f}MB"
