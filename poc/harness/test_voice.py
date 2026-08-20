"""T14 cloned-voice TTS (plan §6 / Phase 1b).

Requires the stack up with POC_TTS_BACKEND=chatterbox and a
Chatterbox-TTS-Server (CUDA, cloned male "marvin" voice) already running
on :8004 — started outside run_poc.sh, never killed by the harness.
Voice identity is asserted by pitch: the reply's median F0 must sit well
below Kokoro af_heart re-synthesizing the same text. Run: pytest -m voice.
"""

from __future__ import annotations

import os
import sys

import httpx
import pytest

from . import audio, results, stt
from .test_matrix import POC_DIR, timed_turn
from .test_smoke import TIMEISH

CHATTERBOX_URL = os.environ.get("POC_CHATTERBOX_URL", "http://127.0.0.1:8004").rstrip("/")
CHATTERBOX_VOICE = os.environ.get("POC_CHATTERBOX_VOICE", "marvin.wav")
CHATTERBOX_SPEECH_URL = (
    f"{CHATTERBOX_URL}/audio/speech"
    if CHATTERBOX_URL.endswith("/v1")
    else f"{CHATTERBOX_URL}/v1/audio/speech"
)
TURN_TIMEOUT_S = 90.0  # chatterbox synthesis is ~1-3s/sentence, slower than kokoro
F0_RATIO_MAX = 0.8


@pytest.fixture(scope="module")
def chatterbox_up() -> None:
    """Fail fast with a clear message if the Chatterbox server is down."""
    assert os.environ.get("POC_TTS_BACKEND") == "chatterbox", (
        "voice tests require POC_TTS_BACKEND=chatterbox"
    )
    try:
        r = httpx.post(
            CHATTERBOX_SPEECH_URL,
            json={"model": "chatterbox", "input": "hi", "voice": CHATTERBOX_VOICE,
                  "response_format": "wav"},
            timeout=90.0,
        )
        r.raise_for_status()
        assert len(r.content) > 1000, f"suspiciously small response ({len(r.content)} bytes)"
    except Exception as e:  # noqa: BLE001 — turn anything into a clear fail
        pytest.fail(
            f"Chatterbox-TTS-Server not responding on {CHATTERBOX_URL} "
            f"(it runs outside run_poc.sh; start it before pytest -m voice): {e}"
        )


@pytest.mark.voice
async def test_t14_cloned_voice(chatterbox_up, session, stubs):
    """T14: T1 turn through the chatterbox backend; reply is the cloned male voice."""
    r = await timed_turn(session, "t1_time.wav", "get_current_time", timeout=TURN_TIMEOUT_S)
    text = stt.transcribe(r["pcm"])
    assert TIMEISH.search(text), f"reply not time-ish: {text!r}"

    f0_reply = audio.median_f0(r["pcm"], 16000)
    assert f0_reply is not None, "no voiced frames in the captured reply"

    # Same text through Kokoro af_heart locally (the voice the reply must NOT be).
    sys.path.insert(0, str(POC_DIR / "stubs"))
    from kokoro_shim import load_kokoro, synth_pcm

    ref_pcm = audio.resample(synth_pcm(load_kokoro(), text, voice="af_heart"), 24000, 16000)
    f0_ref = audio.median_f0(ref_pcm, 16000)
    assert f0_ref is not None, "no voiced frames in the af_heart reference synth"

    ratio = f0_reply / f0_ref
    print(
        f"\nT14: reply_f0={f0_reply:.0f}Hz af_heart_f0={f0_ref:.0f}Hz ratio={ratio:.2f} "
        f"tool->first_audio={r['tool_to_audio']:.2f}s e2e={r['e2e']:.2f}s reply={text!r}"
    )
    results.record(
        "test_t14_cloned_voice",
        {
            "reply_f0_hz": f0_reply,
            "reference_f0_hz": f0_ref,
            "f0_ratio": ratio,
            "tool_to_audio": r["tool_to_audio"],
            "e2e": r["e2e"],
        },
    )
    assert ratio < F0_RATIO_MAX, (
        f"reply pitch {f0_reply:.0f}Hz is not clearly below af_heart's {f0_ref:.0f}Hz "
        f"(ratio {ratio:.2f} >= {F0_RATIO_MAX}) — cloned voice not in use?"
    )
