"""Generate the WAV fixtures from poc/CONTRACT.md via kokoro-onnx.

16 kHz mono s16, ~300 ms leading + ~1.2 s trailing silence, voice af_heart.
Idempotent: existing files are skipped unless --force.

    cd poc && .venv/bin/python -m harness.make_fixtures [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# CONTRACT.md fixture table.
FIXTURES: dict[str, str] = {
    "t1_time.wav": "What time is it?",
    "t2_timer.wav": "Set a timer for five minutes.",
    "t3_music.wav": "Put some music on.",
    "t3_news.wav": "I'd like to listen to the news.",
    "t4_bbc.wav": "Play BBC Radio 4.",
    "t4_stop.wav": "Stop the radio.",
    "t4_spotify.wav": "Play Purple Rain by Prince.",
    "t5_long.wav": "Please count slowly from one to thirty, one number at a time.",
    "t5_interrupt.wav": "Stop. What time is it?",
    "t8_recall.wav": "What did I just ask you about?",
    "t9_weather.wav": "What's the weather like?",
    "t10_date.wav": "What's the date today?",
    "t13_wake.wav": "Hey babel, what time is it?",
}

LEAD_MS = 300
TRAIL_MS = 1200
OUT_RATE = 16000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args(argv)

    from .audio import resample, save_wav, silence

    todo = {
        name: text
        for name, text in FIXTURES.items()
        if args.force or not (FIXTURES_DIR / name).exists()
    }
    if not todo:
        print("all fixtures present; nothing to do (use --force to regenerate)")
        return 0

    sys.path.insert(0, str(POC_DIR / "stubs"))
    from kokoro_shim import load_kokoro, synth_pcm  # shares model download/cache

    kokoro = load_kokoro()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in todo.items():
        pcm24 = synth_pcm(kokoro, text, voice="af_heart")
        pcm16 = resample(pcm24, 24000, OUT_RATE)
        pcm = silence(LEAD_MS, OUT_RATE) + pcm16 + silence(TRAIL_MS, OUT_RATE)
        save_wav(FIXTURES_DIR / name, pcm, OUT_RATE)
        print(f"wrote {name}: {len(pcm) / 2 / OUT_RATE:.2f}s  \"{text}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
