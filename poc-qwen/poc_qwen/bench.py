"""Latency / RTF sweep for Qwen3-TTS cloning on this machine.

Same three sentences as poc-tts/poc_tts/bench.py so rows land next to the
Chatterbox Flash numbers in poc-tts/bench-m4-max.md. Appends one JSON row per
(model, sentence, repeat) to reports/runs.jsonl. The first repeat of every
model is recorded with cold=true and excluded from the summary.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from .config import POC_DIR, load_config, voice_dirs
from .engine import Qwen3Engine, discover_voices, sidecar_transcript

SENTENCES = [
    ("short", "Sure, the kitchen light is on."),
    ("medium", "I checked the calendar for tomorrow and you have three meetings, the first one starting at nine fifteen."),
    (
        "long",
        "Here is the summary you asked for. The build finished in about four minutes and all tests passed, "
        "except for one flaky integration test that succeeded on retry. I have also updated the dependency lock "
        "file, and the deployment to staging is scheduled for six o'clock this evening, so let me know if you "
        "want to hold it.",
    ),
]

RUNS = POC_DIR / "reports" / "runs.jsonl"


def run_matrix(engine: Qwen3Engine, ref: Path, ref_text: str, sizes: list[str], repeats: int, sink=None) -> list[dict]:
    rows: list[dict] = []
    for size in sizes:
        for rep in range(repeats):
            for name, text in SENTENCES:
                r = engine.clone(text, str(ref), ref_text, language="English", size=size)
                row = {
                    "ts": time.time(),
                    "host": platform.node(),
                    "engine": "qwen3-tts-mlx",
                    "size": size,
                    "sentence": name,
                    "repeat": rep,
                    "cold": rep == 0,
                    "voice": ref.name,
                    **r.timings,
                }
                try:
                    import mlx.core as mx

                    row["peak_mem_gb"] = round(mx.get_peak_memory() / 2**30, 2)
                except Exception:
                    pass
                rows.append(row)
                if sink is not None:
                    sink(row)
                print(f"{size} {name:6s} rep{rep} gen {r.timings['gen_s']:.2f}s audio {r.timings['audio_s']:.2f}s rtf {r.timings['rtf']:.2f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(size, sentence) -> median gen_s / rtf over warm rows."""
    out: dict[tuple[str, str], dict] = {}
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("cold"):
            continue
        groups.setdefault((row["size"], row["sentence"]), []).append(row)
    for key, group in groups.items():
        out[key] = {
            "n": len(group),
            "gen_s": statistics.median(r["gen_s"] for r in group),
            "audio_s": statistics.median(r["audio_s"] for r in group),
            "rtf": statistics.median(r["rtf"] for r in group),
            "peak_mem_gb": max(r.get("peak_mem_gb", 0) for r in group),
        }
    return out


def markdown_table(summary: dict[tuple[str, str], dict], sizes: list[str]) -> str:
    lines = ["| sentence | " + " | ".join(f"{s} gen / RTF" for s in sizes) + " |", "| --- | " + " | ".join("---" for _ in sizes) + " |"]
    for name, _ in SENTENCES:
        cells = []
        for s in sizes:
            v = summary.get((s, name))
            cells.append(f"{v['gen_s']:.2f} s / {v['rtf']:.2f}" if v else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.6B,1.7B")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--voice", default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    repeats = args.repeats or int(cfg["bench"].get("repeats", 3))
    voice = args.voice or cfg["bench"]["voice"]
    voices = discover_voices(voice_dirs(cfg))
    ref = voices.get(Path(voice).stem)
    if ref is None:
        print(f"voice {voice!r} not found in {voice_dirs(cfg)}", file=sys.stderr)
        return 1
    ref_text = sidecar_transcript(ref)
    engine = Qwen3Engine(cfg)
    if ref_text is None:
        ref_text = engine.transcribe(ref)
    RUNS.parent.mkdir(parents=True, exist_ok=True)

    def sink(row):
        with open(RUNS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    rows = run_matrix(engine, ref, ref_text, sizes, repeats, sink)
    print()
    print(markdown_table(summarize(rows), sizes))
    print(json.dumps(engine.model_info(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
