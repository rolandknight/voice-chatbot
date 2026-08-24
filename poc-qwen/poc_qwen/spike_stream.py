"""Streaming spike (iteration-2 reconnaissance, not wired into the UI).

Calls mlx-audio with stream=True for the medium bench sentence and records
time-to-first-chunk, chunk cadence and the concatenated wav, per model size.
Appends to reports/stream_spike.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .bench import SENTENCES
from .config import POC_DIR, load_config, voice_dirs
from .engine import Qwen3Engine, discover_voices, sidecar_transcript

OUT = POC_DIR / "reports" / "stream_spike.jsonl"


def measure(chunks_with_times: list[tuple[float, int]], sample_rate: int) -> dict:
    """chunks_with_times: (seconds since start, samples in chunk)."""
    if not chunks_with_times:
        return {"ttfa_s": None}
    times = [t for t, _ in chunks_with_times]
    sizes = [n for _, n in chunks_with_times]
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {
        "ttfa_s": round(times[0], 3),
        "chunks": len(sizes),
        "chunk_s_median": round(float(np.median([n / sample_rate for n in sizes])), 3),
        "gap_s_median": round(float(np.median(gaps)), 3) if gaps else None,
        "gap_s_max": round(max(gaps), 3) if gaps else None,
        "total_s": round(times[-1], 3),
        "audio_s": round(sum(sizes) / sample_rate, 3),
    }


def main() -> int:
    cfg = load_config()
    voices = discover_voices(voice_dirs(cfg))
    ref = voices[Path(cfg["bench"]["voice"]).stem]
    ref_text = sidecar_transcript(ref) or ""
    engine = Qwen3Engine(cfg)
    text = dict(SENTENCES)["medium"]
    for size in ("0.6B", "1.7B"):
        for interval in (0.32, 0.64):
            for rep in range(2):
                t0 = time.perf_counter()
                parts, stamps = [], []
                for chunk in engine.stream_clone(text, str(ref), ref_text, language="English", size=size, interval_s=interval):
                    stamps.append((time.perf_counter() - t0, len(chunk)))
                    parts.append(chunk)
                row = {"size": size, "interval_s": interval, "repeat": rep, "cold": rep == 0, **measure(stamps, 24000)}
                if rep == 1:
                    sf.write(POC_DIR / "reports" / f"stream_{size}_{interval}.wav", np.concatenate(parts), 24000)
                print(json.dumps(row), flush=True)
                with open(OUT, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
