"""Time-to-first-audio bench: the three baseline sentences through
synthesize_stream() with the config.yaml generation defaults.

    make bench-stream   ->   reports/stream_runs.jsonl (one line per sentence)

``--block-stream`` swaps in the Task 16 spike engine
(``engine_blockstream.BlockStreamEngine``), which vocodes each finished T3
block rather than each finished sentence. Rows are tagged ``engine:
"blockstream"`` so both sets can live in the same JSONL.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from poc_tts_streaming.bench import SENTENCES  # the same three sentences every baseline uses
from poc_tts_streaming.config import load_config, voice_paths

REPORT = Path(__file__).resolve().parent.parent / "reports" / "stream_runs.jsonl"


def measure(engine, text: str, voice: str, knobs: dict) -> dict:
    t0 = time.perf_counter()
    chunks = []
    last = t0
    ttfa = None
    for chunk_text, pcm in engine.synthesize_stream(text, voice, **knobs):
        now = time.perf_counter()
        if ttfa is None:
            ttfa = now - t0
        chunks.append({"chars": len(chunk_text), "gen_s": round(now - last, 4),
                       "audio_s": round(len(pcm) / engine.sr, 4)})
        last = now
    total = time.perf_counter() - t0
    return {
        "ttfa_s": round(ttfa or total, 4), "gen_s": round(total, 4),
        "audio_s": round(sum(c["audio_s"] for c in chunks), 4),
        "n_chunks": len(chunks), "first_chunk_chars": chunks[0]["chars"] if chunks else 0,
        "chunks": chunks,
    }


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-stream", action="store_true",
                        help="use the Task 16 spike engine (intra-sentence block streaming)")
    parser.add_argument("--runs", type=int, default=2,
                        help="measured runs per sentence; the best ttfa_s wins (default 2)")
    args = parser.parse_args(argv)

    if args.block_stream:
        from poc_tts_streaming.engine_blockstream import BlockStreamEngine as Engine
    else:
        from poc_tts_streaming.engine_flash import FlashEngine as Engine

    config = load_config()
    paths = voice_paths(config)
    engine = Engine(engine_cfg=config.get("engine", {}), generation_cfg=config.get("generation", {}),
                    voice_paths=paths)
    engine.load()
    gen = config.get("generation", {})
    knobs = {k: gen[k] for k in ("chunk_size", "split_text", "split_on_clauses") if k in gen}
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    REPORT.parent.mkdir(exist_ok=True)
    # warm-up: the first generate() pays CUDA-graph / allocator costs
    list(engine.synthesize_stream("Warm up.", voice, **knobs))
    with open(REPORT, "a", encoding="utf-8") as out:
        for label, text in SENTENCES:   # list of (name, text) tuples in bench.py:39
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            best = min((measure(engine, text, voice, knobs) for _ in range(args.runs)),
                       key=lambda r: r["ttfa_s"])
            row = {"ts": int(time.time()), "host": platform.node(), "sentence": label, "chars": len(text),
                   "engine": "blockstream" if args.block_stream else "sentence",
                   "dtype": str(engine.dtype).replace("torch.", ""), "backend": engine.backend,
                   "drf_block_size": engine.drf_block_size, "generation": gen,
                   "vram_peak_mb": (round(torch.cuda.max_memory_reserved() / 2**20)
                                    if torch.cuda.is_available() else None), **best}
            out.write(json.dumps(row) + "\n")
            print(f"{label:>7}: ttfa {best['ttfa_s']:.3f}s  gen {best['gen_s']:.2f}s  "
                  f"audio {best['audio_s']:.2f}s  chunks {best['n_chunks']}")


if __name__ == "__main__":
    main()
