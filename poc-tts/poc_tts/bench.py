"""Sweep Chatterbox Flash tuning configurations and record RTF.

The three sentences are identical to those used for the Turbo CUDA, Flash
CUDA, and Turbo CPU baselines in the design spec, so results are directly
comparable rather than standalone.

Only `drf_block_size` is a load-time constructor argument for FlashEngine;
`num_steps` and `n_cfm_timesteps` are per-request parameters of
synthesize(). The sweep therefore loads the model once per drf_block_size
value (2 loads total for the default grid) and varies the other two axes
per-request inside that load, instead of rebuilding the engine for every
grid cell.
"""

from __future__ import annotations

import itertools
import json
import platform
import socket
import time
from pathlib import Path

import torch

SENTENCES = [
    ("short", "Sure, the kitchen light is on."),
    ("medium", "I checked the calendar for tomorrow and you have three meetings, "
               "the first one starting at nine fifteen."),
    ("long", "Here is the summary you asked for. The build finished in about four "
             "minutes, all thirty two tests passed, and the only warning came from "
             "the audio device layer, which reported that the sample rate was "
             "renegotiated partway through the session. Nothing else looked out of "
             "the ordinary, so I would call that a clean run."),
]

GRID = {
    "drf_block_size": [16, 32],
    "num_steps": [4, 6, 10],
    "n_cfm_timesteps": [1, 2],
}

REPORTS = Path(__file__).resolve().parent.parent / "reports" / "runs.jsonl"


def sweep_configs(grid: dict) -> list[dict]:
    """Cartesian product of the grid axes."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def record_result(path: Path, row: dict) -> None:
    """Append one result as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    from poc_tts.config import load_config, voice_paths
    from poc_tts.engine_flash import FlashEngine

    config = load_config()
    paths = voice_paths(config)
    voice = config.get("bench", {}).get("voice", "marvin.wav")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    per_request_grid = {
        "num_steps": GRID["num_steps"],
        "n_cfm_timesteps": GRID["n_cfm_timesteps"],
    }
    per_request_configs = sweep_configs(per_request_grid)

    for block_size in GRID["drf_block_size"]:
        engine_cfg = dict(config.get("engine", {}))
        engine_cfg["drf_block_size"] = block_size
        engine = FlashEngine(
            engine_cfg=engine_cfg,
            generation_cfg=config.get("generation", {}),
            voice_paths=paths,
        )
        engine.load()
        engine.synthesize(text="Warming up the voice.", voice=voice)

        for combo in per_request_configs:
            full_config = {"drf_block_size": block_size, **combo}

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            for name, text in SENTENCES:
                timings = []
                for _ in range(2):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    audio, sample_rate = engine.synthesize(
                        text=text,
                        voice=voice,
                        num_steps=combo["num_steps"],
                        n_cfm_timesteps=combo["n_cfm_timesteps"],
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)

                audio_s = len(audio) / sample_rate
                best = min(timings)
                row = {
                    "ts": stamp,
                    "host": socket.gethostname(),
                    "machine": platform.machine(),
                    "model": "chatterbox-flash",
                    "device": engine.device,
                    "dtype": str(engine.dtype).replace("torch.", ""),
                    "backend": engine.backend,
                    "config": full_config,
                    "sentence": name,
                    "results": {
                        "audio_s": round(audio_s, 3),
                        "gen_s": round(best, 3),
                        "rtf": round(best / audio_s, 4),
                    },
                }
                if torch.cuda.is_available():
                    row["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
                record_result(REPORTS, row)
                print(f"{full_config} {name}: RTF {row['results']['rtf']}", flush=True)

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"sweep complete -> {REPORTS}")


if __name__ == "__main__":
    main()
