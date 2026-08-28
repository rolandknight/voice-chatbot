"""Sweep Chatterbox Flash tuning configurations and record RTF.

The three sentences are identical to those used for the Turbo CUDA, Flash
CUDA, and Turbo CPU baselines in the design spec, so results are directly
comparable rather than standalone.

`num_steps` and `n_cfm_timesteps` are per-request parameters of
synthesize(). Everything else -- `drf_block_size` (a FlashEngine constructor
argument), the backend, and MLX weight quantization (read from the
environment when chatterbox_flash builds its MLX engine) -- is fixed at load
time. The sweep therefore loads the model once per load-time combination and
varies the two per-request axes inside that load, instead of rebuilding the
engine for every grid cell.

The backend / dtype / quantization axes come from the environment rather than
config.yaml, because config.yaml is shared with the CUDA box: `backend: mlx`
committed there would break it. The Makefile already sources a gitignored
`.env` into every recipe, which is where machine-specific settings belong.

    POC_TTS_BENCH_BACKENDS=mlx POC_TTS_BENCH_DTYPE=float16 \
        POC_TTS_BENCH_QUANT_BITS=,4 make bench

sweeps the MLX backend twice, unquantized and 4-bit. Unset, all three fall
back to config.yaml and the sweep behaves exactly as it did before.
"""

from __future__ import annotations

import itertools
import json
import os
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

# chatterbox_flash reads this when it constructs its MLX engine -- there is no
# quantize_bits argument anywhere in its Python API, so the env var is the
# whole interface. Set per load-time combination, before the model is built.
QUANT_ENV_VAR = "CHATTERBOX_FLASH_MLX_QUANT_BITS"


def sweep_configs(grid: dict) -> list[dict]:
    """Cartesian product of the grid axes."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _split_env_list(raw: str) -> list[str]:
    """Split a comma-separated env value, preserving empty entries.

    An empty entry is meaningful for the quantization axis: `,4` means "run
    unquantized, then run 4-bit", and dropping the empty half would silently
    lose the baseline the 4-bit numbers are supposed to be compared against.
    """
    return [item.strip() for item in raw.split(",")]


def load_time_configs(config: dict, env: dict | None = None) -> list[dict]:
    """Every combination that requires its own model load.

    Backend and dtype default to the engine section of config.yaml, so an
    unset environment reproduces the pre-existing sweep exactly: one backend,
    no quantization, two block sizes.
    """
    env = os.environ if env is None else env
    engine_cfg = config.get("engine", {})

    backends_raw = env.get("POC_TTS_BENCH_BACKENDS", "").strip()
    backends = (
        _split_env_list(backends_raw) if backends_raw
        else [engine_cfg.get("backend", "auto")]
    )

    dtype = env.get("POC_TTS_BENCH_DTYPE", "").strip() or engine_cfg.get("dtype", "auto")

    quant_raw = env.get("POC_TTS_BENCH_QUANT_BITS", "").strip()
    quant_bits = _split_env_list(quant_raw) if quant_raw else [""]

    return [
        {
            "backend": backend,
            "dtype": dtype,
            "quantize_bits": int(bits) if bits else None,
            "drf_block_size": block_size,
        }
        for backend, bits, block_size in itertools.product(
            backends, quant_bits, GRID["drf_block_size"]
        )
    ]


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

    for load_cfg in load_time_configs(config):
        block_size = load_cfg["drf_block_size"]
        quantize_bits = load_cfg["quantize_bits"]

        engine_cfg = dict(config.get("engine", {}))
        engine_cfg["drf_block_size"] = block_size
        engine_cfg["backend"] = load_cfg["backend"]
        engine_cfg["dtype"] = load_cfg["dtype"]

        # Must be set before the model is built: chatterbox_flash reads it
        # when it lazily constructs the MLX engine on the first generate().
        if quantize_bits is None:
            os.environ.pop(QUANT_ENV_VAR, None)
        else:
            os.environ[QUANT_ENV_VAR] = str(quantize_bits)

        engine = FlashEngine(
            engine_cfg=engine_cfg,
            generation_cfg=config.get("generation", {}),
            voice_paths=paths,
        )
        engine.load()
        engine.synthesize(text="Warming up the voice.", voice=voice)

        for combo in per_request_configs:
            full_config = {"drf_block_size": block_size, **combo}
            # Absent rather than null when unquantized, so rows from the CUDA
            # box and the Mac fp16 sweep stay byte-identical in shape.
            if quantize_bits is not None:
                full_config["quantize_bits"] = quantize_bits

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
                print(
                    f"[{engine.backend}] {full_config} {name}: "
                    f"RTF {row['results']['rtf']}",
                    flush=True,
                )

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.environ.pop(QUANT_ENV_VAR, None)
    print(f"sweep complete -> {REPORTS}")


if __name__ == "__main__":
    main()
