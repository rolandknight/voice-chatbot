"""Persistent per-run performance results (poc/reports/runs.jsonl).

Every test that produces probes or bench stats appends one JSON line with
enough host/config context to compare runs across machines (the Linux dev
box vs the Mac Studio) and configurations (cloud vs local LLM, kokoro vs
chatterbox, CPU vs Metal whisper). View with `make poc-results`.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

RUNS_PATH = Path(__file__).resolve().parent.parent / "reports" / "runs.jsonl"


def _config_snapshot() -> dict[str, Any]:
    return {
        "llm_model": os.environ.get("POC_LLM_MODEL", "(default)"),
        "llm_base": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "whisper": Path(os.environ.get("POC_WHISPER_MODEL", "ggml-base.en.bin")).name,
        "tts_backend": os.environ.get("POC_TTS_BACKEND", "kokoro"),
        "wake": bool(os.environ.get("POC_WAKE_MODEL")),
    }


def record(test: str, data: dict[str, Any]) -> None:
    """Append one result line; never fail the test over bookkeeping."""
    try:
        RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
            "os": platform.system(),
            "machine": platform.machine(),
            "test": test,
            **_config_snapshot(),
            "results": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in data.items()},
        }
        with RUNS_PATH.open("a") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass
