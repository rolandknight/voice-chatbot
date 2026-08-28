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
import re
import subprocess
import time
from pathlib import Path
from typing import Any

POC_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = POC_DIR.parent
RUNS_PATH = POC_DIR / "reports" / "runs.jsonl"


def _build_profile() -> dict[str, str]:
    profile: dict[str, str] = {}
    path = POC_DIR / "logs" / "build-profile.env"
    if not path.exists():
        return profile
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            profile[key] = value
    return profile


def _chatterbox_profile() -> dict[str, str]:
    candidates = (
        REPO_DIR / "vendor" / "chatterbox-tts-server",
        REPO_DIR / "vendor" / "Chatterbox-TTS-Server",
    )
    server_dir = next(
        (
            path
            for path in candidates
            if (path / ".venv" / "bin" / "python").exists()
            or (path / "venv" / "bin" / "python").exists()
        ),
        next((path for path in candidates if (path / ".git").exists()), None),
    )
    if server_dir is None:
        return {}

    profile = {"chatterbox_path": server_dir.name}
    try:
        revision = subprocess.run(
            ["git", "-C", str(server_dir), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision:
            profile["chatterbox_revision"] = revision
    except (OSError, subprocess.CalledProcessError):
        pass

    config_path = server_dir / "config.yaml"
    if config_path.exists():
        match = re.search(r"^\s+device:\s*([^\s#]+)", config_path.read_text(), re.MULTILINE)
        if match:
            profile["chatterbox_device"] = match.group(1)

    for env_name in (".venv", "venv"):
        install_type_path = server_dir / env_name / ".install_type"
        if install_type_path.exists():
            profile["chatterbox_install_type"] = install_type_path.read_text().strip()
            break
        if (server_dir / env_name / "bin" / "python").exists():
            profile["chatterbox_install_type"] = f"{env_name}-legacy"
            break
    return profile


def _config_snapshot() -> dict[str, Any]:
    build = _build_profile()
    stt_backend = os.environ.get(
        "POC_STT_BACKEND", build.get("POC_STT_BACKEND", "whisper")
    )
    if stt_backend == "moonshine":
        stt_model = Path(
            os.environ.get("POC_MOONSHINE_MODEL", "medium-streaming-en")
        ).name
    elif stt_backend in {"nemotron", "nvidia"}:
        stt_model = "nemotron-speech-streaming-en-0.6b.q8_0.gguf"
    else:
        stt_model = Path(
            os.environ.get("POC_WHISPER_MODEL", "ggml-base.en.bin")
        ).name
    snapshot: dict[str, Any] = {
        "llm_model": os.environ.get("POC_LLM_MODEL", "(default)"),
        "stt_backend": stt_backend,
        "stt_model": stt_model,
        # Retain the historical field so existing report readers do not break.
        "whisper": Path(os.environ.get("POC_WHISPER_MODEL", "ggml-base.en.bin")).name,
        "stt_accelerator": build.get("POC_STT_ACCELERATOR", "unknown"),
        "opus": build.get("POC_OPUS_SOURCE", "unknown"),
        "tts_backend": os.environ.get("POC_TTS_BACKEND", "kokoro"),
        "wake": bool(os.environ.get("POC_WAKE_MODEL")),
    }
    if snapshot["tts_backend"] == "chatterbox":
        snapshot["chatterbox_voice"] = os.environ.get(
            "POC_CHATTERBOX_VOICE", "marvin.wav"
        )
        snapshot.update(_chatterbox_profile())
    return snapshot


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
