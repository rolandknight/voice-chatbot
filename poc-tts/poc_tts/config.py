"""Config loading for poc-tts."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Engine settings that may be overridden per machine from the environment.
# config.yaml is shared with the CUDA box, so the Mac's `backend: mlx` /
# `dtype: float16` cannot live there -- committing them would break CUDA on
# the next pull. The Makefile already sources a gitignored `.env` into every
# recipe, so a Mac checkout selects Metal with:
#
#     POC_TTS_ENGINE_BACKEND=mlx
#     POC_TTS_ENGINE_DTYPE=float16
#
# Values are passed through untouched; resolve_device / resolve_dtype /
# resolve_backend in engine_flash.py stay the single place that validates them,
# so a typo here fails the same way a typo in config.yaml would.
ENGINE_ENV_OVERRIDES = {
    "device": "POC_TTS_ENGINE_DEVICE",
    "dtype": "POC_TTS_ENGINE_DTYPE",
    "backend": "POC_TTS_ENGINE_BACKEND",
}


def apply_engine_overrides(config: dict, env: dict | None = None) -> dict:
    """Overlay POC_TTS_ENGINE_* onto the engine section, if any are set.

    Returns a new dict; the input is not mutated. An unset or empty variable
    leaves the config.yaml value alone, so an untouched environment behaves
    exactly as it did before this existed.
    """
    env = os.environ if env is None else env
    overrides = {
        key: env[var].strip()
        for key, var in ENGINE_ENV_OVERRIDES.items()
        if env.get(var, "").strip()
    }
    if not overrides:
        return config
    merged = dict(config)
    merged["engine"] = {**config.get("engine", {}), **overrides}
    return merged


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml. Paths inside it resolve against the poc-tts dir."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return apply_engine_overrides(yaml.safe_load(handle) or {})


def voice_paths(config: dict) -> list[Path]:
    """Resolve configured voice search paths against the poc-tts directory."""
    base = DEFAULT_CONFIG_PATH.parent
    raw = config.get("voices", {}).get("paths", [])
    return [(base / entry).resolve() for entry in raw]
