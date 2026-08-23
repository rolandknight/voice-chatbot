"""Config loading for poc-tts."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml. Paths inside it resolve against the poc-tts dir."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def voice_paths(config: dict) -> list[Path]:
    """Resolve configured voice search paths against the poc-tts directory."""
    base = DEFAULT_CONFIG_PATH.parent
    raw = config.get("voices", {}).get("paths", [])
    return [(base / entry).resolve() for entry in raw]
