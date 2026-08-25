"""config.yaml + POC_GEMMA4_<SECTION>_<KEY> overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "POC_GEMMA4_"


def _coerce(old: Any, raw: str) -> Any:
    if isinstance(old, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        return float(raw)
    return raw


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> dict:
    env = os.environ if env is None else env
    cfg = yaml.safe_load((path or ROOT / "config.yaml").read_text())
    for section, keys in cfg.items():
        if not isinstance(keys, dict):
            continue
        for key, val in list(keys.items()):
            if isinstance(val, dict):
                continue
            raw = env.get(f"{PREFIX}{section}_{key}".upper())
            if raw is not None:
                keys[key] = _coerce(val, raw)
    return cfg
