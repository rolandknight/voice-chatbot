"""Config loading for poc-qwen.

config.yaml is the source of truth. Any scalar can be overridden from the
environment with POC_QWEN_<SECTION>_<KEY>=value (upper-cased); values are
coerced to the type of the yaml value they replace (bool/int/float/str).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

POC_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = POC_DIR / "config.yaml"
ENV_PREFIX = "POC_QWEN_"


def _coerce(raw: str, like):
    if isinstance(like, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    return raw


def apply_env_overrides(config: dict, env: dict | None = None) -> dict:
    """Overlay POC_QWEN_<SECTION>_<KEY> onto matching scalar keys. Returns a new dict."""
    env = os.environ if env is None else env
    merged = {section: (dict(values) if isinstance(values, dict) else values) for section, values in config.items()}
    for section, values in merged.items():
        if not isinstance(values, dict):
            continue
        for key, current in values.items():
            if isinstance(current, (dict, list)):
                continue
            var = f"{ENV_PREFIX}{section}_{key}".upper()
            raw = env.get(var, "")
            if raw.strip():
                values[key] = _coerce(raw, current)
    return merged


def load_config(path: Path | None = None, env: dict | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_base_dir"] = str(path.resolve().parent)
    return apply_env_overrides(config, env)


def voice_dirs(config: dict) -> list[Path]:
    """Configured voice search paths, resolved against the config file's dir. Missing dirs are dropped."""
    base = Path(config.get("_base_dir", POC_DIR))
    raw = config.get("voices", {}).get("paths", [])
    return [p for p in ((base / entry).resolve() for entry in raw) if p.is_dir()]
