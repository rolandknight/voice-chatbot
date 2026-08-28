"""Cached Config loader.

`load()` is the single entry point — call it once at startup or anywhere else
that needs the live config. .env is pulled into os.environ before Config()
is built so the secrets-from-env source sees the values.

Run as a module for a debug dump:
    python -m config.loader --print-effective
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from dotenv import load_dotenv

from config.schema import Config

_cached: Optional[Config] = None


def load() -> Config:
    global _cached
    if _cached is None:
        load_dotenv(override=True)
        _cached = Config()
    return _cached


def get() -> Config:
    return load()


def _print_effective() -> None:
    cfg = load()
    print(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config loader debug entry.")
    parser.add_argument(
        "--print-effective",
        action="store_true",
        help="Dump the resolved Config tree as JSON.",
    )
    args = parser.parse_args()
    if args.print_effective:
        _print_effective()
    else:
        parser.print_help()
