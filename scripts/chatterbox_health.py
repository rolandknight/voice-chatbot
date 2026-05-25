#!/usr/bin/env python3
"""Small health/utility script for the local Chatterbox-TTS-Server.

Usage:
  python scripts/chatterbox_health.py            # ping + list models
  python scripts/chatterbox_health.py reload     # ask the server to reload
                                                 #  (use after editing voices)
  python scripts/chatterbox_health.py speak TEXT [--voice NAME] [--out FILE]

Reads tts.chatterbox.base_url / tts.chatterbox.model from config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

# When run standalone the project root isn't on sys.path; add it so the
# `config` package resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load as load_config  # noqa: E402

_cfg = load_config()
BASE_URL = _cfg.tts.chatterbox.base_url.rstrip("/")
MODEL = _cfg.tts.chatterbox.model


def _ping() -> int:
    try:
        r = httpx.get(f"{BASE_URL}/models", timeout=5.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Chatterbox server unreachable at {BASE_URL}: {e}", file=sys.stderr)
        return 1
    print(f"OK — {BASE_URL}")
    models = data.get("data") or data.get("models") or []
    if models:
        print("Models:")
        for m in models:
            name = m.get("id") if isinstance(m, dict) else str(m)
            print(f"  - {name}")
    return 0


def _reload() -> int:
    # Try the conventional endpoints in order; both have been used by
    # different revs of the server. Treat any 2xx as success.
    for path in ("/admin/reload", "/voices/reload", "/reload"):
        url = BASE_URL.rsplit("/v1", 1)[0] + path
        try:
            r = httpx.post(url, timeout=10.0)
            if 200 <= r.status_code < 300:
                print(f"Reloaded via {url}")
                return 0
        except Exception:
            continue
    print(
        "No reload endpoint responded. Restart the server manually "
        "(Ctrl+C in its terminal, then ./scripts/start_chatterbox.sh).",
        file=sys.stderr,
    )
    return 2


def _speak(text: str, voice: str | None, out: Path) -> int:
    payload = {"model": MODEL, "input": text, "voice": voice or "default"}
    try:
        with httpx.stream(
            "POST", f"{BASE_URL}/audio/speech", json=payload, timeout=60.0
        ) as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except Exception as e:
        print(f"Speech request failed: {e}", file=sys.stderr)
        return 3
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ping", help="ping the server and list models")
    sub.add_parser("reload", help="ask the server to reload voices")
    sp = sub.add_parser("speak", help="synthesize one utterance to a WAV file")
    sp.add_argument("text")
    sp.add_argument("--voice", default=None)
    sp.add_argument("--out", default="chatterbox_out.wav", type=Path)
    args = parser.parse_args()

    if args.cmd in (None, "ping"):
        return _ping()
    if args.cmd == "reload":
        return _reload()
    if args.cmd == "speak":
        return _speak(args.text, args.voice, args.out)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
