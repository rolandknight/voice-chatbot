#!/usr/bin/env python3
"""Print ALSA capture/playback devices known to this Raspberry Pi."""

from __future__ import annotations

import subprocess


def _run(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return f"{args[0]} not found. Install alsa-utils: sudo apt install alsa-utils\n"
    except subprocess.CalledProcessError as exc:
        return exc.output


def main() -> None:
    print("Capture devices")
    print("===============")
    print(_run(["arecord", "-L"]))
    print("Playback devices")
    print("================")
    print(_run(["aplay", "-L"]))


if __name__ == "__main__":
    main()
