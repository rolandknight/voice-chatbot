#!/usr/bin/env python3
"""Standalone on-device wake-word test.

Captures the mic via PortAudio (sounddevice) at 16 kHz mono and runs
openWakeWord locally — no WebRTC, no server. Use it to confirm, on the Mac with
the Jabra (or on the Pi), that "hey babel" / "hey marvin" are detected reliably
before wiring wake into the WebRTC client's connect-on-wake lifecycle.

It also prints a live input-level meter, which doubles as a mic-health check
(a flat -inf/-90 dB meter means the process isn't getting mic audio — check the
Jabra mute button and the OS microphone permission for your terminal).

    make run-wake-test                     # Jabra @ default threshold
    make run-wake-test THRESHOLD=0.6
    python devices/rpi5/wake_test.py --device Jabra --threshold 0.5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from wake import CHUNK_SAMPLES, LocalWakeDetector

# Repo root: devices/rpi5/ -> devices/ -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "wakeword"

# stem -> persona (matches config.yaml wake.models)
DEFAULT_PERSONA_FOR_MODEL = {"hey_babel": "babel", "hey_marvin": "marvin"}


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)


def _meter(db: float, width: int = 30) -> str:
    # Map -60..0 dBFS onto the bar.
    frac = 0.0 if db == -math.inf else max(0.0, min(1.0, (db + 60.0) / 60.0))
    filled = int(frac * width)
    return "#" * filled + "-" * (width - filled)


def _resolve_models(model_dir: Path, names: list[str]) -> list[str]:
    paths = []
    for name in names:
        p = model_dir / f"{name}.onnx"
        if not p.exists():
            sys.exit(f"wake model not found: {p}")
        paths.append(str(p))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=os.environ.get("WAKE_INPUT_DEVICE", "Jabra"),
        help="PortAudio input device: index, name substring (e.g. 'Jabra'), or "
        "'default'. Default: 'Jabra'.",
    )
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory of *.onnx wake models."
    )
    parser.add_argument(
        "--models",
        default="hey_babel,hey_marvin",
        help="Comma-separated model stems to load.",
    )
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("THRESHOLD", "0.5")))
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument(
        "--no-meter", action="store_true", help="Disable the live level meter."
    )
    args = parser.parse_args()

    device = None if args.device == "default" else args.device
    model_paths = _resolve_models(Path(args.model_dir), args.models.split(","))

    print(f"Loading openWakeWord models: {args.models} (threshold={args.threshold})...")
    detector = LocalWakeDetector(
        model_paths=model_paths,
        persona_for_model=DEFAULT_PERSONA_FOR_MODEL,
        threshold=args.threshold,
        cooldown_secs=args.cooldown,
    )
    print(f"Loaded: {detector.model_keys}")
    print(f"Listening on device {args.device!r} @ 16 kHz. Say a wake word. Ctrl+C to stop.\n")

    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=CHUNK_SAMPLES, device=device
    )
    stream.start()
    peak_db = -math.inf
    last_meter = 0.0
    try:
        while True:
            data, overflowed = stream.read(CHUNK_SAMPLES)
            samples = data[:, 0]
            ev = detector.process(samples)
            if ev is not None:
                print(
                    f"\n✓ WAKE  {ev.model_key!r}  score={ev.score:.3f}  "
                    f"persona={ev.persona!r}"
                )
                peak_db = -math.inf
            if not args.no_meter:
                db = _dbfs(samples)
                peak_db = max(peak_db, db)
                now = time.monotonic()
                if now - last_meter >= 0.1:
                    last_meter = now
                    shown = -99.9 if db == -math.inf else db
                    sys.stdout.write(f"\rlevel [{_meter(db)}] {shown:6.1f} dBFS  ")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
