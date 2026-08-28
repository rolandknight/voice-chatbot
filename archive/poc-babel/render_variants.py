"""Render af_heart (Kokoro) reference-clip candidates for cloning "babel" with Chatterbox.

CPU only: forces onnxruntime's CPUExecutionProvider so a GPU session running
elsewhere on the box is never touched. Model files (~340 MB) download into
./models/kokoro/ on first run and are cached.

Each variant is a ~10 s block of assistant-register speech -- Chatterbox Flash
conditions its voice encoder on the first 6 s and its decoder on the first
10 s of the reference, so anything past ~10 s is wasted and question
intonation is avoided. Output per variant: out/<name>.wav (24 kHz mono s16,
lossless -- copy this into voices/ if chosen) and out/<name>.mp3 for
listening, plus out/manifest.json.

    make            # setup + render all three
    .venv/bin/python render_variants.py --only b   # one variant
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Must be set before kokoro_onnx is imported: it reads ONNX_PROVIDER at session
# creation and would otherwise pick CUDA if onnxruntime-gpu were ever installed.
os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "kokoro"
OUT_DIR = HERE / "out"
RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
VOICE = "af_heart"
SR = 24000
TARGET_S = (8.0, 12.0)  # warn outside this; Chatterbox uses only the first 10 s


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    speed: float
    text: str


VARIANTS = [
    Variant(
        "a", "babel-a-intro", 1.0,
        "Hi, I'm Babel. I can set timers, play the radio, check the weather, and "
        "answer your questions. Just say hey Babel, then tell me what you need, and "
        "I'll take it from there.",
    ),
    Variant(
        "b", "babel-b-narration", 0.95,
        "The kitchen light is on, and the kettle should be boiling in about two "
        "minutes. Outside it's cool and clear this morning, so bring a jacket if "
        "you're heading out for a walk.",
    ),
    Variant(
        "c", "babel-c-dialogue", 1.0,
        "Sure, I can do that. Your timer is set for five minutes, and Radio Four is "
        "playing now. I'll let you know when the timer runs out, and then we can "
        "start the next one.",
    ),
]


# --- pure helpers (unit-tested) ---------------------------------------------

def trim_silence(x: np.ndarray, sr: int, threshold: float = 0.01,
                 frame_ms: int = 20, pad_ms: int = 60) -> np.ndarray:
    """Cut leading/trailing frames whose RMS is below `threshold` (float scale),
    keeping `pad_ms` of context on each side. All-silent input is returned as is."""
    n = max(1, int(sr * frame_ms / 1000))
    usable = len(x) - len(x) % n
    if usable == 0:
        return x
    rms = np.sqrt((x[:usable].reshape(-1, n).astype(np.float64) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms >= threshold)
    if loud.size == 0:
        return x
    pad = int(sr * pad_ms / 1000)
    start = max(0, loud[0] * n - pad)
    end = min(len(x), (loud[-1] + 1) * n + pad)
    return x[start:end]


def peak_normalize(x: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak == 0.0:
        return x
    return (x * (10 ** (target_dbfs / 20) / peak)).astype(np.float32)


def fade(x: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n == 0:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


# --- model + I/O ------------------------------------------------------------

def ensure_models(model_dir: Path = MODEL_DIR) -> tuple[Path, Path]:
    import httpx

    model_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"downloading {name} ...", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", f"{RELEASE_BASE}/{name}", follow_redirects=True, timeout=600.0) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)
        print(f"downloaded {name} ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
    return model_dir / MODEL_FILES[0], model_dir / MODEL_FILES[1]


def load_kokoro():
    from kokoro_onnx import Kokoro

    onnx_path, voices_path = ensure_models()
    k = Kokoro(str(onnx_path), str(voices_path))
    providers = k.sess.get_providers()
    if providers != ["CPUExecutionProvider"]:
        sys.exit(f"refusing to run: onnxruntime providers are {providers}, expected CPU only")
    return k


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def wav_to_mp3(wav: Path, mp3: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH (needed for mp3 output)")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
         "-ac", "1", "-ar", str(SR), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)],
        check=True,
    )


def render(kokoro, v: Variant, out_dir: Path) -> dict:
    samples, sr = kokoro.create(v.text, voice=VOICE, speed=v.speed, lang="en-us")
    assert sr == SR, f"unexpected kokoro sample rate {sr}"
    raw_s = len(samples) / sr
    x = fade(peak_normalize(trim_silence(np.asarray(samples, dtype=np.float32), sr)), sr)
    dur = len(x) / sr
    wav, mp3 = out_dir / f"{v.name}.wav", out_dir / f"{v.name}.mp3"
    save_wav(wav, x, sr)
    wav_to_mp3(wav, mp3)
    note = ""
    if not TARGET_S[0] <= dur <= TARGET_S[1]:
        note = f"  <-- outside {TARGET_S[0]:.0f}-{TARGET_S[1]:.0f} s target"
    print(f"{v.name}: {dur:.2f}s (raw {raw_s:.2f}s, speed {v.speed}) -> {mp3.name}{note}")
    return {**asdict(v), "voice": VOICE, "duration_s": round(dur, 2),
            "wav": wav.name, "mp3": mp3.name}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=[v.key for v in VARIANTS], help="render a single variant")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    todo = [v for v in VARIANTS if args.only is None or v.key == args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    kokoro = load_kokoro()
    print(f"kokoro loaded, providers={kokoro.sess.get_providers()}, voice={VOICE}")
    manifest = [render(kokoro, v, args.out) for v in todo]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
