"""Go/no-go gate: clone voices/one-one.mp3 with the 0.6B Base model.

Transcribes the reference with mlx-whisper (the transcript is required by
Qwen3-TTS ICL cloning), generates one sentence, writes reports/smoke.wav and
prints timings. If this does not sound like the reference, stop the plan.
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent
REF = HERE.parent / "voices" / "one-one.mp3"
MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
TEXT = "I checked the calendar for tomorrow and you have three meetings, the first one starting at nine fifteen."


def main() -> int:
    t0 = time.perf_counter()
    import mlx_whisper

    ref_text = mlx_whisper.transcribe(str(REF), path_or_hf_repo="mlx-community/whisper-base.en-mlx")["text"].strip()
    print(f"transcribe {time.perf_counter() - t0:.2f}s: {ref_text!r}")

    from mlx_audio.tts.utils import load_model

    t0 = time.perf_counter()
    model = load_model(MODEL)
    print(f"load {MODEL} {time.perf_counter() - t0:.2f}s")

    for label in ("cold", "warm"):
        t0 = time.perf_counter()
        results = list(model.generate(text=TEXT, ref_audio=str(REF), ref_text=ref_text, lang_code="english"))
        audio = np.concatenate([np.array(r.audio) for r in results]).astype(np.float32)
        mx.eval(mx.array(0))
        gen = time.perf_counter() - t0
        dur = len(audio) / model.sample_rate
        print(f"{label}: gen {gen:.2f}s audio {dur:.2f}s rtf {gen / dur:.2f} peak {mx.get_peak_memory() / 2**30:.2f} GiB")

    out = HERE / "reports" / "smoke.wav"
    sf.write(out, audio, model.sample_rate)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
