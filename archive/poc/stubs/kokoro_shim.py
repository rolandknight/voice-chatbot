"""Kokoro TTS shim speaking the OpenAI /v1/audio/speech protocol (port 8880).

Per poc/CONTRACT.md: request `{"model": "kokoro", "voice": "af_heart",
"input": str, "response_format": "pcm"}` -> raw little-endian s16 mono PCM
at 24000 Hz, no WAV header. Model files are downloaded into
poc/models/kokoro/ on first start and cached. Run with:

    uvicorn kokoro_shim:app --host 127.0.0.1 --port 8880
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

POC_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = POC_DIR / "models" / "kokoro"
RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download"
MODEL_FILES = {
    "kokoro-v1.0.onnx": f"{RELEASE_BASE}/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{RELEASE_BASE}/model-files-v1.0/voices-v1.0.bin",
}
DEFAULT_VOICE = "af_heart"

_kokoro = None  # loaded in lifespan; None means /health is not ok yet


def ensure_models(model_dir: Path = MODEL_DIR) -> tuple[Path, Path]:
    """Download kokoro-onnx model files if missing; return (onnx, voices) paths."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_FILES.items():
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"[kokoro_shim] downloading {name} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)
        print(f"[kokoro_shim] downloaded {name} ({dest.stat().st_size} bytes)")
    return model_dir / "kokoro-v1.0.onnx", model_dir / "voices-v1.0.bin"


def load_kokoro():
    """Ensure model files exist and return a loaded Kokoro instance."""
    from kokoro_onnx import Kokoro

    onnx_path, voices_path = ensure_models()
    return Kokoro(str(onnx_path), str(voices_path))


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _kokoro
    _kokoro = load_kokoro()
    print("[kokoro_shim] model loaded")
    yield


app = FastAPI(title="poc-kokoro-shim", lifespan=_lifespan)


class SpeechReq(BaseModel):
    model: str = "kokoro"
    voice: str = DEFAULT_VOICE
    input: str
    response_format: str = "pcm"
    speed: Optional[float] = 1.0


def synth_pcm(kokoro, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Synthesize text -> raw s16le mono PCM at 24 kHz."""
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    assert sample_rate == 24000, f"unexpected kokoro sample rate {sample_rate}"
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


@app.post("/v1/audio/speech")
def speech(req: SpeechReq) -> Response:
    if _kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input:
        raise HTTPException(status_code=422, detail="empty input")
    pcm = synth_pcm(_kokoro, req.input, voice=req.voice or DEFAULT_VOICE, speed=req.speed or 1.0)
    return Response(content=pcm, media_type="application/octet-stream")


@app.get("/health")
def health() -> dict:
    if _kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8880)
