"""Tiny FastAPI wrapper around stable-audio-tools so the voice-chatbot
can hit it with a plain POST like it does Woosh.

Loaded by scripts/start_stable_audio.sh inside vendor/stable-audio/.venv,
where stable_audio_tools is importable. Listens on STABLE_AUDIO_PORT
(default 8006) and exposes:

    POST /generate     -> {"prompt": str, "seconds": float (optional),
                           "steps": int (optional), "cfg_scale": float (optional),
                           "seed": int (optional)}
                          returns FLAC bytes (audio/flac)
    GET  /docs         -> FastAPI docs (used by run.sh's readiness probe)
    GET  /ping         -> liveness

Model: stabilityai/stable-audio-open-1.0 (44.1 kHz stereo, ~1.2 GB).
Device: MPS when available (Apple Silicon Metal), else CPU. PyTorch ops
without an MPS kernel fall back to CPU via PYTORCH_ENABLE_MPS_FALLBACK=1
(exported by start_stable_audio.sh).
"""

from __future__ import annotations

import io
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stable_audio_server")

MODEL_NAME = os.environ.get(
    "STABLE_AUDIO_MODEL", "stabilityai/stable-audio-open-1.0"
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Module-level so the lifespan handler can populate it once and reuse.
_state: dict = {"model": None, "config": None, "device": _pick_device()}


def _patch_apg_project_for_mps() -> None:
    """stable-audio-tools 0.0.20 calls `.double()` (float64) inside
    DiffusionTransformer.apg_project for numerical stability, but MPS
    refuses float64 outright — `PYTORCH_ENABLE_MPS_FALLBACK` doesn't
    cover dtype casts, so the call raises and aborts generation.

    Patch it to run the projection in the input dtype (float32) on MPS.
    Generation quality at 100 steps is indistinguishable from float64 for
    short SFX clips.
    """
    if _state["device"] != "mps":
        return
    from stable_audio_tools.models import dit as _dit

    def apg_project_mps(self, v0, v1, padding_mask=None):
        dtype = v0.dtype
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).to(dtype)
            v0_masked = v0 * mask
            v1_masked = v1 * mask
            v1_norm = v1_masked.norm(dim=[-1, -2], keepdim=True).clamp(min=1e-8)
            v1_normalized = v1_masked / v1_norm
            v0_parallel = (v0_masked * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized
            v0_orthogonal = (v0 - (v0 * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized) * mask
        else:
            v1n = torch.nn.functional.normalize(v1, dim=[-1, -2])
            v0_parallel = (v0 * v1n).sum(dim=[-1, -2], keepdim=True) * v1n
            v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    _dit.DiffusionTransformer.apg_project = apg_project_mps
    logger.info("Patched stable_audio_tools apg_project (MPS-safe float32 path).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading {MODEL_NAME} on device={_state['device']} ...")
    _patch_apg_project_for_mps()
    model, model_config = get_pretrained_model(MODEL_NAME)
    model = model.to(_state["device"])
    _state["model"] = model
    _state["config"] = model_config
    logger.info(
        f"Model ready: sample_rate={model_config['sample_rate']} "
        f"sample_size={model_config['sample_size']}"
    )
    yield
    _state["model"] = None
    _state["config"] = None


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    seconds: float = Field(default=5.0, ge=0.5, le=47.0)
    steps: int = Field(default=100, ge=10, le=250)
    cfg_scale: float = Field(default=7.0, ge=1.0, le=15.0)
    seed: Optional[int] = None


def _to_flac_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """generate_diffusion_cond returns a [B, C, T] float tensor in [-1, 1].

    Take the first batch, clamp, convert to int16, write as FLAC.
    """
    wav = audio[0].detach().cpu().clamp(-1.0, 1.0)
    # torchaudio.save wants shape [channels, time]. SAO returns stereo.
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(
        buf,
        wav,
        sample_rate=sample_rate,
        format="flac",
        bits_per_sample=16,
    )
    buf.seek(0)
    return buf.read()


@app.get("/ping")
def ping():
    return {"status": "ok", "model": MODEL_NAME, "device": _state["device"]}


@app.post("/generate")
async def generate(req: GenerateRequest):
    model = _state["model"]
    config = _state["config"]
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    sample_rate = config["sample_rate"]
    sample_size = config["sample_size"]
    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    conditioning = [
        {
            "prompt": req.prompt,
            "seconds_start": 0,
            "seconds_total": float(req.seconds),
        }
    ]

    logger.info(
        f"generate prompt={req.prompt!r} seconds={req.seconds} "
        f"steps={req.steps} cfg={req.cfg_scale} seed={seed}"
    )
    start = time.time()

    try:
        audio = generate_diffusion_cond(
            model,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=_state["device"],
            seed=seed,
        )
    except Exception as exc:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    data = _to_flac_bytes(audio, sample_rate)
    elapsed = time.time() - start
    logger.info(f"generated {len(data)} bytes in {elapsed:.2f}s")
    return StreamingResponse(io.BytesIO(data), media_type="audio/flac")
