from __future__ import annotations

import asyncio
import random
import re
import tempfile
import time
from pathlib import Path

import httpx
from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext

# Prompts matching this pattern route to Stable Audio Open by default;
# everything else routes to Woosh. SAO produces 44.1 kHz stereo and handles
# bodily/comedic sounds much better than Woosh's foley training.
_SAO_KEYWORD_RE = re.compile(
    r"\b(fart|farts|farting|"
    r"burp|burps|burping|belch|belches|belching|"
    r"laugh|laughs|laughing|laughter|giggle|giggles|giggling|chuckle|chuckles|"
    r"cough|coughs|coughing|sneeze|sneezes|sneezing|"
    r"hiccup|hiccups|hiccuping|raspberry|raspberries|"
    r"snore|snores|snoring|yawn|yawns|yawning|"
    r"gulp|gulps|gulping|slurp|slurps|slurping)\b",
    re.IGNORECASE,
)


def _route_sfx_backend(
    description: str, available: set[str], override: str | None
) -> str:
    if not available:
        raise ValueError("no SFX backends configured")
    if len(available) == 1:
        return next(iter(available))
    if override in {"woosh", "stable_audio"} and override in available:
        return override
    if "stable_audio" in available and _SAO_KEYWORD_RE.search(description):
        return "stable_audio"
    if "woosh" in available:
        return "woosh"
    return next(iter(available))


def _build_sfx_body(backend: str, description: str) -> dict:
    if backend == "woosh":
        return {
            "version": "0.1",
            "token": "string",
            "args": {
                "model": "Woosh-DFlow",
                "prompt": description,
                "cfg": 3.0,
                "sampler": "heun",
                "num_steps": 5,
                "sigma_min": 0.0001,
                "sigma_max": 80,
                "rho": 7,
                "S_churn": 1,
                "S_min": 0,
                "S_noise": 1,
                "guidance_scale": 7.5,
                "noise_scheduler": "karras",
                "seed": random.randint(0, 2**32 - 1),
            },
        }
    if backend == "stable_audio":
        # 40 steps × 3s lands around ~30s on MPS (vs ~110s at 100×5s).
        # Quality difference for short comedic SFX is hard to notice.
        return {
            "prompt": description,
            "seconds": 3.0,
            "steps": 40,
            "cfg_scale": 7.0,
            "seed": random.randint(0, 2**32 - 1),
        }
    raise ValueError(f"unknown SFX backend: {backend!r}")


async def _generate_and_play_sfx(
    backend: str, backend_url: str, description: str, wait_evt: asyncio.Event
) -> None:
    body = _build_sfx_body(backend, description)
    timeout = 120.0 if backend == "stable_audio" else 60.0
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{backend_url}/generate", json=body)
            r.raise_for_status()
        out = Path(tempfile.gettempdir()) / f"sfx_{int(time.time() * 1000)}.flac"
        out.write_bytes(r.content)
        try:
            await asyncio.wait_for(wait_evt.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning(
                "SFX: timed out waiting for bot to finish speaking; playing anyway"
            )
        await asyncio.create_subprocess_exec(
            "mpv", "--no-terminal", "--no-video", str(out),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning(
            f"SFX generate/play failed for {description!r} via {backend}: {e}"
        )


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    desc = (params.arguments.get("description") or "").strip()
    if not desc:
        await params.result_callback("Tell me what sound to make.")
        return
    spoken_desc = desc.rstrip(".!?")
    await params.result_callback(f"Playing a {spoken_desc}.")
    available = set(ctx.sfx_backends.keys())
    backend = _route_sfx_backend(desc, available, ctx.sfx_backend_override)
    logger.info(f"SFX: routing {desc!r} -> {backend}")
    wait_evt = ctx.sfx_tracker.snapshot_next_silence()
    asyncio.create_task(
        _generate_and_play_sfx(backend, ctx.sfx_backends[backend], desc, wait_evt)
    )
