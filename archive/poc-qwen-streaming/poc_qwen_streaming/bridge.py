"""Python side of the PyO3 embed.

The Rust server owns one OS thread that holds the GIL and calls the methods of
`Bridge`. `Bridge` wraps poc-qwen's `Qwen3Engine` (imported from ../poc-qwen,
never copied) and adds the one thing the Gradio app lacked: a single streaming
generator for all three tabs. Everything MLX still runs on the engine's
`mlx-worker` daemon thread; chunks cross to the caller through a queue, and the
queue waits release the GIL, so the Rust thread blocking here never starves it.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from poc_qwen.config import load_config, voice_dirs
from poc_qwen.engine import LANGUAGES, SAMPLE_RATE, SIZES, Qwen3Engine, discover_voices, lang_key, sidecar_transcript
from poc_qwen.text import chunk_text

_DONE = object()


class Chunk:
    """One piece of streamed audio. `audio` is float32 mono at `sample_rate`."""

    __slots__ = ("audio", "sample_rate", "t")

    def __init__(self, audio: np.ndarray, sample_rate: int, t: float):
        self.audio = audio
        self.sample_rate = sample_rate
        self.t = t


class Seam:
    """Holds back a crossfade tail so consecutive model calls blend at the seam.

    mlx-audio keeps decoder state across the chunks of one `generate` call, so
    those need no treatment; this only applies between sentence-chunk calls.
    """

    def __init__(self, n: int):
        self.n = n
        self.tail: np.ndarray | None = None

    def push(self, audio: np.ndarray, first_of_call: bool) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.tail is not None and len(self.tail):
            if first_of_call:
                k = min(len(self.tail), len(audio))
                ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
                head = self.tail[:k] * (1 - ramp) + audio[:k] * ramp
                audio = np.concatenate([head, self.tail[k:], audio[k:]]) if k < len(self.tail) else np.concatenate([head, audio[k:]])
            else:
                audio = np.concatenate([self.tail, audio])  # same call: decoder state is continuous
            self.tail = None
        if self.n and len(audio) > self.n:
            self.tail, audio = audio[-self.n :].copy(), audio[: -self.n]
        elif self.n:
            self.tail = np.concatenate([self.tail, audio]) if self.tail is not None else audio
            audio = audio[:0]
        return audio

    def flush(self) -> np.ndarray:
        out, self.tail = (self.tail if self.tail is not None else np.zeros(0, np.float32)), None
        return out


class Bridge:
    def __init__(self, config_path: str | None = None, loader=None):
        self.cfg = load_config(Path(config_path) if config_path else None)
        self.engine = Qwen3Engine(self.cfg, loader=loader)
        self._voices = discover_voices(voice_dirs(self.cfg))
        self.gen_cfg = self.cfg.get("generation", {})
        self.preload_status: dict = {"state": "idle", "done": [], "pending": [], "s": 0.0}

    # ---- catalogue ---------------------------------------------------------
    def voices(self) -> list[dict]:
        return [{"name": n, "path": str(p), "transcript": sidecar_transcript(p) or ""} for n, p in self._voices.items()]

    def voice_path(self, name: str) -> str | None:
        p = self._voices.get(name)
        return str(p) if p else None

    def speakers(self) -> list[str]:
        return self.engine.speakers()

    def languages(self) -> list[str]:
        return list(LANGUAGES)

    def sizes(self) -> list[str]:
        return list(SIZES)

    def model_info(self) -> dict:
        info = self.engine.model_info()
        info["preload"] = dict(self.preload_status)
        return info

    def preload(self, wait: bool = False) -> dict:
        """Queue load + warm of the configured models and ICL-cache priming for preset voices.

        Runs on the engine's MLX worker; returns immediately (status in
        `preload_status` / `model_info()["preload"]`) unless `wait`. Later
        generations queue behind it on the same worker, which is the point:
        the first click waits for readiness instead of paying for it.
        """
        pcfg = self.cfg.get("preload", {}) or {}
        if not pcfg.get("enabled", True):
            return self.preload_status
        size = str(pcfg.get("size", "1.7B"))
        kinds = list(pcfg.get("models", []))
        voices = pcfg.get("voices", "all")
        names = list(self._voices) if voices == "all" else [v for v in (voices or []) if v in self._voices]
        steps = [("model", k) for k in kinds] + [("voice", n) for n in names]
        self.preload_status = {"state": "running", "done": [], "pending": [f"{a}:{b}" for a, b in steps], "errors": [], "s": 0.0}
        fut = self.engine._submit(self._preload_run, steps, size)
        if wait:
            fut.result()
        return self.preload_status

    def _preload_run(self, steps, size) -> None:
        """Worker-thread body of preload(); every step is best effort."""
        st = self.preload_status
        t0 = time.perf_counter()
        for what, name in steps:
            try:
                if what == "model":
                    kind = {"clone_default": "clone", "clone_small": "clone", "custom_voice": "custom_voice", "voice_design": "voice_design"}[name]
                    self.engine._load(self.engine.model_id(kind, "0.6B" if name == "clone_small" else size))
                else:
                    clip = self._voices[name]
                    text = sidecar_transcript(clip)
                    if not text:
                        raise ValueError("no sidecar transcript; ICL cache not primed")
                    import mlx.core as mx

                    model = self.engine._load(self.engine.model_id("clone", size))
                    # One tiny clone with the exact (ref_text, audio) the UI will send: fills model._icl_cache.
                    for _ in model.generate(text="Hi.", ref_audio=mx.array(self.engine._ref_audio(str(clip))), ref_text=text, lang_code="auto", stream=True, **self._sampling()):
                        pass
                st["done"].append(f"{what}:{name}")
            except Exception as exc:  # noqa: BLE001
                st["errors"].append(f"{what}:{name}: {exc}")
            st["pending"].remove(f"{what}:{name}")
            st["s"] = round(time.perf_counter() - t0, 2)
        st["state"] = "done"
        print(f"[bridge] preload done in {st['s']} s: {len(st['done'])} ok, {len(st['errors'])} errors {st['errors'] or ''}", file=sys.stderr, flush=True)

    def _sampling(self) -> dict:
        return {"temperature": float(self.gen_cfg.get("temperature", 0.9)), "top_p": float(self.gen_cfg.get("top_p", 0.9))}

    # ---- streaming -----------------------------------------------------------
    def _kwargs(self, tab: str, p: dict) -> tuple[str, dict]:
        """Map a tab's UI params onto (model_id, generate kwargs) — the one adapter for mlx-audio's API."""
        lang = lang_key(p.get("language") or "auto")
        size = p.get("size") or "1.7B"
        sampling = self._sampling()
        if tab == "clone":
            ref = p.get("ref_audio")
            if ref is None or (isinstance(ref, str) and not ref.strip()):
                raise ValueError("Reference audio is required")
            kwargs = {"ref_audio": ref, "lang_code": lang, **sampling}
            if not p.get("xvector_only"):
                if not (p.get("ref_text") or "").strip():
                    raise ValueError("Reference text is required unless 'Use x-vector only' is checked")
                kwargs["ref_text"] = p["ref_text"].strip()
            return self.engine.model_id("clone", size), kwargs
        if tab == "custom":
            kwargs = {"voice": p.get("speaker") or "Ryan", "lang_code": lang, **sampling}
            if (p.get("instruct") or "").strip():
                kwargs["instruct"] = p["instruct"].strip()
            return self.engine.model_id("custom_voice", size), kwargs
        if tab == "design":
            if not (p.get("instruct") or "").strip():
                raise ValueError("Voice description is required")
            return self.engine.model_id("voice_design"), {"instruct": p["instruct"].strip(), "lang_code": lang, **sampling}
        raise ValueError(f"unknown tab {tab!r}")

    def model_for(self, tab: str, params: dict) -> str:
        """HF id the given tab/params would use (for telemetry); '' if the params are invalid."""
        try:
            return self._kwargs(tab, params)[0]
        except Exception:
            return ""

    def stream(self, tab: str, params: dict, stop: threading.Event | None = None) -> Iterator[Chunk]:
        """Yield float32 chunks as mlx-audio emits them, for any tab.

        Runs the model on the engine's MLX thread; this generator may be
        consumed from any thread. `stop.set()` ends generation after the
        current chunk.
        """
        text = (params.get("text") or "").strip()
        if not text:
            raise ValueError("Text is empty")
        model_id, kwargs = self._kwargs(tab, params)
        stop = stop or threading.Event()
        interval = float(params.get("interval_s") or self.gen_cfg.get("streaming_interval_s", 0.32))
        chunks = chunk_text(text, int(self.gen_cfg.get("max_chunk_chars", 300)))
        q: queue.Queue = queue.Queue()
        t0 = time.perf_counter()

        def produce():
            try:
                model = self.engine._load(model_id)
                sr = getattr(model, "sample_rate", SAMPLE_RATE)
                if "ref_audio" in kwargs:
                    import mlx.core as mx

                    kwargs["ref_audio"] = mx.array(self.engine._ref_audio(kwargs["ref_audio"]))
                seam = Seam(int(sr * float(self.gen_cfg.get("crossfade_ms", 20)) / 1000))
                for piece in chunks:
                    if stop.is_set():
                        break
                    first = True
                    for r in model.generate(text=piece, stream=True, streaming_interval=interval, **kwargs):
                        audio = seam.push(np.array(r.audio, dtype=np.float32).reshape(-1), first)
                        first = False
                        if len(audio):
                            q.put(Chunk(audio, sr, time.perf_counter() - t0))
                        if stop.is_set():
                            break
                tail = seam.flush()
                if len(tail):
                    q.put(Chunk(tail, sr, time.perf_counter() - t0))
            except BaseException as exc:  # noqa: BLE001 - delivered to the consumer
                q.put(exc)
            finally:
                q.put(_DONE)

        self.engine._submit(produce)
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
