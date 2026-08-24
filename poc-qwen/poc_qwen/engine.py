"""Qwen3-TTS engine on mlx-audio.

Owns model lifecycle and every mlx_audio import. Never imports Gradio.
`app.py` and `bench.py` only call the public methods of `Qwen3Engine`.
"""

from __future__ import annotations

import importlib
import logging
import platform
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .text import chunk_text

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
WARMUP_TEXT = "Warming up the model."
SIZES = ("0.6B", "1.7B")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


@dataclass
class Result:
    audio: np.ndarray  # float32 mono
    sample_rate: int
    timings: dict = field(default_factory=dict)  # load_s, gen_s, audio_s, rtf, chunks, model

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def crossfade_concat(parts: list[np.ndarray], sample_rate: int, crossfade_ms: float) -> np.ndarray:
    """Concatenate with a linear crossfade at every seam."""
    parts = [p.astype(np.float32) for p in parts if len(p)]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    n = int(sample_rate * crossfade_ms / 1000)
    out = parts[0]
    for nxt in parts[1:]:
        k = min(n, len(out), len(nxt))
        if k == 0:
            out = np.concatenate([out, nxt])
            continue
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        seam = out[-k:] * (1 - ramp) + nxt[:k] * ramp
        out = np.concatenate([out[:-k], seam, nxt[k:]])
    return out


def load_reference(path_or_array, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load a reference clip as float32 mono at `sample_rate` (mic input may be 48 kHz stereo)."""
    if isinstance(path_or_array, np.ndarray):
        audio, sr = path_or_array, sample_rate
    else:
        from mlx_audio.utils import load_audio  # resamples + mono via mlx

        import mlx.core as mx

        arr = load_audio(str(path_or_array), sample_rate=sample_rate)
        mx.eval(arr)
        return np.array(arr, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1 if audio.shape[1] <= 2 else 0)
    return audio


class Qwen3Engine:
    def __init__(self, cfg: dict, loader: Callable | None = None):
        self.cfg = cfg
        self.models_cfg = cfg["models"]
        self.gen_cfg = cfg.get("generation", {})
        self.max_resident = int(self.models_cfg.get("max_resident", 2))
        self._loader = loader  # test seam; defaults to mlx_audio.tts.utils.load_model
        self._models: OrderedDict[str, object] = OrderedDict()
        self._load_s: dict[str, float] = {}
        self._warm_s: dict[str, float] = {}
        self._ref_cache: dict[str, np.ndarray] = {}
        # Every MLX/Metal call runs on this one persistent daemon thread. Gradio
        # (anyio) executes handlers on pooled worker threads that are torn down
        # after a few idle seconds, and MLX keeps per-thread Metal state that
        # is destroyed with the thread -- which segfaulted the app between
        # requests. A daemon thread is also never joined at interpreter exit,
        # so the same teardown cannot crash the process on shutdown.
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, name="mlx-worker", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            fn, args, kwargs, fut = self._jobs.get()
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn(*args, **kwargs))
                except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                    fut.set_exception(exc)

    def _submit(self, fn, *args, **kwargs) -> Future:
        fut: Future = Future()
        self._jobs.put((fn, args, kwargs, fut))
        return fut

    def _on_worker(self, fn, *args, **kwargs):
        """Run fn on the MLX thread and return its result (re-entrant from the worker itself)."""
        if threading.current_thread() is self._worker:
            return fn(*args, **kwargs)
        return self._submit(fn, *args, **kwargs).result()

    # ---- model registry -------------------------------------------------
    def model_id(self, kind: str, size: str = "1.7B") -> str:
        if kind == "clone":
            return self.models_cfg["clone_small"] if size == "0.6B" else self.models_cfg["clone_default"]
        if kind == "custom_voice":
            mid = self.models_cfg["custom_voice"]
            return mid.replace("1.7B", "0.6B") if size == "0.6B" else mid
        if kind == "voice_design":
            return self.models_cfg["voice_design"]
        raise ValueError(f"unknown model kind {kind!r}")

    def _load(self, model_id: str):
        if model_id in self._models:
            self._models.move_to_end(model_id)
            return self._models[model_id]
        while len(self._models) >= self.max_resident:
            evicted, _ = self._models.popitem(last=False)
            log.info("evicting %s", evicted)
            self._clear_cache()
        loader = self._loader
        if loader is None:
            from mlx_audio.tts.utils import load_model as loader  # noqa: N813

        t0 = time.perf_counter()
        model = loader(model_id)
        self._load_s[model_id] = time.perf_counter() - t0
        self._models[model_id] = model
        self._warm(model_id, model)
        return model

    def _warm(self, model_id: str, model) -> None:
        """Absorb Metal kernel compilation so the first demo utterance is not charged for it."""
        t0 = time.perf_counter()
        try:
            kind = getattr(getattr(model, "config", None), "tts_model_type", "base")
            if kind == "voice_design":
                list(model.generate(text=WARMUP_TEXT, instruct="A calm male voice.", lang_code="english"))
            elif kind == "custom_voice":
                speakers = model.get_supported_speakers() or ["ryan"]
                list(model.generate(text=WARMUP_TEXT, voice=speakers[0], lang_code="english"))
            else:
                # Exercise the ICL clone path too (reference encoder + prefill),
                # otherwise the first real clone still pays ~5 s of kernel
                # compilation on top of the plain-generate warm-up.
                import mlx.core as mx

                noise = mx.array((np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32))
                list(model.generate(text=WARMUP_TEXT, ref_audio=noise, ref_text="Warm up.", lang_code="english"))
        except Exception as exc:  # warm-up is best effort
            log.warning("warm-up failed for %s: %s", model_id, exc)
        self._warm_s[model_id] = time.perf_counter() - t0

    def unload_all(self) -> None:
        self._on_worker(self._unload_all)

    def _unload_all(self) -> None:
        self._models.clear()
        self._clear_cache()

    @staticmethod
    def _clear_cache() -> None:
        try:
            import gc

            import mlx.core as mx

            gc.collect()
            mx.clear_cache()
        except Exception:
            pass

    # ---- generation -----------------------------------------------------
    def _run(self, model_id: str, text: str, gen_kwargs: dict) -> Result:
        model = self._load(model_id)
        sr = getattr(model, "sample_rate", SAMPLE_RATE)
        chunks = chunk_text(text, int(self.gen_cfg.get("max_chunk_chars", 300)))
        parts: list[np.ndarray] = []
        t0 = time.perf_counter()
        for chunk in chunks:
            for r in model.generate(text=chunk, **gen_kwargs):
                parts.append(np.array(r.audio, dtype=np.float32).reshape(-1))
        audio = crossfade_concat(parts, sr, float(self.gen_cfg.get("crossfade_ms", 20)))
        gen_s = time.perf_counter() - t0
        audio_s = len(audio) / sr
        return Result(
            audio=audio,
            sample_rate=sr,
            timings={
                "model": model_id,
                "load_s": round(self._load_s.get(model_id, 0.0), 3),
                "warm_s": round(self._warm_s.get(model_id, 0.0), 3),
                "gen_s": round(gen_s, 3),
                "audio_s": round(audio_s, 3),
                "rtf": round(gen_s / audio_s, 3) if audio_s else None,
                "chunks": len(chunks),
                "chars": len(text),
            },
        )

    def _sampling(self) -> dict:
        return {
            "temperature": float(self.gen_cfg.get("temperature", 0.9)),
            "top_p": float(self.gen_cfg.get("top_p", 0.9)),
        }

    def _ref_audio(self, ref) -> np.ndarray:
        key = str(ref) if not isinstance(ref, np.ndarray) else None
        if key and key in self._ref_cache:
            return self._ref_cache[key]
        audio = load_reference(ref)
        if key:
            self._ref_cache[key] = audio
        return audio

    def clone(self, text, ref_audio, ref_text: str | None, language="auto", size="1.7B", *, xvector_only=False) -> Result:
        return self._on_worker(self._clone, text, ref_audio, ref_text, language, size, xvector_only)

    def _clone(self, text, ref_audio, ref_text, language, size, xvector_only) -> Result:
        if not text.strip():
            raise ValueError("Target text is empty")
        ref = self._ref_audio(ref_audio)
        import mlx.core as mx

        kwargs = {"ref_audio": mx.array(ref), "lang_code": lang_key(language), **self._sampling()}
        if not xvector_only:
            if not (ref_text or "").strip():
                raise ValueError("Reference text is required unless 'Use x-vector only' is checked")
            kwargs["ref_text"] = ref_text.strip()
        result = self._run(self.model_id("clone", size), text, kwargs)
        result.timings["mode"] = "xvector" if xvector_only else "icl"
        return result

    def custom_voice(self, text, speaker, language="auto", instruct="", size="1.7B") -> Result:
        return self._on_worker(self._custom_voice, text, speaker, language, instruct, size)

    def _custom_voice(self, text, speaker, language, instruct, size) -> Result:
        if not text.strip():
            raise ValueError("Text is empty")
        kwargs = {"voice": speaker, "lang_code": lang_key(language), **self._sampling()}
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct.strip()
        return self._run(self.model_id("custom_voice", size), text, kwargs)

    def voice_design(self, text, instruct, language="auto") -> Result:
        return self._on_worker(self._voice_design, text, instruct, language)

    def _voice_design(self, text, instruct, language) -> Result:
        if not text.strip():
            raise ValueError("Text is empty")
        if not (instruct or "").strip():
            raise ValueError("Voice description is required")
        kwargs = {"instruct": instruct.strip(), "lang_code": lang_key(language), **self._sampling()}
        return self._run(self.model_id("voice_design"), text, kwargs)

    def stream_clone(self, text, ref_audio, ref_text, language="auto", size="1.7B", interval_s=0.32) -> Iterator[np.ndarray]:
        """Iteration-2 seam: yields float32 chunks as mlx-audio emits them.

        Generation runs on the MLX thread; chunks cross to the caller through a
        queue, so this can be consumed from any thread (or an async loop).
        """
        q: queue.Queue = queue.Queue()
        done = object()

        def produce():
            try:
                model = self._load(self.model_id("clone", size))
                import mlx.core as mx

                for r in model.generate(
                    text=text,
                    ref_audio=mx.array(self._ref_audio(ref_audio)),
                    ref_text=ref_text,
                    lang_code=lang_key(language),
                    stream=True,
                    streaming_interval=interval_s,
                    **self._sampling(),
                ):
                    q.put(np.array(r.audio, dtype=np.float32).reshape(-1))
            except BaseException as exc:  # propagate to the consumer
                q.put(exc)
            finally:
                q.put(done)

        self._submit(produce)
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    # ---- helpers ----------------------------------------------------------
    def transcribe(self, audio_path) -> str:
        return self._on_worker(self._transcribe, audio_path)

    def _transcribe(self, audio_path) -> str:
        tcfg = self.cfg.get("transcribe", {})
        if not tcfg.get("enabled", True):
            return ""
        try:
            import mlx_whisper
        except ImportError:
            return ""
        out = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=tcfg.get("model", "mlx-community/whisper-base.en-mlx"))
        return (out.get("text") or "").strip()

    def speakers(self) -> list[str]:
        """Preset speakers of the CustomVoice model; falls back to the documented list without loading it."""
        for mid, model in self._models.items():
            if "CustomVoice" in mid:
                return [s.capitalize() for s in model.get_supported_speakers()]
        return ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]

    def languages(self) -> list[str]:
        return list(LANGUAGES)

    def model_info(self) -> dict:
        return self._on_worker(self._model_info)

    def _model_info(self) -> dict:
        info = {
            "chip": platform.processor() or platform.machine(),
            "resident": list(self._models),
            "load_s": self._load_s,
            "warm_s": self._warm_s,
        }
        try:
            import mlx.core as mx

            info["mlx"] = mx.__version__
            info["active_gb"] = round(mx.get_active_memory() / 2**30, 2)
            info["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
            info["chip"] = mx.device_info().get("device_name", info["chip"])
        except Exception:
            pass
        try:
            info["mlx_audio"] = importlib.metadata.version("mlx-audio")
        except Exception:
            pass
        return info


# Display name -> mlx-audio lang_code (the codec_language_id keys).
LANGUAGES = {
    "Auto": "auto",
    "English": "english",
    "Chinese": "chinese",
    "Japanese": "japanese",
    "Korean": "korean",
    "German": "german",
    "French": "french",
    "Russian": "russian",
    "Portuguese": "portuguese",
    "Spanish": "spanish",
    "Italian": "italian",
}


def lang_key(language: str) -> str:
    return LANGUAGES.get(language, (language or "auto").lower())


def discover_voices(dirs: list[Path]) -> dict[str, Path]:
    """name -> clip path for every audio file in the given dirs (first wins on name clash)."""
    found: dict[str, Path] = {}
    for d in dirs:
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in AUDIO_EXTS and p.stem not in found:
                found[p.stem] = p
    return found


def sidecar_transcript(clip: Path) -> str | None:
    txt = clip.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else None
