"""On-device wake-word detection (openWakeWord), shared by the wake test and
the WebRTC client's connect-on-wake lifecycle.

Mirrors the server's `wakeword_detector.WakeWordDetector` so device-side wake
uses the same models, thresholds, and cooldown — the difference is only *where*
it runs: here it gates whether the device opens a WebRTC peer at all, instead of
gating a turn inside an always-connected pipeline.

openWakeWord expects 16 kHz mono int16, fed in 1280-sample (80 ms) chunks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

# 80 ms at 16 kHz — openWakeWord's expected per-call chunk.
CHUNK_SAMPLES = 1280


@dataclass
class WakeEvent:
    """A model firing above threshold (after cooldown)."""

    model_key: str
    score: float
    persona: Optional[str] = None
    backend: Optional[str] = None


class LocalWakeDetector:
    """Runs openWakeWord on 16 kHz mono int16 chunks and reports the first
    model that crosses `threshold` (respecting a per-model `cooldown_secs`).

    Args:
        model_paths: filesystem paths to .onnx wake models (or bundled
            openWakeWord keys). The dict key openWakeWord reports is the file
            stem, e.g. "hey_babel".
        persona_for_model / backend_for_model: map that stem to the persona /
            LLM backend to request for the session when the model fires.
        threshold: per-chunk probability (default 0.5, matching config.yaml).
        cooldown_secs: min gap between consecutive fires of the same model.
    """

    def __init__(
        self,
        *,
        model_paths: list[str],
        persona_for_model: Optional[dict[str, str]] = None,
        backend_for_model: Optional[dict[str, str]] = None,
        threshold: float = 0.5,
        cooldown_secs: float = 1.5,
    ) -> None:
        from openwakeword.model import Model

        self._model = Model(wakeword_models=list(model_paths), inference_framework="onnx")
        self._persona_for_model = dict(persona_for_model or {})
        self._backend_for_model = dict(backend_for_model or {})
        self._threshold = threshold
        self._cooldown_secs = cooldown_secs
        self._last_fire: dict[str, float] = {}
        self.model_keys = sorted(self._model.models.keys())

    def scores(self, samples_int16) -> dict[str, float]:
        """Raw per-model probabilities for a 1280-sample chunk (for metering /
        threshold tuning). Advances openWakeWord's streaming state."""
        return {k: float(v) for k, v in self._model.predict(samples_int16).items()}

    def process(self, samples_int16) -> Optional[WakeEvent]:
        """Feed one chunk; return a WakeEvent if a model fired, else None."""
        now = time.monotonic()
        best: Optional[tuple[str, float]] = None
        for key, score in self._model.predict(samples_int16).items():
            if score < self._threshold:
                continue
            if now - self._last_fire.get(key, 0.0) < self._cooldown_secs:
                continue
            if best is None or score > best[1]:
                best = (key, float(score))
        if best is None:
            return None
        key, score = best
        self._last_fire[key] = now
        return WakeEvent(
            model_key=key,
            score=score,
            persona=self._persona_for_model.get(key),
            backend=self._backend_for_model.get(key),
        )
