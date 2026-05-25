"""Audio-based wake-word detection via openwakeword.

Replaces the text-based WakePhraseUserTurnStartStrategy (which matched
Whisper transcriptions against regexes) with raw-audio inference that fires
during the wake word, not after the whole utterance has been transcribed.

Pipeline shape:

  transport.input() -> WakeWordDetector -> stt -> ... -> context_aggregator.user()

WakeWordDetector consumes InputAudioRawFrame, buffers samples into 1280-sample
(80 ms) chunks at 16 kHz, runs openwakeword on each chunk, and on a match:
  1. switches the active persona via apply_skill_persona_switch, and
  2. pushes a WakeWordDetectedFrame downstream.

WakeWordUserTurnStartStrategy is plugged into UserTurnStrategies(start=[...])
in place of WakePhraseUserTurnStartStrategy. It watches for
WakeWordDetectedFrame and signals turn start. Timeout behaviour matches the
text-based strategy: IDLE drops transcriptions, AWAKE forwards everything and
re-arms the inactivity timer on activity frames.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    DataFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager

from persona_router import PersonaConfig, PersonaState, apply_skill_persona_switch


# openwakeword's expected chunk: 80 ms at 16 kHz = 1280 samples = 2560 bytes
# of 16-bit PCM. The transport may deliver larger or smaller chunks, so we
# buffer and slice.
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2


@dataclass
class WakeWordDetectedFrame(DataFrame):
    """Emitted by WakeWordDetector when an openwakeword model fires.

    `model_key` is the openwakeword model identifier as it appears in
    `Model.predict()` output (e.g. "hey_babel", "hey_jarvis_v0.1"). `score`
    is the per-model probability that crossed the threshold.
    """

    model_key: str = ""
    score: float = 0.0


class WakeWordDetector(FrameProcessor):
    """Runs openwakeword on incoming raw audio and emits WakeWordDetectedFrame
    when any registered model crosses its threshold.

    Place between transport.input() and stt. Audio frames are always passed
    through unchanged so Whisper still sees them.

    Args:
        model_paths_or_keys: list of either filesystem paths to .onnx files or
            bundled openwakeword model keys (e.g. "hey_jarvis_v0.1"). Order
            doesn't matter; the openwakeword model dict is keyed by the file
            stem.
        persona_for_model: maps the openwakeword model key (the dict key in
            `model.predict()` output) to the persona id to activate when that
            model fires. Models without a mapping still emit a frame but
            don't switch persona.
        threshold: per-chunk probability threshold (default 0.5).
        cooldown_secs: minimum time between consecutive fires of the SAME
            model. Prevents one wake event from triggering on every chunk
            while the user is still saying the wake phrase.
        persona_config / persona_state: optional, used to apply persona
            switches. If either is None, persona switching is skipped.
    """

    def __init__(
        self,
        *,
        model_paths_or_keys: list[str],
        persona_for_model: dict[str, str],
        threshold: float = 0.5,
        cooldown_secs: float = 1.5,
        persona_config: Optional[PersonaConfig] = None,
        persona_state: Optional[PersonaState] = None,
    ):
        super().__init__()
        # Import lazily so module import doesn't pay the onnxruntime cost when
        # the detector isn't actually used (e.g. unit tests of other modules).
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=model_paths_or_keys,
            inference_framework="onnx",
        )
        self._persona_for_model = dict(persona_for_model)
        self._threshold = threshold
        self._cooldown_secs = cooldown_secs
        self._persona_config = persona_config
        self._persona_state = persona_state

        self._buffer = bytearray()
        # Last-fire monotonic timestamps per model key. Used for cooldown.
        self._last_fire_ts: dict[str, float] = {}

        logger.info(
            f"WakeWordDetector loaded models: {sorted(self._model.models.keys())} "
            f"(threshold={threshold}, cooldown={cooldown_secs}s)"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, InputAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            await self._on_audio(frame)
        await self.push_frame(frame, direction)

    async def _on_audio(self, frame: InputAudioRawFrame) -> None:
        # openwakeword expects 16 kHz mono 16-bit PCM. Transport is configured
        # for exactly that — if a future change introduces resampling, this
        # is where it would go.
        self._buffer.extend(frame.audio)
        while len(self._buffer) >= CHUNK_BYTES:
            chunk = bytes(self._buffer[:CHUNK_BYTES])
            del self._buffer[:CHUNK_BYTES]
            await self._predict_chunk(chunk)

    async def _predict_chunk(self, chunk: bytes) -> None:
        import numpy as np

        samples = np.frombuffer(chunk, dtype=np.int16)
        # predict() is sync and CPU-bound (a few hundred microseconds on
        # Apple Silicon). Keep it inline to preserve frame ordering — running
        # it in a thread would let frames overtake each other.
        scores = self._model.predict(samples)

        now = asyncio.get_event_loop().time()
        for model_key, score in scores.items():
            if score < self._threshold:
                continue
            last = self._last_fire_ts.get(model_key, 0.0)
            if now - last < self._cooldown_secs:
                continue
            self._last_fire_ts[model_key] = now
            await self._fire(model_key, float(score))

    async def _fire(self, model_key: str, score: float) -> None:
        persona_id = self._persona_for_model.get(model_key)
        logger.info(
            f"Wake word detected: {model_key!r} score={score:.3f} "
            f"persona={persona_id!r}"
        )
        if (
            persona_id is not None
            and self._persona_config is not None
            and self._persona_state is not None
        ):
            apply_skill_persona_switch(
                self._persona_config, self._persona_state, persona_id
            )
        await self.push_frame(
            WakeWordDetectedFrame(model_key=model_key, score=score),
            FrameDirection.DOWNSTREAM,
        )


class _WakeState(enum.Enum):
    IDLE = "idle"
    AWAKE = "awake"


class WakeWordUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User-turn-start strategy gated on WakeWordDetectedFrame.

    Mirrors the structure of pipecat's WakePhraseUserTurnStartStrategy but
    listens for an audio-based wake event instead of regex matches on
    transcriptions. Place first in UserTurnStrategies(start=[...]).

    Event handlers available:
      - on_wake_word_detected(strategy, model_key, score)
      - on_wake_word_timeout(strategy)
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        is_busy: Optional[Callable[[], bool]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._timeout = timeout
        # When provided and returning True at timeout-expiry, defer the
        # AWAKE→IDLE transition by another full period. Used to keep the
        # wake session alive while a tool call or LLM response is mid-flight.
        self._is_busy = is_busy

        self._state = _WakeState.IDLE
        self._timeout_event = asyncio.Event()
        self._timeout_task: asyncio.Task | None = None

        self._register_event_handler("on_wake_word_detected")
        self._register_event_handler("on_wake_word_timeout")

    @property
    def state(self) -> _WakeState:
        return self._state

    async def setup(self, task_manager: BaseTaskManager):
        await super().setup(task_manager)
        if not self._timeout_task:
            self._timeout_task = self.task_manager.create_task(
                self._timeout_task_handler(),
                f"{self}::_timeout_task_handler",
            )

    async def cleanup(self):
        await super().cleanup()
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def reset(self):
        await super().reset()
        if self._state == _WakeState.AWAKE:
            self._refresh_timeout()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        await super().process_frame(frame)
        if self._state == _WakeState.IDLE:
            return await self._process_idle(frame)
        return await self._process_awake(frame)

    async def _process_idle(self, frame: Frame) -> ProcessFrameResult:
        # Wake event arrives BEFORE Whisper finishes the utterance (openwakeword
        # is real-time on raw audio; STT is buffered until VAD silence). When
        # the wake frame shows up we flip to AWAKE so the TranscriptionFrame
        # that follows is allowed through.
        if isinstance(frame, WakeWordDetectedFrame):
            self._transition_to_awake(frame.model_key, frame.score)
            return ProcessFrameResult.STOP
        if isinstance(frame, TranscriptionFrame):
            # Speech before the wake event must not leak into the LLM context.
            await self.trigger_reset_aggregation()
        return ProcessFrameResult.STOP

    async def _process_awake(self, frame: Frame) -> ProcessFrameResult:
        # Same activity frames as the text-based strategy. Each one resets the
        # inactivity timer; when the timer expires we drop back to IDLE.
        if isinstance(
            frame,
            (UserSpeakingFrame, BotSpeakingFrame, TranscriptionFrame,
             VADUserStartedSpeakingFrame),
        ):
            self._refresh_timeout()
        return ProcessFrameResult.CONTINUE

    def _transition_to_awake(self, model_key: str, score: float) -> None:
        self._state = _WakeState.AWAKE
        self._refresh_timeout()
        self.task_manager.create_task(
            self._fire_wake_handlers(model_key, score),
            f"{self}::on_wake_word_detected",
        )

    async def _fire_wake_handlers(self, model_key: str, score: float) -> None:
        # Trigger pipecat's turn-start signal AND the user-facing event.
        # Order: trigger turn first so downstream aggregation flips into
        # turn-active mode before any listener side effects.
        await self.trigger_user_turn_started()
        await self._call_event_handler("on_wake_word_detected", model_key, score)

    def _transition_to_idle(self) -> None:
        logger.debug(f"{self} wake-word timeout, returning to IDLE")
        self._state = _WakeState.IDLE
        self.task_manager.create_task(
            self._call_event_handler("on_wake_word_timeout"),
            f"{self}::on_wake_word_timeout",
        )

    def force_idle(self) -> None:
        """Drop back to IDLE immediately, firing the timeout handlers.

        Use when an external event (e.g. the conversation-context idle
        reset) implies the active session is over — without this the wake
        strategy can outlive the LLM's memory of the conversation, leading
        to wake-less follow-up turns once the context has been wiped.
        Safe to call from any state; no-op when already IDLE.
        """
        if self._state == _WakeState.AWAKE:
            self._transition_to_idle()

    def _refresh_timeout(self) -> None:
        self._timeout_event.set()

    async def _timeout_task_handler(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._timeout_event.wait(),
                    timeout=self._timeout,
                )
                self._timeout_event.clear()
            except TimeoutError:
                if self._state == _WakeState.AWAKE:
                    if self._is_busy is not None and self._is_busy():
                        continue
                    self._transition_to_idle()
