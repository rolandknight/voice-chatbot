"""voice-chatbot WebRTC backend.

Build step 2 from docs/web-rtc.md: client-agnostic WebRTC backend with
DataChannel-driven backend (Ollama / Claude) and persona switching.

  Browser/device  ──WebRTC──>  SmallWebRTCTransport
                  ──DataCh──>  ControlChannel (hello / ready / backend / persona / state)
                                   │
                  Pipecat pipeline:  Silero VAD
                                   → Whisper MLX
                                   → PersonaCommandRouter (voice-cmd persona switch)
                                   → user-context aggregator
                                   → LLM dispatch (Ollama  || Claude)   ← `backend` msg
                                   → PersonaTagRouter (LLM-tag persona switch)
                                   → TTS dispatch (Kokoro || Chatterbox)← `persona` msg
                                   → transport.output

Each peer connection gets its own backend_state and persona_state dicts, so
two clients on the same server can route to different backends/personas
without interfering.

Skills, wake words, idle reset, ducking, and SFX are all wired in — see
build_pipeline_task and build_local_pipeline_task.

Run:
    .venv/bin/python server.py            # http://localhost:8080
    make run-server-lan                   # https://<lan-ip>:8080 (HTTPS for LAN clients)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import load as load_config

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from chatterbox_tts import ChatterboxTTSService
from persona_router import (
    CHATTERBOX_BACKEND,
    KOKORO_BACKEND,
    PersonaCommandRouter,
    PersonaConfig,
    PersonaState,
    PersonaTagRouter,
    load_persona_config,
)
from wakeword_detector import WakeWordDetector, WakeWordUserTurnStartStrategy

# `scripts/` is where resolve_from_config (Jabra device lookup) lives, and the
# radio/spotify packages provide the media players the skills wire up to.
import sys
HERE_BOOT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_BOOT / "scripts"))
from _audio_devices import resolve_from_config  # noqa: E402
from radio import RadioPlayer  # noqa: E402
from spotify import SpotifyPlayer  # noqa: E402
from skills import (  # noqa: E402
    BotSpeakingTracker,
    SkillContext,
    SkillFilterProcessor,
    load_skills,
)


VALID_MODES = {"push", "wake"}


HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"

logging.basicConfig(level=logging.INFO)


class _KeepaliveMixer(BaseAudioMixer):
    """No-op mixer that keeps BaseOutputTransport emitting silence between
    utterances. USB speakerphones like the Jabra Speak2 40 power their amp
    down on an idle stream, eating the first ~500ms of the next utterance.
    Continuous silence keeps the device awake. Verbatim from app.py."""

    async def start(self, sample_rate: int) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def process_frame(self, frame) -> None:
        pass

    async def mix(self, audio: bytes) -> bytes:
        return audio


# ─────────────────────────── skill plumbing helpers ──────────────────────


class ClaudeCueEmitter(FrameProcessor):
    """Emits a short TTS cue ("Claude here.") immediately before forwarding the
    LLMContextFrame downstream. Placed in the Claude branch of the
    ParallelPipeline so the cue only plays when Claude actually answers."""

    def __init__(self, cue_text: str = "Claude here.") -> None:
        super().__init__()
        self._cue_text = cue_text

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(TTSSpeakFrame(text=self._cue_text), direction)
        await self.push_frame(frame, direction)


class MediaDuckWatcher(FrameProcessor):
    """Ducks each registered media player when the user starts speaking and
    un-ducks after the bot finishes its reply. Verbatim from app.py.

    Players are duck-compatible if they expose is_playing() plus either
    pause/resume (RadioPlayer) or duck_pause/duck_resume (SpotifyPlayer)."""

    SAFETY_RESUME_SECS = 8.0

    def __init__(self, players: list[Any]) -> None:
        super().__init__()
        self._players = [p for p in players if p is not None]
        self._safety_task: asyncio.Task | None = None

    @staticmethod
    def _duck(player: Any) -> None:
        if hasattr(player, "duck_pause"):
            player.duck_pause()
        else:
            player.pause()

    @staticmethod
    def _unduck(player: Any) -> None:
        if hasattr(player, "duck_resume"):
            player.duck_resume()
        else:
            player.resume()

    def _cancel_safety(self) -> None:
        if self._safety_task and not self._safety_task.done():
            self._safety_task.cancel()
        self._safety_task = None

    async def _safety_resume(self) -> None:
        try:
            await asyncio.sleep(self.SAFETY_RESUME_SECS)
            for player in self._players:
                if player.is_playing():
                    logger.info(
                        f"Duck safety timer fired; resuming {type(player).__name__}."
                    )
                    self._unduck(player)
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            any_playing = False
            for player in self._players:
                if player.is_playing():
                    self._duck(player)
                    any_playing = True
            if any_playing:
                self._cancel_safety()
                self._safety_task = asyncio.create_task(self._safety_resume())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._cancel_safety()
            for player in self._players:
                if player.is_playing():
                    self._unduck(player)
        await self.push_frame(frame, direction)


class InFlightTracker(FrameProcessor):
    """Tracks whether a tool call or LLM response is mid-flight so the wake
    strategy and idle reset can defer teardown across the quiet window
    between an ask_claude dispatch and Claude's first streamed token.
    Verbatim from app.py — see the long comment there for the why."""

    def __init__(self) -> None:
        super().__init__()
        self._fn_calls_outstanding = 0
        self._llm_responding = False
        self._text_frames_this_response = 0
        self._tool_calls_this_response = 0
        self._interrupted_this_response = False

    def is_in_flight(self) -> bool:
        return self._fn_calls_outstanding > 0 or self._llm_responding

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, FunctionCallsStartedFrame):
            n = len(frame.function_calls) if frame.function_calls else 0
            self._fn_calls_outstanding += n
            self._tool_calls_this_response += n
        elif isinstance(frame, FunctionCallResultFrame):
            self._fn_calls_outstanding = max(0, self._fn_calls_outstanding - 1)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_responding = True
            self._text_frames_this_response = 0
            self._tool_calls_this_response = 0
            self._interrupted_this_response = False
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_responding = False
            if (
                not self._interrupted_this_response
                and self._text_frames_this_response == 0
                and self._tool_calls_this_response == 0
            ):
                logger.warning(
                    "LLM response ended with no text and no tool call — "
                    "likely a truncated tool-call JSON (check max_tokens)."
                )
        elif isinstance(frame, LLMTextFrame) and frame.text:
            self._text_frames_this_response += 1
        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._interrupted_this_response = True
        elif isinstance(frame, UserStartedSpeakingFrame):
            if self._llm_responding:
                self._interrupted_this_response = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._fn_calls_outstanding = 0
            self._llm_responding = False
        await self.push_frame(frame, direction)


# ─────────────────────────── control channel ─────────────────────────────


class ControlChannel:
    """JSON-over-DataChannel adapter for the protocol in docs/web-rtc.md.

    Owns the per-connection `backend_state` and `persona_state` dicts.
    Mutating them is enough to redirect routing on the next utterance —
    the pipeline FunctionFilters re-check the dict on every frame.
    """

    def __init__(
        self,
        connection: SmallWebRTCConnection,
        *,
        backend_state: dict[str, str],
        persona_state: PersonaState,
        available_backends: set[str],
        available_personas: set[str],
        mode: str,
    ) -> None:
        self._conn = connection
        self._backend_state = backend_state
        self._persona_state = persona_state
        self._backends = available_backends
        self._personas = available_personas
        self._mode = mode
        self._caps: set[str] = set()
        self._client_kind: str | None = None
        # Set once the peer connection closes. Stops the pipeline-state
        # emitter from queuing post-disconnect messages into Pipecat's send
        # queue (silently dropped, but produces noisy DEBUG log lines).
        self._closed = False

    def close(self) -> None:
        self._closed = True

    @property
    def client_kind(self) -> str | None:
        return self._client_kind

    @property
    def capabilities(self) -> set[str]:
        return set(self._caps)

    def send_state(self, state: str, **extra: Any) -> None:
        self._send({"type": "state", "state": state, **extra})

    def send_transcript(self, text: str, *, final: bool) -> None:
        self._send({"type": "transcript", "text": text, "final": final})

    def send_wake(self, state: str, **extra: Any) -> None:
        self._send({"type": "wake", "state": state, **extra})

    def send_error(self, code: str, message: str) -> None:
        self._send({"type": "error", "code": code, "message": message})

    def _send(self, msg: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._conn.send_app_message(msg)
        except Exception as e:
            logger.warning(f"control send failed: {e!r}")

    async def handle(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            self.send_error("bad_message", "expected JSON object")
            return
        mtype = msg.get("type")
        if mtype == "hello":
            self._client_kind = msg.get("kind") or "unknown"
            caps = msg.get("capabilities") or []
            if isinstance(caps, list):
                self._caps = {str(c) for c in caps}
            logger.info(
                f"hello: kind={self._client_kind!r} caps={sorted(self._caps)!r}"
            )
            self._send(
                {
                    "type": "ready",
                    "session_id": self._conn.pc_id,
                    "backend": self._backend_state["backend"],
                    "persona": self._persona_state.current,
                    "mode": self._mode,
                    "available_backends": sorted(self._backends),
                    "available_personas": sorted(self._personas),
                }
            )
            if self._mode == "wake":
                # Wake sessions start asleep — explicit so the client UI can
                # render "asleep / awake" without inferring from anything.
                self.send_wake("asleep")
        elif mtype == "backend":
            name = msg.get("name")
            if name not in self._backends:
                self.send_error(
                    "unknown_backend",
                    f"{name!r}; expected one of {sorted(self._backends)}",
                )
                return
            prev = self._backend_state["backend"]
            self._backend_state["backend"] = name
            logger.info(f"backend: {prev!r} -> {name!r}")
            self._send({"type": "backend", "name": name})
        elif mtype == "persona":
            name = msg.get("name")
            if name not in self._personas:
                self.send_error(
                    "unknown_persona",
                    f"{name!r}; expected one of {sorted(self._personas)}",
                )
                return
            prev = self._persona_state.current
            self._persona_state.current = name
            self._persona_state.pinned = name
            self._persona_state.one_shot_active = False
            logger.info(f"persona: {prev!r} -> {name!r}")
            self._send({"type": "persona", "name": name})
        elif mtype == "bye":
            logger.info("client sent bye")
        else:
            logger.debug(f"control: ignoring unknown message type {mtype!r}")


class PipelineStateEmitter(FrameProcessor):
    """Pushes pipeline lifecycle frames out to the control channel as `state`
    messages so the client UI can render listening/thinking/speaking without
    inferring from audio levels. When `control` is None (local-audio
    pipeline), still logs transcripts and state — they just don't go over
    a DataChannel."""

    def __init__(
        self,
        control: ControlChannel | None,
        persona_state: PersonaState,
        *,
        label: str = "ctrl",
    ) -> None:
        super().__init__()
        self._control = control
        self._persona_state = persona_state
        self._label = label

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._emit_state("listening")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"[{self._label}] user stopped speaking")
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._emit_state("thinking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._emit_state("speaking", persona=self._persona_state.current)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._emit_state("idle")
        elif isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text:
                logger.info(f"[{self._label}] heard: {frame.text.strip()!r}")
                if self._control is not None:
                    self._control.send_transcript(frame.text, final=True)
        await self.push_frame(frame, direction)

    def _emit_state(self, state: str, **extra: Any) -> None:
        if self._control is not None:
            self._control.send_state(state, **extra)


# ─────────────────────────── persona / TTS plumbing ───────────────────────


def _stage_chatterbox_refs(persona_config: PersonaConfig) -> None:
    """Copy each chatterbox persona's local ref clip into the Chatterbox
    server's reference_audio/ dir (verbatim from app.py — see the long
    comment there for why this is a copy, not a symlink)."""
    chatterbox_ref_dir = (
        HERE / "vendor" / "chatterbox-tts-server" / "reference_audio"
    )
    if not persona_config.chatterbox_personas():
        return
    if not chatterbox_ref_dir.is_dir():
        logger.warning(
            f"Chatterbox reference_audio dir missing at {chatterbox_ref_dir}; "
            "chatterbox personas will 404 until that server is installed."
        )
        return
    for p in persona_config.chatterbox_personas():
        if not p.ref_audio:
            continue
        src = (HERE / p.ref_audio).resolve()
        if not src.is_file():
            logger.warning(
                f"persona {p.id!r}: ref_audio {p.ref_audio} not found "
                f"(looked at {src}); chatterbox voice will 404"
            )
            continue
        dest = chatterbox_ref_dir / p.voice
        try:
            if dest.is_symlink():
                dest.unlink()
            if dest.exists():
                src_stat = src.stat()
                dest_stat = dest.stat()
                if (
                    src_stat.st_size == dest_stat.st_size
                    and int(src_stat.st_mtime) <= int(dest_stat.st_mtime)
                ):
                    continue
            shutil.copy2(src, dest)
            logger.info(f"Copied {p.ref_audio} -> reference_audio/{p.voice}")
        except OSError as e:
            logger.error(f"Could not copy {src} into {chatterbox_ref_dir}: {e}")


async def _probe_chatterbox(base_url: str) -> bool:
    """Return True if Chatterbox-TTS-Server responds at {base_url}/models."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}/models")
            return r.status_code < 500
    except Exception:
        return False


def _build_persona_tts_services(
    persona_config: PersonaConfig,
    *,
    out_sr: int,
    chatterbox_base_url: str,
    chatterbox_api_key: str,
    chatterbox_model: str,
    chatterbox_available: bool,
) -> dict[str, FrameProcessor]:
    services: dict[str, FrameProcessor] = {}
    for persona in persona_config.personas.values():
        if persona.backend == KOKORO_BACKEND:
            services[persona.id] = KokoroTTSService(
                settings=KokoroTTSService.Settings(voice=persona.voice),
                text_aggregation_mode=TextAggregationMode.SENTENCE,
            )
        elif persona.backend == CHATTERBOX_BACKEND:
            if not chatterbox_available:
                logger.warning(
                    f"persona {persona.id!r} uses chatterbox but the server "
                    f"at {chatterbox_base_url} is not reachable; skipping. "
                    "Start it via run.sh or scripts/start_chatterbox.sh."
                )
                continue
            services[persona.id] = ChatterboxTTSService(
                api_key=chatterbox_api_key,
                base_url=chatterbox_base_url,
                sample_rate=out_sr,
                settings=ChatterboxTTSService.Settings(
                    voice=persona.voice,
                    model=chatterbox_model,
                ),
                text_aggregation_mode=TextAggregationMode.SENTENCE,
            )
        else:
            logger.error(
                f"persona {persona.id!r}: unsupported backend {persona.backend!r}"
            )
    return services


def _persona_filter(persona_state: PersonaState, target_id: str):
    async def _f(_frame: Frame) -> bool:
        return persona_state.current == target_id
    return _f


def _build_tts_dispatch(
    persona_tts: dict[str, FrameProcessor],
    persona_state: PersonaState,
) -> FrameProcessor:
    """Single persona -> direct service. Multiple -> ParallelPipeline with a
    FunctionFilter per branch keyed on persona_state.current."""
    if len(persona_tts) == 1:
        return next(iter(persona_tts.values()))
    return ParallelPipeline(
        *[
            [FunctionFilter(filter=_persona_filter(persona_state, pid), direction=None), svc]
            for pid, svc in persona_tts.items()
        ]
    )


# ─────────────────────────── LLM plumbing ────────────────────────────────


def _backend_filter(backend_state: dict[str, str], target: str):
    async def _f(_frame: Frame) -> bool:
        return backend_state.get("backend", "ollama") == target
    return _f


def _build_llm_dispatch(
    ollama_llm: FrameProcessor,
    claude_llm: FrameProcessor | None,
    backend_state: dict[str, str],
    *,
    claude_cue: FrameProcessor | None = None,
) -> FrameProcessor:
    """Single backend -> the service directly. Both -> ParallelPipeline
    sandwiched by top+bottom FunctionFilters per branch (see app.py:760-790
    for why bottom filters are also required). When `claude_cue` is provided
    it sits inside the Claude branch and speaks a short cue ("Claude here.")
    before forwarding the LLMContextFrame downstream."""
    if claude_llm is None:
        return ollama_llm
    claude_branch: list[Any] = [
        FunctionFilter(filter=_backend_filter(backend_state, "claude"), direction=None),
    ]
    if claude_cue is not None:
        claude_branch.append(claude_cue)
    claude_branch.extend([
        claude_llm,
        FunctionFilter(filter=_backend_filter(backend_state, "claude"), direction=None),
    ])
    return ParallelPipeline(
        [
            FunctionFilter(filter=_backend_filter(backend_state, "ollama"), direction=None),
            ollama_llm,
            FunctionFilter(filter=_backend_filter(backend_state, "ollama"), direction=None),
        ],
        claude_branch,
    )


# ─────────────────────────── pipeline factory ────────────────────────────


def _build_skill_runtime(
    runtime: dict[str, Any],
    *,
    persona_state: PersonaState,
    backend_state: dict[str, str],
):
    """Build the per-connection skill plumbing: SkillContext + SkillRegistry +
    optional BotSpeakingTracker. Singleton resources (RadioPlayer,
    SpotifyPlayer) come from the shared runtime dict so two clients sharing
    a Mac don't fight over its speaker.

    Returns (registry, sfx_tracker). Either may be None when skills are
    disabled or when the loader filters everything out."""
    if not runtime.get("skills_enabled"):
        return None, None
    sfx_tracker = BotSpeakingTracker() if runtime["sfx_enabled"] else None
    ctx = SkillContext(
        radio_player=runtime["radio_player"],
        spotify_player=runtime["spotify_player"],
        sfx_tracker=sfx_tracker,
        sfx_backends=runtime["sfx_backends"],
        sfx_backend_override=runtime["sfx_backend_override"],
        persona_config=runtime["persona_config"],
        persona_state=persona_state,
        # ask_claude flips this mid-session; only meaningful when Claude is
        # wired up. Passing None when claude is off keeps that skill out of
        # the registry via its `requires: [backend_state]` gate.
        backend_state=backend_state if runtime["claude_enabled"] else None,
    )
    registry = load_skills(ctx, runtime["cfg"])
    return registry, sfx_tracker


def build_pipeline_task(
    connection: SmallWebRTCConnection,
    control: ControlChannel,
    *,
    runtime: dict[str, Any],
    backend_state: dict[str, str],
    persona_state: PersonaState,
    mode: str,
) -> PipelineTask:
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=runtime["in_sr"],
            audio_out_sample_rate=runtime["out_sr"],
            audio_in_channels=1,
            audio_out_channels=1,
            audio_in_passthrough=True,
        ),
    )

    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=runtime["whisper_model"],
            language=runtime["language"],
            no_speech_prob=0.55,
            temperature=0.0,
        ),
        ttfs_p99_latency=runtime["ttfs_p99_latency"],
    )

    ollama_llm = OLLamaLLMService(
        base_url=runtime["ollama_base_url"],
        settings=OLLamaLLMService.Settings(
            model=runtime["ollama_model"],
            temperature=0.2,
            max_tokens=512,
            system_instruction=runtime["ollama_system_prompt"],
        ),
    )

    claude_llm: FrameProcessor | None = None
    if runtime["claude_enabled"]:
        claude_llm = AnthropicLLMService(
            api_key=runtime["anthropic_api_key"],
            settings=AnthropicLLMService.Settings(
                model=runtime["claude_model"],
                max_tokens=runtime["claude_max_tokens"],
                system_instruction=runtime["claude_system_prompt"],
                extra={"tools": runtime["claude_tools"]} if runtime["claude_tools"] else {},
            ),
        )

    # Skills + sfx_tracker are per-connection: their handlers close over this
    # connection's persona_state/backend_state, so two clients won't switch
    # each other's persona or backend by issuing voice commands.
    skill_registry, sfx_tracker = _build_skill_runtime(
        runtime, persona_state=persona_state, backend_state=backend_state
    )

    claude_cue = ClaudeCueEmitter() if claude_llm is not None else None
    llm_dispatch = _build_llm_dispatch(
        ollama_llm, claude_llm, backend_state, claude_cue=claude_cue
    )
    tts_dispatch = _build_tts_dispatch(runtime["persona_tts"], persona_state)

    # in_flight defers wake teardown across the quiet ask_claude dispatch
    # window. Constructed up front so wake_strategy can read its is_busy hook.
    in_flight = InFlightTracker()
    wake_strategy: WakeWordUserTurnStartStrategy | None = None

    # Wake mode: prepend a WakeWordDetector and gate turn-start on wake firing.
    # Push mode: turn-start is VAD-driven, mic is effectively push-to-talk.
    wake_detector: FrameProcessor | None = None
    turn_start_strategies: list[Any] = [VADUserTurnStartStrategy()]
    if mode == "wake":
        wake_models = runtime["wake_models"]
        if not wake_models:
            logger.warning(
                "wake mode requested but no usable wake models — falling back "
                "to push-mode behavior for this connection."
            )
        else:
            wake_detector = WakeWordDetector(
                model_paths_or_keys=wake_models,
                persona_for_model=runtime["wake_persona_map"],
                threshold=runtime["wake_threshold"],
                cooldown_secs=runtime["wake_cooldown_secs"],
                persona_config=runtime["persona_config"],
                persona_state=persona_state,
            )
            wake_strategy = WakeWordUserTurnStartStrategy(
                timeout=runtime["wake_session_timeout"],
                is_busy=in_flight.is_in_flight,
            )

            @wake_strategy.event_handler("on_wake_word_detected")
            async def _on_wake(_s, model_key, score):
                logger.info(
                    f"Wake fired: {model_key!r} score={score:.3f} "
                    f"persona={persona_state.current!r} "
                    f"backend={backend_state['backend']!r}"
                )
                control.send_wake(
                    "awake",
                    model=model_key,
                    score=round(score, 3),
                    persona=persona_state.current,
                )

            @wake_strategy.event_handler("on_wake_word_timeout")
            async def _on_sleep(_s):
                # Session-scoped backend revert: ask_claude flips backend_state
                # to "claude" for the duration of the wake session, sleep ends
                # it. Mirrors app.py's wake-timeout handler.
                if backend_state["backend"] != "ollama":
                    logger.info(
                        f"Wake timeout — reverting backend "
                        f"{backend_state['backend']!r} -> 'ollama'"
                    )
                    backend_state["backend"] = "ollama"
                    control._send({"type": "backend", "name": "ollama"})
                else:
                    logger.info("Wake session timed out — back to asleep")
                control.send_wake("asleep")

            turn_start_strategies = [wake_strategy, VADUserTurnStartStrategy()]

    context = LLMContext(messages=[])
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    min_volume=runtime["vad_min_volume"],
                    stop_secs=runtime["vad_stop_secs"],
                ),
            ),
            user_turn_strategies=UserTurnStrategies(start=turn_start_strategies),
        ),
    )

    # Skills bind their tool handlers to this connection's Ollama service and
    # seed the always-available tool set on the context. The
    # SkillFilterProcessor below swaps in the per-turn top-K relevant tools.
    if skill_registry is not None:
        skill_registry.register(ollama_llm, context)
        logger.info(
            f"[webrtc {connection.pc_id}] skills loaded: "
            f"{sorted(skill_registry.skills_by_name)}"
        )

    stages: list[Any] = [transport.input()]
    if wake_detector is not None:
        # WakeWordDetector runs ahead of STT so the wake event reaches the
        # user-aggregator before any TranscriptionFrame for that turn.
        stages.append(wake_detector)
    stages += [
        stt,
        PersonaCommandRouter(runtime["persona_config"], persona_state),
        aggregator.user(),
    ]
    if skill_registry is not None:
        stages.append(
            SkillFilterProcessor(
                context,
                skill_registry,
                k=runtime["skills_filter_k"],
                debug=runtime["skills_filter_debug"],
            )
        )
    stages += [
        llm_dispatch,
        in_flight,
        PersonaTagRouter(runtime["persona_config"], persona_state),
        tts_dispatch,
        transport.output(),
    ]
    duckable = [
        p for p in (runtime["radio_player"], runtime["spotify_player"]) if p is not None
    ]
    if duckable:
        # MediaDuckWatcher must sit downstream of transport.output() so it
        # sees the canonical Bot/User speaking lifecycle frames.
        stages.append(MediaDuckWatcher(duckable))
    if sfx_tracker is not None:
        stages.append(sfx_tracker)
    stages += [
        PipelineStateEmitter(control, persona_state, label="webrtc"),
        aggregator.assistant(),
    ]
    pipeline = Pipeline(stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runtime["idle_timeout_secs"],
        cancel_on_idle_timeout=False,
    )

    @task.event_handler("on_idle_timeout")
    async def _on_conversation_idle(_task):
        # Defer reset while a tool call / Claude response is mid-flight — the
        # idle handler re-fires on the next interval, so a silent return is
        # enough. Mirrors app.py.
        if in_flight.is_in_flight():
            return
        if not context.messages:
            return
        logger.info(
            f"[webrtc {connection.pc_id}] conversation idle for "
            f"{runtime['idle_timeout_secs']:.0f}s — resetting context."
        )
        context.set_messages([])
        if wake_strategy is not None:
            wake_strategy.force_idle()

    return task


def build_local_pipeline_task(
    *,
    runtime: dict[str, Any],
    input_device_index: int | None,
    output_device_index: int | None,
) -> PipelineTask:
    """LocalAudioTransport-backed pipeline running wake mode against the
    default (or configured) input/output device — typically the Jabra
    speakerphone the original `app.py` targets.

    This is its own task with its own backend_state/persona_state, so a
    browser client switching backend on its own connection doesn't affect
    the local Jabra channel, and vice versa.
    """
    backend_state: dict[str, str] = {"backend": runtime["default_backend"]}
    persona_state = PersonaState(
        current=runtime["default_persona"], pinned=runtime["default_persona"]
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=runtime["in_sr"],
            audio_out_sample_rate=runtime["out_sr"],
            audio_in_channels=1,
            audio_out_channels=1,
            audio_in_passthrough=True,
            audio_out_mixer=_KeepaliveMixer(),
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )
    )

    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=runtime["whisper_model"],
            language=runtime["language"],
            no_speech_prob=0.55,
            temperature=0.0,
        ),
        ttfs_p99_latency=runtime["ttfs_p99_latency"],
    )

    ollama_llm = OLLamaLLMService(
        base_url=runtime["ollama_base_url"],
        settings=OLLamaLLMService.Settings(
            model=runtime["ollama_model"],
            temperature=0.2,
            max_tokens=512,
            system_instruction=runtime["ollama_system_prompt"],
        ),
    )
    claude_llm: FrameProcessor | None = None
    if runtime["claude_enabled"]:
        claude_llm = AnthropicLLMService(
            api_key=runtime["anthropic_api_key"],
            settings=AnthropicLLMService.Settings(
                model=runtime["claude_model"],
                max_tokens=runtime["claude_max_tokens"],
                system_instruction=runtime["claude_system_prompt"],
                extra={"tools": runtime["claude_tools"]} if runtime["claude_tools"] else {},
            ),
        )
    skill_registry, sfx_tracker = _build_skill_runtime(
        runtime, persona_state=persona_state, backend_state=backend_state
    )

    claude_cue = ClaudeCueEmitter() if claude_llm is not None else None
    llm_dispatch = _build_llm_dispatch(
        ollama_llm, claude_llm, backend_state, claude_cue=claude_cue
    )
    tts_dispatch = _build_tts_dispatch(runtime["persona_tts"], persona_state)

    in_flight = InFlightTracker()
    wake_strategy: WakeWordUserTurnStartStrategy | None = None

    # Local audio always runs in wake mode (matches app.py behavior — wake is
    # the only sane gating against an always-on mic feeding a continuous
    # speech stream into the LLM).
    wake_detector: FrameProcessor | None = None
    turn_start_strategies: list[Any] = [VADUserTurnStartStrategy()]
    if runtime["wake_available"]:
        wake_detector = WakeWordDetector(
            model_paths_or_keys=runtime["wake_models"],
            persona_for_model=runtime["wake_persona_map"],
            threshold=runtime["wake_threshold"],
            cooldown_secs=runtime["wake_cooldown_secs"],
            persona_config=runtime["persona_config"],
            persona_state=persona_state,
        )
        wake_strategy = WakeWordUserTurnStartStrategy(
            timeout=runtime["wake_session_timeout"],
            is_busy=in_flight.is_in_flight,
        )

        @wake_strategy.event_handler("on_wake_word_detected")
        async def _on_wake(_s, model_key, score):
            logger.info(
                f"[local] wake fired: {model_key!r} score={score:.3f} "
                f"persona={persona_state.current!r} "
                f"backend={backend_state['backend']!r}"
            )

        @wake_strategy.event_handler("on_wake_word_timeout")
        async def _on_sleep(_s):
            if backend_state["backend"] != "ollama":
                logger.info(
                    f"[local] wake timeout — reverting backend "
                    f"{backend_state['backend']!r} -> 'ollama'"
                )
                backend_state["backend"] = "ollama"
            else:
                logger.info("[local] wake session timed out — back to asleep")

        turn_start_strategies = [wake_strategy, VADUserTurnStartStrategy()]
    else:
        logger.warning(
            "[local] no wake models — local pipeline will respond to every "
            "VAD-detected utterance (effectively always-on, which is rarely "
            "what you want for a room mic)."
        )

    context = LLMContext(messages=[])
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    min_volume=runtime["vad_min_volume"],
                    stop_secs=runtime["vad_stop_secs"],
                ),
            ),
            user_turn_strategies=UserTurnStrategies(start=turn_start_strategies),
        ),
    )

    if skill_registry is not None:
        skill_registry.register(ollama_llm, context)
        logger.info(
            f"[local] skills loaded: {sorted(skill_registry.skills_by_name)}"
        )

    stages: list[Any] = [transport.input()]
    if wake_detector is not None:
        stages.append(wake_detector)
    stages += [
        stt,
        PersonaCommandRouter(runtime["persona_config"], persona_state),
        aggregator.user(),
    ]
    if skill_registry is not None:
        stages.append(
            SkillFilterProcessor(
                context,
                skill_registry,
                k=runtime["skills_filter_k"],
                debug=runtime["skills_filter_debug"],
            )
        )
    stages += [
        llm_dispatch,
        in_flight,
        PersonaTagRouter(runtime["persona_config"], persona_state),
        tts_dispatch,
        transport.output(),
    ]
    duckable = [
        p for p in (runtime["radio_player"], runtime["spotify_player"]) if p is not None
    ]
    if duckable:
        stages.append(MediaDuckWatcher(duckable))
    if sfx_tracker is not None:
        stages.append(sfx_tracker)
    stages += [
        PipelineStateEmitter(control=None, persona_state=persona_state, label="local"),
        aggregator.assistant(),
    ]

    task = PipelineTask(
        Pipeline(stages),
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runtime["idle_timeout_secs"],
        cancel_on_idle_timeout=False,
    )

    @task.event_handler("on_idle_timeout")
    async def _on_conversation_idle(_task):
        if in_flight.is_in_flight():
            return
        if not context.messages:
            return
        logger.info(
            f"[local] conversation idle for "
            f"{runtime['idle_timeout_secs']:.0f}s — resetting context."
        )
        context.set_messages([])
        if wake_strategy is not None:
            wake_strategy.force_idle()

    return task


# ─────────────────────────── runtime config ──────────────────────────────


def _lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ip not in ips and ":" not in ip and not ip.startswith("127."):
                ips.append(ip)
    except socket.gaierror:
        pass
    return ips


def _ollama_system_prompt() -> str:
    return (
        "You are a fast local voice assistant. "
        "Keep replies brief and conversational. "
        "Prefer one or two short sentences."
    )


def _claude_system_prompt(web_search: bool, web_fetch: bool) -> str:
    base = (
        "You are Claude, a helpful voice assistant on a Mac. "
        "Reply conversationally and stay spoken-friendly: no markdown, no bullet "
        "lists, no code blocks. Be accurate and complete — you can take a "
        "paragraph or two when the question warrants it."
    )
    if web_search and web_fetch:
        base += (
            " You can call web_search for current information and web_fetch to "
            "read a specific URL — use them when the question depends on recent "
            "or external facts, but skip them for things you already know. "
            "When you do search, summarize in plain prose; don't read URLs or "
            "citation markers aloud."
        )
    elif web_search:
        base += (
            " You can call web_search for current information — use it when "
            "the question depends on recent facts. Summarize in plain prose."
        )
    elif web_fetch:
        base += (
            " You can call web_fetch to read a specific URL the user mentions. "
            "Summarize the page in plain prose."
        )
    return base


async def _load_runtime(cfg) -> dict[str, Any]:
    """Build the immutable per-server runtime snapshot. Per-connection state
    (backend_state, persona_state) is constructed separately at each /api/offer."""
    if cfg.huggingface.hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    persona_config, _ = load_persona_config(
        cfg.resolved_personas_path(), cfg.tts.default_persona
    )

    _stage_chatterbox_refs(persona_config)

    has_chatterbox = bool(persona_config.chatterbox_personas())
    chatterbox_base = cfg.tts.chatterbox.base_url
    chatterbox_available = False
    if has_chatterbox:
        chatterbox_available = await _probe_chatterbox(chatterbox_base)
        if not chatterbox_available:
            logger.warning(
                f"Chatterbox-TTS-Server unreachable at {chatterbox_base}; "
                "chatterbox-backed personas will be skipped."
            )
        else:
            logger.info(f"Chatterbox-TTS-Server reachable at {chatterbox_base}")

    persona_tts = _build_persona_tts_services(
        persona_config,
        out_sr=cfg.audio.out_sample_rate,
        chatterbox_base_url=chatterbox_base,
        chatterbox_api_key=cfg.tts.chatterbox.api_key.get_secret_value(),
        chatterbox_model=cfg.tts.chatterbox.model,
        chatterbox_available=chatterbox_available,
    )
    if not persona_tts:
        raise RuntimeError(
            "no usable persona TTS services — check personas.yaml and that "
            "any required external TTS servers are running."
        )

    claude_enabled = cfg.claude.enabled
    claude_tools: list[dict] = []
    if claude_enabled:
        if cfg.claude.web_search_enabled:
            claude_tools.append(
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": cfg.claude.web_search_max_uses,
                }
            )
        if cfg.claude.web_fetch_enabled:
            claude_tools.append(
                {
                    "type": "web_fetch_20260209",
                    "name": "web_fetch",
                    "max_uses": cfg.claude.web_fetch_max_uses,
                }
            )

    # Wake models: resolve each entry to either a file path (existing .onnx)
    # or a bundled openwakeword key. Drop missing files with a warning so
    # wake mode still works with whatever remains.
    wake_models: list[str] = []
    wake_persona_map: dict[str, str] = {}
    for m in cfg.wake.models:
        if m.model.endswith(".onnx"):
            p = (HERE / m.model).resolve()
            if not p.is_file():
                logger.warning(
                    f"wake model {m.model!r} not found at {p}; skipping. "
                    "Train it via scripts/wakeword/ or remove it from config.yaml."
                )
                continue
            wake_models.append(str(p))
            wake_persona_map[Path(m.model).stem] = m.persona
        else:
            wake_models.append(m.model)
            wake_persona_map[m.model] = m.persona
    if wake_models:
        logger.info(
            f"Wake models       : {[Path(m).name for m in wake_models]} "
            f"(threshold={cfg.wake.threshold}, cooldown={cfg.wake.cooldown_secs}s)"
        )
    else:
        logger.warning("No usable wake models — wake mode will fall back to push.")

    # Skills + media players. RadioPlayer and SpotifyPlayer drive real audio on
    # the host Mac, so they're process singletons shared across all WebRTC
    # connections and the optional local-audio pipeline — two clients can't
    # meaningfully each have their own. Per-connection state (SkillContext,
    # registry, SkillFilterProcessor) is built later inside the pipeline
    # factories and references these singletons through the runtime dict.
    skills_enabled = cfg.skills.enabled
    radio_enabled = skills_enabled and cfg.skills.radio.enabled
    spotify_enabled = (
        skills_enabled
        and cfg.skills.spotify.enabled
        and bool(cfg.skills.spotify.client_id.get_secret_value().strip())
    )
    sfx_woosh_enabled = skills_enabled and cfg.skills.sfx.woosh_enabled
    sfx_sao_enabled = skills_enabled and cfg.skills.sfx.sao_enabled
    sfx_enabled = sfx_woosh_enabled or sfx_sao_enabled
    sfx_backends: dict[str, str] = {}
    if sfx_woosh_enabled:
        sfx_backends["woosh"] = cfg.skills.sfx.woosh_url
    if sfx_sao_enabled:
        sfx_backends["stable_audio"] = cfg.skills.sfx.sao_url

    radio_player = RadioPlayer() if radio_enabled else None
    spotify_player = SpotifyPlayer() if spotify_enabled else None

    return {
        "in_sr": cfg.audio.in_sample_rate,
        "out_sr": cfg.audio.out_sample_rate,
        "whisper_model": cfg.stt.whisper_model,
        "language": cfg.stt.language,
        "ttfs_p99_latency": cfg.stt.ttfs_p99_latency_secs,
        "ollama_model": cfg.llm.ollama_model,
        "ollama_base_url": cfg.llm.ollama_base_url,
        "ollama_system_prompt": _ollama_system_prompt(),
        "vad_min_volume": cfg.wake.vad_min_volume,
        "vad_stop_secs": cfg.wake.vad_stop_secs,
        "persona_config": persona_config,
        "persona_tts": persona_tts,
        "available_personas": set(persona_tts.keys()),
        "default_persona": persona_config.default
        if persona_config.default in persona_tts
        else next(iter(persona_tts)),
        "claude_enabled": claude_enabled,
        "anthropic_api_key": cfg.claude.api_key.get_secret_value().strip(),
        "claude_model": cfg.claude.model,
        "claude_max_tokens": cfg.claude.max_tokens,
        "claude_tools": claude_tools,
        "claude_system_prompt": _claude_system_prompt(
            cfg.claude.web_search_enabled and claude_enabled,
            cfg.claude.web_fetch_enabled and claude_enabled,
        ),
        "available_backends": {"ollama", "claude"} if claude_enabled else {"ollama"},
        "default_backend": "ollama",
        "wake_models": wake_models,
        "wake_persona_map": wake_persona_map,
        "wake_threshold": cfg.wake.threshold,
        "wake_cooldown_secs": cfg.wake.cooldown_secs,
        "wake_session_timeout": cfg.conversation.idle_timeout_secs,
        "wake_available": bool(wake_models),
        # Skills + media. cfg is stashed so per-connection load_skills() can
        # evaluate each SKILL.md's `enabled_when` dotted path against it.
        "cfg": cfg,
        "skills_enabled": skills_enabled,
        "sfx_enabled": sfx_enabled,
        "sfx_backends": sfx_backends,
        "sfx_backend_override": cfg.skills.sfx.backend if sfx_enabled else None,
        "skills_filter_k": cfg.skills.filter_k,
        "skills_filter_debug": cfg.skills.filter_debug,
        "radio_player": radio_player,
        "spotify_player": spotify_player,
        "idle_timeout_secs": cfg.conversation.idle_timeout_secs,
    }


# ─────────────────────────── warmups ─────────────────────────────────────


_pipeline_tasks: set[asyncio.Task[Any]] = set()


async def _prewarm_whisper(model: str, language: str, in_sr: int) -> None:
    try:
        import mlx_whisper
        import numpy as np
    except Exception as e:
        logger.debug(f"Whisper warmup skipped: {e}")
        return
    logger.info(f"Pre-warming Whisper MLX ({model})...")
    silent = np.zeros(in_sr // 10, dtype="float32")
    try:
        await asyncio.to_thread(
            mlx_whisper.transcribe,
            silent,
            path_or_hf_repo=model,
            language=language,
        )
    except Exception as e:
        logger.debug(f"Whisper warmup failed: {e}")


async def _prewarm_persona_tts(persona_tts: dict[str, FrameProcessor]) -> None:
    """Run one short utterance through each persona's TTS so ONNX graphs are
    built / weights paged in before the first real reply. Chatterbox warmup
    will silently fail if the server isn't fully up yet — that's fine, the
    real call will still work once it is."""
    for pid, svc in persona_tts.items():
        logger.info(f"Pre-warming TTS for persona {pid!r}...")
        try:
            async for _ in svc.run_tts("Hi.", context_id=f"warmup-{pid}"):
                break
        except Exception as e:
            logger.debug(f"TTS warmup skipped for {pid!r}: {e}")


async def _prewarm_ollama(model: str, base_url: str) -> str:
    import httpx
    host = base_url.rsplit("/v1", 1)[0]
    logger.info(f"Pre-warming Ollama LLM ({model})...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 1},
                },
            )
            try:
                ps = await client.get(f"{host}/api/ps")
                ps.raise_for_status()
                resident = [
                    m.get("name") or m.get("model")
                    for m in ps.json().get("models", [])
                ]
                resident = [n for n in resident if n]
                if model not in resident:
                    logger.error(
                        f"Ollama pre-warm did not leave {model!r} resident. "
                        f"Currently loaded: {resident or '(none)'}."
                    )
                else:
                    co = [n for n in resident if n != model]
                    if co:
                        logger.warning(
                            f"Ollama has {model!r} resident alongside {co} — "
                            "memory pressure may evict one between turns."
                        )
                    else:
                        logger.info(f"Ollama resident: {model} (sole tenant)")
            except Exception as e:
                logger.debug(f"Ollama /api/ps probe skipped: {e}")
    except Exception as e:
        logger.warning(f"Ollama warmup failed: {e}")
    return host


async def _ollama_keepalive(host: str, model: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                await asyncio.sleep(240)
                r = await client.post(
                    f"{host}/api/generate",
                    json={"model": model, "keep_alive": -1},
                )
                r.raise_for_status()
                logger.info(f"Ollama keepalive: {model} pinned")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Ollama keepalive failed: {e}")


# ─────────────────────────── FastAPI app ─────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    rc = await _load_runtime(cfg)
    app.state.runtime = rc

    await _prewarm_whisper(rc["whisper_model"], rc["language"], rc["in_sr"])
    await _prewarm_persona_tts(rc["persona_tts"])
    host = await _prewarm_ollama(rc["ollama_model"], rc["ollama_base_url"])

    heartbeat = asyncio.create_task(
        _ollama_keepalive(host, rc["ollama_model"]),
        name="ollama-keepalive",
    )

    # Optional always-on local-audio pipeline (Jabra etc.) — enabled by the
    # --local-audio CLI flag, plumbed through env so it survives the
    # uvicorn-managed app load.
    local_task: asyncio.Task[Any] | None = None
    if os.environ.get("VOICE_CHATBOT_LOCAL_AUDIO") == "1":
        in_idx, out_idx, in_name, out_name = resolve_from_config(cfg.audio)
        logger.info(
            f"[local] audio devices: in=[{in_idx}] {in_name!r} "
            f"out=[{out_idx}] {out_name!r}"
        )
        local_pipeline = build_local_pipeline_task(
            runtime=rc,
            input_device_index=in_idx,
            output_device_index=out_idx,
        )

        async def _run_local() -> None:
            runner = PipelineRunner(handle_sigint=False)
            try:
                await runner.run(local_pipeline)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("[local] pipeline runner crashed")

        local_task = asyncio.create_task(_run_local(), name="local-audio-pipeline")
        logger.info("[local] pipeline started")

    logger.info(
        "ready: whisper={whisper} ollama={ollama} personas={p} backends={b} local={l}",
        whisper=rc["whisper_model"],
        ollama=rc["ollama_model"],
        p=sorted(rc["available_personas"]),
        b=sorted(rc["available_backends"]),
        l=local_task is not None,
    )
    try:
        yield
    finally:
        if local_task is not None:
            local_task.cancel()
        # Tighter shutdown: clean exits complete in <100ms, the timeouts here
        # are only for the pathological case where a task is stuck. Past ~2s
        # the user is double-Ctrl-C'ing anyway, which would SIGKILL us.
        heartbeat.cancel()
        for t in list(_pipeline_tasks):
            t.cancel()
        pending = [heartbeat, *_pipeline_tasks]
        if local_task is not None:
            pending.append(local_task)
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=1.5,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"shutdown: {len(pending)} task(s) didn't exit within "
                    "1.5s — abandoning"
                )
        # Stop the singleton media players last — anything still mid-tool-call
        # at shutdown has already had its task cancelled above.
        if rc.get("radio_player") is not None:
            rc["radio_player"].stop()
        if rc.get("spotify_player") is not None:
            # api_pause=False: killing mpv silences the speaker; the API pause
            # is what produces "rate/request limit" stdout spam from spotipy.
            rc["spotify_player"].stop(api_pause=False)


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
async def options() -> dict[str, Any]:
    """Backends + personas the browser UI can pick from. Driven by
    personas.yaml + Claude availability — no client-side hardcoding."""
    rc = app.state.runtime
    modes = ["push"]
    if rc["wake_available"]:
        modes.append("wake")
    return {
        "backends": sorted(rc["available_backends"]),
        "default_backend": rc["default_backend"],
        "personas": sorted(rc["available_personas"]),
        "default_persona": rc["default_persona"],
        "modes": modes,
        "wake_models": [Path(m).name for m in rc["wake_models"]],
    }


@app.post("/api/offer")
async def offer(request: Request) -> JSONResponse:
    payload = await request.json()
    if "sdp" not in payload or "type" not in payload:
        return JSONResponse({"error": "expected {sdp, type}"}, status_code=400)

    rc = app.state.runtime

    mode = (payload.get("mode") or "push").lower()
    if mode not in VALID_MODES:
        return JSONResponse(
            {"error": f"unknown mode {mode!r}; expected one of {sorted(VALID_MODES)}"},
            status_code=400,
        )
    if mode == "wake" and not rc["wake_available"]:
        logger.warning("wake mode requested but no wake models — using push mode")
        mode = "push"

    connection = SmallWebRTCConnection()
    await connection.initialize(sdp=payload["sdp"], type=payload["type"])

    # Per-connection state — two clients can be on different backends/personas
    # without stepping on each other.
    backend_state: dict[str, str] = {"backend": rc["default_backend"]}
    persona_state = PersonaState(
        current=rc["default_persona"],
        pinned=rc["default_persona"],
    )

    control = ControlChannel(
        connection,
        backend_state=backend_state,
        persona_state=persona_state,
        available_backends=rc["available_backends"],
        available_personas=rc["available_personas"],
        mode=mode,
    )

    @connection.event_handler("app-message")
    async def _on_app(_conn: SmallWebRTCConnection, message: Any) -> None:
        await control.handle(message)

    @connection.event_handler("closed")
    async def _on_closed(conn: SmallWebRTCConnection) -> None:
        control.close()
        logger.info(f"connection closed pc_id={conn.pc_id}")

    task = build_pipeline_task(
        connection,
        control,
        runtime=rc,
        backend_state=backend_state,
        persona_state=persona_state,
        mode=mode,
    )
    logger.info(f"connection accepted: mode={mode!r} pc_id={connection.pc_id}")

    async def _run() -> None:
        runner = PipelineRunner(handle_sigint=False)
        try:
            await runner.run(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("pipeline runner crashed")
        finally:
            try:
                await asyncio.wait_for(connection.disconnect(), timeout=2.0)
            except (Exception, asyncio.TimeoutError):
                pass

    bg = asyncio.create_task(_run(), name=f"pipeline-{connection.pc_id}")
    _pipeline_tasks.add(bg)
    bg.add_done_callback(_pipeline_tasks.discard)

    answer = connection.get_answer()
    if answer is None:
        return JSONResponse({"error": "no SDP answer produced"}, status_code=500)
    return JSONResponse(answer)


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="voice-chatbot WebRTC backend (with optional always-on local audio)."
    )
    parser.add_argument(
        "--local-audio",
        action="store_true",
        help="Also boot a LocalAudioTransport pipeline against the configured "
        "input/output device (typically the Jabra) in wake mode. Mirrors the "
        "behavior of the original app.py.",
    )
    args = parser.parse_args()

    if args.local_audio:
        os.environ["VOICE_CHATBOT_LOCAL_AUDIO"] = "1"

    host = os.environ.get("WEBRTC_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBRTC_PORT", "8080"))
    cert = os.environ.get("WEBRTC_SSL_CERT") or None
    key = os.environ.get("WEBRTC_SSL_KEY") or None
    scheme = "https" if cert and key else "http"

    print()
    print(f"  {scheme}://localhost:{port}")
    for ip in _lan_ips():
        print(f"  {scheme}://{ip}:{port}")
    if scheme == "http":
        print()
        print("  NOTE: browsers only grant mic access on http://localhost.")
        print("  For LAN clients use `make run-server-lan` (HTTPS).")
    if args.local_audio:
        print()
        print("  --local-audio: also running an always-on local pipeline")
        print("  against the configured Jabra device (wake mode).")
    print()

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=cert,
        ssl_keyfile=key,
    )


if __name__ == "__main__":
    main()
