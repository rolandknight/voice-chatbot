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

Skills, wake words, idle reset, ducking, SFX — not yet wired (later steps).

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
    Frame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
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
from pipecat.transports.base_transport import TransportParams
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


VALID_MODES = {"push", "wake"}


HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"

logging.basicConfig(level=logging.INFO)


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
    """Pushes pipeline lifecycle frames out to the control channel as
    `state` messages so the client UI can render listening/thinking/speaking
    without inferring from audio levels. Also logs transcripts so a missing
    transcription is obvious in stderr."""

    def __init__(self, control: ControlChannel) -> None:
        super().__init__()
        self._control = control

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._control.send_state("listening")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("user stopped speaking")
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._control.send_state("thinking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._control.send_state(
                "speaking", persona=self._control._persona_state.current
            )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._control.send_state("idle")
        elif isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text:
                logger.info(f"heard: {frame.text.strip()!r}")
                self._control.send_transcript(frame.text, final=True)
        await self.push_frame(frame, direction)


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
) -> FrameProcessor:
    """Single backend -> the service directly. Both -> ParallelPipeline
    sandwiched by top+bottom FunctionFilters per branch (see app.py:760-790
    for why bottom filters are also required)."""
    if claude_llm is None:
        return ollama_llm
    return ParallelPipeline(
        [
            FunctionFilter(filter=_backend_filter(backend_state, "ollama"), direction=None),
            ollama_llm,
            FunctionFilter(filter=_backend_filter(backend_state, "ollama"), direction=None),
        ],
        [
            FunctionFilter(filter=_backend_filter(backend_state, "claude"), direction=None),
            claude_llm,
            FunctionFilter(filter=_backend_filter(backend_state, "claude"), direction=None),
        ],
    )


# ─────────────────────────── pipeline factory ────────────────────────────


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

    llm_dispatch = _build_llm_dispatch(ollama_llm, claude_llm, backend_state)
    tts_dispatch = _build_tts_dispatch(runtime["persona_tts"], persona_state)

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
            )

            @wake_strategy.event_handler("on_wake_word_detected")
            async def _on_wake(_s, model_key, score):
                logger.info(
                    f"Wake fired: {model_key!r} score={score:.3f} "
                    f"persona={persona_state.current!r}"
                )
                control.send_wake(
                    "awake",
                    model=model_key,
                    score=round(score, 3),
                    persona=persona_state.current,
                )

            @wake_strategy.event_handler("on_wake_word_timeout")
            async def _on_sleep(_s):
                logger.info("Wake session timed out — back to asleep")
                control.send_wake("asleep")

            # Wake strategy MUST come before VAD strategy so it gates start.
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

    stages: list[Any] = [transport.input()]
    if wake_detector is not None:
        # WakeWordDetector runs ahead of STT so the wake event reaches the
        # user-aggregator before any TranscriptionFrame for that turn.
        stages.append(wake_detector)
    stages += [
        stt,
        PersonaCommandRouter(runtime["persona_config"], persona_state),
        aggregator.user(),
        llm_dispatch,
        PersonaTagRouter(runtime["persona_config"], persona_state),
        tts_dispatch,
        transport.output(),
        PipelineStateEmitter(control),
        aggregator.assistant(),
    ]
    pipeline = Pipeline(stages)

    return PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )


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

    logger.info(
        "ready: whisper={whisper} ollama={ollama} personas={p} backends={b}",
        whisper=rc["whisper_model"],
        ollama=rc["ollama_model"],
        p=sorted(rc["available_personas"]),
        b=sorted(rc["available_backends"]),
    )
    try:
        yield
    finally:
        # Tighter shutdown: clean exits complete in <100ms, the timeouts here
        # are only for the pathological case where a task is stuck. Past ~2s
        # the user is double-Ctrl-C'ing anyway, which would SIGKILL us.
        heartbeat.cancel()
        for t in list(_pipeline_tasks):
            t.cancel()
        pending = [heartbeat, *_pipeline_tasks]
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
    import uvicorn

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
