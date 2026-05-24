#!/usr/bin/env python3
"""
Local Pipecat voice-to-voice prototype for Apple Silicon + Jabra USB speakerphone.

Pipeline:
  Jabra mic -> Pipecat LocalAudioTransport -> Whisper MLX STT -> Ollama LLM -> Kokoro TTS -> Jabra speaker

Notes:
  - First run downloads model weights and can take a while.
  - For best Jabra behavior, set it as the macOS default input/output device.
  - You can also force device indexes in .env after running ./run.sh --devices.
"""

import asyncio
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from _audio_devices import resolve_from_env  # noqa: E402
from radio import RadioPlayer  # noqa: E402
from spotify import SpotifyPlayer  # noqa: E402

from skills import (  # noqa: E402
    BotSpeakingTracker,
    SkillContext,
    SkillFilterProcessor,
    load_skills,
)
from persona_router import (  # noqa: E402
    CHATTERBOX_BACKEND,
    KOKORO_BACKEND,
    PersonaCommandRouter,
    PersonaConfig,
    PersonaState,
    PersonaTagRouter,
    load_persona_config,
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.tts_service import TextAggregationMode

from chatterbox_tts import ChatterboxTTSService  # noqa: E402

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMContextFrame,
    MixerControlFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)

from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import (
    WakePhraseUserTurnStartStrategy,
)


class KeepaliveMixer(BaseAudioMixer):
    """No-op mixer whose only purpose is to make BaseOutputTransport's audio
    loop continuously emit silence when no TTS audio is queued. USB
    speakerphones like the Jabra Speak2 40 power their amp/DAC down when
    they see a fully idle stream, eating the first 500ms+ of the next
    utterance. Continuous silence keeps the device awake."""

    async def start(self, sample_rate: int) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def process_frame(self, frame: MixerControlFrame) -> None:
        pass

    async def mix(self, audio: bytes) -> bytes:
        return audio


class TranscriptionTap(FrameProcessor):
    """Logs final transcriptions as they pass through. Useful for debugging
    wake-phrase mismatches — shows exactly what Whisper produced so aliases
    can be added to WAKE_PHRASES / CLAUDE_WAKE_PHRASES."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text
        ):
            logger.info(f"Heard: {frame.text.strip()!r}")
        await self.push_frame(frame, direction)


class BackendRouter(FrameProcessor):
    """Switches backend_state based on wake-phrase content in every final
    transcription, not just the IDLE→AWAKE transition.

    pipecat's WakePhraseUserTurnStartStrategy only fires on_wake_phrase_detected
    once per IDLE→AWAKE transition. While AWAKE it just refreshes its timeout
    and passes frames through, so a mid-conversation 'hey babel' after an
    initial 'hey claude' never flips the backend. This processor mirrors the
    strategy's regex/punctuation matching and updates backend_state on every
    transcription; last wake phrase in the utterance wins."""

    def __init__(self, backend_state: dict, claude_phrases, ollama_phrases):
        super().__init__()
        self._state = backend_state
        self._claude_patterns = self._compile(claude_phrases)
        self._ollama_patterns = self._compile(ollama_phrases)

    @staticmethod
    def _compile(phrases):
        # Same shape as WakePhraseUserTurnStartStrategy: word-boundary
        # anchored, whitespace-tolerant between words, case-insensitive.
        out = []
        for phrase in phrases:
            words = phrase.split()
            if not words:
                continue
            out.append(
                re.compile(
                    r"\b" + r"\s*".join(re.escape(w) for w in words) + r"\b",
                    re.IGNORECASE,
                )
            )
        return out

    @staticmethod
    def _last_match_pos(text: str, patterns) -> int:
        best = -1
        for pattern in patterns:
            for m in pattern.finditer(text):
                if m.start() > best:
                    best = m.start()
        return best

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text
        ):
            stripped = re.sub(r"[^\w\s]", "", frame.text)
            claude_pos = self._last_match_pos(stripped, self._claude_patterns)
            ollama_pos = self._last_match_pos(stripped, self._ollama_patterns)
            new_backend = None
            if claude_pos >= 0 and claude_pos >= ollama_pos:
                new_backend = "claude"
            elif ollama_pos >= 0 and ollama_pos > claude_pos:
                new_backend = "ollama"
            if new_backend and self._state.get("backend") != new_backend:
                logger.info(
                    f"Backend switched -> {new_backend} (in-turn wake phrase)"
                )
                self._state["backend"] = new_backend
        await self.push_frame(frame, direction)


class ClaudeCueEmitter(FrameProcessor):
    """Emits a short TTS cue ("Claude here.") immediately before forwarding the
    LLMContextFrame downstream. Placed in the Claude branch of the ParallelPipeline
    so the cue only plays when Claude actually answers."""

    def __init__(self, cue_text: str = "Claude here."):
        super().__init__()
        self._cue_text = cue_text

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(
                TTSSpeakFrame(text=self._cue_text), direction
            )
        await self.push_frame(frame, direction)


class MediaDuckWatcher(FrameProcessor):
    """Ducks each registered media player when the user starts speaking and
    un-ducks after the bot finishes its reply. Placed after transport.output()
    so it sees the canonical Bot/User speaking lifecycle frames.

    Players are duck-compatible if they expose:
      - is_playing() -> bool
      - a pause callable (pause() for RadioPlayer; duck_pause() for
        SpotifyPlayer, which reserves pause() for the Web API "user
        explicitly paused" semantics)
      - a matching resume callable (resume() / duck_resume())

    A safety timer guards against the case where the user makes a noise that
    never triggers a wake/LLM turn — without it, media would stay paused
    until the next bot reply, which could be never."""

    SAFETY_RESUME_SECS = 8.0

    def __init__(self, players):
        super().__init__()
        self._players = [p for p in players if p is not None]
        self._safety_task: Optional[asyncio.Task] = None

    @staticmethod
    def _duck(player) -> None:
        if hasattr(player, "duck_pause"):
            player.duck_pause()
        else:
            player.pause()

    @staticmethod
    def _unduck(player) -> None:
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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


def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name, "")
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "")
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value == "":
        return default
    return value in ("1", "true", "yes", "on")


async def main():
    load_dotenv(override=True)

    logger.remove()
    logger.add(
        sys.stderr,
        filter=lambda record: record["message"] != "Idle timeout detected.",
    )

    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"

    input_device_index, output_device_index, input_device_name, output_device_name = (
        resolve_from_env()
    )

    audio_in_sample_rate = env_int("AUDIO_IN_SAMPLE_RATE", 16000)
    audio_out_sample_rate = env_int("AUDIO_OUT_SAMPLE_RATE", 24000)

    whisper_model = os.getenv("WHISPER_MLX_MODEL", "mlx-community/whisper-base.en-mlx")
    language = os.getenv("LANGUAGE", "en")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:26b")
    conversation_idle_timeout = env_float("CONVERSATION_IDLE_TIMEOUT_SECS", 10.0)
    babel_skills_enabled = env_bool("BABEL_SKILLS_ENABLED", True)
    radio_enabled = babel_skills_enabled and env_bool("BABEL_RADIO_ENABLED", True)
    shows_enabled = radio_enabled and env_bool("BABEL_SHOWS_ENABLED", True)
    # Spotify is gated on both the env flag and a configured client id —
    # without credentials, the handlers would just emit the bootstrap prompt
    # for every voice command, which is worse than not exposing them at all.
    spotify_enabled = (
        babel_skills_enabled
        and env_bool("BABEL_SPOTIFY_ENABLED", True)
        and bool(os.getenv("SPOTIPY_CLIENT_ID", "").strip())
    )
    # SFX backends both default to off — each is multi-GB on disk and not
    # everyone wants them. When on, the URLs point at the local FastAPI
    # servers under vendor/ (auto-started by run.sh).
    #   - Woosh: foley/ambience/animals/machines (Sony text-to-foley).
    #   - Stable Audio Open: high-quality non-musical sounds incl.
    #     bodily/comedic effects where Woosh underperforms.
    # When both are on, BABEL_SFX_BACKEND chooses routing (auto = keyword
    # routing handled inside skills/sfx/generate_sound_effect/handler.py).
    sfx_woosh_enabled = babel_skills_enabled and env_bool("BABEL_SFX_ENABLED", False)
    sfx_sao_enabled = babel_skills_enabled and env_bool("BABEL_SAO_ENABLED", False)
    sfx_enabled = sfx_woosh_enabled or sfx_sao_enabled
    sfx_woosh_url = (
        os.getenv("WOOSH_URL", "http://127.0.0.1:8005")
        if sfx_woosh_enabled
        else None
    )
    sfx_stable_audio_url = (
        os.getenv("STABLE_AUDIO_URL", "http://127.0.0.1:8006")
        if sfx_sao_enabled
        else None
    )
    sfx_backend_override = os.getenv("BABEL_SFX_BACKEND", "auto").strip().lower() or None
    # Tracker is a FrameProcessor inserted into the pipeline below; the SFX
    # handler uses it to defer mpv playback until the bot's TTS has stopped,
    # so the spoken ack and the SFX clip don't overlap on the speaker.
    sfx_tracker = BotSpeakingTracker() if sfx_enabled else None
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    personas_config_path = Path(
        os.getenv("PERSONAS_CONFIG", "personas.yaml")
    )
    if not personas_config_path.is_absolute():
        personas_config_path = Path(__file__).parent / personas_config_path
    persona_config, persona_state = load_persona_config(
        personas_config_path, os.getenv("DEFAULT_PERSONA")
    )

    # Legacy env override: pre-personas config, the active voice was set
    # via KOKORO_VOICE. If it's still set we apply it to the babel persona
    # so existing .env files don't silently drift.
    kokoro_voice_override = os.getenv("KOKORO_VOICE", "").strip()
    if kokoro_voice_override:
        babel_persona = persona_config.personas.get("babel")
        if babel_persona and babel_persona.backend == KOKORO_BACKEND:
            if babel_persona.voice != kokoro_voice_override:
                logger.info(
                    f"KOKORO_VOICE={kokoro_voice_override!r} overrides babel persona voice"
                )
                babel_persona.voice = kokoro_voice_override

    chatterbox_base_url = os.getenv(
        "CHATTERBOX_BASE_URL", "http://127.0.0.1:8004/v1"
    )
    chatterbox_model = os.getenv("CHATTERBOX_MODEL", "chatterbox-turbo")
    chatterbox_api_key = os.getenv("CHATTERBOX_API_KEY", "not-needed")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    # Multiple aliases because whisper-tiny.en often mishears "claude" as
    # "cloud", "claud", "clod", etc. All of these route to Claude.
    claude_wake_phrases = [
        p.strip().lower()
        for p in os.getenv(
            "CLAUDE_WAKE_PHRASES",
            "hey claude,hey claud,hey cloud,hey clod,hey clog",
        ).split(",")
        if p.strip()
    ]
    claude_max_tokens = env_int("CLAUDE_MAX_TOKENS", 1024) or 1024
    claude_enabled = bool(anthropic_api_key)

    # Claude server tools. Both are GA on the Anthropic API; passing them in
    # `tools` lets Claude decide per-turn whether to search/fetch. max_uses
    # caps per-turn invocations to keep voice latency bounded — each search
    # round-trip adds ~1-2s before the first audible token.
    claude_web_search = claude_enabled and env_bool("CLAUDE_WEB_SEARCH", True)
    claude_web_fetch = claude_enabled and env_bool("CLAUDE_WEB_FETCH", True)
    claude_web_search_max_uses = env_int("CLAUDE_WEB_SEARCH_MAX_USES", 3) or 3
    claude_web_fetch_max_uses = env_int("CLAUDE_WEB_FETCH_MAX_USES", 2) or 2

    claude_tools: list[dict] = []
    if claude_web_search:
        claude_tools.append(
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": claude_web_search_max_uses,
            }
        )
    if claude_web_fetch:
        claude_tools.append(
            {
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "max_uses": claude_web_fetch_max_uses,
            }
        )

    logger.info("Starting local voice prototype")
    logger.info(f"Input device  : [{input_device_index}] {input_device_name}")
    logger.info(f"Output device : [{output_device_index}] {output_device_name}")
    logger.info(f"Whisper MLX model  : {whisper_model}")
    logger.info(f"Ollama model       : {ollama_model}")
    persona_summary = ", ".join(
        f"{p.id}({p.backend}:{p.voice})" for p in persona_config.personas.values()
    )
    logger.info(f"Personas           : {persona_summary}")
    if claude_enabled:
        logger.info(f"Claude model       : {anthropic_model} (wake: {claude_wake_phrases})")
        enabled_tools = [t["name"] for t in claude_tools]
        if enabled_tools:
            logger.info(f"Claude tools       : {enabled_tools}")
        else:
            logger.info("Claude tools       : none")
    else:
        logger.info("Claude routing     : disabled (ANTHROPIC_API_KEY not set)")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio_in_sample_rate,
            audio_out_sample_rate=audio_out_sample_rate,
            audio_in_channels=1,
            audio_out_channels=1,
            # Must be True: this controls whether captured audio frames are
            # pushed downstream to STT/VAD/turn analyzer. It is NOT a mic→
            # speaker loopback switch. With False, no transcription happens.
            audio_in_passthrough=True,
            # Drives continuous silence to the output device to keep the
            # Jabra amp from sleeping between utterances.
            audio_out_mixer=KeepaliveMixer(),
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )
    )

    # ttfs_p99_latency: measured p99 from speech end → final transcript. Used
    # by the turn analyzer to size its post-VAD wait window. Default 1.0s is
    # too conservative for whisper-tiny.en-mlx on Apple Silicon; observed TTFB
    # here is ~0.28s, so 0.5s is a safe p99 estimate. Override with
    # STT_TTFS_P99_LATENCY after running the stt-benchmark for your settings.
    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=whisper_model,
            language=language,
            no_speech_prob=0.55,
            temperature=0.0,
        ),
        ttfs_p99_latency=env_float("STT_TTFS_P99_LATENCY", 0.5),
    )

    # Both babel and Claude share one LLMContext for conversation history but
    # each carries its own system instruction via service settings. Putting the
    # system message into the context too would trigger the adapter's
    # "Both system_instruction and an initial system message" warning every
    # turn and would force babel's prompt onto Claude (or vice versa).
    babel_system_prompt = (
        "You are a fast local voice assistant running on a Mac. "
        "Keep replies brief and conversational. "
        "Prefer one or two short sentences. "
        "Do not mention internal implementation details unless asked."
    )
    if babel_skills_enabled:
        # Per-tool guidance lives in each skill's SKILL.md description (sent
        # to Ollama as part of the OpenAI tools payload). The system prompt
        # only carries the meta-instruction to actually call tools.
        babel_system_prompt += (
            " Call tools whenever the user asks for the time, the date, a "
            "timer, the weather, music, radio, sound effects, or any "
            "current/recent information. Don't guess answers you can look up. "
            "After a tool returns, repeat its result back in one short spoken "
            "sentence."
        )

    # max_tokens bumped from 80 to 200 because tool-call turns need room
    # for the tool call payload plus a one- or two-sentence spoken response
    # after the tool result comes back.
    llm = OLLamaLLMService(
        base_url=ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=ollama_model,
            temperature=0.2,
            max_tokens=200,
            system_instruction=babel_system_prompt,
        ),
    )

    claude_llm = None
    if claude_enabled:
        tool_hint = ""
        if claude_web_search and claude_web_fetch:
            tool_hint = (
                " You can call web_search for current information and web_fetch to "
                "read a specific URL — use them when the question depends on "
                "recent or external facts, but skip them for things you already "
                "know. When you do search, summarize the answer in plain prose; "
                "don't read URLs or citation markers aloud."
            )
        elif claude_web_search:
            tool_hint = (
                " You can call web_search for current information — use it when "
                "the question depends on recent facts, but skip it for things "
                "you already know. Summarize results in plain prose; don't read "
                "URLs or citation markers aloud."
            )
        elif claude_web_fetch:
            tool_hint = (
                " You can call web_fetch to read a specific URL the user "
                "mentions. Summarize the page in plain prose; don't read URLs "
                "or citation markers aloud."
            )
        claude_system_prompt = (
            "You are Claude, a helpful voice assistant on a Mac. "
            "Reply conversationally and stay spoken-friendly: no markdown, no bullet "
            "lists, no code blocks. Be accurate and complete — you can take a "
            "paragraph or two when the question warrants it."
            + tool_hint
        )
        claude_extra: dict = {}
        if claude_tools:
            # Pipecat's AnthropicLLMService merges settings.extra into the
            # request params *after* the context's (empty) tools list, so
            # this puts our server tools into the actual API call.
            claude_extra["tools"] = claude_tools
        claude_llm = AnthropicLLMService(
            api_key=anthropic_api_key,
            settings=AnthropicLLMService.Settings(
                model=anthropic_model,
                max_tokens=claude_max_tokens,
                system_instruction=claude_system_prompt,
                extra=claude_extra,
            ),
        )

    # Chatterbox-TTS-Server resolves its `voice` parameter against its
    # own reference_audio/ + voices/ directories — it doesn't read our
    # personas.yaml ref_audio field. Copy each chatterbox persona's
    # local ref clip into the server's reference_audio dir using the
    # persona's `voice` value as the destination name.
    #
    # We have to copy (not symlink) because the server uses Path.resolve()
    # + a "stays inside reference_audio" check (utils.safe_resolve_within);
    # any symlink whose target lives elsewhere fails as "path traversal".
    # Re-copies only when source mtime/size has changed, so restarts stay
    # fast.
    chatterbox_ref_dir = (
        Path(__file__).parent / "vendor" / "chatterbox-tts-server" / "reference_audio"
    )
    if persona_config.chatterbox_personas():
        if not chatterbox_ref_dir.is_dir():
            logger.warning(
                f"Chatterbox reference_audio dir missing at {chatterbox_ref_dir}; "
                f"voices may 404 until the server is fully installed."
            )
        else:
            for p in persona_config.chatterbox_personas():
                if not p.ref_audio:
                    continue
                src = (Path(__file__).parent / p.ref_audio).resolve()
                if not src.is_file():
                    logger.warning(
                        f"persona {p.id!r}: ref_audio {p.ref_audio} not found "
                        f"(looked at {src}); the chatterbox voice will 404"
                    )
                    continue
                dest = chatterbox_ref_dir / p.voice
                try:
                    if dest.is_symlink():
                        # Clean up older symlink-based installs that the
                        # server now rejects as path traversal.
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
                    logger.info(
                        f"Copied {p.ref_audio} -> reference_audio/{p.voice}"
                    )
                except OSError as e:
                    logger.error(
                        f"Could not copy {src} into {chatterbox_ref_dir}: {e}"
                    )

    # SENTENCE mode: Kokoro gets full clauses/sentences for natural prosody.
    # Costs ~600ms vs TOKEN mode since the aggregator waits for sentence
    # boundaries before sending text to TTS, but TOKEN mode produced choppy
    # audio with Kokoro because each LLM token was synthesized independently.
    # Chatterbox runs through the same sentence aggregator for parity.
    persona_tts_services: dict[str, object] = {}
    for persona in persona_config.personas.values():
        if persona.backend == KOKORO_BACKEND:
            persona_tts_services[persona.id] = KokoroTTSService(
                settings=KokoroTTSService.Settings(voice=persona.voice),
                text_aggregation_mode=TextAggregationMode.SENTENCE,
            )
        elif persona.backend == CHATTERBOX_BACKEND:
            # Chatterbox-TTS-Server speaks the OpenAI /v1/audio/speech
            # protocol. The `voice` is the server-side registered name
            # for the reference clip; the actual clone is served by
            # Chatterbox using the ref_audio file declared in
            # personas.yaml (registered with the server out of band, see
            # scripts/chatterbox_health.py).
            persona_tts_services[persona.id] = ChatterboxTTSService(
                api_key=chatterbox_api_key,
                base_url=chatterbox_base_url,
                sample_rate=audio_out_sample_rate,
                settings=ChatterboxTTSService.Settings(
                    voice=persona.voice,
                    model=chatterbox_model,
                ),
                text_aggregation_mode=TextAggregationMode.SENTENCE,
            )
        else:
            raise RuntimeError(
                f"persona {persona.id!r}: unsupported backend {persona.backend!r}"
            )

    radio_player: Optional[RadioPlayer] = RadioPlayer() if radio_enabled else None
    spotify_player: Optional[SpotifyPlayer] = (
        SpotifyPlayer() if spotify_enabled else None
    )

    # Context starts empty (no system message). Each LLM service carries its
    # own system_instruction; see babel_system_prompt above and
    # claude_system_prompt below.
    context = LLMContext(messages=[])
    skill_registry = None
    if babel_skills_enabled:
        sfx_backends: dict[str, str] = {}
        if sfx_woosh_url:
            sfx_backends["woosh"] = sfx_woosh_url
        if sfx_stable_audio_url:
            sfx_backends["stable_audio"] = sfx_stable_audio_url
        # Shows are gated by env (BABEL_SHOWS_ENABLED, default on) inside the
        # SKILL.md frontmatter. The radio_player gate handles the deps; if
        # shows are disabled explicitly we drop the player reference so the
        # play_bbc_show skill's `requires: [radio_player]` check fails. The
        # other radio skills still register because shows_enabled defaults
        # to True so the env gate is unset.
        skill_ctx = SkillContext(
            radio_player=radio_player,
            spotify_player=spotify_player,
            sfx_tracker=sfx_tracker,
            sfx_backends=sfx_backends,
            sfx_backend_override=sfx_backend_override,
            persona_config=persona_config,
            persona_state=persona_state,
        )
        skill_registry = load_skills(skill_ctx)
        skill_registry.register(llm, context)
        logger.info(f"Babel skills      : {sorted(skill_registry.skills_by_name)}")
    else:
        logger.info("Babel skills      : disabled (BABEL_SKILLS_ENABLED=0)")
    # Silero's `min_volume` is normalized EBU R128 loudness (pyloudnorm), not
    # RMS. For low-gain USB speakerphones like the Jabra Speak2 40, integrated
    # loudness frequently clamps to ~0, so the 0.6 default gates everything
    # out. Disable the volume floor and rely on Silero's confidence threshold.
    vad_min_volume = env_float("VAD_MIN_VOLUME", 0.0)
    vad_stop_secs = env_float("VAD_STOP_SECS", 0.2)

    wake_phrases = [
        p.strip()
        for p in os.getenv("WAKE_PHRASES", "hey babel,hey babe,hey baby").split(",")
        if p.strip()
    ]
    if claude_enabled:
        existing_lower = {p.lower() for p in wake_phrases}
        for phrase in claude_wake_phrases:
            if phrase and phrase not in existing_lower:
                wake_phrases.append(phrase)
                existing_lower.add(phrase)
    claude_wake_set = set(claude_wake_phrases)
    wake_timeout_secs = env_float("WAKE_TIMEOUT_SECS", 30.0)
    wake_strategy = WakePhraseUserTurnStartStrategy(
        phrases=wake_phrases,
        timeout=wake_timeout_secs,
        single_activation=False,
    )

    # Mutable state shared with the routing filters below. Wake handler is the
    # single writer; filters read this when an LLMContextFrame arrives.
    backend_state = {"backend": "ollama"}

    @wake_strategy.event_handler("on_wake_phrase_detected")
    async def _on_wake(_strategy, phrase):
        matched = (phrase or "").strip().lower()
        if claude_enabled and matched in claude_wake_set:
            backend_state["backend"] = "claude"
        else:
            backend_state["backend"] = "ollama"
        logger.info(f"Wake phrase detected: {phrase!r} -> {backend_state['backend']}")

    @wake_strategy.event_handler("on_wake_phrase_timeout")
    async def _on_sleep(_strategy):
        logger.info("Wake timeout — going back to sleep")

    logger.info(f"Wake phrases       : {wake_phrases} (timeout={wake_timeout_secs}s)")

    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(min_volume=vad_min_volume, stop_secs=vad_stop_secs),
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[
                    wake_strategy,
                    VADUserTurnStartStrategy(),
                    TranscriptionUserTurnStartStrategy(),
                ],
            ),
        ),
    )

    if claude_enabled and claude_llm is not None:
        async def _route_to_ollama(_frame: Frame) -> bool:
            return backend_state.get("backend", "ollama") != "claude"

        async def _route_to_claude(_frame: Frame) -> bool:
            return backend_state.get("backend", "ollama") == "claude"

        # Filters MUST sandwich each LLM. A single top filter only blocks
        # downstream frames; the assistant aggregator pushes the tool-result
        # re-trigger LLMContextFrame UPSTREAM, and inside a branch upstream
        # flow goes sink -> llm -> filter, so the LLM runs inference BEFORE
        # the top filter sees the frame. Without a bottom filter both LLMs
        # answer the second round (babel speaks the tool result, Claude
        # then speaks its own paraphrase). The bottom filter intercepts the
        # upstream frame before it reaches the LLM in the inactive branch.
        # direction=None on every filter so both directions are gated.
        llm_stage = ParallelPipeline(
            [
                FunctionFilter(filter=_route_to_ollama, direction=None),
                llm,
                FunctionFilter(filter=_route_to_ollama, direction=None),
            ],
            [
                FunctionFilter(filter=_route_to_claude, direction=None),
                ClaudeCueEmitter(),
                claude_llm,
                FunctionFilter(filter=_route_to_claude, direction=None),
            ],
        )
    else:
        llm_stage = llm

    # Build the TTS dispatch: when there's just one persona we use the
    # service directly (zero overhead, matches pre-personas behaviour);
    # otherwise we fan out via a ParallelPipeline with a FunctionFilter
    # per branch keyed on persona_state.current. This mirrors the
    # Ollama/Claude routing pattern above.
    if len(persona_tts_services) == 1:
        tts_dispatch = next(iter(persona_tts_services.values()))
    else:
        def _make_persona_filter(target_id: str):
            async def _f(_frame: Frame) -> bool:
                return persona_state.current == target_id
            return _f

        tts_dispatch = ParallelPipeline(
            *[
                [
                    FunctionFilter(filter=_make_persona_filter(pid), direction=None),
                    svc,
                ]
                for pid, svc in persona_tts_services.items()
            ]
        )

    pipeline_stages = [
        transport.input(),
        stt,
        TranscriptionTap(),
        PersonaCommandRouter(persona_config, persona_state),
    ]
    if claude_enabled:
        # Ollama wake phrases as a set distinct from Claude's. Anything in
        # `wake_phrases` that's not a Claude phrase is treated as Ollama.
        ollama_only_phrases = [
            p for p in wake_phrases if p.strip().lower() not in claude_wake_set
        ]
        pipeline_stages.append(
            BackendRouter(
                backend_state=backend_state,
                claude_phrases=claude_wake_phrases,
                ollama_phrases=ollama_only_phrases,
            )
        )
    pipeline_stages.append(context_aggregator.user())
    if skill_registry is not None:
        # Per-turn tool filter: reads the latest user message off the shared
        # LLMContext and swaps in the top-K relevant tools. Runs before the
        # LLM stage so Ollama sees a tight, on-topic tool list. Claude's
        # tools come from settings.extra so this mutation is a no-op for
        # the Claude branch.
        pipeline_stages.append(
            SkillFilterProcessor(
                context,
                skill_registry,
                k=env_int("BABEL_SKILL_FILTER_K", 15),
                debug=env_bool("BABEL_SKILL_FILTER_DEBUG", False),
            )
        )
    pipeline_stages += [
        llm_stage,
        PersonaTagRouter(persona_config, persona_state),
        tts_dispatch,
        transport.output(),
    ]
    duckable_players = [p for p in (radio_player, spotify_player) if p is not None]
    if duckable_players:
        # Duck active media while babel is listening/replying. Placed after
        # transport.output() so it sees the User/Bot speaking lifecycle frames
        # the transport emits.
        pipeline_stages.append(MediaDuckWatcher(duckable_players))
    if sfx_tracker is not None:
        # Same placement requirement as RadioDuckWatcher: must be downstream
        # of transport.output() to see canonical Bot/User speaking frames.
        pipeline_stages.append(sfx_tracker)
    pipeline_stages.append(context_aggregator.assistant())
    pipeline = Pipeline(pipeline_stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=conversation_idle_timeout,
        cancel_on_idle_timeout=False,
    )

    @task.event_handler("on_idle_timeout")
    async def _on_conversation_idle(task):
        if not context.messages:
            return
        logger.info(
            f"Conversation idle for {conversation_idle_timeout:.0f}s — resetting context."
        )
        context.set_messages([])

    runner = PipelineRunner()

    # Pre-warm Whisper MLX: the first mlx_whisper.transcribe call downloads
    # weights from HF and builds the MLX graph (~2s the first time). Doing it
    # here keeps that cost off the first user turn.
    try:
        import mlx_whisper
        import numpy as np
        logger.info(f"Pre-warming Whisper MLX ({whisper_model})...")
        silent = np.zeros(audio_in_sample_rate // 10, dtype=np.float32)  # 100ms
        await asyncio.to_thread(
            mlx_whisper.transcribe,
            silent,
            path_or_hf_repo=whisper_model,
            language=language,
        )
    except Exception as e:
        logger.debug(f"Whisper warmup skipped: {e}")

    # Pre-warm each persona's TTS service so the first real turn doesn't pay
    # ONNX graph warm-up (Kokoro) or model-load cost (Chatterbox).
    # Chatterbox needs real text — its text chunker drops whitespace-only
    # input as "no usable chunks". A short token like "Hi." warms both
    # services correctly. Chatterbox warm-up will still fail silently if
    # the server isn't running yet; that's expected — babel keeps working.
    for pid, svc in persona_tts_services.items():
        try:
            async for _ in svc.run_tts("Hi.", context_id=f"warmup-{pid}"):
                break
        except Exception as e:
            logger.debug(f"TTS warmup skipped for persona {pid!r}: {e}")

    # Pre-warm the Ollama model AND pin it resident with keep_alive=-1. Without
    # this the first user turn pays a cold-load cost (~1-3s) and any 5-minute
    # idle stretch evicts the model and pays it again. /api/chat (not /v1) is
    # the native endpoint that honors keep_alive.
    #
    # Pipecat's OLLamaLLMService talks to /v1/chat/completions, which does not
    # accept keep_alive — so each real turn resets the model's TTL to Ollama's
    # 5-minute default. The heartbeat task below re-pins it every 4 minutes.
    ollama_host = ollama_base_url.rsplit("/v1", 1)[0]
    try:
        import httpx
        logger.info(f"Pre-warming Ollama LLM ({ollama_model})...")
        async with httpx.AsyncClient(timeout=30.0) as _warm_client:
            await _warm_client.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 1},
                },
            )
    except Exception as e:
        logger.debug(f"Ollama warmup skipped: {e}")

    async def _ollama_keepalive_heartbeat():
        # /api/generate with no prompt and keep_alive=-1 is the canonical
        # "pin this model in memory" call — doesn't generate, just refreshes
        # the TTL. 240s leaves a 60s margin under Ollama's 5-minute default.
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    await asyncio.sleep(240)
                    await client.post(
                        f"{ollama_host}/api/generate",
                        json={"model": ollama_model, "keep_alive": -1},
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"Ollama keepalive heartbeat skipped: {e}")

    keepalive_task = asyncio.create_task(_ollama_keepalive_heartbeat())

    if claude_enabled:
        claude_display = claude_wake_phrases[0] if claude_wake_phrases else "hey claude"
        logger.info(
            f"Ready. Say {wake_phrases[0]!r} for the local model, "
            f"{claude_display!r} for Claude. Press Ctrl+C to stop."
        )
    else:
        logger.info(
            f"Ready. Say {wake_phrases[0]!r} to wake the bot. Press Ctrl+C to stop."
        )

    try:
        await runner.run(task)
    finally:
        keepalive_task.cancel()
        try:
            await keepalive_task
        except (asyncio.CancelledError, Exception):
            pass
        if radio_player is not None:
            radio_player.stop()
        if spotify_player is not None:
            # api_pause=False on shutdown: killing mpv already silences the
            # speaker, and the API pause is the part that produces the
            # "Your application has reached a rate/request limit..." stdout
            # spam from spotipy when we're throttled.
            spotify_player.stop(api_pause=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
