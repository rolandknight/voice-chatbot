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
import shutil
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config import load as load_config  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from _audio_devices import resolve_from_config  # noqa: E402
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
    CancelFrame,
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
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

from wakeword_detector import (  # noqa: E402
    WakeWordDetector,
    WakeWordUserTurnStartStrategy,
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
    intent matching — shows exactly what Whisper produced so skill triggers
    or persona voice-command rules can be tuned to real STT output."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text
        ):
            logger.info(f"Heard: {frame.text.strip()!r}")
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


class InFlightTracker(FrameProcessor):
    """Tracks whether a tool call or LLM response is mid-flight.

    The conversation idle timer and the wake-word IDLE transition both run on
    a fixed budget measured from the user's last spoken activity. During the
    gap between a function call dispatch (e.g. ask_claude) and the first
    streamed LLM token from the now-active backend, no Bot/User speaking or
    transcription frames flow — so both clocks tick without being refreshed
    and can fire mid-response, killing the in-flight Claude request.

    Plumbing this processor into the pipeline after the LLM stage lets the
    idle handlers consult is_in_flight() and defer teardown while work is
    pending.

    The Start/End ControlFrames and the FunctionCallResultFrame DataFrame are
    subject to the route_to_{ollama,claude} FunctionFilters that sandwich
    each branch of the ParallelPipeline — so when ask_claude flips
    backend_state mid-tool-call, the matching End/Result frames from the
    now-inactive branch get silently dropped, leaving the counters stuck.
    BotStoppedSpeakingFrame (SystemFrame, always passes filters) is used as
    a guaranteed reset: once TTS has finished playback, there is nothing
    pending."""

    def __init__(self):
        super().__init__()
        self._fn_calls_outstanding = 0
        self._llm_responding = False
        # Per-response counters reset on LLMFullResponseStartFrame. Used to
        # detect "silent" responses where the LLM emitted neither text nor
        # a dispatchable tool call — the common cause is a tool-call JSON
        # truncated by max_tokens, which pipecat drops via json.loads in
        # base_llm.py without surfacing anything user-visible. The
        # _interrupted flag suppresses the warning when the response was
        # cancelled by the user starting to speak — base_llm.py's finally
        # block pushes LLMFullResponseEndFrame on interruption too, so we
        # need the extra signal to tell a real silent failure from a
        # legitimate barge-in.
        self._text_frames_this_response = 0
        self._tool_calls_this_response = 0
        self._interrupted_this_response = False

    def is_in_flight(self) -> bool:
        return self._fn_calls_outstanding > 0 or self._llm_responding

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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
                    "likely a truncated tool-call JSON (check max_tokens) or "
                    "a malformed completion. Babel will be silent this turn."
                )
        elif isinstance(frame, LLMTextFrame) and frame.text:
            self._text_frames_this_response += 1
        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._interrupted_this_response = True
        elif isinstance(frame, UserStartedSpeakingFrame):
            # Belt-and-braces: some pipecat versions broadcast interruption
            # via UserStartedSpeakingFrame directly rather than a dedicated
            # InterruptionFrame. Either way, an in-flight LLM response is
            # about to be cancelled — don't warn about its empty output.
            if self._llm_responding:
                self._interrupted_this_response = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._fn_calls_outstanding = 0
            self._llm_responding = False
        await self.push_frame(frame, direction)


async def main():
    cfg = load_config()

    logger.remove()
    logger.add(
        sys.stderr,
        filter=lambda record: record["message"] != "Idle timeout detected.",
    )

    if cfg.huggingface.hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    input_device_index, output_device_index, input_device_name, output_device_name = (
        resolve_from_config(cfg.audio)
    )

    audio_in_sample_rate = cfg.audio.in_sample_rate
    audio_out_sample_rate = cfg.audio.out_sample_rate

    whisper_model = cfg.stt.whisper_model
    language = cfg.stt.language
    ollama_model = cfg.llm.ollama_model
    conversation_idle_timeout = cfg.conversation.idle_timeout_secs
    babel_skills_enabled = cfg.skills.enabled
    radio_enabled = babel_skills_enabled and cfg.skills.radio.enabled
    # Spotify is gated on both the config flag and a configured client id —
    # without credentials, the handlers would just emit the bootstrap prompt
    # for every voice command, which is worse than not exposing them at all.
    spotify_enabled = (
        babel_skills_enabled
        and cfg.skills.spotify.enabled
        and bool(cfg.skills.spotify.client_id.get_secret_value().strip())
    )
    # SFX backends both default to off — each is multi-GB on disk and not
    # everyone wants them. When on, the URLs point at the local FastAPI
    # servers under vendor/ (auto-started by run.sh).
    #   - Woosh: foley/ambience/animals/machines (Sony text-to-foley).
    #   - Stable Audio Open: high-quality non-musical sounds incl.
    #     bodily/comedic effects where Woosh underperforms.
    # When both are on, skills.sfx.backend chooses routing (auto = keyword
    # routing handled inside skills/sfx/generate_sound_effect/handler.py).
    sfx_woosh_enabled = babel_skills_enabled and cfg.skills.sfx.woosh_enabled
    sfx_sao_enabled = babel_skills_enabled and cfg.skills.sfx.sao_enabled
    sfx_enabled = sfx_woosh_enabled or sfx_sao_enabled
    sfx_woosh_url = cfg.skills.sfx.woosh_url if sfx_woosh_enabled else None
    sfx_stable_audio_url = cfg.skills.sfx.sao_url if sfx_sao_enabled else None
    sfx_backend_override = cfg.skills.sfx.backend
    # Tracker is a FrameProcessor inserted into the pipeline below; the SFX
    # handler uses it to defer mpv playback until the bot's TTS has stopped,
    # so the spoken ack and the SFX clip don't overlap on the speaker.
    sfx_tracker = BotSpeakingTracker() if sfx_enabled else None
    ollama_base_url = cfg.llm.ollama_base_url

    persona_config, persona_state = load_persona_config(
        cfg.resolved_personas_path(), cfg.tts.default_persona
    )

    chatterbox_base_url = cfg.tts.chatterbox.base_url
    chatterbox_model = cfg.tts.chatterbox.model
    chatterbox_api_key = cfg.tts.chatterbox.api_key.get_secret_value()

    anthropic_api_key = cfg.claude.api_key.get_secret_value().strip()
    anthropic_model = cfg.claude.model
    claude_max_tokens = cfg.claude.max_tokens
    claude_enabled = cfg.claude.enabled

    # Claude server tools. Both are GA on the Anthropic API; passing them in
    # `tools` lets Claude decide per-turn whether to search/fetch. max_uses
    # caps per-turn invocations to keep voice latency bounded — each search
    # round-trip adds ~1-2s before the first audible token.
    claude_web_search = claude_enabled and cfg.claude.web_search_enabled
    claude_web_fetch = claude_enabled and cfg.claude.web_fetch_enabled
    claude_web_search_max_uses = cfg.claude.web_search_max_uses
    claude_web_fetch_max_uses = cfg.claude.web_fetch_max_uses

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
        logger.info(f"Claude model       : {anthropic_model} (via ask_claude skill)")
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
    # here is ~0.28s, so 0.5s is a safe p99 estimate. Override via
    # stt.ttfs_p99_latency_secs in config.yaml after running the stt-benchmark.
    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=whisper_model,
            language=language,
            no_speech_prob=0.55,
            temperature=0.0,
        ),
        ttfs_p99_latency=cfg.stt.ttfs_p99_latency_secs,
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

    # max_tokens bumped to 512: 200 was tight enough that tool-call JSON
    # for skills with non-trivial arg shapes (or the LLM second-guessing
    # itself before emitting the call) could truncate mid-stream. A
    # truncated tool call fails json.loads downstream and is dropped
    # silently — Babel produces no audible response and the wake session
    # times out. 512 leaves headroom for the JSON plus the brief spoken
    # reply after the tool result returns.
    llm = OLLamaLLMService(
        base_url=ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=ollama_model,
            temperature=0.2,
            max_tokens=512,
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

    # Shared mutable backend selector consumed by the ParallelPipeline routing
    # filters below. Declared up here so the SkillContext (built next) can
    # see the same dict the wake-timeout handler later resets. Only meaningful
    # when claude_enabled; the routing filters are only built in that branch.
    backend_state = {"backend": "ollama"}

    skill_registry = None
    if babel_skills_enabled:
        sfx_backends: dict[str, str] = {}
        if sfx_woosh_url:
            sfx_backends["woosh"] = sfx_woosh_url
        if sfx_stable_audio_url:
            sfx_backends["stable_audio"] = sfx_stable_audio_url
        # Shows are gated by skills.shows.enabled via the SKILL.md `enabled_when`
        # field; play_bbc_show also has `requires: [radio_player]`, so it
        # disappears if radio is off too.
        skill_ctx = SkillContext(
            radio_player=radio_player,
            spotify_player=spotify_player,
            sfx_tracker=sfx_tracker,
            sfx_backends=sfx_backends,
            sfx_backend_override=sfx_backend_override,
            persona_config=persona_config,
            persona_state=persona_state,
            # backend_state is only useful when there's actually a Claude
            # branch to switch to. Passing None when claude_enabled is False
            # makes the ask_claude skill's `requires: [backend_state]` gate
            # filter it out.
            backend_state=backend_state if claude_enabled else None,
        )
        skill_registry = load_skills(skill_ctx, cfg)
        skill_registry.register(llm, context)
        logger.info(f"Babel skills      : {sorted(skill_registry.skills_by_name)}")
    else:
        logger.info("Babel skills      : disabled (skills.enabled = false)")
    # Silero's `min_volume` is normalized EBU R128 loudness (pyloudnorm), not
    # RMS. For low-gain USB speakerphones like the Jabra Speak2 40, integrated
    # loudness frequently clamps to ~0, so the 0.6 default gates everything
    # out. Disable the volume floor and rely on Silero's confidence threshold.
    vad_min_volume = cfg.wake.vad_min_volume
    vad_stop_secs = cfg.wake.vad_stop_secs

    wake_model_paths = [m.model for m in cfg.wake.models]
    wake_persona_map = {
        # openwakeword keys the predict() output dict by the model file's
        # stem (or the bundled-model key without the .onnx suffix). For
        # 'models/wakeword/hey_babel.onnx' that's 'hey_babel'; for
        # 'hey_jarvis_v0.1' it's 'hey_jarvis_v0.1' unchanged.
        Path(m.model).stem if m.model.endswith(".onnx") else m.model: m.persona
        for m in cfg.wake.models
    }
    wake_detector = WakeWordDetector(
        model_paths_or_keys=wake_model_paths,
        persona_for_model=wake_persona_map,
        threshold=cfg.wake.threshold,
        cooldown_secs=cfg.wake.cooldown_secs,
        persona_config=persona_config,
        persona_state=persona_state,
    )
    # Tracks tool-call and LLM-response lifecycle frames so the idle/wake
    # clocks can defer their teardown while Claude (or any backend) is still
    # producing a response. Inserted into the pipeline after llm_stage below.
    in_flight = InFlightTracker()

    # One session clock for everything: same value drives the LLM context
    # idle reset (via PipelineTask below) and the wake strategy's IDLE/AWAKE
    # transition. Splitting them is what caused the "I just got reset but
    # I'm still awake" bug. is_busy keeps the wake session alive across the
    # quiet window between an ask_claude dispatch and Claude's first token.
    wake_strategy = WakeWordUserTurnStartStrategy(
        timeout=conversation_idle_timeout,
        is_busy=in_flight.is_in_flight,
    )

    @wake_strategy.event_handler("on_wake_word_detected")
    async def _on_wake(_strategy, model_key, score):
        logger.info(
            f"Wake activated: {model_key!r} (score={score:.3f}) — "
            f"persona={persona_state.current!r}, backend={backend_state['backend']!r}"
        )

    @wake_strategy.event_handler("on_wake_word_timeout")
    async def _on_sleep(_strategy):
        # Session-scoped backend revert: ask_claude flipped this to "claude"
        # for the duration of the wake session; sleep ends the session.
        if backend_state["backend"] != "ollama":
            logger.info(
                f"Wake timeout — reverting backend "
                f"{backend_state['backend']!r} -> 'ollama'"
            )
            backend_state["backend"] = "ollama"
        else:
            logger.info("Wake timeout — going back to sleep")

    logger.info(
        f"Wake models        : {wake_model_paths} "
        f"(timeout={conversation_idle_timeout}s, threshold={cfg.wake.threshold})"
    )

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

    # WakeWordDetector runs ahead of STT on the raw 16 kHz audio frames:
    # openwakeword fires during the wake word itself, not after Whisper has
    # buffered the full utterance, so the wake event reaches the user
    # aggregator before any TranscriptionFrame for that turn.
    pipeline_stages = [
        transport.input(),
        wake_detector,
        stt,
        TranscriptionTap(),
        PersonaCommandRouter(persona_config, persona_state),
        context_aggregator.user(),
    ]
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
                k=cfg.skills.filter_k,
                debug=cfg.skills.filter_debug,
            )
        )
    pipeline_stages += [
        llm_stage,
        in_flight,
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
        if in_flight.is_in_flight():
            # Work pending (function call dispatched, or Claude still streaming).
            # PipelineTask re-fires on_idle_timeout each interval, so a silent
            # return is enough — we re-check on the next tick.
            return
        if not context.messages:
            return
        logger.info(
            f"Conversation idle for {conversation_idle_timeout:.0f}s — resetting context."
        )
        context.set_messages([])
        # Force the wake strategy back to IDLE: if the LLM has forgotten the
        # conversation, the user should have to re-wake too. Otherwise the
        # 30s wake timeout outlives the 10s conversation idle and follow-up
        # speech gets processed without a wake word.
        wake_strategy.force_idle()

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
        # Logged at INFO so missing heartbeats are obvious when chasing
        # cold-load TTFB spikes (a 30s+ first-token delay almost always
        # means the model was evicted between turns).
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    await asyncio.sleep(240)
                    response = await client.post(
                        f"{ollama_host}/api/generate",
                        json={"model": ollama_model, "keep_alive": -1},
                    )
                    response.raise_for_status()
                    logger.info(
                        f"Ollama keepalive: {ollama_model} pinned (keep_alive=-1)"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Ollama keepalive heartbeat failed: {e}")

    keepalive_task = asyncio.create_task(_ollama_keepalive_heartbeat())

    wake_display = ", ".join(sorted(wake_persona_map.keys())) or "(no models)"
    if claude_enabled:
        logger.info(
            f"Ready. Wake words: {wake_display}. "
            f"Say 'ask Claude' mid-session to route to Claude. Press Ctrl+C to stop."
        )
    else:
        logger.info(
            f"Ready. Wake words: {wake_display}. Press Ctrl+C to stop."
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
