"""Persona routing for the voice chatbot.

All routing behaviour is driven by personas.yaml — no persona names,
backends, or trigger phrases are hardcoded here. This module loads the
config, exposes a shared `PersonaState` for the rest of the app to read,
and provides two small FrameProcessor subclasses that update that state
based on what flows past them:

  - PersonaCommandRouter: watches TranscriptionFrame (placed after STT,
    before the context aggregator). Matches voice_command rules from
    personas.yaml against the user's transcript.
  - PersonaTagRouter: watches TTSSpeakFrame / TextFrame (placed
    immediately before the TTS dispatch branches). Matches llm_tag
    rules, strips the tag from the text so TTS never voices it.

The actual TTS dispatch (one service per persona, gated by persona
state) is built in app.py via a ParallelPipeline of FunctionFilters —
this matches the existing Ollama/Claude routing pattern in the codebase
and keeps the routing implementation visible at the pipeline level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml
from loguru import logger

from pipecat.frames.frames import Frame, LLMTextFrame, TextFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


KOKORO_BACKEND = "kokoro"
CHATTERBOX_BACKEND = "chatterbox"
VALID_BACKENDS = (KOKORO_BACKEND, CHATTERBOX_BACKEND)


@dataclass
class PersonaDef:
    """One persona entry from personas.yaml."""

    id: str
    backend: str
    voice: str
    ref_audio: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Rule:
    type: str
    action: str
    # voice_command rules
    pattern: re.Pattern | None = None
    # llm_tag rules
    tag_pattern: re.Pattern | None = None
    # skill_intent rules
    skill: str | None = None


@dataclass
class PersonaState:
    """Mutable persona state shared across the pipeline.

    `current` is the active persona id. `pinned` is the persona we revert
    to after a `set_persona_until_next_tag` action — the inline-tag
    trigger sets this so a tag is a one-shot voice change rather than a
    sticky switch.
    """

    current: str
    pinned: str
    # True between a `set_persona_until_next_tag` firing and the next
    # assistant turn boundary; PersonaTagRouter clears it.
    one_shot_active: bool = False


class PersonaConfig:
    """Parsed personas.yaml. Loaded once at startup; immutable thereafter."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}

        personas_raw = raw.get("personas") or {}
        if not isinstance(personas_raw, dict) or not personas_raw:
            raise ValueError(
                f"{path}: 'personas' section is required and must not be empty"
            )

        personas: dict[str, PersonaDef] = {}
        for pid, body in personas_raw.items():
            if not isinstance(body, dict):
                raise ValueError(f"{path}: persona {pid!r} must be a mapping")
            backend = body.get("backend")
            if backend not in VALID_BACKENDS:
                raise ValueError(
                    f"{path}: persona {pid!r} has unknown backend {backend!r}; "
                    f"expected one of {VALID_BACKENDS}"
                )
            voice = body.get("voice")
            if not voice:
                raise ValueError(f"{path}: persona {pid!r} is missing 'voice'")
            ref_audio = body.get("ref_audio")
            if backend == CHATTERBOX_BACKEND and not ref_audio:
                raise ValueError(
                    f"{path}: persona {pid!r} uses backend 'chatterbox' but has "
                    f"no ref_audio — chatterbox needs a reference clip"
                )
            personas[pid] = PersonaDef(
                id=pid,
                backend=backend,
                voice=voice,
                ref_audio=ref_audio,
                settings=body.get("settings") or {},
            )
        self.personas = personas

        routing_raw = raw.get("routing") or {}
        default = routing_raw.get("default")
        if not default:
            raise ValueError(f"{path}: 'routing.default' is required")
        if default not in personas:
            raise ValueError(
                f"{path}: routing.default = {default!r} is not a declared persona"
            )
        self.default = default

        rules_raw = routing_raw.get("rules") or []
        rules: list[_Rule] = []
        for idx, r in enumerate(rules_raw):
            rules.append(self._compile_rule(idx, r, set(personas.keys())))
        self.rules = rules

    @staticmethod
    def _compile_rule(idx: int, r: dict, persona_ids: set[str]) -> _Rule:
        rtype = r.get("type")
        action = r.get("action")
        if not rtype or not action:
            raise ValueError(f"routing.rules[{idx}]: type and action are required")
        if rtype == "voice_command":
            match = r.get("match")
            if not match:
                raise ValueError(f"routing.rules[{idx}]: voice_command needs 'match'")
            pattern = _compile_voice_command(match, persona_ids)
            return _Rule(type=rtype, action=action, pattern=pattern)
        if rtype == "llm_tag":
            pat = r.get("pattern")
            if not pat:
                raise ValueError(f"routing.rules[{idx}]: llm_tag needs 'pattern'")
            return _Rule(
                type=rtype,
                action=action,
                tag_pattern=re.compile(pat, re.IGNORECASE),
            )
        if rtype == "skill_intent":
            skill = r.get("skill")
            if not skill:
                raise ValueError(f"routing.rules[{idx}]: skill_intent needs 'skill'")
            return _Rule(type=rtype, action=action, skill=skill)
        raise ValueError(f"routing.rules[{idx}]: unknown type {rtype!r}")

    def known(self, persona_id: str) -> bool:
        return persona_id in self.personas

    def get(self, persona_id: str) -> PersonaDef:
        return self.personas[persona_id]

    def chatterbox_personas(self) -> list[PersonaDef]:
        return [p for p in self.personas.values() if p.backend == CHATTERBOX_BACKEND]

    def kokoro_personas(self) -> list[PersonaDef]:
        return [p for p in self.personas.values() if p.backend == KOKORO_BACKEND]

    def voice_command_rules(self) -> Iterable[_Rule]:
        return (r for r in self.rules if r.type == "voice_command")

    def llm_tag_rules(self) -> Iterable[_Rule]:
        return (r for r in self.rules if r.type == "llm_tag")

    def skill_intent_target(self, skill_name: str) -> str | None:
        for r in self.rules:
            if r.type == "skill_intent" and r.skill == skill_name:
                return r.action
        return None


def _compile_voice_command(template: str, persona_ids: set[str]) -> re.Pattern:
    """Compile a "switch to {persona}" style template into a regex.

    Whitespace between words is liberal (matches BackendRouter's tolerance
    for STT punctuation). The {persona} placeholder captures any
    declared persona id, case-insensitively.
    """
    persona_alt = "|".join(sorted((re.escape(p) for p in persona_ids), key=len, reverse=True))
    parts = []
    i = 0
    while i < len(template):
        if template.startswith("{persona}", i):
            parts.append(rf"(?P<persona>{persona_alt})")
            i += len("{persona}")
        else:
            ch = template[i]
            if ch.isspace():
                parts.append(r"\s+")
                while i < len(template) and template[i].isspace():
                    i += 1
                continue
            parts.append(re.escape(ch))
            i += 1
    return re.compile(r"\b" + "".join(parts) + r"\b", re.IGNORECASE)


class PersonaCommandRouter(FrameProcessor):
    """Reads TranscriptionFrame and applies voice_command rules.

    Place after STT/TranscriptionTap, before the context aggregator. The
    transcript still flows downstream untouched — we just update shared
    state when a command pattern matches. Last match in the utterance
    wins, mirroring BackendRouter's last-wake-phrase-wins behaviour.
    """

    def __init__(self, config: PersonaConfig, state: PersonaState):
        super().__init__()
        self._config = config
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text
        ):
            self._maybe_switch(frame.text)
        await self.push_frame(frame, direction)

    def _maybe_switch(self, text: str) -> None:
        stripped = re.sub(r"[^\w\s]", "", text)
        last_pos = -1
        last_persona: str | None = None
        for rule in self._config.voice_command_rules():
            if rule.pattern is None:
                continue
            for m in rule.pattern.finditer(stripped):
                pid = m.group("persona").lower()
                if pid == m.group("persona") or self._config.known(pid):
                    target = pid if self._config.known(pid) else None
                else:
                    target = None
                if target is None:
                    candidate = m.group("persona")
                    if self._config.known(candidate):
                        target = candidate
                if target and m.start() > last_pos:
                    last_pos = m.start()
                    last_persona = target
        if last_persona and last_persona != self._state.current:
            logger.info(
                f"Persona switched -> {last_persona} (voice_command)"
            )
            self._state.current = last_persona
            self._state.pinned = last_persona
            self._state.one_shot_active = False


class PersonaTagRouter(FrameProcessor):
    """Reads text frames heading to TTS, applies llm_tag rules, strips tags.

    Place immediately before the TTS dispatch branches. For each
    [persona:X] tag it finds, it sets current persona to X (one-shot:
    reverts to `pinned` at the next assistant turn boundary) and removes
    the tag from the outgoing text so TTS never speaks it.
    """

    _ASSISTANT_TURN_BOUNDARIES: tuple[type, ...] = ()

    def __init__(self, config: PersonaConfig, state: PersonaState):
        super().__init__()
        self._config = config
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSSpeakFrame) and frame.text:
                cleaned = self._handle_text(frame.text)
                if cleaned != frame.text:
                    frame = TTSSpeakFrame(text=cleaned)
            elif isinstance(frame, TextFrame) and frame.text:
                cleaned = self._handle_text(frame.text)
                if cleaned != frame.text:
                    # Preserve LLMTextFrame subclass so downstream aggregation
                    # still treats this as LLM-emitted text (which keeps
                    # includes_inter_frame_spaces=True). Falling back to a
                    # plain TextFrame would flip that flag to False and break
                    # word spacing in TTS output.
                    cls = type(frame) if isinstance(frame, LLMTextFrame) else TextFrame
                    frame = cls(text=cleaned)
        await self.push_frame(frame, direction)

    def _handle_text(self, text: str) -> str:
        cleaned = text
        tag_removed = False
        first_match_persona: str | None = None
        for rule in self._config.llm_tag_rules():
            if rule.tag_pattern is None:
                continue
            for m in rule.tag_pattern.finditer(cleaned):
                pid = m.group(1)
                if self._config.known(pid) and first_match_persona is None:
                    first_match_persona = pid
            new_cleaned = rule.tag_pattern.sub("", cleaned)
            if new_cleaned != cleaned:
                tag_removed = True
                cleaned = new_cleaned
        # Only normalise whitespace when we actually removed a tag — a tag
        # deletion can leave a "word  word" double space. Per-token streamed
        # frames carry meaningful leading/trailing spaces (" set", " a", ...);
        # stripping them concatenates words together in the TTS output.
        if tag_removed:
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
        if first_match_persona and first_match_persona != self._state.current:
            logger.info(
                f"Persona switched -> {first_match_persona} (llm_tag, one-shot)"
            )
            self._state.current = first_match_persona
            self._state.one_shot_active = True
        return cleaned


def load_persona_config(path: Path, default_env: str | None) -> tuple[PersonaConfig, PersonaState]:
    """Read personas.yaml and build initial state.

    `default_env` (typically `os.getenv("DEFAULT_PERSONA")`) overrides
    routing.default when set and valid.
    """
    config = PersonaConfig(path)
    boot = config.default
    if default_env:
        env_choice = default_env.strip()
        if env_choice:
            if not config.known(env_choice):
                raise ValueError(
                    f"DEFAULT_PERSONA={env_choice!r} is not a declared persona in {path}"
                )
            boot = env_choice
    state = PersonaState(current=boot, pinned=boot)
    logger.info(f"Persona boot       : {boot} (from {path.name})")
    return config, state


def apply_skill_persona_switch(
    config: PersonaConfig, state: PersonaState, persona_id: str
) -> bool:
    """Called by the switch_persona skill. Returns True on success."""
    if not config.known(persona_id):
        return False
    if persona_id != state.current:
        logger.info(f"Persona switched -> {persona_id} (skill_intent)")
    state.current = persona_id
    state.pinned = persona_id
    state.one_shot_active = False
    return True
