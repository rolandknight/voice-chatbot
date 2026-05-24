"""Resources that skill handlers can access at invocation time.

Handlers declare in their SKILL.md `requires:` list the attribute names they
need (e.g. `requires: [radio_player]`); the loader skips registration if any
required attribute is missing/empty on the SkillContext instance built by app.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from persona_router import PersonaConfig, PersonaState
    from radio import RadioPlayer
    from spotify import SpotifyPlayer

    from skills._tracker import BotSpeakingTracker


@dataclass
class SkillContext:
    radio_player: Optional["RadioPlayer"] = None
    spotify_player: Optional["SpotifyPlayer"] = None
    sfx_tracker: Optional["BotSpeakingTracker"] = None
    sfx_backends: dict[str, str] = field(default_factory=dict)
    sfx_backend_override: Optional[str] = None
    persona_config: Optional["PersonaConfig"] = None
    persona_state: Optional["PersonaState"] = None

    def has(self, name: str) -> bool:
        """True when the attribute is set and non-empty.

        Used by the loader to gate skills whose `requires:` list names this
        attribute. Dict/list types are non-empty by truthiness; player objects
        only need to be non-None. If the name resolves to a method, that method
        is called and its boolean result is returned — that's how the
        persona-switch gate (multiple personas declared) is expressed in SKILL.md.
        """
        value = getattr(self, name, None)
        if value is None:
            return False
        if callable(value):
            return bool(value())
        if isinstance(value, (dict, list, tuple, set, str)) and not value:
            return False
        return True

    def persona_switch_available(self) -> bool:
        return (
            self.persona_config is not None
            and self.persona_state is not None
            and len(self.persona_config.personas) > 1
        )
