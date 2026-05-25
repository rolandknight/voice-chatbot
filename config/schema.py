"""Typed config schema.

All non-secret config comes from `config.yaml` at the project root. A small,
explicit allowlist of env vars is mapped into the tree for secrets — see
`_SECRET_ENV_MAP` below. The mapping is explicit (rather than via
`env_nested_delimiter`) so the env namespace stays small and predictable.

Precedence (first wins): explicit kwargs → secrets-from-env → YAML → defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AudioConfig(BaseModel):
    in_sample_rate: int = 16000
    out_sample_rate: int = 24000
    input_device_name: Optional[str] = None
    input_device_index: Optional[int] = None
    output_device_name: Optional[str] = None
    output_device_index: Optional[int] = None


class STTConfig(BaseModel):
    whisper_model: str = "mlx-community/whisper-base.en-mlx"
    language: str = "en"
    ttfs_p99_latency_secs: float = 0.5


class LLMConfig(BaseModel):
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:26b"
    ollama_keep_alive: str = "-1"


class WakewordModelConfig(BaseModel):
    """One openwakeword model + the persona to activate on a hit.

    `model` is either a filesystem path (relative to project root) to an
    .onnx file or a bundled openwakeword model key like 'hey_jarvis_v0.1'.
    `persona` must match a declared persona id in personas.yaml.
    """
    model: str
    persona: str


class WakeConfig(BaseModel):
    # Audio-based wake detection via openwakeword. Each entry is one model
    # with its associated persona. The detector loads them all into a single
    # openwakeword.Model instance — the shared melspec + embedding backbone
    # makes per-model overhead negligible.
    models: list[WakewordModelConfig] = Field(
        default_factory=lambda: [
            WakewordModelConfig(
                model="models/wakeword/hey_babel.onnx", persona="babel"
            ),
            WakewordModelConfig(model="hey_jarvis_v0.1", persona="marvin"),
        ]
    )
    # Per-chunk probability threshold (0..1). Raise if a model mis-fires on
    # similar-sounding speech; lower if real wake attempts get missed.
    threshold: float = 0.5
    # Minimum gap between consecutive fires of the same model. Prevents one
    # spoken wake word from triggering on every 80 ms chunk.
    cooldown_secs: float = 1.5
    # Note: silence-to-sleep is driven by conversation.idle_timeout_secs so
    # the wake-IDLE transition and the LLM-context reset land on one clock.
    vad_min_volume: float = 0.0
    vad_stop_secs: float = 0.2


class ConversationConfig(BaseModel):
    idle_timeout_secs: float = 10.0


class ClaudeConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    web_search_enabled: bool = True
    web_search_max_uses: int = 3
    web_fetch_enabled: bool = True
    web_fetch_max_uses: int = 2
    api_key: SecretStr = Field(default=SecretStr(""))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.get_secret_value().strip())


class ChatterboxConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8004/v1"
    model: str = "chatterbox-turbo"
    api_key: SecretStr = Field(default=SecretStr("not-needed"))


class TTSConfig(BaseModel):
    personas_config: Path = Path("personas.yaml")
    default_persona: Optional[str] = None
    chatterbox: ChatterboxConfig = Field(default_factory=ChatterboxConfig)


class RadioSkillConfig(BaseModel):
    enabled: bool = True


class ShowsSkillConfig(BaseModel):
    enabled: bool = True


class SpotifySkillConfig(BaseModel):
    enabled: bool = True
    redirect_uri: str = "http://127.0.0.1:8765/callback"
    client_id: SecretStr = Field(default=SecretStr(""))
    client_secret: SecretStr = Field(default=SecretStr(""))


class SFXConfig(BaseModel):
    woosh_enabled: bool = False
    woosh_url: str = "http://127.0.0.1:8005"
    woosh_port: int = 8005
    sao_enabled: bool = False
    sao_url: str = "http://127.0.0.1:8006"
    sao_port: int = 8006
    backend: Literal["auto", "woosh", "stable_audio"] = "auto"


class WebSearchConfig(BaseModel):
    provider: Literal["duckduckgo", "brave", "tavily"] = "duckduckgo"
    brave_api_key: SecretStr = Field(default=SecretStr(""))
    tavily_api_key: SecretStr = Field(default=SecretStr(""))


class WeatherConfig(BaseModel):
    default_location: str = ""


class SkillsConfig(BaseModel):
    enabled: bool = True
    filter_k: int = 15
    filter_debug: bool = False
    radio: RadioSkillConfig = Field(default_factory=RadioSkillConfig)
    shows: ShowsSkillConfig = Field(default_factory=ShowsSkillConfig)
    spotify: SpotifySkillConfig = Field(default_factory=SpotifySkillConfig)
    sfx: SFXConfig = Field(default_factory=SFXConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)


class HFConfig(BaseModel):
    hub_offline: bool = False
    token: SecretStr = Field(default=SecretStr(""))


_SECRET_ENV_MAP: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": ("claude", "api_key"),
    "CHATTERBOX_API_KEY": ("tts", "chatterbox", "api_key"),
    "SPOTIPY_CLIENT_ID": ("skills", "spotify", "client_id"),
    "SPOTIPY_CLIENT_SECRET": ("skills", "spotify", "client_secret"),
    "BRAVE_API_KEY": ("skills", "web_search", "brave_api_key"),
    "TAVILY_API_KEY": ("skills", "web_search", "tavily_api_key"),
    "HF_TOKEN": ("huggingface", "token"),
}


class _SecretsEnvSource(PydanticBaseSettingsSource):
    """Maps the secret env-var allowlist into nested Config paths."""

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for env_name, path in _SECRET_ENV_MAP.items():
            raw = os.environ.get(env_name)
            if raw is None or raw.strip() == "":
                continue
            node = out
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = raw
        return out


class Config(BaseSettings):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    huggingface: HFConfig = Field(default_factory=HFConfig)

    model_config = SettingsConfigDict(extra="forbid")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        yaml_path = os.environ.get(
            "BABEL_CONFIG", str(PROJECT_ROOT / "config.yaml")
        )
        return (
            init_settings,
            _SecretsEnvSource(settings_cls),
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),
        )

    def resolved_personas_path(self) -> Path:
        """tts.personas_config resolved against the project root if relative."""
        p = self.tts.personas_config
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
