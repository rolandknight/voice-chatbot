"""Request models for poc-tts.

Field names mirror the vendored CustomTTSRequest so the copied ui/script.js
payload works unchanged, plus the four Flash-specific knobs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FlashTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice_mode: Literal["predefined", "clone"] = "predefined"
    predefined_voice_id: Optional[str] = None
    reference_audio_filename: Optional[str] = None
    output_format: Literal["wav", "mp3", "opus"] = Field(
        "wav",
        description=(
            "The PoC only ever encodes and returns WAV. 'mp3' and 'opus' are "
            "accepted -- ui/script.js always sends a value, and its default "
            "select option is 'mp3' -- but coerced to 'wav' rather than "
            "rejected with a 422."
        ),
    )
    split_text: bool = True
    chunk_size: int = Field(120, ge=50, le=500)

    # Shared with the Turbo UI.
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    exaggeration: Optional[float] = Field(None, ge=0.0, le=2.0)
    cfg_weight: Optional[float] = Field(
        None, ge=0.0, le=5.0, description="Maps to Flash's cfg_scale."
    )

    # Flash-specific speed/quality knobs.
    num_steps: Optional[int] = Field(None, ge=1, le=32)
    n_cfm_timesteps: Optional[int] = Field(None, ge=1, le=8)

    @field_validator("output_format")
    @classmethod
    def _coerce_to_wav(cls, value: str) -> str:
        """The PoC has no encoder for mp3/opus; accept the request but make
        sure the stored value can never mislead anything downstream that
        might start trusting it."""
        return "wav"
