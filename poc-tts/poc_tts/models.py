"""Request models for poc-tts.

Field names mirror the vendored CustomTTSRequest so the copied ui/script.js
payload works unchanged, plus the four Flash-specific knobs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FlashTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice_mode: Literal["predefined", "clone"] = "predefined"
    predefined_voice_id: Optional[str] = None
    reference_audio_filename: Optional[str] = None
    output_format: Literal["wav"] = Field(
        "wav", description="PoC serves WAV only; opus/mp3 are out of scope."
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
