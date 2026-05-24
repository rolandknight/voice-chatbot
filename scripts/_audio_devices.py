"""Shared helpers for picking PyAudio devices by name or index."""
from __future__ import annotations

import os
from typing import Optional

import pyaudio


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None


def find_device(pa: pyaudio.PyAudio, *, name_substr: Optional[str], index: Optional[int], direction: str) -> int:
    """Return a PyAudio device index for the requested direction ("input" or "output").

    Priority:
      1. name_substr (case-insensitive substring match on device name)
      2. index (validated for the requested direction)
      3. PyAudio default for the direction
    """
    want_in = direction == "input"
    channels_key = "maxInputChannels" if want_in else "maxOutputChannels"

    if name_substr:
        needle = name_substr.lower()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get(channels_key, 0)) <= 0:
                continue
            if needle in str(info.get("name", "")).lower():
                return i
        raise RuntimeError(
            f"No {direction} device found matching name substring '{name_substr}'. "
            f"Run ./run.sh --devices to see what's available."
        )

    if index is not None:
        info = pa.get_device_info_by_index(index)
        if int(info.get(channels_key, 0)) <= 0:
            raise RuntimeError(
                f"Device index {index} ({info.get('name')}) has no {direction} channels. "
                f"Run ./run.sh --devices to see what's available."
            )
        return index

    if want_in:
        return int(pa.get_default_input_device_info()["index"])
    return int(pa.get_default_output_device_info()["index"])


def resolve_from_env() -> tuple[int, int, str, str]:
    """Resolve input/output device indexes using env vars. Returns (in_idx, out_idx, in_name, out_name)."""
    pa = pyaudio.PyAudio()
    try:
        in_name = _env("INPUT_DEVICE_NAME")
        out_name = _env("OUTPUT_DEVICE_NAME")
        in_idx_env = _env("INPUT_DEVICE_INDEX")
        out_idx_env = _env("OUTPUT_DEVICE_INDEX")
        in_idx = find_device(
            pa,
            name_substr=in_name,
            index=int(in_idx_env) if in_idx_env else None,
            direction="input",
        )
        out_idx = find_device(
            pa,
            name_substr=out_name,
            index=int(out_idx_env) if out_idx_env else None,
            direction="output",
        )
        in_info = pa.get_device_info_by_index(in_idx)
        out_info = pa.get_device_info_by_index(out_idx)
        return in_idx, out_idx, str(in_info.get("name", "")), str(out_info.get("name", ""))
    finally:
        pa.terminate()
