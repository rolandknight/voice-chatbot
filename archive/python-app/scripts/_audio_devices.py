"""Shared helpers for picking PyAudio devices by name or index."""
from __future__ import annotations

from typing import Optional

import pyaudio


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


def resolve_from_config(audio) -> tuple[int, int, str, str]:
    """Resolve input/output device indexes from a typed AudioConfig.

    `audio` must expose input_device_name / input_device_index /
    output_device_name / output_device_index attributes (the AudioConfig
    pydantic model from `config.schema`).

    A fresh PyAudio instance is created (and terminated) on every call, so
    repeated calls re-enumerate devices — that's what makes the server's
    hot-attach watcher pick up a Jabra plugged in after startup.
    """
    pa = pyaudio.PyAudio()
    try:
        in_idx = find_device(
            pa,
            name_substr=audio.input_device_name,
            index=audio.input_device_index,
            direction="input",
        )
        out_idx = find_device(
            pa,
            name_substr=audio.output_device_name,
            index=audio.output_device_index,
            direction="output",
        )
        in_info = pa.get_device_info_by_index(in_idx)
        out_info = pa.get_device_info_by_index(out_idx)
        return in_idx, out_idx, str(in_info.get("name", "")), str(out_info.get("name", ""))
    finally:
        pa.terminate()


def try_resolve_from_config(audio) -> Optional[tuple[int, int, str, str]]:
    """Like `resolve_from_config` but returns None instead of raising when the
    requested device can't be found (or PortAudio itself fails to init).

    Used by the server's optional local-audio watcher, which must tolerate the
    Jabra being absent at boot and hot-plugged later — so a missing device is a
    "scan again in a moment" condition, not a fatal startup error.
    """
    try:
        return resolve_from_config(audio)
    except Exception:
        return None
