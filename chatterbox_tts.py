"""Pipecat TTS service for the local Chatterbox-TTS-Server.

Chatterbox exposes the OpenAI /v1/audio/speech protocol, so we reuse
pipecat's OpenAITTSService for the HTTP client, auth, streaming, and
metrics — but two things differ:

  1. Voice names: OpenAITTSService validates the voice against OpenAI's
     canonical list (alloy, ash, ballad, ...) and rejects anything else.
     Cloned persona names like 'marvin' would never reach the server.

  2. Response format: Chatterbox-TTS-Server only accepts wav/opus/mp3,
     not OpenAI's raw 'pcm'. We request 'wav' and strip the RIFF/WAVE
     header on the fly so the rest of the pipecat audio pipeline still
     sees raw PCM samples — exactly what TTSAudioRawFrame expects.

This subclass overrides run_tts() to handle both differences while
keeping every other piece of OpenAITTSService behaviour intact.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

from loguru import logger
from openai import BadRequestError

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.utils.tracing.service_decorators import traced_tts


# Largest plausible RIFF/WAVE header we'd need to buffer before locating
# the "data" subchunk. A vanilla 16-bit PCM WAV is 44 bytes; servers
# that prepend a "LIST" or "INFO" subchunk push it higher. 4 KiB is
# overkill but cheap — Chatterbox's first audio chunk is much larger
# than that anyway, so it never delays the first frame.
_MAX_HEADER_SCAN = 4096
_CLOCK_RE = re.compile(
    r"\b(0?[1-9]|1[0-2]):([0-5]\d)\s*([AaPp])\.?\s*([Mm])\.?\b"
)
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty"]


def _strip_wav_header(buf: bytes) -> tuple[bytes, bool]:
    """Return (payload_after_header, header_complete).

    If the "data" subchunk hasn't been seen yet, returns (empty, False)
    and the caller keeps buffering. Once found, returns the audio bytes
    that follow the 8-byte 'data' + size fields, and True.
    """
    idx = buf.find(b"data")
    if idx < 0:
        return b"", False
    payload_start = idx + 8  # skip "data" + 4-byte size
    if payload_start > len(buf):
        return b"", False
    return buf[payload_start:], True


def _two_digit_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n % 10 == 0:
        return _TENS[n // 10]
    return f"{_TENS[n // 10]} {_ONES[n % 10]}"


def _normalize_clock_times(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        meridiem = "A M" if match.group(3).lower() == "a" else "P M"
        hour_words = _ONES[hour]
        if minute == 0:
            clock = f"{hour_words} o'clock"
        elif minute < 10:
            clock = f"{hour_words} oh {_ONES[minute]}"
        else:
            clock = f"{hour_words} {_two_digit_words(minute)}"
        return f"{clock} {meridiem}"

    return _CLOCK_RE.sub(repl, text)


class ChatterboxTTSService(OpenAITTSService):
    """OpenAI-compatible TTS pointed at a local Chatterbox-TTS-Server.

    Behaves identically to OpenAITTSService except the voice name is
    passed through verbatim (no whitelist), 'wav' is requested instead
    of 'pcm', and the WAV header is stripped before frames hit the
    audio pipeline.
    """

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = _normalize_clock_times(text)
        logger.debug(f"{self}: Generating TTS [{text}]")
        voice = self._settings.voice
        if not voice:
            yield ErrorFrame(error="Chatterbox TTS voice must be specified")
            return

        create_params = {
            "input": text,
            "model": self._settings.model,
            "voice": voice,
            "response_format": "wav",
        }
        if self._settings.instructions:
            create_params["instructions"] = self._settings.instructions
        if self._settings.speed:
            create_params["speed"] = self._settings.speed

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio "
                        f"(status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                # self.chunk_size is derived from sample_rate, which the
                # base service only sets when the pipeline starts. At
                # warm-up time it's still 0, and iter_bytes(0) fails
                # internally with "range() arg 3 must not be zero".
                # 4096 is small enough to keep TTFB low and large enough
                # to be efficient — used as a fallback only.
                chunk_size = self.chunk_size or 4096
                header_buf = bytearray()
                header_done = False

                async for chunk in r.iter_bytes(chunk_size):
                    if not chunk:
                        continue
                    if not header_done:
                        header_buf.extend(chunk)
                        payload, header_done = _strip_wav_header(bytes(header_buf))
                        if not header_done:
                            if len(header_buf) > _MAX_HEADER_SCAN:
                                yield ErrorFrame(
                                    error=(
                                        f"Chatterbox WAV header not found in "
                                        f"first {_MAX_HEADER_SCAN} bytes"
                                    )
                                )
                                return
                            continue
                        if payload:
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(
                                payload, self.sample_rate, 1, context_id=context_id
                            )
                        header_buf.clear()
                        continue
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        chunk, self.sample_rate, 1, context_id=context_id
                    )
        except BadRequestError as e:
            yield ErrorFrame(error=f"Chatterbox TTS request rejected: {e}")
