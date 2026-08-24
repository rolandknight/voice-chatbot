"""PCM helpers shared by the WebRTC track and the chunked-PCM endpoint.

Everything here is numpy-only: no aiortc, no torch.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16, clipped. Same scaling as poc-tts's WAV path."""
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


class FrameSlicer:
    """Re-frame arbitrary-length int16 PCM into exact FRAME_SAMPLES frames.

    aiortc stamps every RTP packet cut from one AudioFrame with the same
    timestamp, so frames larger than 20 ms lose all but one packet
    (docs/web-rtc.md). Exact 480-sample frames are therefore a hard rule,
    and this is the one place that rule is enforced.
    """

    def __init__(self) -> None:
        self._carry = np.zeros(0, dtype=np.int16)

    def push(self, pcm_int16: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._carry, np.asarray(pcm_int16, dtype=np.int16)])
        n_full = len(buf) // FRAME_SAMPLES
        frames = [buf[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n_full)]
        self._carry = buf[n_full * FRAME_SAMPLES:]
        return frames

    def flush(self) -> list[np.ndarray]:
        if len(self._carry) == 0:
            return []
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        frame[:len(self._carry)] = self._carry
        self._carry = np.zeros(0, dtype=np.int16)
        return [frame]

    def clear(self) -> None:
        self._carry = np.zeros(0, dtype=np.int16)


# --- chunk-edge silence --------------------------------------------------------

# Every generate() draw is an independent utterance: it opens with a beat of
# silence and closes with one, and those two stack into a double pause at every
# chunk join. Worse, roughly 1-2 % of draws never emit the stop token and run
# the whole _speech_len_for_text_tokens budget -- a 300-token / 12 s floor for
# short text -- whose excess decodes to digital silence (RMS 0.000; measured on
# both engines, see results-rtx-2060.md). Trimming each chunk's edges is the fix
# for both, and the two constants below are the whole policy:
#
#   -45 dBFS (0.0056 in float32 [-1, 1]) is comfortably above this vocoder's
#   room tone -- the runaway tail is exact zeros and inter-word silence sits
#   near -60 dBFS -- and comfortably below any voiced sample, so nothing
#   audible is ever cut. A tighter line (-60) would start keeping the noise
#   floor; a looser one (-30) would start clipping quiet consonant tails.
#
#   120 ms is one natural breath. Keeping it means a trimmed chunk still
#   *sounds* like a sentence rather than a splice, while two joined chunks
#   contribute 240 ms instead of the second or more they used to.
TRIM_THRESHOLD_DB = -45.0
TRIM_KEEP_MS = 120


def _linear_threshold(threshold_db: float) -> float:
    """dBFS -> the |amplitude| a float32 [-1, 1] sample must exceed."""
    return float(10.0 ** (threshold_db / 20.0))


def _keep_samples(sr: int, keep_ms: int) -> int:
    return max(0, int(round(sr * keep_ms / 1000.0)))


def trim_edge_silence(
    pcm: np.ndarray,
    sr: int,
    *,
    threshold_db: float = TRIM_THRESHOLD_DB,
    keep_ms: int = TRIM_KEEP_MS,
) -> np.ndarray:
    """Cut leading and trailing silence back to at most ``keep_ms`` each.

    Interior silence is untouched -- a pause the model put *inside* a sentence
    is prosody, not padding. An entirely silent clip yields at most ``keep_ms``
    of itself, so a chunk that produced only silence still contributes a beat
    rather than disappearing; only an empty input gives an empty result.
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0:
        return pcm
    keep = _keep_samples(sr, keep_ms)
    loud = np.flatnonzero(np.abs(pcm) > _linear_threshold(threshold_db))
    if loud.size == 0:
        return pcm[:keep]
    start = max(0, int(loud[0]) - keep)
    end = min(pcm.size, int(loud[-1]) + 1 + keep)
    return pcm[start:end]


class TrailingSilenceGate:
    """Streaming ``trim_edge_silence`` for a chunk delivered piece by piece.

    The block-streaming engine emits a chunk as several already-vocoded,
    cross-faded windows, and a runaway silent tail spans many of them -- by the
    time the chunk ends, most of that silence has already been handed to the
    caller, so trimming the last window fixes nothing. This gate sits on the
    emission path instead and holds silence back until it knows whether speech
    follows it:

    * a piece's speech span (first to last sample over the threshold) is
      emitted immediately, so nothing is delayed that a listener would notice;
    * silence after it is buffered, and released in full the moment more speech
      arrives -- that is how a real mid-chunk pause survives;
    * silence still buffered at :meth:`flush` is the chunk's trailing silence
      and is cut to ``keep_ms``;
    * silence before the chunk's first speech is likewise cut to ``keep_ms``.

    The result is exact: concatenating everything this gate emits for a chunk
    gives byte-for-byte what ``trim_edge_silence`` would have returned for the
    whole chunk (``tests/test_audio.py::
    test_gate_matches_trim_edge_silence_on_the_whole_chunk``). It never touches
    sample values, so cross-fade continuity between the windows it *does* emit
    is unaffected -- it only decides what is emitted, and when.
    """

    def __init__(
        self,
        *,
        sr: int = SAMPLE_RATE,
        threshold_db: float = TRIM_THRESHOLD_DB,
        keep_ms: int = TRIM_KEEP_MS,
    ) -> None:
        self._threshold = _linear_threshold(threshold_db)
        self._keep = _keep_samples(sr, keep_ms)
        self._pending: list[np.ndarray] = []
        self._seen_speech = False

    def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Feed one window; return the pieces that may be emitted now."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size == 0:
            return []
        loud = np.flatnonzero(np.abs(pcm) > self._threshold)
        if loud.size == 0:
            self._pending.append(pcm)
            return []
        first, last = int(loud[0]), int(loud[-1])
        silence = np.concatenate([*self._pending, pcm[:first]])
        self._pending = [pcm[last + 1:]]
        if not self._seen_speech:
            self._seen_speech = True
            # max(0, ...): a negative start would wrap round and keep the *last*
            # keep-len samples of a shorter run, cutting silence that was
            # already within budget.
            silence = silence[max(0, len(silence) - self._keep):]
        speech = pcm[first:last + 1]
        return [np.concatenate([silence, speech]) if silence.size else speech]

    def flush(self) -> list[np.ndarray]:
        """End the chunk: release at most ``keep_ms`` of its trailing silence."""
        pending, self._pending = self._pending, []
        self._seen_speech = False
        if not pending:
            return []
        tail = np.concatenate(pending)[:self._keep]
        return [tail] if tail.size else []
