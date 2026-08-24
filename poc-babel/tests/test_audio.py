"""Pure-numpy helpers in render_variants -- no model, no ffmpeg."""
import numpy as np

from render_variants import peak_normalize, trim_silence

SR = 24000


def _tone(seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_silence_strips_leading_and_trailing_silence():
    lead, tail = np.zeros(SR // 2, np.float32), np.zeros(SR, np.float32)
    x = np.concatenate([lead, _tone(1.0), tail])
    y = trim_silence(x, SR, pad_ms=0)
    # Within one 20 ms analysis frame of the true 1.0 s of tone.
    assert abs(len(y) - SR) <= SR * 0.02


def test_trim_silence_keeps_pad_on_both_sides():
    x = np.concatenate([np.zeros(SR, np.float32), _tone(1.0), np.zeros(SR, np.float32)])
    y = trim_silence(x, SR, pad_ms=50)
    pad = int(SR * 0.05)
    assert SR + 2 * pad - SR * 0.02 <= len(y) <= SR + 2 * pad + SR * 0.02


def test_trim_silence_on_all_silence_returns_input_unchanged():
    x = np.zeros(SR, np.float32)
    assert len(trim_silence(x, SR)) == SR


def test_peak_normalize_hits_target_dbfs():
    y = peak_normalize(_tone(0.5, amp=0.1), target_dbfs=-3.0)
    assert abs(np.abs(y).max() - 10 ** (-3 / 20)) < 1e-3


def test_peak_normalize_of_silence_is_silence():
    y = peak_normalize(np.zeros(100, np.float32))
    assert not y.any()
