import numpy as np

from poc_tts_streaming.audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    TRIM_KEEP_MS,
    FrameSlicer,
    TrailingSilenceGate,
    silence_frame,
    to_int16,
    trim_edge_silence,
)


def test_to_int16_clips_and_scales():
    out = to_int16(np.array([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out.tolist() == [-32767, -32767, 0, 16383, 32767]


def test_slicer_emits_full_frames_and_carries_the_remainder():
    s = FrameSlicer()
    frames = s.push(np.arange(1001, dtype=np.int16))
    assert [len(f) for f in frames] == [480, 480]
    assert frames[0][0] == 0 and frames[1][0] == 480
    tail = s.flush()
    assert len(tail) == 1 and len(tail[0]) == FRAME_SAMPLES
    assert tail[0][:41].tolist() == list(range(960, 1001))
    assert not tail[0][41:].any()


def test_slicer_joins_across_pushes():
    s = FrameSlicer()
    assert s.push(np.ones(300, dtype=np.int16)) == []
    frames = s.push(np.ones(300, dtype=np.int16))
    assert len(frames) == 1 and frames[0].all()


def test_flush_on_empty_slicer_emits_nothing():
    assert FrameSlicer().flush() == []


def test_clear_drops_the_carry():
    s = FrameSlicer()
    s.push(np.ones(300, dtype=np.int16))
    s.clear()
    assert s.flush() == []


def test_silence_frame_shape():
    f = silence_frame()
    assert f.dtype == np.int16 and f.shape == (FRAME_SAMPLES,) and not f.any()


# --- chunk-edge silence -------------------------------------------------------

SR = SAMPLE_RATE
KEEP = SR * TRIM_KEEP_MS // 1000          # 2880 samples at 24 kHz / 120 ms


def _speech(n, level=0.3):
    return np.full(n, level, dtype=np.float32)


def _silence(n):
    return np.zeros(n, dtype=np.float32)


def test_trim_edge_silence_keeps_only_keep_ms_on_each_edge():
    pcm = np.concatenate([_silence(SR), _speech(SR), _silence(2 * SR)])
    out = trim_edge_silence(pcm, SR)
    assert len(out) == KEEP + SR + KEEP
    # the speech survives intact and lands where the keep window predicts
    assert np.array_equal(out[KEEP:KEEP + SR], _speech(SR))
    assert not out[:KEEP].any() and not out[KEEP + SR:].any()


def test_trim_edge_silence_leaves_a_clip_with_no_silence_alone():
    pcm = _speech(SR)
    out = trim_edge_silence(pcm, SR)
    assert np.array_equal(out, pcm)


def test_trim_edge_silence_on_an_all_silent_clip_returns_at_most_keep_ms():
    out = trim_edge_silence(_silence(10 * SR), SR)
    assert len(out) == KEEP and not out.any()


def test_trim_edge_silence_on_a_short_all_silent_clip_returns_it_whole():
    """Shorter than keep_ms: never return an empty array for non-empty input."""
    out = trim_edge_silence(_silence(100), SR)
    assert len(out) == 100


def test_trim_edge_silence_on_empty_input_returns_empty():
    assert trim_edge_silence(np.zeros(0, dtype=np.float32), SR).size == 0


def test_trim_edge_silence_treats_sub_threshold_noise_as_silence():
    """-45 dBFS is the line: a -60 dBFS noise floor is silence, not speech."""
    floor = np.full(SR, 10 ** (-60 / 20), dtype=np.float32)
    pcm = np.concatenate([floor, _speech(SR), floor])
    assert len(trim_edge_silence(pcm, SR)) == KEEP + SR + KEEP


def test_trim_edge_silence_keep_ms_zero_removes_every_edge_sample():
    pcm = np.concatenate([_silence(SR), _speech(SR), _silence(SR)])
    assert len(trim_edge_silence(pcm, SR, keep_ms=0)) == SR


# --- TrailingSilenceGate ------------------------------------------------------

def _drain_len(gate, windows):
    return sum(len(p) for p in _drain(gate, windows))


def _drain(gate, windows):
    out = [p for w in windows for p in gate.push(w)]
    out += gate.flush()
    return out


def test_gate_preserves_a_real_mid_chunk_pause():
    """speech -> silence -> speech: nothing between the outer speech edges is
    dropped, or a deliberate pause inside a sentence would vanish."""
    gate = TrailingSilenceGate(sr=SR)
    pieces = _drain(gate, [_speech(SR), _silence(SR), _speech(SR)])
    total = np.concatenate(pieces)
    assert len(total) == 3 * SR
    assert not total[SR:2 * SR].any()


def test_gate_drops_a_long_trailing_silence_at_flush():
    gate = TrailingSilenceGate(sr=SR)
    emitted = gate.push(_speech(SR))
    assert sum(len(p) for p in emitted) == SR
    assert gate.push(_silence(4 * SR)) == []          # buffered, not emitted
    assert gate.push(_silence(4 * SR)) == []
    tail = gate.flush()
    assert sum(len(p) for p in tail) == KEEP
    assert not np.concatenate(tail).any()


def test_gate_keeps_a_short_leading_silence_whole():
    """Less leading silence than keep_ms is already within budget and must be
    passed through untouched -- not re-sliced from the end."""
    gate = TrailingSilenceGate(sr=SR)
    emitted = np.concatenate(gate.push(
        np.concatenate([_silence(KEEP // 2), _speech(1000)])
    ))
    assert len(emitted) == KEEP // 2 + 1000


def test_gate_matches_trim_when_the_edges_are_shorter_than_keep_ms():
    whole = np.concatenate([_silence(500), _speech(4000), _silence(600)])
    gate = TrailingSilenceGate(sr=SR)
    windows = [whole[i:i + 700] for i in range(0, len(whole), 700)]
    assert np.array_equal(
        np.concatenate(_drain(gate, windows)), trim_edge_silence(whole, SR)
    )


def test_gate_caps_the_leading_silence():
    gate = TrailingSilenceGate(sr=SR)
    assert gate.push(_silence(2 * SR)) == []
    emitted = np.concatenate(gate.push(_speech(SR)))
    assert len(emitted) == KEEP + SR
    assert not emitted[:KEEP].any()


def test_gate_on_an_all_silent_chunk_emits_at_most_keep_ms():
    gate = TrailingSilenceGate(sr=SR)
    assert _drain_len(gate, [_silence(4 * SR)] * 3) == KEEP


def test_gate_conserves_every_speechful_sample():
    """The speech spans are emitted byte-for-byte; only edge silence moves."""
    gate = TrailingSilenceGate(sr=SR)
    a, b = _speech(1000, 0.3), _speech(1000, 0.7)
    pieces = _drain(gate, [_silence(SR), a, _silence(500), b, _silence(SR)])
    total = np.concatenate(pieces)
    assert len(total) == KEEP + 1000 + 500 + 1000 + KEEP
    assert np.array_equal(total[KEEP:KEEP + 1000], a)
    assert np.array_equal(total[KEEP + 1500:KEEP + 2500], b)


def test_gate_matches_trim_edge_silence_on_the_whole_chunk():
    """The streaming gate and the one-shot trim are the same function.

    This is the property that lets the block path be reasoned about as if it
    trimmed the finished chunk, even though it can only see one window at a
    time.
    """
    rng = np.random.default_rng(0)
    whole = np.concatenate([
        _silence(3000),
        (rng.standard_normal(9000) * 0.2).astype(np.float32),
        _silence(1500),
        (rng.standard_normal(4000) * 0.2).astype(np.float32),
        _silence(60000),
    ])
    gate = TrailingSilenceGate(sr=SR)
    windows = [whole[i:i + 5000] for i in range(0, len(whole), 5000)]
    streamed = np.concatenate(_drain(gate, windows))
    assert np.array_equal(streamed, trim_edge_silence(whole, SR))


def test_gate_is_reusable_after_flush():
    """One gate per chunk is the wiring, but flush must leave a clean slate."""
    gate = TrailingSilenceGate(sr=SR)
    _drain(gate, [_speech(SR), _silence(4 * SR)])
    assert gate.push(_silence(2 * SR)) == []
    assert len(np.concatenate(gate.push(_speech(SR)))) == KEEP + SR


def test_gate_ignores_empty_pushes():
    gate = TrailingSilenceGate(sr=SR)
    assert gate.push(np.zeros(0, dtype=np.float32)) == []
    assert gate.flush() == []
