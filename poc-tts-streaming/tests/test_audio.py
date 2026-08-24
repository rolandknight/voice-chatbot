import numpy as np

from poc_tts_streaming.audio import FRAME_SAMPLES, FrameSlicer, silence_frame, to_int16


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
