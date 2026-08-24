import numpy as np

from poc_tts_streaming.bench_stream import measure


class FakeEngine:
    sr = 24000
    def synthesize_stream(self, text, voice, **kw):
        for s in ("One.", "Two."):
            yield s, np.zeros(12000, dtype=np.float32)


def test_measure_reports_ttfa_total_audio_and_chunks():
    row = measure(FakeEngine(), "One. Two.", "v.wav", {"num_steps": 4})
    assert row["n_chunks"] == 2
    assert row["first_chunk_chars"] == 4
    assert row["audio_s"] == 1.0
    assert 0 <= row["ttfa_s"] <= row["gen_s"]
    assert [c["audio_s"] for c in row["chunks"]] == [0.5, 0.5]
