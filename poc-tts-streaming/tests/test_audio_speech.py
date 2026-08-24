import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.server import create_app


class GatedStream:
    """Second chunk waits on a gate so the test can prove the first chunk
    was flushed to the client before the second was generated."""
    def __init__(self):
        self.gate = threading.Event()
        self.second_started = threading.Event()
    def __call__(self, text, voice, *, cancel=None, **knobs):
        yield "One.", np.full(2400, 0.25, dtype=np.float32)
        self.second_started.set()
        self.gate.wait(5)
        yield "Two.", np.full(2400, 0.25, dtype=np.float32)


@pytest.fixture
def setup(tmp_path):
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    eng.synthesize_stream = GatedStream()
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    app = create_app(eng, {"generation": {}, "realtime": {"default_voice": "one-one.mp3"}}, voice_paths=[voices])
    return app, eng, TestClient(app)


def test_pcm_is_streamed_chunk_by_chunk(setup):
    app, eng, client = setup
    stream = eng.synthesize_stream
    with client.stream("POST", "/v1/audio/speech",
                       json={"input": "One. Two.", "voice": "one-one.mp3", "response_format": "pcm"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/pcm")
        it = r.iter_bytes(4800)
        first = next(it)
        assert len(first) == 4800, "first sentence arrives on its own"
        assert stream.second_started.wait(5)
        stream.gate.set()
        rest = b"".join(it)
        assert len(rest) == 4800
    assert np.frombuffer(first, dtype=np.int16)[0] == 8191


def test_wav_returns_a_whole_file(setup):
    app, eng, client = setup
    eng.synthesize_stream.gate.set()
    r = client.post("/v1/audio/speech", json={"input": "One. Two.", "voice": "one-one.mp3",
                                              "response_format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and len(r.content) == 44 + 2 * 4800


def test_unknown_voice_and_missing_input_use_the_openai_error_shape(setup):
    _, _, client = setup
    r = client.post("/v1/audio/speech", json={"input": "x", "voice": "ghost.wav", "response_format": "pcm"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "voice"
    r = client.post("/v1/audio/speech", json={"voice": "one-one.mp3"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "input"


def test_x_chatterbox_overrides_reach_the_engine(setup):
    app, eng, client = setup
    calls = []
    def spy(text, voice, *, cancel=None, **knobs):
        calls.append(knobs)
        yield "x.", np.zeros(480, dtype=np.float32)
    eng.synthesize_stream = spy
    client.post("/v1/audio/speech", json={"input": "x.", "voice": "one-one.mp3",
                                          "response_format": "pcm", "x_chatterbox": {"num_steps": 2}})
    assert calls[0]["num_steps"] == 2
