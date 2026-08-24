import socket
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import uvicorn
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


@contextmanager
def live_server(app):
    """uvicorn on an ephemeral loopback port, in a daemon thread. Only a real
    HTTP server proves chunked delivery: Starlette's TestClient and httpx's
    ASGITransport both buffer the whole body before returning it."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    # ws="none": the app never opens a WebSocket route, and uvicorn's "auto"
    # protocol probing imports websockets.legacy at Config.load() time --
    # which raises a DeprecationWarning this app has no reason to trigger.
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="none")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.025)
    else:
        raise RuntimeError("uvicorn did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)


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
    app, eng, _ = setup
    stream = eng.synthesize_stream
    with live_server(app) as base:
        started = time.monotonic()
        with httpx.stream("POST", f"{base}/v1/audio/speech",
                          json={"input": "One. Two.", "voice": "one-one.mp3", "response_format": "pcm"},
                          timeout=10) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("audio/pcm")
            raw = r.iter_raw()
            first = b""
            while len(first) < 4800:
                first += next(raw)
            elapsed = time.monotonic() - started
            assert elapsed < 2.0, f"first chunk arrived after {elapsed:.2f}s -- delivery is not incremental"
            assert stream.second_started.wait(5)
            assert not stream.gate.is_set()
            stream.gate.set()
            rest = first[4800:] + b"".join(raw)
            first = first[:4800]
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
