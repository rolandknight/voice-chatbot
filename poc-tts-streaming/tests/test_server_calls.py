from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.server import create_app


@pytest.fixture
def app(tmp_path):
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    return create_app(eng, {"generation": {}, "realtime": {"default_voice": "one-one.mp3"}},
                      voice_paths=[voices])


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def fake_calls(app):
    app.state.calls = MagicMock()
    app.state.calls.create = AsyncMock(return_value=("call_abc", "v=0\r\nanswer"))
    app.state.calls.hangup = AsyncMock(return_value=True)
    return app.state.calls


def _token(client):
    return client.post("/v1/realtime/client_secrets", json={}).json()["value"]


def test_calls_requires_a_valid_bearer(client, fake_calls):
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp"})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_request_error"
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp", "authorization": "Bearer ek_bogus"})
    assert r.status_code == 401


def test_calls_accepts_application_sdp(client, fake_calls):
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp", "authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 201
    assert r.headers["content-type"].startswith("application/sdp")
    assert r.headers["location"] == "/v1/realtime/calls/call_abc"
    assert r.text == "v=0\r\nanswer"
    args, kwargs = fake_calls.create.call_args
    assert args[0] == "v=0\r\noffer" and kwargs["session_patch"] is None


def test_calls_accepts_multipart_with_session(client, fake_calls):
    r = client.post("/v1/realtime/calls",
                    files={"sdp": (None, "v=0\r\noffer"),
                           "session": (None, '{"x_chatterbox": {"num_steps": 2}}')},
                    headers={"authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 201
    _, kwargs = fake_calls.create.call_args
    assert kwargs["session_patch"] == {"x_chatterbox": {"num_steps": 2}}


def test_calls_rejects_other_content_types(client, fake_calls):
    r = client.post("/v1/realtime/calls", json={"sdp": "x"},
                    headers={"authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 415


def test_calls_503_when_model_not_loaded(app, client, fake_calls):
    app.state.engine.loaded = False
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp", "authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 503


def test_hangup(client, fake_calls):
    assert client.delete("/v1/realtime/calls/call_abc").status_code == 200
    fake_calls.hangup.return_value = False
    assert client.delete("/v1/realtime/calls/call_zzz").status_code == 404


def test_multipart_with_a_bad_session_patch_is_a_400_and_creates_no_call(app, client):
    r = client.post("/v1/realtime/calls",
                    files={"sdp": (None, "v=0\r\noffer"),
                           "session": (None, '{"audio": {"output": {"voice": "ghost.wav"}}}')},
                    headers={"authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_value" and err["param"] == "session.audio.output.voice"
    assert len(app.state.calls) == 0
