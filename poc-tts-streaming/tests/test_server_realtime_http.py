from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.server import ClientSecretStore, create_app


@pytest.fixture
def engine():
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    return eng


@pytest.fixture
def client(engine, tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    config = {"server": {"port": 8006},
              "generation": {"num_steps": 4, "n_cfm_timesteps": 1},
              "realtime": {"model": "chatterbox-flash", "default_voice": "one-one.mp3",
                           "client_secret_ttl_s": 600}}
    return TestClient(create_app(engine, config, voice_paths=[voices]))


def test_client_secret_shape(client):
    r = client.post("/v1/realtime/client_secrets", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["value"].startswith("ek_")
    assert isinstance(body["expires_at"], int)
    assert body["session"]["type"] == "realtime"
    assert body["session"]["audio"]["output"]["voice"] == "one-one.mp3"
    assert body["session"]["x_chatterbox"]["num_steps"] == 4


def test_client_secret_applies_the_session_patch(client):
    r = client.post("/v1/realtime/client_secrets",
                    json={"session": {"x_chatterbox": {"num_steps": 2}}})
    assert r.json()["session"]["x_chatterbox"]["num_steps"] == 2


def test_client_secret_rejects_a_bad_patch_with_the_openai_error_shape(client):
    r = client.post("/v1/realtime/client_secrets",
                    json={"session": {"audio": {"output": {"voice": "ghost.wav"}}}})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_value"
    assert err["param"] == "session.audio.output.voice"


def test_rejected_patch_does_not_mint_a_token(client):
    store = client.app.state.secrets
    before = len(store._tokens)
    r = client.post("/v1/realtime/client_secrets",
                    json={"session": {"audio": {"output": {"voice": "ghost.wav"}}}})
    assert r.status_code == 400
    assert len(store._tokens) == before


def test_store_expires_tokens():
    t = [1000]
    store = ClientSecretStore(ttl_s=10, clock=lambda: t[0])
    tok = store.issue(None)["value"]
    assert store.verify(tok)
    t[0] = 1011
    assert not store.verify(tok)
    assert not store.verify("ek_nope") and not store.verify(None)


def test_tts_endpoint_is_gone(client):
    assert client.post("/tts", json={"text": "x"}).status_code == 404


def test_initial_data_still_serves_the_ui_shape(client):
    body = client.get("/api/ui/initial-data").json()
    for key in ("config", "reference_files", "predefined_voices", "presets", "initial_gen_result", "model_info"):
        assert key in body
    assert body["config"]["realtime"]["model"] == "chatterbox-flash"
