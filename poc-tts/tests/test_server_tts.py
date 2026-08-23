from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poc_tts.engine_flash import OutOfMemoryError
from poc_tts.server import create_app


@pytest.fixture
def engine():
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    eng.synthesize.return_value = (np.zeros(2400, dtype=np.float32), 24000)
    return eng


@pytest.fixture
def client(engine, tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "marvin.wav").write_bytes(b"x")
    config = {"server": {"port": 8005}, "generation": {}}
    return TestClient(create_app(engine, config, voice_paths=[voices]))


def test_tts_returns_wav_audio(client):
    r = client.post("/tts", json={
        "text": "Hello there.",
        "voice_mode": "predefined",
        "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_tts_forwards_the_four_flash_knobs(client, engine):
    client.post("/tts", json={
        "text": "Hello there.",
        "voice_mode": "predefined",
        "predefined_voice_id": "marvin.wav",
        "num_steps": 4,
        "n_cfm_timesteps": 1,
        "temperature": 0.7,
        "cfg_weight": 1.5,
    })
    _, kwargs = engine.synthesize.call_args
    assert kwargs["num_steps"] == 4
    assert kwargs["n_cfm_timesteps"] == 1
    assert kwargs["temperature"] == 0.7
    assert kwargs["cfg_scale"] == 1.5, "UI sends cfg_weight; Flash takes cfg_scale"


def test_tts_clone_mode_uses_reference_audio_filename(client, engine):
    client.post("/tts", json={
        "text": "Hello.",
        "voice_mode": "clone",
        "reference_audio_filename": "marvin.wav",
    })
    _, kwargs = engine.synthesize.call_args
    assert kwargs["voice"] == "marvin.wav"


def test_tts_rejects_empty_text(client):
    r = client.post("/tts", json={
        "text": "", "voice_mode": "predefined", "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 422


def test_tts_missing_voice_id_is_a_400(client):
    r = client.post("/tts", json={"text": "Hi.", "voice_mode": "predefined"})
    assert r.status_code == 400
    assert "predefined_voice_id" in r.json()["detail"]


def test_tts_unknown_voice_is_a_404_naming_paths(client, engine):
    engine.synthesize.side_effect = FileNotFoundError("reference voice 'nope.wav' not found. Searched: /tmp/voices")
    r = client.post("/tts", json={
        "text": "Hi.", "voice_mode": "predefined", "predefined_voice_id": "nope.wav",
    })
    assert r.status_code == 404
    assert "Searched" in r.json()["detail"]


def test_tts_oom_is_a_507_with_vram_detail(client, engine):
    engine.synthesize.side_effect = OutOfMemoryError(
        "ran out of VRAM during generation. VRAM 0.40 GB free of 6.14 GB total"
    )
    r = client.post("/tts", json={
        "text": "Hi.", "voice_mode": "predefined", "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 507
    assert "VRAM" in r.json()["detail"]


def test_tts_when_model_not_loaded_is_503(client, engine):
    engine.loaded = False
    r = client.post("/tts", json={
        "text": "Hi.", "voice_mode": "predefined", "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 503


def test_save_and_reset_settings_round_trip(client):
    saved = client.post("/save_settings", json={"last_text": "remembered"})
    assert saved.status_code == 200
    assert client.post("/reset_settings").status_code == 200
