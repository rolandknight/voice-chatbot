from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from poc_tts.server import _voice_record, create_app


@pytest.fixture
def client(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "marvin.wav").write_bytes(b"x")

    engine = MagicMock()
    engine.loaded = True
    engine.model_info.return_value = {
        "loaded": True, "type": "flash", "class_name": "ChatterboxFlashTTS",
        "device": "cuda", "sample_rate": 24000,
        "supports_paralinguistic_tags": False, "available_paralinguistic_tags": [],
        "supports_multilingual": False, "supported_languages": {"en": "English"},
    }
    config = {
        "server": {"host": "127.0.0.1", "port": 8005},
        "generation": {
            "temperature": 0.6, "exaggeration": 0.5, "cfg_scale": 1.0,
            "num_steps": 10, "n_cfm_timesteps": 2,
        },
    }
    return TestClient(create_app(engine, config, voice_paths=[voices]))


def test_index_serves_the_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_static_assets_are_served(client):
    assert client.get("/script.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


def test_model_info_endpoint(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    assert r.json()["type"] == "flash"


def test_initial_data_has_the_keys_script_js_reads(client):
    """script.js destructures these on load; a missing key breaks the page
    silently rather than raising."""
    body = client.get("/api/ui/initial-data").json()
    for key in (
        "config", "reference_files", "predefined_voices",
        "presets", "initial_gen_result", "model_info",
    ):
        assert key in body, f"missing key: {key}"


def test_initial_data_reports_flash_model(client):
    assert client.get("/api/ui/initial-data").json()["model_info"]["type"] == "flash"


def test_reference_files_lists_discovered_voices(client):
    assert client.get("/get_reference_files").json() == ["marvin.wav"]


def test_predefined_voices_returns_ui_shaped_records(client):
    voices = client.get("/get_predefined_voices").json()
    assert voices and all("display_name" in v and "filename" in v for v in voices)


def test_restart_server_is_a_clear_noop_not_a_404(client):
    """The UI calls this; a 404 would read as a bug."""
    r = client.post("/restart_server")
    assert r.status_code == 200
    assert "not supported" in r.json()["message"].lower()


def test_predefined_voices_agree_between_endpoints(client):
    """Both endpoints derive display_name from the same filenames; a change
    applied to one and missed in the other would be a silent UI mismatch."""
    direct = client.get("/get_predefined_voices").json()
    embedded = client.get("/api/ui/initial-data").json()["predefined_voices"]
    assert direct == embedded


def test_voice_record_strips_actual_extension_not_just_wav():
    """_voice_record used to do name.replace('.wav', ''), which left '.mp3'
    in the display name -- voices/ ships mp3 reference clips."""
    assert _voice_record("some_voice.mp3")["display_name"] == "Some Voice"
