import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from poc_tts.engine_flash import OutOfMemoryError as EngineOutOfMemoryError
from poc_tts.server import _voice_record, create_app, main


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


def test_initial_data_config_matches_ui_contract(client):
    """ui/script.js:699 reads config.audio_output.format (falling back to
    'mp3' when absent -- and config.yaml never had an audio_output section,
    so every page load silently selected mp3, which models.FlashTTSRequest
    used to reject outright with a 422 on every Generate click).
    ui/script.js:689-694 reads config.generation_defaults, with cfg_scale
    renamed to cfg_weight at script.js:694 -- the raw 'generation:' block in
    config.yaml has no such key and was dead for all GUI traffic."""
    config = client.get("/api/ui/initial-data").json()["config"]
    assert config["audio_output"]["format"] == "wav"
    assert config["generation_defaults"]["cfg_weight"] == 1.0
    assert config["generation_defaults"]["temperature"] == 0.6


def test_index_html_same_origin_assets_all_resolve(client):
    """Every src=/href= ui/index.html ships must actually be servable.
    Earlier verification only grepped served files for substrings, which
    cannot catch a missing asset -- ui/vendor/wavesurfer.min.js 404'd this
    way and broke playback (and thus the success toast) on every generation,
    silently, because the failure lands inside submitTTSRequest's try/catch."""
    html = client.get("/").text
    references = re.findall(r'(?:src|href)="([^"]+)"', html)
    checked = 0
    for ref in references:
        if ref.startswith(("http://", "https://", "//", "#", "mailto:")):
            continue
        path = ref if ref.startswith("/") else f"/{ref}"
        assert client.get(path).status_code == 200, f"{ref} -> {path} did not resolve"
        checked += 1
    assert checked >= 4, "asset-reference regex matched suspiciously few refs"


def test_main_exits_cleanly_on_load_time_oom(monkeypatch):
    """engine.synthesize() already translated OOM into the good
    _vram_report() message; engine.load() didn't, and main() called it bare
    -- so a load-time OOM gave a raw traceback and a dead server. Load time
    is the likelier failure point: the PoC is designed to run beside the
    Turbo server, which holds 4.7 GB of a 6 GB card, while Flash needs
    roughly 3 GB of weights."""
    fake_config = {
        "server": {"host": "127.0.0.1", "port": 8005},
        "engine": {}, "generation": {},
    }
    monkeypatch.setattr("poc_tts.server.load_config", lambda: fake_config)
    monkeypatch.setattr("poc_tts.server.configured_voice_paths", lambda cfg: [])
    with patch("poc_tts.engine_flash.FlashEngine") as engine_cls, patch("uvicorn.run") as run:
        engine_cls.return_value.load.side_effect = EngineOutOfMemoryError(
            "ran out of VRAM loading the model. VRAM 0.10 GB free of 6.14 GB total"
        )
        with pytest.raises(SystemExit):
            main()
        run.assert_not_called()
