import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.engine_flash import OutOfMemoryError as EngineOutOfMemoryError
from poc_tts_streaming.server import _voice_record, create_app, main


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
    monkeypatch.setattr("poc_tts_streaming.server.load_config", lambda: fake_config)
    monkeypatch.setattr("poc_tts_streaming.server.configured_voice_paths", lambda cfg: [])
    with patch("poc_tts_streaming.engine_flash.FlashEngine") as engine_cls, patch("uvicorn.run") as run:
        engine_cls.return_value.load.side_effect = EngineOutOfMemoryError(
            "ran out of VRAM loading the model. VRAM 0.10 GB free of 6.14 GB total"
        )
        with pytest.raises(SystemExit):
            main()
        run.assert_not_called()


def test_initial_data_round_trips_saved_ui_state(client):
    """Saved UI state must come back as config.ui_state.

    script.js:621 reads config.ui_state and restores last_text from it. When
    /save_settings was write-only, every load saw an empty textarea, took the
    "no text" branch at script.js:711 and re-applied the default preset -- so
    a preset could never be cleared. This is that round-trip.
    """
    before = client.get("/api/ui/initial-data").json()["config"]
    assert before["ui_state"] == {}

    # The real client posts {"ui_state": {...}} (ui/script.js:278). Posting a
    # flat dict here would pass against a server that double-nests -- which is
    # exactly the bug this test failed to catch the first time it was written.
    saved = client.post(
        "/save_settings",
        json={"ui_state": {
            "last_text": "cleared by the user",
            "last_preset_name": "Standard Narration",
        }},
    )
    assert saved.status_code == 200

    after = client.get("/api/ui/initial-data").json()["config"]
    assert after["ui_state"]["last_text"] == "cleared by the user"
    assert after["ui_state"]["last_preset_name"] == "Standard Narration"


def test_reset_settings_clears_round_tripped_ui_state(client):
    client.post("/save_settings", json={"ui_state": {"last_text": "something"}})
    assert client.post("/reset_settings").status_code == 200
    assert client.get("/api/ui/initial-data").json()["config"]["ui_state"] == {}


def test_standard_narration_preset_exists_and_is_neutral(client):
    """script.js:714 prefers a preset named exactly 'Standard Narration' over
    presets[0]. Without it the load-time default fell through to the Turbo
    tech-support piece, whose paralinguistic tags Flash does not support."""
    presets = client.get("/api/ui/initial-data").json()["presets"]
    match = [p for p in presets if p["name"] == "Standard Narration"]
    assert match, "the neutral default preset script.js looks for is missing"
    assert "[sigh]" not in match[0]["text"] and "[laugh]" not in match[0]["text"]


def test_tag_using_presets_are_identifiable_by_their_text(client):
    """The GUI hides presets whose text uses paralinguistic tags when the model
    reports supports_paralinguistic_tags=false (Flash does).

    ui/script.js filters on preset TEXT, not on the name: every tag-using preset
    is named "⚡ Turbo: ..." and the leading emoji made the original
    startsWith('turbo') check never match. This pins the two to agree, so a
    future Turbo-named preset without tags -- or a tag-using preset named
    something else -- cannot silently slip past the filter.
    """
    import re

    tag = re.compile(r"\[(laugh|chuckle|sigh|gasp|cough|clear throat|sniff|groan|shush)\]", re.I)
    presets = client.get("/api/ui/initial-data").json()["presets"]
    assert presets, "no presets served"

    named_turbo = {p["name"] for p in presets if "turbo:" in p["name"].lower()}
    uses_tags = {p["name"] for p in presets if tag.search(p["text"])}
    assert named_turbo == uses_tags, (
        f"naming and tag usage disagree; named-only={named_turbo - uses_tags}, "
        f"tagged-only={uses_tags - named_turbo}"
    )
    assert "Standard Narration" not in uses_tags, "the neutral default must survive the filter"


def test_flash_reports_no_paralinguistic_support(client):
    """The filter above is keyed on this flag, so it must actually be false."""
    assert client.get("/api/model-info").json()["supports_paralinguistic_tags"] is False
