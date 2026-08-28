from pathlib import Path

import numpy as np

from poc_qwen.app import Handlers, build_demo, status_line
from poc_qwen.config import load_config, voice_dirs
from poc_qwen.engine import discover_voices

REF = np.zeros(24000, dtype=np.float32)


def make_handlers(engine, tmp_path):
    (tmp_path / "bob.wav").write_bytes(b"")
    (tmp_path / "bob.txt").write_text("bob says hi")
    return Handlers(engine, discover_voices([tmp_path]))


def test_clone_handler_returns_audio_and_status(engine, tmp_path):
    h = make_handlers(engine, tmp_path)
    (sr, audio), status = h.voice_clone(REF, "transcript", False, "Hello.", "English", "0.6B")
    assert sr == 24000 and isinstance(audio, np.ndarray) and len(audio) > 0
    assert "Voice Clone" in status and "RTF" in status


def test_clone_without_reference_is_a_status_not_crash(engine, tmp_path):
    h = make_handlers(engine, tmp_path)
    audio, status = h.voice_clone(None, "", False, "Hello.", "Auto", "1.7B")
    assert audio is None and status.startswith("❌")


def test_engine_errors_are_reported(engine, tmp_path):
    h = make_handlers(engine, tmp_path)
    audio, status = h.voice_clone(REF, "", False, "Hello.", "Auto", "1.7B")  # ref text required
    assert audio is None and "failed" in status


def test_design_and_custom_voice(engine, tmp_path):
    h = make_handlers(engine, tmp_path)
    (sr, a), s = h.voice_design("Hi.", "Auto", "A deep voice")
    assert "Voice Design" in s
    (sr, a), s = h.custom_voice("Hi.", "English", "Ryan", "", "1.7B")
    assert "TTS" in s


def test_pick_preset_uses_sidecar(engine, tmp_path):
    h = make_handlers(engine, tmp_path)
    path, text = h.pick_preset("bob")
    assert path.endswith("bob.wav") and text == "bob says hi"
    assert h.pick_preset("") == (None, "")


def test_repo_presets_listed(engine):
    voices = discover_voices(voice_dirs(load_config(env={})))
    assert {"one-one", "babel", "marvin"} <= set(voices)


def test_build_demo_constructs(engine, tmp_path):
    demo = build_demo(make_handlers(engine, tmp_path))
    assert demo is not None


def test_status_line_format():
    s = status_line("X", {"model": "a/b", "chars": 3, "audio_s": 1.0, "gen_s": 0.5, "rtf": 0.5, "chunks": 1})
    assert "`b`" in s and "0.50 s" in s
