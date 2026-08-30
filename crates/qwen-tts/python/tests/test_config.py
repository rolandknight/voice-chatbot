from pathlib import Path

from qwen_tts.config import apply_env_overrides, load_config, voice_dirs


def test_default_is_the_server_profile():
    cfg = load_config(env={})
    assert cfg["_base_dir"].endswith("/config")
    assert isinstance(cfg["server"]["port"], int)
    assert cfg["models"]["clone_default"].endswith("1.7B-Base-bf16")
    assert cfg["preload"]["enabled"] is True


def test_env_override_coerces_types():
    cfg = load_config(env={"QWEN_SERVER_PORT": "8010", "QWEN_TRANSCRIBE_ENABLED": "false", "QWEN_GENERATION_TEMPERATURE": "0.5"})
    assert cfg["server"]["port"] == 8010 and isinstance(cfg["server"]["port"], int)
    assert cfg["transcribe"]["enabled"] is False
    assert cfg["generation"]["temperature"] == 0.5


def test_empty_env_value_is_ignored_and_input_not_mutated():
    base = {"server": {"port": 1}}
    out = apply_env_overrides(base, {"QWEN_SERVER_PORT": "  "})
    assert out["server"]["port"] == 1 and out is not base


def test_voice_dirs_resolve_and_drop_missing(tmp_path):
    cfg = {"_base_dir": str(tmp_path), "voices": {"paths": ["a", "missing"]}}
    (tmp_path / "a").mkdir()
    assert voice_dirs(cfg) == [(tmp_path / "a").resolve()]


def test_repo_voices_dir_exists():
    assert any(Path(d).name == "voices" for d in voice_dirs(load_config(env={})))
