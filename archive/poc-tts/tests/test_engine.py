from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from poc_tts.config import load_config
from poc_tts.engine_flash import FlashEngine, OutOfMemoryError


def test_load_config_reads_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("server:\n  port: 8005\nengine:\n  device: auto\n")
    loaded = load_config(cfg)
    assert loaded["server"]["port"] == 8005
    assert loaded["engine"]["device"] == "auto"


def _engine(tmp_path, **engine_overrides):
    engine_cfg = {
        "device": "cpu",
        "dtype": "auto",
        "backend": "auto",
        "drf_block_size": 16,
    }
    engine_cfg.update(engine_overrides)
    return FlashEngine(
        engine_cfg=engine_cfg,
        generation_cfg={
            "temperature": 0.6,
            "exaggeration": 0.5,
            "cfg_scale": 1.0,
            "num_steps": 10,
            "n_cfm_timesteps": 2,
        },
        voice_paths=[tmp_path],
    )


def test_model_info_before_load_reports_not_loaded(tmp_path):
    info = _engine(tmp_path).model_info()
    assert info["loaded"] is False
    assert info["type"] == "flash"


def test_model_info_has_every_key_the_ui_reads(tmp_path):
    """ui/script.js updateModelUI reads these directly; a missing key is a
    silent UI break, not an exception."""
    info = _engine(tmp_path).model_info()
    for key in (
        "loaded", "type", "class_name", "device", "sample_rate",
        "supports_paralinguistic_tags", "available_paralinguistic_tags",
        "supports_multilingual", "supported_languages",
    ):
        assert key in info, f"missing UI key: {key}"


def test_model_info_type_is_flash_so_ui_stays_english_only(tmp_path):
    info = _engine(tmp_path).model_info()
    assert info["type"] == "flash"
    assert info["supports_multilingual"] is False
    assert info["supported_languages"] == {"en": "English"}


def test_synthesize_before_load_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not loaded"):
        _engine(tmp_path).synthesize(text="hi", voice="a.wav")


def test_load_passes_resolved_dtype_and_block_size(tmp_path):
    eng = _engine(tmp_path, drf_block_size=32)
    fake_model = MagicMock()
    fake_model.sr = 24000
    with patch("poc_tts.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
    _, kwargs = cls.from_pretrained.call_args
    assert kwargs["device"] == "cpu"
    assert kwargs["dtype"] is torch.float32
    assert kwargs["drf_block_size"] == 32
    assert eng.loaded is True


def test_synthesize_forwards_generation_params(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    eng = _engine(tmp_path)
    fake_model = MagicMock()
    fake_model.sr = 24000
    fake_model.generate.return_value = torch.zeros(1, 2400)
    with patch("poc_tts.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
        audio, sr = eng.synthesize(text="hi", voice="a.wav", num_steps=4)
    _, kwargs = fake_model.generate.call_args
    assert kwargs["num_steps"] == 4
    assert kwargs["backend"] == "torch"
    assert kwargs["n_cfm_timesteps"] == 2
    assert sr == 24000
    assert isinstance(audio, np.ndarray)


def test_synthesize_concatenates_chunks(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    eng = _engine(tmp_path)
    fake_model = MagicMock()
    fake_model.sr = 24000
    fake_model.generate.return_value = torch.zeros(1, 1000)
    text = "First sentence here. Second sentence here. Third sentence here."
    with patch("poc_tts.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
        audio, _ = eng.synthesize(text=text, voice="a.wav", chunk_size=25)
    assert fake_model.generate.call_count == 3
    assert audio.shape[0] == 3000


def test_load_cuda_oom_is_translated_with_actionable_detail(tmp_path):
    """synthesize() already translates OOM into the good _vram_report()
    message; load() didn't, so a load-time OOM (the likelier failure point
    -- the PoC is designed to run beside the Turbo server, which already
    holds most of a 6 GB card) produced a raw traceback instead."""
    eng = _engine(tmp_path)
    with patch("poc_tts.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        with pytest.raises(OutOfMemoryError, match="VRAM"):
            eng.load()
    assert eng.loaded is False


def test_cuda_oom_is_translated_with_actionable_detail(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    eng = _engine(tmp_path)
    fake_model = MagicMock()
    fake_model.sr = 24000
    fake_model.generate.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
    with patch("poc_tts.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
        with pytest.raises(OutOfMemoryError, match="VRAM"):
            eng.synthesize(text="hi", voice="a.wav")


def test_synthesize_rejects_blank_text_when_splitting_disabled(tmp_path):
    """split_text=False took [text.strip()] verbatim, so blank input became
    [""] -- a non-empty list that slipped past the empty-text guard and sent
    an empty string to generate()."""
    (tmp_path / "a.wav").write_bytes(b"x")
    eng = _engine(tmp_path)
    fake_model = MagicMock()
    fake_model.sr = 24000
    with patch("poc_tts.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
        with pytest.raises(ValueError, match="text is empty"):
            eng.synthesize(text="   ", voice="a.wav", split_text=False)
    fake_model.generate.assert_not_called()
