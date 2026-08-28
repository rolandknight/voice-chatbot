from pathlib import Path
from unittest.mock import MagicMock, patch
import threading

import numpy as np
import pytest
import torch

from poc_tts_streaming.audio import TRIM_KEEP_MS
from poc_tts_streaming.config import load_config
from poc_tts_streaming.engine_flash import FlashEngine, OutOfMemoryError


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
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
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
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
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
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
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
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
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
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
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
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
        with pytest.raises(ValueError, match="text is empty"):
            eng.synthesize(text="   ", voice="a.wav", split_text=False)
    fake_model.generate.assert_not_called()


def _loaded_engine(tmp_path, samples_per_chunk=1000, **engine_overrides):
    (tmp_path / "a.wav").write_bytes(b"x")
    eng = _engine(tmp_path, **engine_overrides)
    fake_model = MagicMock()
    fake_model.sr = 24000
    fake_model.generate.return_value = torch.zeros(1, samples_per_chunk)
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
    return eng, fake_model


def test_synthesize_stream_yields_one_chunk_per_sentence_in_order(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    text = "First sentence here. Second sentence here. Third sentence here."
    out = list(eng.synthesize_stream(text, "a.wav", chunk_size=25))
    assert [t for t, _ in out] == [
        "First sentence here.", "Second sentence here.", "Third sentence here."]
    assert all(pcm.dtype == np.float32 and pcm.shape == (1000,) for _, pcm in out)
    assert model.generate.call_count == 3


def test_synthesize_stream_is_lazy(tmp_path):
    """The first chunk must come back before the second is generated --
    that is the whole point of streaming."""
    eng, model = _loaded_engine(tmp_path)
    gen = eng.synthesize_stream("One. Two.", "a.wav", chunk_size=4)
    next(gen)
    assert model.generate.call_count == 1


def test_synthesize_stream_stops_after_current_chunk_on_cancel(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    cancel = threading.Event()
    gen = eng.synthesize_stream("One. Two. Three.", "a.wav", chunk_size=4, cancel=cancel)
    next(gen)
    cancel.set()
    assert list(gen) == []
    assert model.generate.call_count == 1


def test_synthesize_stream_missing_voice_raises_before_first_yield(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    with pytest.raises(FileNotFoundError):
        next(eng.synthesize_stream("Hello.", "missing.wav"))
    model.generate.assert_not_called()


def test_synthesize_stream_forwards_split_on_clauses(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    text = "alpha beta, gamma delta, epsilon zeta, eta theta."
    whole = list(eng.synthesize_stream(text, "a.wav", chunk_size=20, split_on_clauses=False))
    split = list(eng.synthesize_stream(text, "a.wav", chunk_size=20, split_on_clauses=True))
    assert len(whole) == 1 and len(split) > 1


def test_synthesize_stream_trims_the_chunk_edge_silence(tmp_path):
    """A draw that pads its budget with silence must not stream the padding.

    This is the runaway case in miniature: 0.5 s of speech followed by 2 s of
    digital silence, which is what ~1-2 % of real draws look like when the stop
    token never lands. What reaches the caller is the speech plus one 120 ms
    breath on each edge.
    """
    keep = int(24000 * TRIM_KEEP_MS / 1000)
    speech = torch.full((1, 12000), 0.3)
    padded = torch.cat([torch.zeros(1, 4800), speech, torch.zeros(1, 48000)], dim=1)
    eng, model = _loaded_engine(tmp_path)
    model.generate.return_value = padded

    (label, pcm), = list(eng.synthesize_stream("Hello there.", "a.wav"))

    assert label == "Hello there."
    assert len(pcm) == keep + 12000 + keep
    assert np.array_equal(pcm[keep:keep + 12000], np.full(12000, 0.3, dtype=np.float32))


def test_synthesize_stream_normalises_clause_fragments_for_the_model(tmp_path):
    """chunk_text keeps a clause mark (, ; :) on an over-long sentence's
    fragment, so the model would otherwise see "it was the age of wisdom,"
    with no sentence-final punctuation. speakable() must be applied at the
    generate() boundary while the yielded label -- what transcripts and
    bench_stream's chars count -- stays the original chunk text."""
    eng, model = _loaded_engine(tmp_path)
    chunk = "it was the age of wisdom,"

    (label, _pcm), = list(eng.synthesize_stream(chunk, "a.wav", split_text=False))

    args, _kwargs = model.generate.call_args
    assert args[0] == "it was the age of wisdom."
    assert label == chunk


def test_trim_keep_ms_zero_is_rejected(tmp_path):
    """A zero keep lets an all-silent chunk emit nothing at all, and the block
    path puts the chunk's transcript label on the first piece it emits -- so
    the sentence would vanish from output_audio_transcript.delta while the
    sentence path still labelled its (empty) yield. Fail loudly at construction
    instead of diverging the two engines at runtime."""
    with pytest.raises(ValueError, match="trim_keep_ms"):
        _engine(tmp_path, trim_keep_ms=0)


def test_trim_keep_ms_zero_is_allowed_when_trimming_is_off(tmp_path):
    """The guard is about what the trim emits; with trim_silence off there is
    no trim, so the value is inert rather than dangerous."""
    eng = _engine(tmp_path, trim_silence=False, trim_keep_ms=0)
    assert eng.trim_silence is False


def test_trim_silence_can_be_switched_off(tmp_path):
    """engine.trim_silence: false streams the raw draw -- the escape hatch the
    sample-count identity tests use to pin the un-gated length."""
    eng, model = _loaded_engine(tmp_path, trim_silence=False)
    model.generate.return_value = torch.cat(
        [torch.full((1, 12000), 0.3), torch.zeros(1, 48000)], dim=1
    )
    (_label, pcm), = list(eng.synthesize_stream("Hello there.", "a.wav"))
    assert len(pcm) == 60000
