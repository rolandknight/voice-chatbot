import numpy as np
import pytest

from poc_qwen.engine import LANGUAGES, crossfade_concat, discover_voices, lang_key, sidecar_transcript

REF = np.zeros(24000, dtype=np.float32)


def test_clone_maps_kwargs_and_times(engine, fake_loader):
    r = engine.clone("Hello world.", REF, "ref transcript", language="English", size="0.6B")
    model = fake_loader.loaded[0]
    assert model.model_id.endswith("0.6B-Base-bf16")
    call = model.calls[-1]  # warm-up call came first
    assert call["ref_text"] == "ref transcript" and call["lang_code"] == "english"
    assert call["temperature"] == 0.9 and call["top_p"] == 0.9
    assert r.sample_rate == 24000 and r.duration_s > 0
    assert set(r.timings) >= {"gen_s", "audio_s", "rtf", "chunks", "model", "load_s", "warm_s"}
    assert r.timings["mode"] == "icl"


def test_xvector_only_omits_ref_text(engine, fake_loader):
    r = engine.clone("Hi.", REF, "", xvector_only=True)
    assert "ref_text" not in fake_loader.loaded[0].calls[-1]
    assert r.timings["mode"] == "xvector"


def test_clone_requires_ref_text_unless_xvector(engine):
    with pytest.raises(ValueError):
        engine.clone("Hi.", REF, "")


def test_custom_voice_and_design_route_to_right_models(engine, fake_loader):
    engine.custom_voice("Hi.", "Ryan", instruct="Happy")
    engine.voice_design("Hi.", "A deep voice")
    ids = [m.model_id for m in fake_loader.loaded]
    assert any("CustomVoice" in i for i in ids) and any("VoiceDesign" in i for i in ids)
    cv = next(m for m in fake_loader.loaded if "CustomVoice" in m.model_id)
    assert cv.calls[-1]["voice"] == "Ryan" and cv.calls[-1]["instruct"] == "Happy"


def test_lru_evicts_oldest(engine, fake_loader):
    engine.clone("a.", REF, "t", size="0.6B")
    engine.custom_voice("b.", "Ryan")
    engine.voice_design("c.", "x")
    assert len(engine.model_info()["resident"]) == 2
    assert not any("0.6B" in m for m in engine.model_info()["resident"])


def test_long_text_is_chunked_and_concatenated(engine, fake_loader):
    text = " ".join(f"This is sentence number {i}." for i in range(30))
    r = engine.clone(text, REF, "t")
    model = fake_loader.loaded[0]
    assert r.timings["chunks"] > 1
    assert len(model.calls) - 1 == r.timings["chunks"]


def test_crossfade_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    out = crossfade_concat([a, b], 24000, crossfade_ms=10)  # 240 samples overlap
    assert len(out) == 2000 - 240
    assert np.allclose(out, 1.0)


def test_lang_key():
    assert lang_key("English") == "english" and lang_key("Auto") == "auto" and lang_key("weird") == "weird"
    assert "Auto" in LANGUAGES


def test_discover_voices_and_sidecar(tmp_path):
    (tmp_path / "bob.wav").write_bytes(b"")
    (tmp_path / "bob.txt").write_text("hello")
    (tmp_path / "notes.md").write_text("x")
    voices = discover_voices([tmp_path])
    assert list(voices) == ["bob"]
    assert sidecar_transcript(voices["bob"]) == "hello"
    assert sidecar_transcript(tmp_path / "none.wav") is None


def test_warmup_exercises_icl_path(engine, fake_loader):
    engine.clone("Hi.", REF, "t", size="0.6B")
    warm = fake_loader.loaded[0].calls[0]
    assert "ref_audio" in warm and warm["ref_text"] == "Warm up."


def test_all_mlx_work_runs_on_one_persistent_thread(engine, fake_loader, cfg):
    import threading

    seen = set()
    orig = fake_loader

    def loader(model_id):
        seen.add(threading.current_thread().name)
        return orig(model_id)

    from poc_qwen.engine import Qwen3Engine

    eng = Qwen3Engine(cfg, loader=loader)
    threads = [threading.Thread(target=lambda: eng.clone("Hi.", REF, "t", size="0.6B")), threading.Thread(target=lambda: eng.voice_design("Hi.", "x"))]
    for t in threads:
        t.start(); t.join()
    eng.model_info(); eng.unload_all()
    assert seen == {"mlx-worker"}


def test_stream_clone_yields_chunks_from_worker(engine, fake_loader):
    chunks = list(engine.stream_clone("Hello there.", REF, "t", size="0.6B"))
    assert len(chunks) == 1 and chunks[0].dtype == np.float32
    call = fake_loader.loaded[0].calls[-1]
    assert call["stream"] is True and call["streaming_interval"] == 0.32
