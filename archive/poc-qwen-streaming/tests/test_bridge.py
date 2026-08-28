import threading

import numpy as np
import pytest

from poc_qwen_streaming.bridge import Seam

REF = np.zeros(24000, dtype=np.float32)


def test_stream_clone_yields_chunks_and_maps_kwargs(bridge, fake_loader):
    chunks = list(bridge.stream("clone", {"text": "Hello world.", "ref_audio": REF, "ref_text": "ref", "language": "English", "size": "0.6B"}))
    model = fake_loader.loaded[0]
    assert model.model_id.endswith("0.6B-Base-bf16")
    call = model.calls[-1]
    assert call["stream"] is True and call["ref_text"] == "ref" and call["lang_code"] == "english"
    assert call["temperature"] == 0.9 and call["top_p"] == 0.9
    assert len(chunks) >= 3 and all(c.sample_rate == 24000 for c in chunks)
    assert chunks[0].t >= 0 and chunks[-1].t >= chunks[0].t
    total = sum(len(c.audio) for c in chunks)
    assert total == int(24000 * 0.05 * len("Hello world."))  # nothing lost across the seam holdback


def test_xvector_only_omits_ref_text(bridge, fake_loader):
    list(bridge.stream("clone", {"text": "Hi.", "ref_audio": REF, "xvector_only": True}))
    assert "ref_text" not in fake_loader.loaded[0].calls[-1]


def test_clone_requires_ref_text_unless_xvector(bridge):
    with pytest.raises(ValueError):
        list(bridge.stream("clone", {"text": "Hi.", "ref_audio": REF}))


def test_custom_and_design_route(bridge, fake_loader):
    list(bridge.stream("custom", {"text": "Hi.", "speaker": "Ryan", "instruct": "Happy"}))
    list(bridge.stream("design", {"text": "Hi.", "instruct": "A deep voice"}))
    ids = [m.model_id for m in fake_loader.loaded]
    assert any("CustomVoice" in i for i in ids) and any("VoiceDesign" in i for i in ids)
    cv = next(m for m in fake_loader.loaded if "CustomVoice" in m.model_id)
    assert cv.calls[-1]["voice"] == "Ryan" and cv.calls[-1]["instruct"] == "Happy"


def test_design_requires_instruct(bridge):
    with pytest.raises(ValueError):
        list(bridge.stream("design", {"text": "Hi."}))


def test_long_text_is_chunked_and_length_preserved(bridge, fake_loader):
    text = " ".join(f"This is sentence number {i}." for i in range(30))
    chunks = list(bridge.stream("custom", {"text": text}))
    model = fake_loader.loaded[0]
    calls = [c for c in model.calls if c.get("stream")]
    assert len(calls) > 1
    expected = sum(int(24000 * 0.05 * len(c["text"])) for c in calls)
    got = sum(len(c.audio) for c in chunks)
    overlap = 480 * (len(calls) - 1)  # 20 ms crossfade per seam
    assert got == expected - overlap


def test_stop_event_ends_early(bridge, fake_loader):
    text = " ".join(f"Sentence {i} is here." for i in range(40))
    stop = threading.Event()
    n = 0
    for _ in bridge.stream("custom", {"text": text}, stop=stop):
        n += 1
        if n == 2:
            stop.set()
    calls = [c for c in fake_loader.loaded[0].calls if c.get("stream")]
    assert len(calls) < 5


def test_seam_crossfade():
    seam = Seam(4)
    a = seam.push(np.ones(10, np.float32), True)
    assert len(a) == 6 and seam.tail is not None and len(seam.tail) == 4
    b = seam.push(np.full(10, 3.0, np.float32), True)
    assert len(b) == 6
    assert b[0] == pytest.approx(1.0) and b[3] == pytest.approx(3.0)
    assert len(seam.flush()) == 4 and seam.tail is None


def test_catalogue(bridge):
    assert "Auto" in bridge.languages()
    assert bridge.sizes() == ["0.6B", "1.7B"]
    assert isinstance(bridge.speakers(), list) and bridge.speakers()
    names = [v["name"] for v in bridge.voices()]
    assert "one-one" in names
    assert bridge.voice_path("one-one").endswith("one-one.mp3")


def test_model_for(bridge):
    assert bridge.model_for("custom", {"size": "0.6B"}).endswith("0.6B-CustomVoice-bf16")
    assert bridge.model_for("clone", {}) == ""


def test_preload_loads_models_and_primes_presets(bridge, fake_loader):
    st = bridge.preload(wait=True)
    assert st["state"] == "done" and not st["pending"]
    ids = [m.model_id for m in fake_loader.loaded]
    assert any("1.7B-Base" in i for i in ids) and any("CustomVoice" in i for i in ids) and any("VoiceDesign" in i for i in ids)
    base = next(m for m in fake_loader.loaded if "1.7B-Base" in m.model_id)
    primed = [c for c in base.calls if c.get("stream") and c["text"] == "Hi."]
    assert len(primed) >= 1 and all("ref_text" in c for c in primed)
    assert "model:clone_default" in st["done"]
    assert bridge.model_info()["preload"]["state"] == "done"


def test_preload_disabled(bridge, fake_loader):
    bridge.cfg["preload"] = {"enabled": False}
    bridge.preload(wait=True)
    assert fake_loader.loaded == []
