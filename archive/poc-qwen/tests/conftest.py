import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc_qwen.config import load_config  # noqa: E402


@dataclass
class FakeGen:
    audio: np.ndarray


class FakeModel:
    """Stands in for an mlx-audio Qwen3 model. Yields a 24 kHz tone: 0.05 s per char."""

    sample_rate = 24000

    def __init__(self, kind="base"):
        self.config = types.SimpleNamespace(tts_model_type=kind)
        self.calls = []

    def get_supported_speakers(self):
        return ["ryan", "aiden"]

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        n = int(self.sample_rate * 0.05 * len(text))
        yield FakeGen(audio=np.sin(np.arange(n) * 0.1).astype(np.float32))


@pytest.fixture
def cfg():
    return load_config(env={})


@pytest.fixture
def fake_loader():
    loaded = []

    def loader(model_id):
        kind = "voice_design" if "VoiceDesign" in model_id else "custom_voice" if "CustomVoice" in model_id else "base"
        m = FakeModel(kind)
        m.model_id = model_id
        loaded.append(m)
        return m

    loader.loaded = loaded
    return loader


@pytest.fixture
def engine(cfg, fake_loader, monkeypatch):
    import mlx.core as mx  # noqa: F401 - available in the venv; ref audio is passed as ndarray in tests

    from poc_qwen.engine import Qwen3Engine

    eng = Qwen3Engine(cfg, loader=fake_loader)
    return eng
