"""GPU-free fixtures: a fake mlx-audio model behind the engine's loader hook."""

import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from qwen_tts.config import load_config

HERE = Path(__file__).resolve().parent


@dataclass
class FakeGen:
    audio: np.ndarray


class FakeModel:
    """Stands in for an mlx-audio Qwen3 model: a 24 kHz tone, 0.05 s per char, in 3 chunks when streaming."""

    sample_rate = 24000

    def __init__(self, kind="base"):
        self.config = types.SimpleNamespace(tts_model_type=kind)
        self.calls = []

    def get_supported_speakers(self):
        return ["ryan", "aiden"]

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        n = int(self.sample_rate * 0.05 * len(text))
        tone = np.sin(np.arange(n) * 0.1).astype(np.float32)
        if kwargs.get("stream"):
            for part in np.array_split(tone, 3):
                yield FakeGen(audio=part)
        else:
            yield FakeGen(audio=tone)


@pytest.fixture
def cfg():
    """The test profile (tests/config.yaml): LRU of 2, all tabs, preload of all three models."""
    return load_config(HERE / "config.yaml", env={})


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
def engine(cfg, fake_loader):
    from qwen_tts.engine import Qwen3Engine

    return Qwen3Engine(cfg, loader=fake_loader)


@pytest.fixture
def bridge(fake_loader):
    from qwen_tts.bridge import Bridge

    return Bridge(str(HERE / "config.yaml"), loader=fake_loader)
