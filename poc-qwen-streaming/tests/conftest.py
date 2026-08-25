import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "poc-qwen"))


@dataclass
class FakeGen:
    audio: np.ndarray


class FakeModel:
    """Stands in for an mlx-audio Qwen3 model. Streams 3 chunks of 0.05 s per char total."""

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
def bridge(fake_loader):
    from poc_qwen_streaming.bridge import Bridge

    return Bridge(str(HERE / "config.yaml"), loader=fake_loader)
