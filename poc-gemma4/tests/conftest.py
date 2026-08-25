import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc_gemma4.config import ROOT, load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config(env={})


@pytest.fixture
def skills_root(cfg):
    return (ROOT / cfg["skills"]["root"]).resolve()
