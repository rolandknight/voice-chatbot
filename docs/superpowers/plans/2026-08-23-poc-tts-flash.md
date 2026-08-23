# poc-tts Chatterbox Flash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `poc-tts/` — a standalone FastAPI server on port 8005 that runs Chatterbox Flash on the fastest backend available, driven through a committed copy of the Chatterbox web GUI, with a tuning sweep that records results alongside the existing Turbo baselines.

**Architecture:** Three units with hard boundaries. `engine_flash.py` owns model lifecycle and all hardware decisions and never imports FastAPI. `server.py` owns the HTTP surface and never imports `chatterbox_flash`. `bench.py` drives the engine directly. That boundary is what lets every endpoint test run with the engine mocked and no GPU present.

**Tech Stack:** Python 3.10 (mise-pinned, scoped to `poc-tts/`), `chatterbox-flash==0.1.0`, torch >= 2.6 with CUDA, FastAPI + uvicorn, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-poc-tts-flash-design.md`

## Global Constraints

- Python is pinned to **3.10** via `poc-tts/mise.toml`. This applies to `poc-tts/` only — the rest of the repo stays hermit-managed (`bin/activate-hermit`, `python3@3.10`, `rust-1.97.1`). Do not add a root `mise.toml` and do not remove any `bin/` hermit shim.
- `chatterbox-flash` is pinned **exactly** to `0.1.0`. It is a single-release package; do not use a range.
- **Never modify `vendor/chatterbox-tts-server/` or its venv.** That environment is pinned at `torch 2.5.1+cu121`; `chatterbox-flash` requires `torch>=2.6`. Upgrading it breaks the working Turbo server. `poc-tts/.venv` is a separate environment and that separation is the point.
- The server binds **port 8005**. Turbo owns 8004 and both must run simultaneously for A/B.
- Model repo id is `ResembleAI/chatterbox-flash`. First run downloads 3.2 GB.
- Dtype resolution order is exactly: explicit config override, then `bfloat16` if CUDA and `torch.cuda.is_bf16_supported()`, then `float16` if CUDA, then `float32`. Never take the library's `bfloat16` default unconditionally.
- Backend `auto` resolves to `flashinfer` when importable, else `torch`. `mlx` is never selected automatically.
- `voices/` (repo root, git-tracked) is the source of truth for reference clips. `vendor/chatterbox-tts-server/reference_audio/` is scanned only when it exists. The PoC must run correctly with the vendor clone absent.
- Commit after every task.

---

### Task 1: Toolchain, skeleton, and setup

**Files:**
- Create: `poc-tts/mise.toml`
- Create: `poc-tts/requirements.txt`
- Create: `poc-tts/setup.sh`
- Create: `poc-tts/config.yaml`
- Create: `poc-tts/.gitignore`
- Create: `poc-tts/README.md`
- Modify: `Makefile` (add targets alongside the existing `poc-*` block at lines 242-292)

**Interfaces:**
- Consumes: nothing.
- Produces: a working `poc-tts/.venv` containing `chatterbox_flash`; `poc-tts/reports/flashinfer_probe.json` recording whether FlashInfer built.

- [ ] **Step 1: Create the mise pin**

`poc-tts/mise.toml`:

```toml
[tools]
python = "3.10"
```

- [ ] **Step 2: Create requirements**

`poc-tts/requirements.txt`:

```
chatterbox-flash==0.1.0
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pyyaml==6.0.2
soundfile==0.12.1
pytest==8.3.4
httpx==0.28.1
```

`chatterbox-flash` pulls `torch>=2.6`, `torchaudio`, `chatterbox-tts>=0.1.7`, and `transformers` transitively. Do not pin torch here — let the resolver pick the CUDA build matching the driver.

- [ ] **Step 3: Create config.yaml**

`poc-tts/config.yaml`:

```yaml
server:
  host: 127.0.0.1
  port: 8005

engine:
  # device: auto | cuda | cpu
  device: auto
  # dtype: auto | bfloat16 | float16 | float32
  # auto = bfloat16 if the GPU supports it, else float16 on CUDA, else float32.
  # The library default is bfloat16 unconditionally, which is wrong on sm_75.
  dtype: auto
  # backend: auto | flashinfer | torch | mlx
  # auto = flashinfer when importable, else torch. mlx is never automatic.
  backend: auto
  drf_block_size: 16

generation:
  temperature: 0.6
  exaggeration: 0.5
  cfg_scale: 1.0
  num_steps: 10
  n_cfm_timesteps: 2

voices:
  # Searched in order. Missing directories are skipped, not an error.
  paths:
    - ../voices
    - ../vendor/chatterbox-tts-server/reference_audio
```

- [ ] **Step 4: Create .gitignore**

`poc-tts/.gitignore`:

```
.venv/
model_cache/
outputs/
__pycache__/
.pytest_cache/
```

`reports/` is deliberately NOT ignored — the sweep results are the deliverable.

- [ ] **Step 5: Create setup.sh**

`poc-tts/setup.sh`:

```bash
#!/usr/bin/env bash
# Bootstrap the poc-tts environment. Idempotent.
set -euo pipefail

POC_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$POC_DIR"

command -v mise >/dev/null 2>&1 || {
    echo "ERROR: mise is not installed. See https://mise.jdx.dev" >&2
    exit 1
}

mise install

if [ ! -d .venv ]; then
    echo "Creating .venv with mise python 3.10 ..."
    mise exec -- python -m venv .venv
fi

./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt

# FlashInfer is the paper's fast path. sm_75 (Turing) is its documented floor,
# but its kernels lean on newer tensor-core features, so support is unproven
# there. Probe it and record the result either way -- never fail the setup.
mkdir -p reports
if ./.venv/bin/python -m pip install -q flashinfer-python 2>reports/flashinfer_install.log; then
    PROBE_OK=true
else
    PROBE_OK=false
fi
./.venv/bin/python - <<'PY' > reports/flashinfer_probe.json
import importlib.util, json, platform
try:
    import torch
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    cc, gpu = None, None
print(json.dumps({
    "flashinfer_importable": importlib.util.find_spec("flashinfer") is not None,
    "gpu": gpu,
    "compute_capability": f"sm_{cc[0]}{cc[1]}" if cc else None,
    "host": platform.node(),
}, indent=2))
PY

echo "--- flashinfer probe ---"
cat reports/flashinfer_probe.json
echo "poc-tts setup done"
```

- [ ] **Step 6: Make it executable and run it**

Run:
```bash
chmod +x poc-tts/setup.sh && ./poc-tts/setup.sh
```
Expected: ends with `poc-tts setup done`, and `reports/flashinfer_probe.json` reports `flashinfer_importable` true or false plus `compute_capability`. On the RTX 2060 development box expect `"compute_capability": "sm_75"`. Either probe result is acceptable — record it, do not chase it.

- [ ] **Step 7: Verify the environment is isolated**

Run:
```bash
./poc-tts/.venv/bin/python -c "import torch, chatterbox_flash; print('poc-tts torch', torch.__version__)"
./vendor/chatterbox-tts-server/venv/bin/python -c "import torch; print('turbo torch', torch.__version__)"
```
Expected: poc-tts reports torch 2.6.x or newer; turbo still reports `2.5.1+cu121`. If the Turbo line changed, the isolation constraint was violated — stop and fix.

- [ ] **Step 8: Add Makefile targets**

Add to `Makefile` after the existing `poc-results` target, following the established `poc-*` convention:

```makefile
POC_TTS_PY := poc-tts/.venv/bin/python

poc-tts-setup:  ## poc-tts: mise python 3.10, venv, deps, flashinfer probe (idempotent)
	@./poc-tts/setup.sh

poc-tts:    ## poc-tts: run the Chatterbox Flash server + GUI on :8005
	@$(POC_TTS_PY) -m poc_tts.server

poc-tts-bench:  ## poc-tts: sweep Flash tuning configs, append poc-tts/reports/runs.jsonl
	@$(POC_TTS_PY) -m poc_tts.bench

poc-tts-test:  ## poc-tts: GPU-free unit tests
	@cd poc-tts && .venv/bin/python -m pytest tests -v
```

Add `poc-tts-setup poc-tts poc-tts-bench poc-tts-test` to the `.PHONY` line that already lists the `poc-*` targets.

- [ ] **Step 9: Create README**

`poc-tts/README.md`:

```markdown
# poc-tts — Chatterbox Flash

Standalone Chatterbox Flash server on :8005, serving a copy of the Chatterbox
web GUI. Turbo keeps :8004, so both run side by side for A/B.

    make poc-tts-setup    # mise python 3.10, venv, deps, flashinfer probe
    make poc-tts          # server + GUI on http://127.0.0.1:8005
    make poc-tts-bench    # tuning sweep -> reports/runs.jsonl
    make poc-tts-test     # GPU-free unit tests

Python here is mise-pinned (`mise.toml`); the rest of the repo stays on hermit.

Never install into `vendor/chatterbox-tts-server/venv` — it is pinned at
torch 2.5.1+cu121 and Flash needs >= 2.6.

Design: `docs/superpowers/specs/2026-08-23-poc-tts-flash-design.md`
```

- [ ] **Step 10: Commit**

```bash
git add poc-tts/mise.toml poc-tts/requirements.txt poc-tts/setup.sh \
        poc-tts/config.yaml poc-tts/.gitignore poc-tts/README.md \
        poc-tts/reports/flashinfer_probe.json Makefile
git commit -m "feat(poc-tts): mise-pinned python 3.10 skeleton and setup"
```

---

### Task 2: Hardware resolution

**Files:**
- Create: `poc-tts/poc_tts/__init__.py` (empty)
- Create: `poc-tts/poc_tts/engine_flash.py`
- Create: `poc-tts/tests/__init__.py` (empty)
- Create: `poc-tts/tests/test_resolution.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `resolve_device(requested: str, cuda_available: bool) -> str`
  - `resolve_dtype(requested: str, device: str, bf16_supported: bool) -> torch.dtype`
  - `resolve_backend(requested: str, flashinfer_available: bool) -> str`
  - `UnsupportedDtypeError(Exception)`

These take capability flags as plain arguments rather than probing `torch.cuda` internally, which is what makes them testable without a GPU.

- [ ] **Step 1: Write the failing tests**

`poc-tts/tests/test_resolution.py`:

```python
import pytest
import torch

from poc_tts.engine_flash import (
    UnsupportedDtypeError,
    resolve_backend,
    resolve_device,
    resolve_dtype,
)


def test_device_auto_prefers_cuda_when_available():
    assert resolve_device("auto", cuda_available=True) == "cuda"


def test_device_auto_falls_back_to_cpu():
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_device_explicit_cuda_without_cuda_is_an_error():
    with pytest.raises(ValueError, match="CUDA requested but not available"):
        resolve_device("cuda", cuda_available=False)


def test_dtype_auto_uses_bf16_when_gpu_supports_it():
    got = resolve_dtype("auto", device="cuda", bf16_supported=True)
    assert got is torch.bfloat16


def test_dtype_auto_falls_back_to_fp16_on_sm75():
    """The RTX 2060 is sm_75: bf16 is unsupported.

    chatterbox-flash defaults to bfloat16 unconditionally, which on Turing
    means emulated speeds or an outright failure. Auto must never pick it
    when the hardware says no. This is the single most likely way this PoC
    silently runs slow.
    """
    got = resolve_dtype("auto", device="cuda", bf16_supported=False)
    assert got is torch.float16


def test_dtype_auto_uses_fp32_on_cpu():
    got = resolve_dtype("auto", device="cpu", bf16_supported=False)
    assert got is torch.float32


def test_dtype_explicit_bf16_on_unsupported_gpu_raises():
    with pytest.raises(UnsupportedDtypeError, match="bfloat16"):
        resolve_dtype("bfloat16", device="cuda", bf16_supported=False)


def test_dtype_explicit_fp16_is_honoured():
    got = resolve_dtype("float16", device="cuda", bf16_supported=True)
    assert got is torch.float16


def test_backend_auto_prefers_flashinfer():
    assert resolve_backend("auto", flashinfer_available=True) == "flashinfer"


def test_backend_auto_falls_back_to_torch():
    assert resolve_backend("auto", flashinfer_available=False) == "torch"


def test_backend_auto_never_selects_mlx():
    assert resolve_backend("auto", flashinfer_available=False) != "mlx"


def test_backend_explicit_flashinfer_when_absent_raises():
    with pytest.raises(ValueError, match="flashinfer requested but not installed"):
        resolve_backend("flashinfer", flashinfer_available=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_resolution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc_tts'`

- [ ] **Step 3: Write the implementation**

`poc-tts/poc_tts/engine_flash.py`:

```python
"""Chatterbox Flash engine wrapper.

Owns model lifecycle and every hardware decision. Never imports FastAPI --
that boundary is what lets the server tests run with this module mocked.
"""

from __future__ import annotations

import torch

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class UnsupportedDtypeError(Exception):
    """Raised when a dtype is explicitly requested that the device cannot run."""


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Resolve 'auto' | 'cuda' | 'cpu' against what the machine actually has."""
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA requested but not available on this machine")
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"invalid device {requested!r} (expected auto, cuda, or cpu)")
    return requested


def resolve_dtype(requested: str, device: str, bf16_supported: bool) -> torch.dtype:
    """Resolve the compute dtype.

    chatterbox-flash's from_pretrained defaults to bfloat16 unconditionally.
    On sm_75 (Turing, e.g. RTX 2060) torch.cuda.is_bf16_supported() is False,
    and taking that default gives emulated speeds or a hard failure. Auto
    therefore steps down to float16 on CUDA rather than trusting the library.
    """
    if requested == "auto":
        if device != "cuda":
            return torch.float32
        return torch.bfloat16 if bf16_supported else torch.float16

    if requested not in _DTYPES:
        raise ValueError(
            f"invalid dtype {requested!r} "
            "(expected auto, bfloat16, float16, or float32)"
        )
    if requested == "bfloat16" and device == "cuda" and not bf16_supported:
        raise UnsupportedDtypeError(
            "bfloat16 was requested but this GPU does not support it "
            "(sm_75 and older). Use dtype: auto or dtype: float16."
        )
    return _DTYPES[requested]


def resolve_backend(requested: str, flashinfer_available: bool) -> str:
    """Resolve the inference backend.

    'auto' picks flashinfer when importable, else torch SDPA. mlx is never
    selected automatically -- upstream marks it experimental and it must be
    named explicitly.
    """
    if requested == "auto":
        return "flashinfer" if flashinfer_available else "torch"
    if requested == "flashinfer" and not flashinfer_available:
        raise ValueError("flashinfer requested but not installed")
    if requested not in ("flashinfer", "torch", "mlx"):
        raise ValueError(
            f"invalid backend {requested!r} "
            "(expected auto, flashinfer, torch, or mlx)"
        )
    return requested
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_resolution.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add poc-tts/poc_tts/__init__.py poc-tts/poc_tts/engine_flash.py \
        poc-tts/tests/__init__.py poc-tts/tests/test_resolution.py
git commit -m "feat(poc-tts): device, dtype, and backend resolution"
```

---

### Task 3: Text chunking and voice discovery

**Files:**
- Modify: `poc-tts/poc_tts/engine_flash.py` (append)
- Create: `poc-tts/tests/test_chunking.py`
- Create: `poc-tts/tests/test_voices.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `chunk_text(text: str, chunk_size: int) -> list[str]`
  - `discover_voices(paths: list[Path]) -> list[str]` — sorted, de-duplicated `.wav` filenames
  - `resolve_voice_path(name: str, paths: list[Path]) -> Path` — raises `FileNotFoundError`

- [ ] **Step 1: Write the failing chunking tests**

`poc-tts/tests/test_chunking.py`:

```python
from poc_tts.engine_flash import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello there.", chunk_size=120) == ["Hello there."]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", chunk_size=120) == []


def test_splits_on_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, chunk_size=25)
    assert len(chunks) == 3
    assert chunks[0] == "First sentence here."
    assert all(not c.startswith(" ") for c in chunks)


def test_sentences_are_packed_up_to_chunk_size():
    text = "One. Two. Three. Four."
    chunks = chunk_text(text, chunk_size=120)
    assert chunks == ["One. Two. Three. Four."]


def test_sentence_longer_than_chunk_size_is_not_dropped():
    long_sentence = "word " * 60 + "end."
    chunks = chunk_text(long_sentence, chunk_size=50)
    assert len(chunks) >= 1
    assert "end." in " ".join(chunks)


def test_no_text_is_lost():
    text = "Alpha beta. Gamma delta. Epsilon zeta."
    chunks = chunk_text(text, chunk_size=15)
    rejoined = " ".join(chunks)
    for word in ("Alpha", "beta", "Gamma", "delta", "Epsilon", "zeta"):
        assert word in rejoined
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_text'`

- [ ] **Step 3: Implement chunking**

Append to `poc-tts/poc_tts/engine_flash.py`:

```python
import re
from pathlib import Path

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of roughly chunk_size characters.

    Splits on sentence boundaries and packs whole sentences together up to
    the target size. A single sentence longer than chunk_size is emitted
    whole rather than cut mid-word -- Flash handles long blocks better than
    it handles a severed clause.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 4: Run to verify pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: 6 passed

- [ ] **Step 5: Write the failing voice tests**

`poc-tts/tests/test_voices.py`:

```python
import pytest

from poc_tts.engine_flash import discover_voices, resolve_voice_path


def test_discovers_wavs_sorted(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    (d / "zeta.wav").write_bytes(b"x")
    (d / "alpha.wav").write_bytes(b"x")
    assert discover_voices([d]) == ["alpha.wav", "zeta.wav"]


def test_ignores_non_wav_files(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    (d / "a.wav").write_bytes(b"x")
    (d / "notes.md").write_bytes(b"x")
    (d / "b.mp3").write_bytes(b"x")
    assert discover_voices([d]) == ["a.wav"]


def test_missing_directory_is_skipped_not_an_error(tmp_path):
    """The vendor clone is gitignored and may be absent entirely."""
    present = tmp_path / "voices"
    present.mkdir()
    (present / "a.wav").write_bytes(b"x")
    absent = tmp_path / "does-not-exist"
    assert discover_voices([present, absent]) == ["a.wav"]


def test_duplicate_names_across_paths_are_deduplicated(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "marvin.wav").write_bytes(b"x")
    (second / "marvin.wav").write_bytes(b"x")
    assert discover_voices([first, second]) == ["marvin.wav"]


def test_resolve_returns_first_match_in_path_order(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "marvin.wav").write_bytes(b"x")
    (second / "marvin.wav").write_bytes(b"x")
    assert resolve_voice_path("marvin.wav", [first, second]).parent == first


def test_resolve_missing_voice_names_the_paths_searched(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="voices"):
        resolve_voice_path("nope.wav", [d])
```

- [ ] **Step 6: Run to verify failure**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_voices.py -v`
Expected: FAIL — `ImportError: cannot import name 'discover_voices'`

- [ ] **Step 7: Implement voice discovery**

Append to `poc-tts/poc_tts/engine_flash.py`:

```python
def discover_voices(paths: list[Path]) -> list[str]:
    """List available reference .wav filenames across the search paths.

    Missing directories are skipped rather than raising: the vendor clone
    under vendor/chatterbox-tts-server/ is gitignored and may be absent.
    Names are de-duplicated, keeping first-path-wins ordering.
    """
    seen: dict[str, None] = {}
    for path in paths:
        if not path.is_dir():
            continue
        for wav in sorted(path.glob("*.wav")):
            seen.setdefault(wav.name, None)
    return sorted(seen)


def resolve_voice_path(name: str, paths: list[Path]) -> Path:
    """Resolve a reference filename to a concrete path, first match wins."""
    for path in paths:
        candidate = path / name
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"reference voice {name!r} not found. Searched: {searched}")
```

- [ ] **Step 8: Run to verify pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests -v`
Expected: 24 passed

- [ ] **Step 9: Commit**

```bash
git add poc-tts/poc_tts/engine_flash.py poc-tts/tests/test_chunking.py poc-tts/tests/test_voices.py
git commit -m "feat(poc-tts): text chunking and reference voice discovery"
```

---

### Task 4: Engine lifecycle and synthesis

**Files:**
- Modify: `poc-tts/poc_tts/engine_flash.py` (append)
- Create: `poc-tts/poc_tts/config.py`
- Create: `poc-tts/tests/test_engine.py`

**Interfaces:**
- Consumes: `resolve_device`, `resolve_dtype`, `resolve_backend`, `chunk_text`, `resolve_voice_path`.
- Produces:
  - `load_config(path: Path) -> dict`
  - `class FlashEngine` with `.load()`, `.model_info() -> dict`, `.synthesize(...) -> tuple[numpy.ndarray, int]`, `.loaded: bool`, `.sr: int`
  - `OutOfMemoryError(Exception)`

`model_info()` must return the exact keys `ui/script.js` reads in `updateModelUI` (`script.js:310-388`): `loaded`, `type`, `class_name`, `device`, `sample_rate`, `supports_paralinguistic_tags`, `available_paralinguistic_tags`, `supports_multilingual`, `supported_languages`. `type` is the literal string `"flash"` — this makes the existing UI keep exaggeration and CFG visible (which Flash has) and force English-only, both correct.

- [ ] **Step 1: Write the config loader test**

`poc-tts/tests/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc_tts.config'`

- [ ] **Step 3: Write the config loader**

`poc-tts/poc_tts/config.py`:

```python
"""Config loading for poc-tts."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml. Paths inside it resolve against the poc-tts dir."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def voice_paths(config: dict) -> list[Path]:
    """Resolve configured voice search paths against the poc-tts directory."""
    base = DEFAULT_CONFIG_PATH.parent
    raw = config.get("voices", {}).get("paths", [])
    return [(base / entry).resolve() for entry in raw]
```

- [ ] **Step 4: Write the engine**

Append to `poc-tts/poc_tts/engine_flash.py`:

```python
import importlib.util
import logging

import numpy as np
from chatterbox_flash import ChatterboxFlashTTS

logger = logging.getLogger(__name__)


class OutOfMemoryError(Exception):
    """Raised when generation runs out of VRAM, with actionable detail."""


def _flashinfer_available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def _vram_report() -> str:
    """Describe VRAM pressure. A bare 'CUDA out of memory' wastes the reader."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return (
        f"VRAM {free / 2**30:.2f} GB free of {total / 2**30:.2f} GB total; "
        f"this process holds {allocated / 2**30:.2f} GB. Flash needs roughly "
        f"2.0 GB of weights plus about 0.8 GB of working set at float16. "
        f"Check `nvidia-smi` for other processes holding the card."
    )


class FlashEngine:
    """Owns the Chatterbox Flash model for the process lifetime."""

    def __init__(self, engine_cfg: dict, generation_cfg: dict, voice_paths: list[Path]):
        self._engine_cfg = engine_cfg
        self._generation_cfg = generation_cfg
        self._voice_paths = voice_paths
        self._model = None

        self.device = resolve_device(
            engine_cfg.get("device", "auto"), torch.cuda.is_available()
        )
        bf16 = torch.cuda.is_bf16_supported() if self.device == "cuda" else False
        self.dtype = resolve_dtype(engine_cfg.get("dtype", "auto"), self.device, bf16)
        self.backend = resolve_backend(
            engine_cfg.get("backend", "auto"), _flashinfer_available()
        )
        self.drf_block_size = int(engine_cfg.get("drf_block_size", 16))
        self.sr = 24000

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model once. First call downloads ~3.2 GB."""
        logger.info(
            "loading Chatterbox Flash: device=%s dtype=%s backend=%s block=%d",
            self.device, self.dtype, self.backend, self.drf_block_size,
        )
        if self.backend == "torch":
            logger.info(
                "flashinfer not in use -- running the portable SDPA path, not "
                "the CUDA-graph path the published RTF figures come from."
            )
        self._model = ChatterboxFlashTTS.from_pretrained(
            device=self.device,
            dtype=self.dtype,
            drf_block_size=self.drf_block_size,
        )
        self.sr = getattr(self._model, "sr", 24000)

    def model_info(self) -> dict:
        """Exactly the keys ui/script.js updateModelUI reads.

        type == "flash" makes the existing UI keep exaggeration and CFG
        visible (Flash has both) and force English-only (Flash is English-
        only by construction). No UI branch is needed for either.
        """
        return {
            "loaded": self.loaded,
            "type": "flash",
            "class_name": "ChatterboxFlashTTS",
            "device": self.device,
            "sample_rate": self.sr if self.loaded else None,
            "supports_paralinguistic_tags": False,
            "available_paralinguistic_tags": [],
            "supports_multilingual": False,
            "supported_languages": {"en": "English"},
            "dtype": str(self.dtype).replace("torch.", ""),
            "backend": self.backend,
            "drf_block_size": self.drf_block_size,
        }

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        temperature: float | None = None,
        exaggeration: float | None = None,
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        n_cfm_timesteps: int | None = None,
        chunk_size: int = 120,
        split_text: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text with a reference voice. Returns (mono float32, sr)."""
        if not self.loaded:
            raise RuntimeError("model is not loaded -- call load() first")

        gen = self._generation_cfg
        prompt = str(resolve_voice_path(voice, self._voice_paths))
        chunks = chunk_text(text, chunk_size) if split_text else [text.strip()]
        if not chunks:
            raise ValueError("text is empty")

        pieces: list[np.ndarray] = []
        try:
            for chunk in chunks:
                wav = self._model.generate(
                    chunk,
                    audio_prompt_path=prompt,
                    temperature=temperature if temperature is not None else gen["temperature"],
                    exaggeration=exaggeration if exaggeration is not None else gen["exaggeration"],
                    cfg_scale=cfg_scale if cfg_scale is not None else gen["cfg_scale"],
                    num_steps=num_steps if num_steps is not None else gen["num_steps"],
                    n_cfm_timesteps=(
                        n_cfm_timesteps if n_cfm_timesteps is not None
                        else gen["n_cfm_timesteps"]
                    ),
                    backend=self.backend,
                )
                pieces.append(wav.detach().float().cpu().numpy().reshape(-1))
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError(
                f"ran out of VRAM during generation. {_vram_report()}"
            ) from exc

        return np.concatenate(pieces), self.sr
```

- [ ] **Step 5: Run to verify pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests -v`
Expected: 33 passed

- [ ] **Step 6: Commit**

```bash
git add poc-tts/poc_tts/config.py poc-tts/poc_tts/engine_flash.py poc-tts/tests/test_engine.py
git commit -m "feat(poc-tts): Flash engine lifecycle, synthesis, and OOM reporting"
```

---

### Task 5: Server — static UI and info endpoints

**Files:**
- Create: `poc-tts/poc_tts/server.py`
- Create: `poc-tts/tests/test_server_info.py`
- Copy: `vendor/chatterbox-tts-server/ui/` → `poc-tts/ui/`

**Interfaces:**
- Consumes: `FlashEngine`, `load_config`, `voice_paths`, `discover_voices`.
- Produces: `create_app(engine, config) -> fastapi.FastAPI`. Taking the engine as an argument is what lets every test inject a mock and run GPU-free.

- [ ] **Step 1: Copy the UI**

Run:
```bash
mkdir -p poc-tts/ui
cp vendor/chatterbox-tts-server/ui/index.html \
   vendor/chatterbox-tts-server/ui/script.js \
   vendor/chatterbox-tts-server/ui/styles.css \
   vendor/chatterbox-tts-server/ui/presets.yaml \
   poc-tts/ui/
```
Expected: four files present in `poc-tts/ui/`. These are now ours and will not track upstream.

- [ ] **Step 2: Write the failing tests**

`poc-tts/tests/test_server_info.py`:

```python
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from poc_tts.server import create_app


@pytest.fixture
def client(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "marvin.wav").write_bytes(b"x")

    engine = MagicMock()
    engine.loaded = True
    engine.model_info.return_value = {
        "loaded": True, "type": "flash", "class_name": "ChatterboxFlashTTS",
        "device": "cuda", "sample_rate": 24000,
        "supports_paralinguistic_tags": False, "available_paralinguistic_tags": [],
        "supports_multilingual": False, "supported_languages": {"en": "English"},
    }
    config = {
        "server": {"host": "127.0.0.1", "port": 8005},
        "generation": {
            "temperature": 0.6, "exaggeration": 0.5, "cfg_scale": 1.0,
            "num_steps": 10, "n_cfm_timesteps": 2,
        },
    }
    return TestClient(create_app(engine, config, voice_paths=[voices]))


def test_index_serves_the_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_static_assets_are_served(client):
    assert client.get("/script.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


def test_model_info_endpoint(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    assert r.json()["type"] == "flash"


def test_initial_data_has_the_keys_script_js_reads(client):
    """script.js destructures these on load; a missing key breaks the page
    silently rather than raising."""
    body = client.get("/api/ui/initial-data").json()
    for key in (
        "config", "reference_files", "predefined_voices",
        "presets", "initial_gen_result", "model_info",
    ):
        assert key in body, f"missing key: {key}"


def test_initial_data_reports_flash_model(client):
    assert client.get("/api/ui/initial-data").json()["model_info"]["type"] == "flash"


def test_reference_files_lists_discovered_voices(client):
    assert client.get("/get_reference_files").json() == ["marvin.wav"]


def test_predefined_voices_returns_ui_shaped_records(client):
    voices = client.get("/get_predefined_voices").json()
    assert voices and all("display_name" in v and "filename" in v for v in voices)


def test_restart_server_is_a_clear_noop_not_a_404(client):
    """The UI calls this; a 404 would read as a bug."""
    r = client.post("/restart_server")
    assert r.status_code == 200
    assert "not supported" in r.json()["message"].lower()
```

- [ ] **Step 3: Run to verify failure**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_server_info.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc_tts.server'`

- [ ] **Step 4: Write the server**

`poc-tts/poc_tts/server.py`:

```python
"""FastAPI surface for poc-tts.

Never imports chatterbox_flash. The engine arrives as a constructor argument
so tests can inject a mock and run without a GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from poc_tts.config import load_config, voice_paths as configured_voice_paths
from poc_tts.engine_flash import discover_voices

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def create_app(engine, config: dict, voice_paths: list[Path]) -> FastAPI:
    app = FastAPI(title="poc-tts: Chatterbox Flash", version="0.1.0")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/script.js", include_in_schema=False)
    async def script_js():
        return FileResponse(UI_DIR / "script.js", media_type="application/javascript")

    @app.get("/styles.css", include_in_schema=False)
    async def styles_css():
        return FileResponse(UI_DIR / "styles.css", media_type="text/css")

    @app.get("/api/model-info")
    async def model_info():
        return engine.model_info()

    @app.get("/get_reference_files")
    async def get_reference_files():
        return discover_voices(voice_paths)

    @app.get("/get_predefined_voices")
    async def get_predefined_voices():
        return [
            {"display_name": name.replace(".wav", "").replace("_", " ").title(),
             "filename": name}
            for name in discover_voices(voice_paths)
        ]

    @app.get("/api/ui/initial-data")
    async def initial_data():
        presets = []
        presets_file = UI_DIR / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
                if isinstance(loaded, list):
                    presets = loaded
        names = discover_voices(voice_paths)
        return {
            "config": config,
            "reference_files": names,
            "predefined_voices": [
                {"display_name": n.replace(".wav", "").replace("_", " ").title(),
                 "filename": n}
                for n in names
            ],
            "presets": presets,
            "initial_gen_result": {
                "outputUrl": None, "filename": None, "genTime": None,
                "submittedVoiceMode": None, "submittedPredefinedVoice": None,
                "submittedCloneFile": None,
            },
            "model_info": engine.model_info(),
        }

    @app.post("/restart_server")
    async def restart_server():
        return JSONResponse(
            {"message": "Restarting is not supported in the poc-tts PoC. "
                        "Stop and rerun `make poc-tts`."}
        )

    return app


def main() -> None:
    import uvicorn

    from poc_tts.engine_flash import FlashEngine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    paths = configured_voice_paths(config)
    engine = FlashEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=config.get("generation", {}),
        voice_paths=paths,
    )
    engine.load()
    app = create_app(engine, config, voice_paths=paths)
    uvicorn.run(
        app,
        host=config.get("server", {}).get("host", "127.0.0.1"),
        port=config.get("server", {}).get("port", 8005),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run to verify pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_server_info.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add poc-tts/poc_tts/server.py poc-tts/ui poc-tts/tests/test_server_info.py
git commit -m "feat(poc-tts): server static UI and info endpoints"
```

---

### Task 6: Server — generation and settings endpoints

**Files:**
- Modify: `poc-tts/poc_tts/server.py` (add endpoints inside `create_app`)
- Create: `poc-tts/poc_tts/models.py`
- Create: `poc-tts/tests/test_server_tts.py`

**Interfaces:**
- Consumes: `FlashEngine.synthesize`, `OutOfMemoryError`.
- Produces: `FlashTTSRequest` pydantic model; `POST /tts`, `POST /save_settings`, `POST /reset_settings`.

`FlashTTSRequest` mirrors the vendored `CustomTTSRequest` field names that `ui/script.js` already sends (`text`, `voice_mode`, `predefined_voice_id`, `reference_audio_filename`, `output_format`, `split_text`, `chunk_size`, `temperature`, `exaggeration`, `cfg_weight`) and adds the four Flash knobs (`num_steps`, `n_cfm_timesteps`, `drf_block_size`, `backend`). `cfg_weight` maps to Flash's `cfg_scale`.

- [ ] **Step 1: Write the failing tests**

`poc-tts/tests/test_server_tts.py`:

```python
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poc_tts.engine_flash import OutOfMemoryError
from poc_tts.server import create_app


@pytest.fixture
def engine():
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    eng.synthesize.return_value = (np.zeros(2400, dtype=np.float32), 24000)
    return eng


@pytest.fixture
def client(engine, tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "marvin.wav").write_bytes(b"x")
    config = {"server": {"port": 8005}, "generation": {}}
    return TestClient(create_app(engine, config, voice_paths=[voices]))


def test_tts_returns_wav_audio(client):
    r = client.post("/tts", json={
        "text": "Hello there.",
        "voice_mode": "predefined",
        "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_tts_forwards_the_four_flash_knobs(client, engine):
    client.post("/tts", json={
        "text": "Hello there.",
        "voice_mode": "predefined",
        "predefined_voice_id": "marvin.wav",
        "num_steps": 4,
        "n_cfm_timesteps": 1,
        "temperature": 0.7,
        "cfg_weight": 1.5,
    })
    _, kwargs = engine.synthesize.call_args
    assert kwargs["num_steps"] == 4
    assert kwargs["n_cfm_timesteps"] == 1
    assert kwargs["temperature"] == 0.7
    assert kwargs["cfg_scale"] == 1.5, "UI sends cfg_weight; Flash takes cfg_scale"


def test_tts_clone_mode_uses_reference_audio_filename(client, engine):
    client.post("/tts", json={
        "text": "Hello.",
        "voice_mode": "clone",
        "reference_audio_filename": "marvin.wav",
    })
    _, kwargs = engine.synthesize.call_args
    assert kwargs["voice"] == "marvin.wav"


def test_tts_rejects_empty_text(client):
    r = client.post("/tts", json={
        "text": "", "voice_mode": "predefined", "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 422


def test_tts_missing_voice_id_is_a_400(client):
    r = client.post("/tts", json={"text": "Hi.", "voice_mode": "predefined"})
    assert r.status_code == 400
    assert "predefined_voice_id" in r.json()["detail"]


def test_tts_unknown_voice_is_a_404_naming_paths(client, engine):
    engine.synthesize.side_effect = FileNotFoundError("reference voice 'nope.wav' not found. Searched: /tmp/voices")
    r = client.post("/tts", json={
        "text": "Hi.", "voice_mode": "predefined", "predefined_voice_id": "nope.wav",
    })
    assert r.status_code == 404
    assert "Searched" in r.json()["detail"]


def test_tts_oom_is_a_507_with_vram_detail(client, engine):
    engine.synthesize.side_effect = OutOfMemoryError(
        "ran out of VRAM during generation. VRAM 0.40 GB free of 6.14 GB total"
    )
    r = client.post("/tts", json={
        "text": "Hi.", "voice_mode": "predefined", "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 507
    assert "VRAM" in r.json()["detail"]


def test_tts_when_model_not_loaded_is_503(client, engine):
    engine.loaded = False
    r = client.post("/tts", json={
        "text": "Hi.", "voice_mode": "predefined", "predefined_voice_id": "marvin.wav",
    })
    assert r.status_code == 503


def test_save_and_reset_settings_round_trip(client):
    saved = client.post("/save_settings", json={"last_text": "remembered"})
    assert saved.status_code == 200
    assert client.post("/reset_settings").status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_server_tts.py -v`
Expected: FAIL — 404 on `/tts` (endpoint does not exist yet)

- [ ] **Step 3: Write the request model**

`poc-tts/poc_tts/models.py`:

```python
"""Request models for poc-tts.

Field names mirror the vendored CustomTTSRequest so the copied ui/script.js
payload works unchanged, plus the four Flash-specific knobs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FlashTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice_mode: Literal["predefined", "clone"] = "predefined"
    predefined_voice_id: Optional[str] = None
    reference_audio_filename: Optional[str] = None
    output_format: Literal["wav"] = Field(
        "wav", description="PoC serves WAV only; opus/mp3 are out of scope."
    )
    split_text: bool = True
    chunk_size: int = Field(120, ge=50, le=500)

    # Shared with the Turbo UI.
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    exaggeration: Optional[float] = Field(None, ge=0.0, le=2.0)
    cfg_weight: Optional[float] = Field(
        None, ge=0.0, le=5.0, description="Maps to Flash's cfg_scale."
    )

    # Flash-specific speed/quality knobs.
    num_steps: Optional[int] = Field(None, ge=1, le=32)
    n_cfm_timesteps: Optional[int] = Field(None, ge=1, le=8)
```

- [ ] **Step 4: Add the endpoints**

Add these imports to the top of `poc-tts/poc_tts/server.py`:

```python
import io
import wave

import numpy as np
from fastapi import HTTPException, Response

from poc_tts.engine_flash import OutOfMemoryError
from poc_tts.models import FlashTTSRequest
```

Add this helper at module level in `server.py`:

```python
def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 [-1, 1] as 16-bit PCM WAV."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
```

Add these endpoints inside `create_app`, before the `return app`:

```python
    settings_store: dict = {}

    @app.post("/tts")
    async def tts(request: FlashTTSRequest):
        if not engine.loaded:
            raise HTTPException(status_code=503, detail="Flash model is not loaded.")

        if request.voice_mode == "predefined":
            voice = request.predefined_voice_id
            if not voice:
                raise HTTPException(
                    status_code=400,
                    detail="predefined_voice_id is required when voice_mode is 'predefined'.",
                )
        else:
            voice = request.reference_audio_filename
            if not voice:
                raise HTTPException(
                    status_code=400,
                    detail="reference_audio_filename is required when voice_mode is 'clone'.",
                )

        try:
            audio, sample_rate = engine.synthesize(
                text=request.text,
                voice=voice,
                temperature=request.temperature,
                exaggeration=request.exaggeration,
                cfg_scale=request.cfg_weight,
                num_steps=request.num_steps,
                n_cfm_timesteps=request.n_cfm_timesteps,
                chunk_size=request.chunk_size,
                split_text=request.split_text,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OutOfMemoryError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(content=_wav_bytes(audio, sample_rate), media_type="audio/wav")

    @app.post("/save_settings")
    async def save_settings(payload: dict):
        settings_store.update(payload)
        return {"message": "Settings saved."}

    @app.post("/reset_settings")
    async def reset_settings():
        settings_store.clear()
        return {"message": "Settings reset."}
```

- [ ] **Step 5: Run to verify pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests -v`
Expected: 51 passed

- [ ] **Step 6: Commit**

```bash
git add poc-tts/poc_tts/models.py poc-tts/poc_tts/server.py poc-tts/tests/test_server_tts.py
git commit -m "feat(poc-tts): /tts generation and settings endpoints"
```

---

### Task 7: UI controls for the Flash knobs

**Files:**
- Modify: `poc-tts/ui/index.html`
- Modify: `poc-tts/ui/script.js`

**Interfaces:**
- Consumes: `POST /tts` accepting `num_steps` and `n_cfm_timesteps`.
- Produces: nothing consumed by later tasks.

No pytest here — this is browser-verified. `type: "flash"` already falls through `updateModelUI`'s turbo/multilingual branches, so exaggeration and CFG stay visible and the language selector stays hidden, both correct for Flash. The work is purely additive.

- [ ] **Step 1: Add the controls to index.html**

Find the block containing `id="exaggeration-group"` and insert immediately after its closing `</div>`:

```html
<div class="setting-group" id="flash-knobs-group">
    <label for="num-steps">Diffusion steps
        <span id="num-steps-value">10</span>
    </label>
    <input type="range" id="num-steps" min="1" max="16" step="1" value="10">
    <small>Lower is faster, at some cost to quality.</small>

    <label for="cfm-timesteps">Vocoder CFM steps
        <span id="cfm-timesteps-value">2</span>
    </label>
    <input type="range" id="cfm-timesteps" min="1" max="4" step="1" value="2">
    <small>The meanflow-distilled vocoder is tuned for 2.</small>
</div>
```

- [ ] **Step 2: Wire the value displays**

In `script.js`, near the other slider declarations around line 95, add:

```javascript
const numStepsSlider = document.getElementById('num-steps');
const numStepsValueDisplay = document.getElementById('num-steps-value');
const cfmTimestepsSlider = document.getElementById('cfm-timesteps');
const cfmTimestepsValueDisplay = document.getElementById('cfm-timesteps-value');
```

Alongside the existing slider listeners, add:

```javascript
if (numStepsSlider && numStepsValueDisplay) {
    numStepsSlider.addEventListener('input', () => {
        numStepsValueDisplay.textContent = numStepsSlider.value;
    });
}
if (cfmTimestepsSlider && cfmTimestepsValueDisplay) {
    cfmTimestepsSlider.addEventListener('input', () => {
        cfmTimestepsValueDisplay.textContent = cfmTimestepsSlider.value;
    });
}
```

- [ ] **Step 3: Include them in the /tts payload**

In the function that builds the `/tts` request body, add:

```javascript
num_steps: numStepsSlider ? parseInt(numStepsSlider.value, 10) : undefined,
n_cfm_timesteps: cfmTimestepsSlider ? parseInt(cfmTimestepsSlider.value, 10) : undefined,
```

- [ ] **Step 4: Show the group only for Flash**

Inside `updateModelUI`, after the existing `if (modelInfo.type === 'turbo')` block near line 374, add:

```javascript
const flashKnobsGroup = document.getElementById('flash-knobs-group');
if (flashKnobsGroup) {
    flashKnobsGroup.classList.toggle('hidden', modelInfo.type !== 'flash');
}
```

- [ ] **Step 5: Verify in the browser**

Run:
```bash
make poc-tts
```
Then open `http://127.0.0.1:8005`. Confirm: the page loads; the model status line reads `ChatterboxFlashTTS loaded on cuda`; `marvin.wav` appears in the voice list; the two new sliders are visible and their numbers track; the language selector is hidden; generating produces audible speech in the cloned voice.

Note that `drf_block_size` and `backend` are load-time arguments, not per-request ones — they stay in `config.yaml` and are swept by Task 8 rather than exposed as live UI controls. Changing them requires a server restart.

- [ ] **Step 6: Commit**

```bash
git add poc-tts/ui/index.html poc-tts/ui/script.js
git commit -m "feat(poc-tts): expose Flash diffusion and CFM step controls in the GUI"
```

---

### Task 8: Tuning sweep

**Files:**
- Create: `poc-tts/poc_tts/bench.py`
- Create: `poc-tts/tests/test_bench.py`

**Interfaces:**
- Consumes: `FlashEngine`, `load_config`, `voice_paths`.
- Produces: `sweep_configs(grid: dict) -> list[dict]`, `record_result(path: Path, row: dict) -> None`; appends to `poc-tts/reports/runs.jsonl`.

Uses the same three sentences as the recorded Turbo and CPU baselines. That is what makes the sweep comparable to `poc/reports/runs.jsonl` rather than a standalone curiosity.

- [ ] **Step 1: Write the failing tests**

`poc-tts/tests/test_bench.py`:

```python
import json

from poc_tts.bench import SENTENCES, record_result, sweep_configs


def test_sweep_is_the_cartesian_product():
    grid = {"num_steps": [4, 10], "n_cfm_timesteps": [1, 2]}
    configs = sweep_configs(grid)
    assert len(configs) == 4
    assert {"num_steps": 4, "n_cfm_timesteps": 1} in configs


def test_sweep_of_single_valued_axes_is_one_config():
    assert len(sweep_configs({"num_steps": [10]})) == 1


def test_sentences_match_the_recorded_baselines():
    """These three are what the Turbo CUDA, Flash CUDA, and Turbo CPU rows
    were measured on. Changing them invalidates the comparison."""
    assert [name for name, _ in SENTENCES] == ["short", "medium", "long"]
    assert len(dict(SENTENCES)["long"]) > 300


def test_record_result_appends_one_json_line(tmp_path):
    path = tmp_path / "runs.jsonl"
    record_result(path, {"rtf": 0.58, "backend": "torch"})
    record_result(path, {"rtf": 0.72, "backend": "torch"})
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1])["rtf"] == 0.72


def test_record_result_creates_parent_directory(tmp_path):
    path = tmp_path / "reports" / "runs.jsonl"
    record_result(path, {"rtf": 0.5})
    assert path.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc-tts && .venv/bin/python -m pytest tests/test_bench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc_tts.bench'`

- [ ] **Step 3: Write the bench**

`poc-tts/poc_tts/bench.py`:

```python
"""Sweep Chatterbox Flash tuning configurations and record RTF.

The three sentences are identical to those used for the Turbo CUDA, Flash
CUDA, and Turbo CPU baselines in the design spec, so results are directly
comparable rather than standalone.
"""

from __future__ import annotations

import itertools
import json
import platform
import socket
import time
from pathlib import Path

import torch

SENTENCES = [
    ("short", "Sure, the kitchen light is on."),
    ("medium", "I checked the calendar for tomorrow and you have three meetings, "
               "the first one starting at nine fifteen."),
    ("long", "Here is the summary you asked for. The build finished in about four "
             "minutes, all thirty two tests passed, and the only warning came from "
             "the audio device layer, which reported that the sample rate was "
             "renegotiated partway through the session. Nothing else looked out of "
             "the ordinary, so I would call that a clean run."),
]

GRID = {
    "drf_block_size": [16, 32],
    "num_steps": [4, 6, 10],
    "n_cfm_timesteps": [1, 2],
}

REPORTS = Path(__file__).resolve().parent.parent / "reports" / "runs.jsonl"


def sweep_configs(grid: dict) -> list[dict]:
    """Cartesian product of the grid axes."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def record_result(path: Path, row: dict) -> None:
    """Append one result as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    from poc_tts.config import load_config, voice_paths
    from poc_tts.engine_flash import FlashEngine

    config = load_config()
    paths = voice_paths(config)
    voice = config.get("bench", {}).get("voice", "marvin.wav")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    for combo in sweep_configs(GRID):
        engine_cfg = dict(config.get("engine", {}))
        engine_cfg["drf_block_size"] = combo["drf_block_size"]
        engine = FlashEngine(
            engine_cfg=engine_cfg,
            generation_cfg=config.get("generation", {}),
            voice_paths=paths,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        engine.load()
        engine.synthesize(text="Warming up the voice.", voice=voice)

        for name, text in SENTENCES:
            timings = []
            for _ in range(2):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                audio, sample_rate = engine.synthesize(
                    text=text,
                    voice=voice,
                    num_steps=combo["num_steps"],
                    n_cfm_timesteps=combo["n_cfm_timesteps"],
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)

            audio_s = len(audio) / sample_rate
            best = min(timings)
            row = {
                "ts": stamp,
                "host": socket.gethostname(),
                "machine": platform.machine(),
                "model": "chatterbox-flash",
                "device": engine.device,
                "dtype": str(engine.dtype).replace("torch.", ""),
                "backend": engine.backend,
                "config": combo,
                "sentence": name,
                "results": {
                    "audio_s": round(audio_s, 3),
                    "gen_s": round(best, 3),
                    "rtf": round(best / audio_s, 4),
                },
            }
            if torch.cuda.is_available():
                row["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
            record_result(REPORTS, row)
            print(f"{combo} {name}: RTF {row['results']['rtf']}", flush=True)

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"sweep complete -> {REPORTS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd poc-tts && .venv/bin/python -m pytest tests -v`
Expected: 56 passed

- [ ] **Step 5: Run the real sweep**

Free the GPU first — the RTX 2060 has 6 GB and the Turbo server alone holds 4.7 GB:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
Stop the Turbo server if it is running, then:
```bash
make poc-tts-bench
```
Expected: 12 configurations × 3 sentences = 36 rows appended to `poc-tts/reports/runs.jsonl`. The reference point is the recorded Flash default of RTF 0.579 on the long sentence; a faster config beats that number.

- [ ] **Step 6: Commit**

```bash
git add poc-tts/poc_tts/bench.py poc-tts/tests/test_bench.py poc-tts/reports/runs.jsonl
git commit -m "feat(poc-tts): Flash tuning sweep recording RTF and VRAM"
```

---

### Task 9: Local Makefile as the PoC entry point

**Files:**
- Create: `poc-tts/Makefile`
- Modify: `Makefile` (repo root — the four `poc-tts-*` targets added in Task 1)
- Modify: `poc-tts/README.md`

**Interfaces:**
- Consumes: `poc-tts/setup.sh` (Task 1), `poc_tts.server` (Task 5), `poc_tts.bench` (Task 8), `tests/` (Tasks 2-8).
- Produces: nothing consumed by later tasks — this is the final task.

`cd poc-tts && make` with no arguments must do everything needed to run the app: install the toolchain and dependencies if absent, then start the server. Repeat runs must skip the install work rather than redoing it.

The root Makefile's `poc-tts-*` targets are refactored to delegate here so the PoC's lifecycle has exactly one source of truth. Task 1 wrote them as direct `$(POC_TTS_PY)` invocations; this task replaces those bodies.

- [ ] **Step 1: Create the local Makefile**

`poc-tts/Makefile`:

```makefile
# poc-tts — Chatterbox Flash PoC.
# Bare `make` installs whatever is missing and starts the app on :8005.

.DEFAULT_GOAL := run

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp

.PHONY: run setup bench test clean help

run: setup  ## Install if needed, then serve the Flash GUI on http://127.0.0.1:8005
	@$(PY) -m poc_tts.server

setup: $(STAMP)  ## mise python 3.10, venv, deps, FlashInfer probe (idempotent)

# The stamp records a completed setup. It is newer than requirements.txt and
# mise.toml after a successful run, so repeat `make` invocations skip straight
# to run; editing either file makes setup rerun.
$(STAMP): requirements.txt mise.toml setup.sh
	@./setup.sh
	@touch $(STAMP)

bench: setup  ## Sweep Flash tuning configs -> reports/runs.jsonl
	@$(PY) -m poc_tts.bench

test: setup  ## GPU-free unit tests
	@$(PY) -m pytest tests -v

clean:  ## Remove the venv and caches. Leaves reports/ and the HF model cache.
	rm -rf $(VENV) .pytest_cache poc_tts/__pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
```

- [ ] **Step 2: Verify the default target installs and runs**

Run:
```bash
cd poc-tts && make clean && make
```
Expected: `setup.sh` runs (venv rebuilt, deps installed, probe written), then the server starts and logs `Uvicorn running on http://127.0.0.1:8005`. Stop it with Ctrl-C.

- [ ] **Step 3: Verify repeat runs skip the install**

Run:
```bash
cd poc-tts && make
```
Expected: no pip output at all — the stamp is newer than `requirements.txt`, `mise.toml`, and `setup.sh`, so `setup` is satisfied and make goes straight to starting the server. Stop it with Ctrl-C.

- [ ] **Step 4: Verify editing requirements retriggers setup**

Run:
```bash
cd poc-tts && touch requirements.txt && make setup
```
Expected: `setup.sh` runs again and completes.

- [ ] **Step 5: Verify the remaining targets**

Run:
```bash
cd poc-tts && make help && make test
```
Expected: `help` lists run, setup, bench, test, clean; `make test` reports the full suite passing.

- [ ] **Step 6: Refactor the root Makefile to delegate**

Replace the four `poc-tts-*` target bodies added in Task 1 with delegating versions, so the local Makefile is the only place the lifecycle is defined. Also delete the now-unused `POC_TTS_PY` variable.

```makefile
poc-tts-setup:  ## poc-tts: mise python 3.10, venv, deps, flashinfer probe (idempotent)
	@$(MAKE) -C poc-tts setup

poc-tts:    ## poc-tts: run the Chatterbox Flash server + GUI on :8005
	@$(MAKE) -C poc-tts run

poc-tts-bench:  ## poc-tts: sweep Flash tuning configs, append poc-tts/reports/runs.jsonl
	@$(MAKE) -C poc-tts bench

poc-tts-test:  ## poc-tts: GPU-free unit tests
	@$(MAKE) -C poc-tts test
```

Leave both `.PHONY` lines as Task 1 set them.

- [ ] **Step 7: Verify delegation works from the repo root**

Run:
```bash
cd /home/rolandknight/github.com/rolandknight/voice-chatbot && make poc-tts-test
```
Expected: the same full suite passes, driven through the local Makefile.

- [ ] **Step 8: Update the README**

In `poc-tts/README.md`, replace the command block with:

```markdown
    make              # install anything missing, then serve on :8005
    make test         # GPU-free unit tests
    make bench        # tuning sweep -> reports/runs.jsonl
    make clean        # drop the venv

Run from this directory. The repo-root `make poc-tts*` targets delegate here.
```

- [ ] **Step 9: Commit**

```bash
git add poc-tts/Makefile poc-tts/README.md Makefile
git commit -m "feat(poc-tts): local Makefile entry point, default target installs and runs"
```

---

## Self-review

**Spec coverage.** Layout and mise → Task 1. Engine and dtype resolution → Tasks 2 and 4. Backend resolution → Tasks 2 and 4. Reference voices with vendor-absent tolerance → Task 3. Eleven endpoints → Tasks 5 and 6. Knob mapping and the four new controls → Tasks 6 and 7. Bench → Task 8. Testing → Tasks 2, 3, 4, 5, 6, 8. Error handling: OOM → Tasks 4 and 6; bf16 guard → Task 2; missing wav → Tasks 3 and 6; flashinfer absent → Tasks 1 and 4.

**Deviation from spec, recorded deliberately.** The spec listed `drf_block_size` and `backend` as new UI selects. They are load-time constructor arguments, not per-request parameters, so exposing them as live controls would require a model reload per request. Task 7 keeps them in `config.yaml` and Task 8 sweeps them; `num_steps` and `n_cfm_timesteps`, which are genuinely per-request, remain live UI controls. Worth confirming this trade is acceptable.

**Type consistency.** `cfg_weight` (UI and `FlashTTSRequest`) maps to `cfg_scale` (`FlashEngine.synthesize` and the library) at exactly one place, `server.py`'s `/tts` handler, and Task 6 has a test asserting it. `voice` is the parameter name throughout the engine; `predefined_voice_id` / `reference_audio_filename` are collapsed to it in the handler. `create_app(engine, config, voice_paths)` has the same three-argument signature in Tasks 5 and 6. `model_info()` keys are defined in Task 4 and asserted in Task 5.
