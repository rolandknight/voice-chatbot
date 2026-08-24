#!/usr/bin/env bash
# Bootstrap the poc-tts-streaming environment. Idempotent.
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

# aiortc pulls PyAV with its own ffmpeg/libopus. Prove it imports beside
# torch in this venv; a wheel mismatch here would otherwise surface as a
# confusing failure on the first /calls.
./.venv/bin/python - <<'PY'
import av, aiortc, torch
print(f"aiortc {aiortc.__version__}, av {av.__version__}, torch {torch.__version__}")
PY

echo "poc-tts-streaming setup done"
