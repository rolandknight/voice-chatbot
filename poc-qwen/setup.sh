#!/usr/bin/env bash
# Bootstrap the poc-qwen environment. Idempotent.
set -euo pipefail

POC_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$POC_DIR"

command -v mise >/dev/null 2>&1 || {
    echo "ERROR: mise is not installed. brew install mise" >&2
    exit 1
}

mise install

if [ ! -d .venv ]; then
    echo "Creating .venv with mise python 3.12 ..."
    mise exec -- python -m venv .venv
fi

./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt

mkdir -p reports
./.venv/bin/python - <<'PY' > reports/env_probe.json || true
import json, platform
out = {"macos": platform.mac_ver()[0], "python": platform.python_version()}
try:
    import mlx.core as mx
    out["mlx"] = mx.__version__
    try:
        out["metal"] = mx.device_info()
    except Exception as e:  # pragma: no cover
        out["metal_error"] = str(e)
except Exception as e:
    out["mlx_error"] = str(e)
try:
    import mlx_audio
    out["mlx_audio"] = __import__("importlib.metadata").metadata.version("mlx-audio")
except Exception as e:
    out["mlx_audio_error"] = str(e)
print(json.dumps(out, indent=2, default=str))
PY
echo "--- env probe ---"
cat reports/env_probe.json
echo "poc-qwen setup done"
