#!/usr/bin/env bash
# Bootstrap the qwen-tts Python half: mise python 3.12, .venv, deps, and the
# qwen_tts package installed editable. Idempotent. The server build links
# against .venv/bin/python (PYO3_PYTHON, set by the root Makefile).
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CRATE_DIR"

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
./.venv/bin/python -m pip install -q -e python

./.venv/bin/python - <<'PY'
import importlib.metadata as md, platform
import mlx.core as mx
print(f"python {platform.python_version()}  mlx {mx.__version__}  mlx-audio {md.version('mlx-audio')}  metal {mx.device_info().get('device_name', '?')}")
PY
echo "qwen-tts setup done: $CRATE_DIR/.venv"
