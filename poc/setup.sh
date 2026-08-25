#!/usr/bin/env bash
# Bootstrap the FlowCat PoC Python side + models. Idempotent.
# rust/cargo also come from mise (mise.toml); recipes use `mise exec -- cargo`.
set -euo pipefail

POC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$POC_DIR/.." && pwd)"
cd "$POC_DIR"

command -v mise >/dev/null 2>&1 || {
    echo "ERROR: mise is not installed. brew install mise" >&2
    exit 1
}

[ -f .env ] || {
    echo "ERROR: poc/.env missing — cp .env.example .env and edit (see README.md)" >&2
    exit 1
}
set -a; . ./.env; set +a

mise install

if [ ! -d .venv ]; then
    echo "Creating .venv with mise python 3.12 ..."
    mise exec -- python -m venv .venv
fi

./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt

mkdir -p models logs reports
[ -s models/silero_vad.onnx ] || curl -sL -o models/silero_vad.onnx \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
[ -s models/ggml-base.en.bin ] || curl -sL -o models/ggml-base.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

case "${POC_STT_BACKEND:-whisper}" in
    moonshine) "$REPO_DIR/scripts/setup_moonshine.sh" ;;
    nemotron | nvidia) "$REPO_DIR/scripts/setup_nemotron.sh" ;;
esac

./.venv/bin/python -m harness.make_fixtures

./.venv/bin/python - <<'PY' > reports/env_probe.json || true
import json, platform, importlib.metadata as md
out = {"macos": platform.mac_ver()[0], "python": platform.python_version(), "arch": platform.machine()}
for pkg in ("aiortc", "kokoro-onnx", "faster-whisper", "fastapi"):
    try:
        out[pkg] = md.version(pkg)
    except Exception as e:
        out[pkg + "_error"] = str(e)
print(json.dumps(out, indent=2, default=str))
PY
echo "--- env probe ---"
cat reports/env_probe.json
echo "poc setup done"
