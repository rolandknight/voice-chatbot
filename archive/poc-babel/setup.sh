#!/usr/bin/env bash
# Bootstrap poc-babel: mise python 3.10, venv, CPU-only deps. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

command -v mise >/dev/null 2>&1 || { echo "ERROR: mise is not installed. See https://mise.jdx.dev" >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg is required for mp3 output" >&2; exit 1; }

mise install
[ -d .venv ] || mise exec -- python -m venv .venv
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt

# Guard: kokoro-onnx must have pulled plain onnxruntime, never the GPU build.
if ./.venv/bin/python -m pip show -q onnxruntime-gpu 2>/dev/null; then
    echo "ERROR: onnxruntime-gpu is installed; this poc is CPU only" >&2; exit 1
fi
echo "poc-babel setup done"
