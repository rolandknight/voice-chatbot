#!/usr/bin/env bash
# Bootstrap + launch the Stable Audio Open FastAPI server.
#
# Idempotent: first run clones stable-audio-tools (for reference/docs),
# bootstraps uv if missing, creates vendor/stable-audio/.venv on Python
# 3.10, and pip-installs stable-audio-tools (+ fastapi + uvicorn +
# torchaudio) from PyPI. Then launches uvicorn against
# scripts/stable_audio_server.py. The model itself (~1.2GB) is
# downloaded from Hugging Face on the first /generate call by
# get_pretrained_model.
#
# We deliberately *don't* `uv sync` against the cloned repo's
# pyproject: its `[all]` extra fails to cross-resolve under uv on
# non-current platforms (depends on stable-audio-tools-dev[train] for
# linux-x86_64, which has no versions). Installing the PyPI package
# directly into a clean venv sidesteps that.
#
# stable-audio-tools pins its own torch/torchaudio combo which would
# conflict with the project's main .venv (Whisper-MLX, Chatterbox
# client). Keeping it in its own venv under vendor/stable-audio/ is
# the same isolation pattern as vendor/woosh/ and
# vendor/chatterbox-tts-server/.
#
# The model is gated on Hugging Face: accept the terms at
# https://huggingface.co/stabilityai/stable-audio-open-1.0 and set
# HF_TOKEN in .env (or `huggingface-cli login`).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/vendor"
SERVER_DIR="$VENDOR_DIR/stable-audio"
REPO_URL="https://github.com/Stability-AI/stable-audio-tools.git"
STABLE_AUDIO_PORT="${STABLE_AUDIO_PORT:-8006}"
# Marker file written after a successful pip install. Lets us tell a
# half-installed venv (from a failed run) apart from a complete one.
DEPS_MARKER="$SERVER_DIR/.venv/.deps-installed"

if [[ ! -d "$SERVER_DIR/.git" ]]; then
  mkdir -p "$VENDOR_DIR"
  echo "Cloning stable-audio-tools into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
fi

# Ensure uv is on PATH (only needed for venv creation).
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (one-time) via https://astral.sh/uv/install.sh ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# If venv exists but deps weren't installed (previous run failed), wipe
# and start over so we don't ship a half-broken environment.
if [[ -d "$SERVER_DIR/.venv" && ! -f "$DEPS_MARKER" ]]; then
  echo "Removing incomplete venv from a previous failed install..."
  rm -rf "$SERVER_DIR/.venv"
fi

if [[ ! -d "$SERVER_DIR/.venv" ]]; then
  echo "First Stable Audio launch — creating Python 3.10 venv and installing deps..."
  echo "(downloads ~2-3 GB of torch/torchaudio wheels, can take several minutes)"
  uv venv --python 3.10 "$SERVER_DIR/.venv"
  # Install stable-audio-tools from PyPI (avoids the cloned repo's
  # broken [all] extra). Extras we have to add ourselves:
  #   fastapi/uvicorn — needed by our server wrapper.
  #   pywavelets>=1.6 — older pin in stable-audio-tools' deps was
  #     compiled against numpy 1.x and crashes at import on numpy 2.x
  #     ("dtype size changed" ABI mismatch).
  #   pytorch-lightning — imported by stable_audio_tools.models.lora
  #     but not declared as a runtime dep of the PyPI package.
  #   soundfile — torchaudio 2.x ships no audio backends by default;
  #     soundfile wraps libsndfile so torchaudio.save can write FLAC.
  uv pip install --python "$SERVER_DIR/.venv/bin/python" \
    stable-audio-tools fastapi uvicorn \
    'pywavelets>=1.6' pytorch-lightning soundfile
  touch "$DEPS_MARKER"
fi

# PyTorch ops without an MPS kernel fall back to CPU instead of raising.
# Same pattern as scripts/start_woosh.sh and scripts/start_chatterbox.sh.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Project's .env often sets HF_HUB_OFFLINE=1 (faster boots once other
# models are cached). That blocks the first SAO download. Unset it here
# so huggingface_hub can fetch model_config.json + the ~1.2GB weights;
# subsequent boots load from the HF cache without hitting the network.
unset HF_HUB_OFFLINE

# Run scripts/stable_audio_server.py through the vendored venv. Running
# from the project's scripts/ dir keeps the module name bare.
cd "$PROJECT_DIR/scripts"

echo "Launching Stable Audio Open FastAPI server on 127.0.0.1:$STABLE_AUDIO_PORT ..."
exec "$SERVER_DIR/.venv/bin/uvicorn" stable_audio_server:app \
  --host 127.0.0.1 --port "$STABLE_AUDIO_PORT"
