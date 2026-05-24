#!/usr/bin/env bash
# Bootstrap + launch the Woosh foley model's FastAPI server.
#
# Idempotent: first run clones the repo, bootstraps uv if missing, runs
# `uv sync` to build vendor/woosh/.venv with torch==2.8.0 and friends,
# downloads ~3.4GB of model checkpoints from the v1.0.0 GitHub release,
# then launches uvicorn. Subsequent runs skip straight to uvicorn.
#
# Woosh pins torch==2.8.0 which would conflict with the project's main
# .venv (Whisper-MLX, Chatterbox client). Keeping it in its own uv venv
# under vendor/woosh/ is the same isolation pattern as
# vendor/chatterbox-tts-server/.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/vendor"
SERVER_DIR="$VENDOR_DIR/woosh"
REPO_URL="https://github.com/SonyResearch/Woosh.git"
WOOSH_PORT="${WOOSH_PORT:-8005}"

# Three checkpoint dirs the DFlow API server needs at runtime. Sizes are
# the compressed-zip sizes from the v1.0.0 release; uncompressed weights
# are similar.
RELEASE_TAG="v1.0.0"
RELEASE_URL="https://github.com/SonyResearch/Woosh/releases/download/${RELEASE_TAG}"
REQUIRED_CHECKPOINTS=("Woosh-DFlow" "Woosh-AE" "TextConditionerA")

if [[ ! -d "$SERVER_DIR/.git" ]]; then
  mkdir -p "$VENDOR_DIR"
  echo "Cloning Woosh into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
fi

cd "$SERVER_DIR"

# Ensure uv is on PATH. Woosh's pyproject.toml is uv-native (defines
# tool.uv.sources, dependency-groups, extras) — pip alone won't honor it.
if ! command -v uv >/dev/null 2>&1; then
  # The official installer drops uv at ~/.local/bin/uv with no sudo
  # needed and no global Python pollution.
  echo "Installing uv (one-time) via https://astral.sh/uv/install.sh ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Build vendor/woosh/.venv if missing. --extra cpu picks CPU wheels; the
# inference code separately detects MPS at runtime (see
# vendor/woosh/test_Woosh-Flow.py), so this still uses the GPU on Mac
# where supported. --group api pulls in fastapi+uvicorn for the server.
if [[ ! -d "$SERVER_DIR/.venv" ]]; then
  echo "First Woosh launch — running uv sync (this may take several minutes)..."
  uv sync --extra cpu --group api
fi

# Download model weights on first run. The release assets are public
# (CC-BY-NC), so plain curl works without gh-cli auth.
#
# Each release zip is laid out as `checkpoints/<MODEL>/weights.safetensors`
# + `checkpoints/<MODEL>/config.yaml`, so we unzip at the *repo root*
# (not under checkpoints/) to avoid double-nesting.
#
# We can't gate on "directory exists" — `git clone` ships a `config.yaml`
# placeholder in every `checkpoints/<MODEL>/` so the dirs always appear
# present. Gate on the actual weight file instead.
mkdir -p checkpoints
for ckpt in "${REQUIRED_CHECKPOINTS[@]}"; do
  if [[ -f "checkpoints/$ckpt/weights.safetensors" \
     || -f "checkpoints/$ckpt/weights.pt" ]]; then
    continue
  fi
  zip_path="checkpoints/${ckpt}.zip"
  if [[ ! -f "$zip_path" ]]; then
    echo "Downloading $ckpt from $RELEASE_TAG (this is large)..."
    curl -L --fail --retry 3 -o "$zip_path" "$RELEASE_URL/${ckpt}.zip"
  fi
  echo "Extracting $ckpt ..."
  unzip -q -o "$zip_path"
  rm -f "$zip_path"
done

# PyTorch ops that don't have an MPS kernel fall back to CPU instead of
# raising. Same pattern as scripts/start_chatterbox.sh.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Launching Woosh FastAPI server on 127.0.0.1:$WOOSH_PORT ..."
exec uv run uvicorn api.api_server:app --host 127.0.0.1 --port "$WOOSH_PORT"
