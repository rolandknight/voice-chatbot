#!/usr/bin/env bash
# Optional pre-stage step: clones the Stability-AI/stable-audio-tools repo
# into vendor/stable-audio/ so the first ./run.sh doesn't have to. The
# actual uv-managed dependency install + model weight download (~1.2GB)
# happens inside scripts/start_stable_audio.sh on first launch.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/vendor"
SERVER_DIR="$VENDOR_DIR/stable-audio"
REPO_URL="https://github.com/Stability-AI/stable-audio-tools.git"

mkdir -p "$VENDOR_DIR"

if [[ ! -d "$SERVER_DIR/.git" ]]; then
  echo "Cloning stable-audio-tools into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
else
  echo "stable-audio-tools already cloned; pulling latest..."
  git -C "$SERVER_DIR" pull --ff-only || echo "(skipping pull, leaving as-is)"
fi

echo ""
echo "Repo ready at $SERVER_DIR."
echo "Next: ./scripts/start_stable_audio.sh (or just ./run.sh — it auto-starts when BABEL_SAO_ENABLED=1)."
echo "First launch installs deps via uv and downloads ~1.2GB of Stable Audio Open 1.0 weights (several minutes)."
echo "Note: the model is gated on Hugging Face — accept the terms at"
echo "https://huggingface.co/stabilityai/stable-audio-open-1.0 and set HF_TOKEN in .env."
