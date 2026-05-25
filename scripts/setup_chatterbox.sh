#!/usr/bin/env bash
# Optional pre-stage step: clones the Chatterbox-TTS-Server repo into
# vendor/ so the first ./run.sh doesn't have to. The actual dependency
# install + model download happens inside the upstream `start.py --cpu`
# launcher, which is invoked by scripts/start_chatterbox.sh (and by
# run.sh on demand).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/vendor"
SERVER_DIR="$VENDOR_DIR/chatterbox-tts-server"
REPO_URL="https://github.com/devnen/Chatterbox-TTS-Server.git"

mkdir -p "$VENDOR_DIR"

if [[ ! -d "$SERVER_DIR/.git" ]]; then
  echo "Cloning Chatterbox-TTS-Server into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
else
  echo "Chatterbox-TTS-Server already cloned; pulling latest..."
  git -C "$SERVER_DIR" pull --ff-only || echo "(skipping pull, leaving as-is)"
fi

# Remove a stale .venv from an older install path before upstream
# start.py rebuilds its own ./venv.
if [[ -d "$SERVER_DIR/.venv" && ! -d "$SERVER_DIR/venv" ]]; then
  echo "Removing stale .venv (upstream uses ./venv)..."
  rm -rf "$SERVER_DIR/.venv"
fi

echo ""
echo "Repo ready at $SERVER_DIR."
echo "Next: ./scripts/start_chatterbox.sh (or just ./run.sh — it auto-starts)."
echo "First launch installs deps and downloads the model (~GB-scale, several minutes)."
