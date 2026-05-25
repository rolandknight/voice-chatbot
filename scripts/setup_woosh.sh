#!/usr/bin/env bash
# Optional pre-stage step: clones the SonyResearch/Woosh foley model repo
# into vendor/ so the first ./run.sh doesn't have to. The actual uv-managed
# dependency install + model weight download (~3.4GB) happens inside
# scripts/start_woosh.sh on first launch.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/vendor"
SERVER_DIR="$VENDOR_DIR/woosh"
REPO_URL="https://github.com/SonyResearch/Woosh.git"

mkdir -p "$VENDOR_DIR"

if [[ ! -d "$SERVER_DIR/.git" ]]; then
  echo "Cloning Woosh into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
else
  echo "Woosh already cloned; pulling latest..."
  git -C "$SERVER_DIR" pull --ff-only || echo "(skipping pull, leaving as-is)"
fi

echo ""
echo "Repo ready at $SERVER_DIR."
echo "Next: ./scripts/start_woosh.sh (or just ./run.sh — it auto-starts when BABEL_SFX_ENABLED=1)."
echo "First launch installs deps via uv and downloads ~3.4GB of model weights (several minutes)."
