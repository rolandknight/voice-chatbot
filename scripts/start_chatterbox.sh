#!/usr/bin/env bash
# Bootstrap + launch the Chatterbox-TTS-Server using the upstream
# `start.py` cross-platform launcher. start.py is idempotent: first run
# creates venv/, installs deps (including the chatterbox-tts package
# that needs --no-deps), and downloads the model on the first server
# start. Subsequent runs just launch.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/vendor"
SERVER_DIR="$VENDOR_DIR/chatterbox-tts-server"
REPO_URL="https://github.com/devnen/Chatterbox-TTS-Server.git"

if [[ ! -d "$SERVER_DIR/.git" ]]; then
  mkdir -p "$VENDOR_DIR"
  echo "Cloning Chatterbox-TTS-Server into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
fi

cd "$SERVER_DIR"

# Remove a previous half-built .venv from an earlier (broken) install
# path — upstream start.py uses ./venv, not ./.venv, and the two can
# confuse newcomers if both exist.
if [[ -d .venv && ! -d venv ]]; then
  echo "Removing stale .venv (upstream uses ./venv)..."
  rm -rf .venv
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Suppress the upstream server's automatic browser open. Python's
# webbrowser module reads BROWSER as the command to "launch"; pointing
# it at /usr/bin/true turns webbrowser.open() into a successful no-op,
# so no Safari/Chrome window appears every time the server boots.
export BROWSER=true

# Chatterbox's torchaudio Resample step crashes on Apple's MPS backend
# in PyTorch 2.5.x: torch.nn.functional.conv1d raises
# `NotImplementedError: Output channels > 65536` for the inner resample
# kernel (despite the kernel having only 2 output channels — it's a
# misleading MPS error). PYTORCH_ENABLE_MPS_FALLBACK=1 is *supposed* to
# divert that op to CPU but isn't honored for this code path in 2.5.x.
# The reliable workaround is to force the whole TTS engine to CPU.
# On an M4 Max the CPU path is still fast enough for interactive use;
# revisit (flip back to mps or auto) when PyTorch ships the fix.
export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_FILE="config.yaml"
if [[ -f "$CONFIG_FILE" ]]; then
  if grep -qE '^[[:space:]]+device:[[:space:]]+(auto|mps)[[:space:]]*$' "$CONFIG_FILE"; then
    # In-place sed compatible with both macOS BSD sed and GNU sed.
    sed -i.bak -E 's/^([[:space:]]+)device:[[:space:]]+(auto|mps)[[:space:]]*$/\1device: cpu/' "$CONFIG_FILE"
    rm -f "$CONFIG_FILE.bak"
    echo "Forced tts_engine.device: cpu in $CONFIG_FILE (MPS path broken on PyTorch 2.5.x)"
  fi
fi

exec "$PYTHON_BIN" start.py --cpu
