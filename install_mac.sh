#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "== Pipecat Jabra Mac prototype installer =="

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: This was written for Apple Silicon Macs. Continuing anyway."
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Installing Xcode command line tools..."
  xcode-select --install || true
  echo "After Xcode tools finish installing, rerun this script."
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh, then rerun this script."
  exit 1
fi

echo "Installing system dependencies (Homebrew)..."
# mpv plays the BBC HLS radio streams targeted at the Jabra via CoreAudio.
# librespot is a headless Spotify Connect endpoint whose PCM we stream into the
# pipeline. The package list lives in the Makefile (BREW_PKGS).
brew update
make install-server-os

echo "Activating Hermit toolchain (provides python + pip)..."
# Hermit manages the Python binary; packages install into its environment —
# no separate virtualenv.
. bin/activate-hermit
python -m pip install --upgrade pip wheel setuptools

echo "Installing Python dependencies (requirements.txt)..."
make install-server

echo "Downloading openwakeword shared backbone + hey_jarvis model..."
# openwakeword ships the melspec/embedding/silero_vad backbone files via a
# CDN, not the wheel. WakeWordDetector tries to use them at startup, so we
# fetch them now — also downloads the bundled 'hey_jarvis_v0.1' model that
# the marvin persona is wired to in config.yaml.
python -c "import openwakeword.utils as u; u.download_models(['hey_jarvis_v0.1'])"

echo "Starting Ollama if needed..."
if ! pgrep -x ollama >/dev/null 2>&1; then
  brew services start ollama || true
  sleep 3
fi

echo "Pulling local LLMs..."
# Default. Gemma 4 26B MoE (~4B active per token, ~17 GB). Best tool-call
# reliability (85.5% tau2-bench) and fires tools without emitting
# chain-of-thought tokens first.
ollama pull gemma4:26b
# E4B fallback for low-RAM machines (~9.6 GB).
ollama pull gemma4:latest
# Smallest fallback (~2 GB) for tight RAM or fast TTFB experiments.
ollama pull qwen2.5:3b

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo ""
echo "Install complete."
echo ""
echo "Next:"
echo "  1) Plug in the Jabra USB speakerphone."
echo "  2) In macOS System Settings > Sound, set Jabra as input and output."
echo "  3) Run: ./run.sh --devices"
echo "  4) Put the Jabra input/output device indexes into config.yaml"
echo "     (audio.input_device_index / audio.output_device_index) if needed."
echo "  5) Run: ./run.sh"
echo "  Note: the first weather query will trigger a macOS Location Services permission prompt."
echo ""
echo "Configuration:"
echo "  Non-secret config lives in config.yaml (committed)."
echo "  Secrets (API keys) live in .env (gitignored; see .env.example)."
echo "  Dump the resolved config: python -m config.loader --print-effective"
echo ""
echo "Optional - Spotify:"
echo "  1) Create an app at https://developer.spotify.com/dashboard."
echo "  2) Put SPOTIPY_CLIENT_ID (and optionally SPOTIPY_CLIENT_SECRET) into .env."
echo "     The redirect URI in the dashboard must match skills.spotify.redirect_uri"
echo "     in config.yaml exactly."
echo "  3) Run: python scripts/spotify.py --bootstrap   (one-time OAuth)"
echo "  4) Run: python scripts/spotify.py --start-sink  (blocking),"
echo "     then open Spotify on your phone and pick 'Babel' from the Connect menu."
echo ""
echo "Optional - Chatterbox cloned voices:"
echo "  The default 'babel' persona uses Kokoro and needs no extra setup."
echo "  To enable additional cloned personas (e.g. 'marvin' in personas.yaml):"
echo "    1) Run: ./scripts/setup_chatterbox.sh   (one-time, ~5 min download)"
echo "    2) Drop reference WAVs (5-15s, mono) into voices/"
echo "    3) Run: ./scripts/start_chatterbox.sh   (leave running in another tab)"
echo "    4) Say 'switch to marvin', or set tts.default_persona: marvin in config.yaml."
