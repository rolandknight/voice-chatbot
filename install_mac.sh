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

echo "Installing system dependencies..."
brew update
# mpv plays the BBC HLS radio streams targeted at the Jabra via CoreAudio.
# librespot is a headless Spotify Connect endpoint; we pipe its PCM into mpv
# so Spotify playback can target the Jabra the same way radio does.
brew install python@3.12 portaudio ffmpeg mpv librespot git cmake pkg-config ollama corelocationcli

echo "Creating Python virtualenv..."
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "Installing Python dependencies..."
# Extras used:
# local       -> PyAudio local audio transport
# mlx-whisper -> Apple Silicon optimized Whisper STT
# kokoro      -> local Kokoro ONNX TTS (kept for the babel persona)
# openai      -> OpenAI-compatible TTS client used to talk to the local
#                Chatterbox-TTS-Server for the cloned chatterbox personas
# ollama      -> local LLM service via Ollama's OpenAI-compatible API
# anthropic   -> Claude routing for the secondary "hey claude" wake phrase
# spotipy     -> Spotify Web API client used to control playback on the
#                librespot Connect device
python -m pip install \
  "pipecat-ai[local,mlx-whisper,kokoro,openai,ollama,silero,anthropic]" \
  python-dotenv loguru pyaudio pyyaml websockets yt-dlp spotipy \
  "pydantic>=2" "pydantic-settings>=2"

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
echo "  Dump the resolved config: .venv/bin/python -m config.loader --print-effective"
echo ""
echo "Optional - Spotify:"
echo "  1) Create an app at https://developer.spotify.com/dashboard."
echo "  2) Put SPOTIPY_CLIENT_ID (and optionally SPOTIPY_CLIENT_SECRET) into .env."
echo "     The redirect URI in the dashboard must match skills.spotify.redirect_uri"
echo "     in config.yaml exactly."
echo "  3) Run: .venv/bin/python scripts/spotify.py --bootstrap   (one-time OAuth)"
echo "  4) Run: .venv/bin/python scripts/spotify.py --start-sink  (blocking),"
echo "     then open Spotify on your phone and pick 'Babel' from the Connect menu."
echo ""
echo "Optional - Chatterbox cloned voices:"
echo "  The default 'babel' persona uses Kokoro and needs no extra setup."
echo "  To enable additional cloned personas (e.g. 'marvin' in personas.yaml):"
echo "    1) Run: ./scripts/setup_chatterbox.sh   (one-time, ~5 min download)"
echo "    2) Drop reference WAVs (5-15s, mono) into voices/"
echo "    3) Run: ./scripts/start_chatterbox.sh   (leave running in another tab)"
echo "    4) Say 'switch to marvin', or set tts.default_persona: marvin in config.yaml."
