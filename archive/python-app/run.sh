#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f bin/activate-hermit ]]; then
  echo "Missing Hermit (bin/activate-hermit). Run ./install_mac.sh first."
  exit 1
fi

# Hermit provides python + the installed packages on PATH.
. bin/activate-hermit

# Pull the handful of config values that this script (sidecar launch
# gating) needs, as shell-quoted exports. config/shell.py validates
# config.yaml + .env via Pydantic, so a malformed config fails here
# rather than mid-app boot.
eval "$(python -m config.shell)"

if [[ "${1:-}" == "--devices" ]]; then
  python scripts/list_audio_devices.py
  exit 0
fi

if ! pgrep -x ollama >/dev/null 2>&1; then
  echo "Starting Ollama..."
  brew services start ollama || true
  sleep 3
fi

# Chatterbox-TTS-Server: started only when personas.yaml declares at
# least one chatterbox-backed persona. Mirrors the Ollama block above —
# checks if it's reachable, launches it in the background if not, waits
# for /v1/models to respond before handing off to app.py.
PERSONAS_FILE="${PERSONAS_CONFIG:-personas.yaml}"
CHATTERBOX_BASE_URL="${CHATTERBOX_BASE_URL:-http://127.0.0.1:8004/v1}"
CHATTERBOX_DIR="$PROJECT_DIR/vendor/chatterbox-tts-server"
CHATTERBOX_LOG="$PROJECT_DIR/vendor/chatterbox.log"
CHATTERBOX_PID_FILE="$PROJECT_DIR/vendor/chatterbox.pid"

needs_chatterbox=0
if [[ -f "$PERSONAS_FILE" ]] && grep -qE '^[[:space:]]*backend:[[:space:]]*chatterbox[[:space:]]*$' "$PERSONAS_FILE"; then
  needs_chatterbox=1
fi

if [[ $needs_chatterbox -eq 1 ]]; then
  if curl -sS -m 2 "$CHATTERBOX_BASE_URL/models" >/dev/null 2>&1; then
    echo "Chatterbox-TTS-Server already running at $CHATTERBOX_BASE_URL."
  else
    first_run=0
    if [[ ! -d "$CHATTERBOX_DIR/venv" ]]; then
      first_run=1
      echo ""
      echo "First Chatterbox launch — installing deps and (on first server"
      echo "start) downloading the model. This can take several minutes."
      echo ""
    fi
    echo "Starting Chatterbox-TTS-Server in the background (logs: $CHATTERBOX_LOG)..."
    mkdir -p "$(dirname "$CHATTERBOX_LOG")"
    # nohup so it survives the parent shell exit; PID written for tooling.
    # start_chatterbox.sh delegates to the upstream `python start.py
    # --cpu` launcher, which handles cloning, deps, and the
    # chatterbox-tts --no-deps install on first run.
    nohup "$PROJECT_DIR/scripts/start_chatterbox.sh" >"$CHATTERBOX_LOG" 2>&1 &
    echo $! > "$CHATTERBOX_PID_FILE"
    # First run = clone + pip install + multi-GB model download. Give it
    # up to 30 minutes. Subsequent launches return in seconds.
    if [[ $first_run -eq 1 ]]; then
      deadline=$(( $(date +%s) + 1800 ))
    else
      deadline=$(( $(date +%s) + 300 ))
    fi
    until curl -sS -m 2 "$CHATTERBOX_BASE_URL/models" >/dev/null 2>&1; do
      if [[ $(date +%s) -ge $deadline ]]; then
        echo "Chatterbox didn't come up before deadline. Check $CHATTERBOX_LOG." >&2
        exit 1
      fi
      if ! kill -0 "$(cat "$CHATTERBOX_PID_FILE")" 2>/dev/null; then
        echo "Chatterbox launcher exited. Last log lines:" >&2
        tail -n 40 "$CHATTERBOX_LOG" >&2 || true
        exit 1
      fi
      sleep 3
    done
    echo "Chatterbox-TTS-Server is up."
  fi
fi

# Woosh foley server: started only when BABEL_SFX_ENABLED=1. Heavy:
# vendor/woosh/.venv (~2GB), checkpoints (~3.4GB), first-run install +
# weight download can take 10–30 minutes. Mirrors the Chatterbox block
# above — check if reachable, launch in background otherwise, wait for
# /docs to respond before handing off to app.py.
need_sfx="$(printf '%s' "${BABEL_SFX_ENABLED:-0}" | tr '[:upper:]' '[:lower:]')"
if [[ "$need_sfx" == "1" || "$need_sfx" == "true" || "$need_sfx" == "yes" ]]; then
  WOOSH_PORT="${WOOSH_PORT:-8005}"
  WOOSH_URL="${WOOSH_URL:-http://127.0.0.1:$WOOSH_PORT}"
  WOOSH_DIR="$PROJECT_DIR/vendor/woosh"
  WOOSH_LOG="$PROJECT_DIR/vendor/woosh.log"
  WOOSH_PID_FILE="$PROJECT_DIR/vendor/woosh.pid"
  export WOOSH_PORT

  if curl -sS -m 2 "$WOOSH_URL/docs" >/dev/null 2>&1; then
    echo "Woosh foley server already running at $WOOSH_URL."
  else
    first_run=0
    # First-run signal: missing .venv OR missing any of the three
    # checkpoint weight files the DFlow API needs. Gate on the .safetensors
    # itself (not the dir) because `git clone` ships a config.yaml in each
    # checkpoints/<MODEL>/ as a placeholder.
    if [[ ! -d "$WOOSH_DIR/.venv" \
       || ! -f "$WOOSH_DIR/checkpoints/Woosh-DFlow/weights.safetensors" \
       || ! -f "$WOOSH_DIR/checkpoints/Woosh-AE/weights.safetensors" \
       || ! -f "$WOOSH_DIR/checkpoints/TextConditionerA/weights.safetensors" ]]; then
      first_run=1
      echo ""
      echo "First Woosh launch — cloning, installing uv deps, and downloading"
      echo "~3.4GB of model weights. This can take 10–30 minutes."
      echo ""
    fi
    echo "Starting Woosh foley server in the background (logs: $WOOSH_LOG)..."
    mkdir -p "$(dirname "$WOOSH_LOG")"
    nohup "$PROJECT_DIR/scripts/start_woosh.sh" >"$WOOSH_LOG" 2>&1 &
    echo $! > "$WOOSH_PID_FILE"
    # First run = clone + uv sync + 3.4GB download. Allow 45 minutes.
    # Subsequent launches load weights from disk (~30–60s on M-series).
    if [[ $first_run -eq 1 ]]; then
      deadline=$(( $(date +%s) + 2700 ))
    else
      deadline=$(( $(date +%s) + 180 ))
    fi
    until curl -sS -m 2 "$WOOSH_URL/docs" >/dev/null 2>&1; do
      if [[ $(date +%s) -ge $deadline ]]; then
        echo "Woosh didn't come up before deadline. Check $WOOSH_LOG." >&2
        exit 1
      fi
      if ! kill -0 "$(cat "$WOOSH_PID_FILE")" 2>/dev/null; then
        echo "Woosh launcher exited. Last log lines:" >&2
        tail -n 40 "$WOOSH_LOG" >&2 || true
        exit 1
      fi
      sleep 3
    done
    echo "Woosh foley server is up at $WOOSH_URL."
  fi
fi

# Stable Audio Open server: started only when BABEL_SAO_ENABLED=1. Same
# isolation pattern as the Woosh block above. ~1.2GB of model weights are
# pulled from Hugging Face on the first /generate call (not on launch),
# so first-run readiness is dominated by the uv sync deps install.
need_sao="$(printf '%s' "${BABEL_SAO_ENABLED:-0}" | tr '[:upper:]' '[:lower:]')"
if [[ "$need_sao" == "1" || "$need_sao" == "true" || "$need_sao" == "yes" ]]; then
  STABLE_AUDIO_PORT="${STABLE_AUDIO_PORT:-8006}"
  STABLE_AUDIO_URL="${STABLE_AUDIO_URL:-http://127.0.0.1:$STABLE_AUDIO_PORT}"
  SAO_DIR="$PROJECT_DIR/vendor/stable-audio"
  SAO_LOG="$PROJECT_DIR/vendor/stable-audio.log"
  SAO_PID_FILE="$PROJECT_DIR/vendor/stable-audio.pid"
  export STABLE_AUDIO_PORT

  if curl -sS -m 2 "$STABLE_AUDIO_URL/docs" >/dev/null 2>&1; then
    echo "Stable Audio Open server already running at $STABLE_AUDIO_URL."
  else
    first_run=0
    # First-run signal: missing .venv. Model weights live in the HF cache,
    # not in vendor/stable-audio/, so we don't gate on weight files.
    if [[ ! -d "$SAO_DIR/.venv" ]]; then
      first_run=1
      echo ""
      echo "First Stable Audio Open launch — cloning stable-audio-tools and"
      echo "installing uv deps. The ~1.2GB model is downloaded from"
      echo "Hugging Face on the first /generate call (you must accept the"
      echo "terms at https://huggingface.co/stabilityai/stable-audio-open-1.0"
      echo "and set HF_TOKEN in .env)."
      echo ""
    fi
    echo "Starting Stable Audio Open server in the background (logs: $SAO_LOG)..."
    mkdir -p "$(dirname "$SAO_LOG")"
    nohup "$PROJECT_DIR/scripts/start_stable_audio.sh" >"$SAO_LOG" 2>&1 &
    echo $! > "$SAO_PID_FILE"
    # First run = clone + uv sync. Allow 15 minutes. Subsequent launches
    # return in seconds.
    if [[ $first_run -eq 1 ]]; then
      deadline=$(( $(date +%s) + 900 ))
    else
      deadline=$(( $(date +%s) + 180 ))
    fi
    until curl -sS -m 2 "$STABLE_AUDIO_URL/docs" >/dev/null 2>&1; do
      if [[ $(date +%s) -ge $deadline ]]; then
        echo "Stable Audio Open didn't come up before deadline. Check $SAO_LOG." >&2
        exit 1
      fi
      if ! kill -0 "$(cat "$SAO_PID_FILE")" 2>/dev/null; then
        echo "Stable Audio Open launcher exited. Last log lines:" >&2
        tail -n 40 "$SAO_LOG" >&2 || true
        exit 1
      fi
      sleep 3
    done
    echo "Stable Audio Open server is up at $STABLE_AUDIO_URL."
  fi
fi

export PYTHONUNBUFFERED=1
# OLLAMA_KEEP_ALIVE is exported by `python -m config.shell` above (sourced
# from llm.ollama_keep_alive in config.yaml; default "-1" pins the model).
python app.py
