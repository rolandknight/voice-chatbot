#!/usr/bin/env bash
# Start Chatterbox-TTS-Server on macOS or Linux without destroying an existing
# environment. The launcher reuses both current `venv/` and legacy `.venv/`
# installs, including the older capitalized vendor directory.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_chatterbox_common.sh
. "$PROJECT_DIR/scripts/_chatterbox_common.sh"

REPO_URL="https://github.com/devnen/Chatterbox-TTS-Server.git"
PLATFORM="${CHATTERBOX_PLATFORM_OVERRIDE:-$(uname -s)}"
MODE="${1:-start}"
SERVER_DIR="$(chatterbox_resolve_server_dir "$PROJECT_DIR")"
REFERENCE_SOURCE="${CHATTERBOX_REFERENCE_SOURCE:-$PROJECT_DIR/voices/look-at-this-door-all-the-doors-in-this-spacecraft-have-a-cheerful-and-sunny-disposition-it-is-their-pleasure-to-open-for-you-and-their-satisfaction-to-close-again-with-the-knowledge-of-a-job-well-done.mp3}"
REFERENCE_NAME="${CHATTERBOX_REFERENCE_NAME:-marvin.wav}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

case "$MODE" in
start | --doctor) ;;
*) fail "usage: $0 [--doctor]" ;;
esac

case "$PLATFORM" in
Darwin | Linux) ;;
*) fail "unsupported platform '$PLATFORM' (supported: Darwin, Linux)" ;;
esac

detect_install_type() {
    local requested="${CHATTERBOX_INSTALL_TYPE:-auto}"

    case "$requested" in
    auto)
        if [ "$PLATFORM" = "Linux" ] \
            && command -v nvidia-smi >/dev/null 2>&1 \
            && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
            INSTALL_TYPE="nvidia"
            DEVICE="cuda"
            DEVICE_REASON="Linux NVIDIA GPU detected"
        else
            INSTALL_TYPE="cpu"
            DEVICE="cpu"
            if [ "$PLATFORM" = "Darwin" ]; then
                DEVICE_REASON="macOS safe default (Chatterbox MPS resampling is unreliable)"
            else
                DEVICE_REASON="no NVIDIA GPU detected"
            fi
        fi
        ;;
    cpu)
        INSTALL_TYPE="cpu"
        DEVICE="cpu"
        DEVICE_REASON="explicit CHATTERBOX_INSTALL_TYPE=cpu"
        ;;
    nvidia | nvidia-cu128)
        [ "$PLATFORM" = "Linux" ] || fail "$requested is supported only on Linux"
        command -v nvidia-smi >/dev/null 2>&1 || fail "$requested requires a working NVIDIA driver (nvidia-smi)"
        INSTALL_TYPE="$requested"
        DEVICE="cuda"
        DEVICE_REASON="explicit CHATTERBOX_INSTALL_TYPE=$requested"
        ;;
    *)
        fail "invalid CHATTERBOX_INSTALL_TYPE '$requested' (expected auto, cpu, nvidia, or nvidia-cu128)"
        ;;
    esac

    case "${CHATTERBOX_DEVICE:-auto}" in
    auto) ;;
    cpu)
        DEVICE="cpu"
        DEVICE_REASON="explicit CHATTERBOX_DEVICE=cpu"
        ;;
    cuda)
        [ "$PLATFORM" = "Linux" ] || fail "CHATTERBOX_DEVICE=cuda is supported only on Linux"
        DEVICE="cuda"
        DEVICE_REASON="explicit CHATTERBOX_DEVICE=cuda"
        ;;
    mps)
        [ "$PLATFORM" = "Darwin" ] || fail "CHATTERBOX_DEVICE=mps is supported only on macOS"
        DEVICE="mps"
        DEVICE_REASON="explicit CHATTERBOX_DEVICE=mps"
        ;;
    *) fail "invalid CHATTERBOX_DEVICE (expected auto, cpu, cuda, or mps)" ;;
    esac
}

python_supports_device() {
    local python_path="$1"
    case "$DEVICE" in
    cpu) return 0 ;;
    cuda) "$python_path" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1 ;;
    mps) "$python_path" -c 'import torch, sys; sys.exit(0 if torch.backends.mps.is_available() else 1)' >/dev/null 2>&1 ;;
    esac
}

find_compatible_environment() {
    local candidate
    for candidate in \
        "$SERVER_DIR/.venv/bin/python" \
        "$SERVER_DIR/venv/bin/python"; do
        if chatterbox_python_is_usable "$candidate" && python_supports_device "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

find_python_310() {
    local candidate=""

    if [ -n "${PYTHON_BIN:-}" ]; then
        candidate="$PYTHON_BIN"
    elif command -v python3.10 >/dev/null 2>&1; then
        candidate="$(command -v python3.10)"
    elif command -v uv >/dev/null 2>&1; then
        candidate="$(uv python find 3.10 2>/dev/null || true)"
    fi

    [ -n "$candidate" ] && [ -x "$candidate" ] || return 1
    "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' >/dev/null 2>&1 || return 1
    printf '%s\n' "$candidate"
}

configure_device() {
    local config_file="$SERVER_DIR/config.yaml"
    [ -f "$config_file" ] || return 0
    grep -qE '^[[:space:]]+device:' "$config_file" || fail "tts_engine.device missing from $config_file"

    # sed -i.bak works with both BSD and GNU sed. Keep the platform choice in
    # upstream's config because the server has no environment override for it.
    sed -i.bak -E \
        "s/^([[:space:]]+)device:[[:space:]].*$/\\1device: $DEVICE/" \
        "$config_file"
    rm -f "$config_file.bak"
}

stage_reference() {
    local reference_dir="$SERVER_DIR/reference_audio"
    local destination="$reference_dir/$REFERENCE_NAME"
    local temporary="$reference_dir/.${REFERENCE_NAME}.tmp.wav"

    [ -f "$destination" ] && return 0
    [ -f "$REFERENCE_SOURCE" ] || fail "reference source not found: $REFERENCE_SOURCE"
    mkdir -p "$reference_dir"

    case "$REFERENCE_NAME" in
    *.wav | *.WAV)
        command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is required to create $REFERENCE_NAME"
        ffmpeg -nostdin -hide_banner -loglevel error -y \
            -i "$REFERENCE_SOURCE" -ac 1 -ar 24000 -c:a pcm_s16le "$temporary"
        mv "$temporary" "$destination"
        ;;
    *)
        cp "$REFERENCE_SOURCE" "$destination"
        ;;
    esac
    echo "Staged cloned-voice reference: $destination"
}

print_plan() {
    local environment_python="${1:-}"
    echo "Chatterbox platform: $PLATFORM"
    echo "Server directory:    $SERVER_DIR"
    echo "Runtime device:      $DEVICE ($DEVICE_REASON)"
    echo "Installer profile:   $INSTALL_TYPE"
    if [ -n "$environment_python" ]; then
        echo "Python environment:  $environment_python"
    else
        echo "Python environment:  not installed (fresh install requires Python 3.10)"
    fi
    if [ -f "$SERVER_DIR/reference_audio/$REFERENCE_NAME" ]; then
        echo "Reference voice:     $REFERENCE_NAME (ready)"
    else
        echo "Reference voice:     $REFERENCE_NAME (will stage from voices/)"
    fi
}

detect_install_type
ENVIRONMENT_PYTHON="$(find_compatible_environment || true)"

if [ "$MODE" = "--doctor" ]; then
    print_plan "$ENVIRONMENT_PYTHON"
    [ -d "$SERVER_DIR/.git" ] || fail "Chatterbox is not cloned; run ./scripts/setup_chatterbox.sh"
    if [ -z "$ENVIRONMENT_PYTHON" ]; then
        find_python_310 >/dev/null || fail "no compatible environment or Python 3.10 found; install it with 'uv python install 3.10'"
    fi
    exit 0
fi

if [ ! -d "$SERVER_DIR/.git" ]; then
    mkdir -p "$(dirname "$SERVER_DIR")"
    echo "Cloning Chatterbox-TTS-Server into $SERVER_DIR ..."
    git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
fi

configure_device
stage_reference
ENVIRONMENT_PYTHON="$(find_compatible_environment || true)"
print_plan "$ENVIRONMENT_PYTHON"

export BROWSER="${BROWSER:-true}"
if [ "$PLATFORM" = "Darwin" ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

cd "$SERVER_DIR"
if [ -n "$ENVIRONMENT_PYTHON" ]; then
    exec "$ENVIRONMENT_PYTHON" server.py
fi

BOOTSTRAP_PYTHON="$(find_python_310 || true)"
if [ -z "$BOOTSTRAP_PYTHON" ]; then
    fail "fresh Chatterbox installs require Python 3.10; install it with 'uv python install 3.10', then rerun"
fi

case "$INSTALL_TYPE" in
cpu) exec "$BOOTSTRAP_PYTHON" start.py --cpu ;;
nvidia) exec "$BOOTSTRAP_PYTHON" start.py --nvidia ;;
nvidia-cu128) exec "$BOOTSTRAP_PYTHON" start.py --nvidia-cu128 ;;
esac
