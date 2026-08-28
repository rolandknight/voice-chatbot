#!/usr/bin/env bash
# Start the PoC-local NeMo-Speech.cpp realtime transcription sidecar.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POC_DIR="$PROJECT_DIR" # runtime root (models/, .deps/, .env); the PoC trees are archived

DEVICE_OVERRIDE_SET="${POC_NEMOTRON_DEVICE+set}"
DEVICE_OVERRIDE="${POC_NEMOTRON_DEVICE-}"
HOST_OVERRIDE_SET="${POC_NEMOTRON_HOST+set}"
HOST_OVERRIDE="${POC_NEMOTRON_HOST-}"
PORT_OVERRIDE_SET="${POC_NEMOTRON_PORT+set}"
PORT_OVERRIDE="${POC_NEMOTRON_PORT-}"
MODEL_ROOT_OVERRIDE_SET="${NEMO_SPEECH_MODEL_DIR+set}"
MODEL_ROOT_OVERRIDE="${NEMO_SPEECH_MODEL_DIR-}"
if [ -f "$POC_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$POC_DIR/.env"
    set +a
fi
[ -n "$DEVICE_OVERRIDE_SET" ] && POC_NEMOTRON_DEVICE="$DEVICE_OVERRIDE"
[ -n "$HOST_OVERRIDE_SET" ] && POC_NEMOTRON_HOST="$HOST_OVERRIDE"
[ -n "$PORT_OVERRIDE_SET" ] && POC_NEMOTRON_PORT="$PORT_OVERRIDE"
[ -n "$MODEL_ROOT_OVERRIDE_SET" ] && NEMO_SPEECH_MODEL_DIR="$MODEL_ROOT_OVERRIDE"

NATIVE_ROOT="$POC_DIR/.deps/nemo-speech/v0.1.0"
NEMO_SPEECH_BIN="$NATIVE_ROOT/bin/nemo-speech"
MODEL_ROOT="${NEMO_SPEECH_MODEL_DIR:-$POC_DIR/models/nemotron}"
DEVICE="${POC_NEMOTRON_DEVICE:-auto}"
HOST="${POC_NEMOTRON_HOST:-127.0.0.1}"
PORT="${POC_NEMOTRON_PORT:-8178}"
RIGHT_CONTEXT="${POC_NEMOTRON_RIGHT_CONTEXT:-6}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

case "$DEVICE" in
auto | cuda:0 | cpu | metal) ;;
*) fail "invalid POC_NEMOTRON_DEVICE '$DEVICE' (expected auto, cuda:0, cpu, or metal)" ;;
esac

case "${POC_PLATFORM_OVERRIDE:-$(uname -s)}" in
Linux)
    [ "$DEVICE" != "metal" ] || fail "POC_NEMOTRON_DEVICE=metal is only supported on macOS"
    ;;
Darwin)
    [ "$DEVICE" != "cuda:0" ] || fail "POC_NEMOTRON_DEVICE=cuda:0 is only supported on Linux"
    if [ "$(uname -m)" = "x86_64" ] && [ "$DEVICE" = "metal" ]; then
        fail "POC_NEMOTRON_DEVICE=metal requires Apple Silicon"
    fi
    ;;
*) fail "unsupported operating system (expected Linux or macOS)" ;;
esac

case "$PORT" in
'' | *[!0-9]*) fail "POC_NEMOTRON_PORT must be an integer from 1 through 65535" ;;
esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || \
    fail "POC_NEMOTRON_PORT must be an integer from 1 through 65535"

# The English model was evaluated at these native cache geometries: 80, 160,
# 560, and 1120 ms. Six right-context frames gives the 560 ms operating point,
# whose published WER is within 0.14 points of the maximum-context setting.
case "$RIGHT_CONTEXT" in
0 | 1 | 6 | 13 | -1) ;;
*) fail "POC_NEMOTRON_RIGHT_CONTEXT must be one of 0, 1, 6, 13, or -1" ;;
esac

[ -x "$NEMO_SPEECH_BIN" ] || \
    fail "NeMo-Speech.cpp is not installed; run ./scripts/setup_nemotron.sh"
[ -d "$MODEL_ROOT" ] || \
    fail "Nemotron model cache is missing; run ./scripts/setup_nemotron.sh"

export NEMO_SPEECH_MODEL_DIR="$MODEL_ROOT"

echo "Starting Nemotron Speech sidecar"
echo "WebSocket: ws://$HOST:$PORT/v1/realtime"
echo "Health:    http://$HOST:$PORT/ready"
echo "Device:    $DEVICE"
echo "ASR window: $(( (RIGHT_CONTEXT < 0 ? 13 : RIGHT_CONTEXT) + 1 )) x 80 ms"

# FlowCat owns end-of-utterance detection and sends commit after its configured
# silence. Disable server batching to minimize latency and GPU state for this
# single-stream laptop profile. Warmup remains enabled and is paid only once.
exec "$NEMO_SPEECH_BIN" serve \
    --asr-model nemotron-en \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    --threads 2 \
    --no-ui \
    --asr.batching.enabled=false \
    --asr.streaming.rnnt_right_context="$RIGHT_CONTEXT" \
    --asr.endpointing.enable=false
