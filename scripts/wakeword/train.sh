#!/usr/bin/env bash
# Host-side launcher: builds the training image and runs the full pipeline.
# Run this on the Linux box with the RTX 2060 (must have NVIDIA driver +
# nvidia-container-toolkit installed). Datasets and model checkpoints are
# bind-mounted to ./_work so re-runs reuse downloads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WAKEWORD_WORK_DIR:-$SCRIPT_DIR/_work}"
IMAGE="${WAKEWORD_IMAGE:-wakeword-trainer:latest}"
CONFIG_NAME="${WAKEWORD_CONFIG:-hey_babel.yml}"
MODEL_NAME="${CONFIG_NAME%.yml}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2; exit 1
fi
if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "Warning: nvidia container runtime not detected. The build will succeed but training will fall back to CPU and be unusably slow." >&2
fi

mkdir -p "$WORK_DIR/data" "$WORK_DIR/output" "$WORK_DIR/config"
cp "$SCRIPT_DIR/$CONFIG_NAME" "$WORK_DIR/config/$CONFIG_NAME"

echo "==> Building image $IMAGE"
docker build -t "$IMAGE" "$SCRIPT_DIR"

echo "==> Running training (this can take several hours on first run; datasets are cached in $WORK_DIR/data for re-runs)"
docker run --rm -it \
    --gpus all \
    --shm-size=2g \
    -v "$WORK_DIR:/work" \
    -e CONFIG="/work/config/$CONFIG_NAME" \
    -e MODEL_NAME="$MODEL_NAME" \
    "$IMAGE"

echo
echo "==> Artifacts produced:"
find "$WORK_DIR/output" -maxdepth 3 \( -name '*.onnx' -o -name '*.tflite' \) -print
echo
echo "Copy the .onnx and .tflite files to voice-chatbot/models/wakeword/ to use them."
