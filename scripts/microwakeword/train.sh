#!/usr/bin/env bash
# Host-side launcher: builds the microWakeWord training image and runs the full
# pipeline. Same shape as scripts/wakeword/train.sh but targets microWakeWord
# (TFLite Micro int8 output) instead of openWakeWord.
#
# Run this on the Linux box with the RTX 2060 (must have NVIDIA driver +
# nvidia-container-toolkit installed). Datasets and checkpoints bind-mount to
# ./_work so re-runs reuse downloads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${MWW_WORK_DIR:-$SCRIPT_DIR/_work}"
IMAGE="${MWW_IMAGE:-microwakeword-trainer:latest}"
CONFIG_NAME="${MWW_CONFIG:-hey_babel.yml}"

# If a sibling scripts/wakeword/_work/data exists, reuse its datasets instead
# of re-downloading the ~30 GB of MIT RIRs + FMA + AudioSet. The two trainers
# share the same raw audio corpora, only the feature extraction differs.
SIBLING_DATA="${SCRIPT_DIR}/../wakeword/_work/data"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2; exit 1
fi
if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "Warning: nvidia container runtime not detected. Build will succeed but training falls back to CPU and is unusably slow." >&2
fi

mkdir -p "$WORK_DIR/data" "$WORK_DIR/output" "$WORK_DIR/config"
cp "$SCRIPT_DIR/$CONFIG_NAME" "$WORK_DIR/config/$CONFIG_NAME"

# Symlink shared corpora if the openWakeWord pipeline has already downloaded them.
for shared in mit_rirs background_clips; do
    if [ -d "$SIBLING_DATA/$shared" ] && [ ! -e "$WORK_DIR/data/$shared" ]; then
        ln -s "$(realpath "$SIBLING_DATA/$shared")" "$WORK_DIR/data/$shared"
        echo "Reusing $shared from scripts/wakeword/_work/data/"
    fi
done

echo "==> Building image $IMAGE"
docker build -t "$IMAGE" "$SCRIPT_DIR"

echo "==> Running training (first run can take several hours; datasets cached in $WORK_DIR/data)"
docker run --rm -it \
    --gpus all \
    --shm-size=2g \
    -v "$WORK_DIR:/work" \
    -e CONFIG="/work/config/$CONFIG_NAME" \
    "$IMAGE"

echo
echo "==> Artifacts produced:"
find "$WORK_DIR/output" -maxdepth 3 -name '*.tflite' -print
echo
echo "Copy hey_*.tflite into firmware/box3/main/models/ and rebuild the firmware."
