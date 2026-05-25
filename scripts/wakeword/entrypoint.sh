#!/usr/bin/env bash
# Runs inside the training container. Stages every dataset openwakeword's
# train.py expects on disk, then runs the full generate -> augment -> train
# -> convert pipeline. Idempotent: skips any step whose output already exists,
# so re-running after a crash resumes instead of restarting from scratch.

set -euo pipefail

CONFIG="${CONFIG:-/work/config/hey_babel.yml}"
DATA_DIR="/work/data"
OUT_DIR="/work/output/hey_babel"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$HF_HOME"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. MIT room impulse responses (used by augmentation for far-field realism).
# ---------------------------------------------------------------------------
if [ ! -d "$DATA_DIR/mit_rirs" ] || [ -z "$(ls -A "$DATA_DIR/mit_rirs" 2>/dev/null)" ]; then
    log "Downloading MIT RIRs"
    python - <<'PY'
import os, glob, shutil
from huggingface_hub import snapshot_download
local = snapshot_download(
    repo_id="davidscripka/MIT_environmental_impulse_responses",
    repo_type="dataset",
    local_dir="/work/data/mit_rirs",
    local_dir_use_symlinks=False,
)
print("MIT RIRs ->", local, "files:", len(glob.glob(local + "/**/*.wav", recursive=True)))
PY
else
    log "MIT RIRs already present, skipping"
fi

# ---------------------------------------------------------------------------
# 2. Pre-computed openwakeword features (ACAV100M training negatives + the
#    validation-set features). These are big (~30 GB for ACAV100M).
# ---------------------------------------------------------------------------
ACAV="$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VAL="$DATA_DIR/validation_set_features.npy"

if [ ! -f "$ACAV" ] || [ ! -f "$VAL" ]; then
    log "Downloading pre-computed openwakeword features"
    python - <<'PY'
import shutil
from huggingface_hub import hf_hub_download
for fname in ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
              "validation_set_features.npy"):
    path = hf_hub_download(
        repo_id="davidscripka/openwakeword_features",
        repo_type="dataset",
        filename=fname,
    )
    shutil.copy(path, f"/work/data/{fname}")
    print(fname, "->", f"/work/data/{fname}")
PY
else
    log "Pre-computed features already present, skipping"
fi

# ---------------------------------------------------------------------------
# 3. Background audio for augmentation: one AudioSet shard + FMA-small.
#    Both are streamed via the `datasets` library and dumped to wav.
# ---------------------------------------------------------------------------
BG_DIR="$DATA_DIR/background_clips"
if [ ! -d "$BG_DIR" ] || [ "$(find "$BG_DIR" -name '*.wav' 2>/dev/null | head -n 1)" = "" ]; then
    log "Building background audio corpus (AudioSet shard + FMA small)"
    mkdir -p "$BG_DIR"
    python - <<'PY'
import os, soundfile as sf
from datasets import load_dataset

out_dir = "/work/data/background_clips"
os.makedirs(out_dir, exist_ok=True)

def dump(ds, prefix, limit):
    n = 0
    for row in ds:
        if n >= limit:
            break
        audio = row["audio"]
        sr = audio["sampling_rate"]
        arr = audio["array"]
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sf.write(f"{out_dir}/{prefix}_{n:05d}.wav", arr, 16000, subtype="PCM_16")
        n += 1
    print(prefix, "wrote", n, "clips")

# FMA small (~8000 30-sec music clips). The dataset's loader fetches a ZIP
# archive whose central directory must be seekable, so streaming is unsupported.
fma = load_dataset("rudraml/fma", "small", split="train", trust_remote_code=True)
dump(fma, "fma", 2000)

# AudioSet balanced shard ("balanced" is a config, not a split)
aset = load_dataset("agkphysics/AudioSet", "balanced", split="train", streaming=True, trust_remote_code=True)
dump(aset, "audioset", 2000)
PY
else
    log "Background clips already present, skipping"
fi

# ---------------------------------------------------------------------------
# 4. Run the openwakeword training pipeline.
#    Each phase writes to $OUT_DIR; train.py skips a phase if its output
#    is already there UNLESS --overwrite is set, so we can resume.
# ---------------------------------------------------------------------------
log "Generating synthetic positive/negative clips"
python -m openwakeword.train \
    --training_config "$CONFIG" \
    --generate_clips

# openwakeword 0.6.0 wraps all four feature-extraction calls under a single
# guard that only checks for positive_features_train.npy. If a previous run
# crashed partway, that file may exist while the other three don't — the next
# run then skips augmentation entirely and the train phase blows up looking
# for a missing .npy. Detect that state and clear the partial set so augment
# regenerates everything.
FEATURE_DIR="$OUT_DIR/hey_babel"
EXPECTED_FEATURES=(positive_features_train.npy negative_features_train.npy
                   positive_features_test.npy  negative_features_test.npy)
present=0
for f in "${EXPECTED_FEATURES[@]}"; do
    [ -f "$FEATURE_DIR/$f" ] && present=$((present+1))
done
if [ "$present" -gt 0 ] && [ "$present" -lt "${#EXPECTED_FEATURES[@]}" ]; then
    log "Detected partial feature set ($present/${#EXPECTED_FEATURES[@]}); clearing so augment re-runs"
    for f in "${EXPECTED_FEATURES[@]}"; do rm -f "$FEATURE_DIR/$f"; done
fi

log "Augmenting clips (RIR + background mix + feature extraction)"
python -m openwakeword.train \
    --training_config "$CONFIG" \
    --augment_clips

log "Training the classifier"
python -m openwakeword.train \
    --training_config "$CONFIG" \
    --train_model

log "Converting ONNX -> TFLite"
python -m openwakeword.train \
    --training_config "$CONFIG" \
    --convert_to_tflite

log "Done. Artifacts:"
find "$OUT_DIR" -maxdepth 2 \( -name '*.onnx' -o -name '*.tflite' \) -print
