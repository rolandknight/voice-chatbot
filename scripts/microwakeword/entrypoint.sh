#!/usr/bin/env bash
# Runs inside the microWakeWord training container.
#
# Stages datasets, generates synthetic positives via Piper, augments + extracts
# 40-band log-mel features, trains the streaming DS-CNN classifier, and
# emits an int8 TFLite Micro model.
#
# Idempotent: every stage skips if its output already exists.

set -euo pipefail

CONFIG="${CONFIG:-/work/config/hey_babel.yml}"
DATA_DIR="/work/data"

# Pull the phrase / model name out of the YAML so $OUT_DIR matches what the
# trainer will write to. Falls back to hey_babel if the field is missing.
MODEL_NAME="$(python -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('model_name', 'hey_babel'))
")"
OUT_DIR="/work/output/${MODEL_NAME}"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$HF_HOME"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. MIT room impulse responses (same source as the openWakeWord pipeline).
# ---------------------------------------------------------------------------
if [ ! -d "$DATA_DIR/mit_rirs" ] || [ -z "$(ls -A "$DATA_DIR/mit_rirs" 2>/dev/null)" ]; then
    log "Downloading MIT RIRs"
    python - <<'PY'
import glob
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
# 2. Background audio: AudioSet balanced shard + FMA small. Same corpora as
#    the openWakeWord pipeline so the negative distribution is comparable.
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

fma = load_dataset("rudraml/fma", "small", split="train", trust_remote_code=True)
dump(fma, "fma", 2000)

aset = load_dataset("agkphysics/AudioSet", "balanced", split="train",
                    streaming=True, trust_remote_code=True)
dump(aset, "audioset", 2000)
PY
else
    log "Background clips already present, skipping"
fi

# ---------------------------------------------------------------------------
# 3. Synthetic positives via piper-sample-generator. microWakeWord wants 16
#    kHz mono WAV in a flat directory; the generator's defaults match that.
# ---------------------------------------------------------------------------
POS_DIR="$DATA_DIR/positives/${MODEL_NAME}"
PHRASE="$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['target_phrase'][0])
")"
N_POS="$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('n_samples', 20000))
")"

if [ ! -d "$POS_DIR" ] || [ "$(find "$POS_DIR" -name '*.wav' 2>/dev/null | head -n 1)" = "" ]; then
    log "Generating $N_POS synthetic positives for '$PHRASE'"
    mkdir -p "$POS_DIR"
    python /opt/piper-sample-generator/generate_samples.py \
        --text "$PHRASE" \
        --max-samples "$N_POS" \
        --batch-size 16 \
        --output-dir "$POS_DIR" \
        --model /opt/piper-sample-generator/models/en-us-libritts-high.pt
else
    log "Positives already present at $POS_DIR, skipping"
fi

# ---------------------------------------------------------------------------
# 4. Feature extraction + training + quantization.
#    microWakeWord ships a CLI that takes a YAML config and runs end-to-end;
#    we invoke it once per phase so a crash mid-train can resume.
# ---------------------------------------------------------------------------
log "Extracting 40-band log-mel features"
python -m microwakeword.feature_extraction \
    --config "$CONFIG" \
    --output_dir "$OUT_DIR/features"

log "Training the streaming DS-CNN classifier"
python -m microwakeword.train \
    --config "$CONFIG" \
    --features_dir "$OUT_DIR/features" \
    --output_dir "$OUT_DIR/checkpoints"

log "Quantizing to int8 + converting to TFLite Micro"
python -m microwakeword.quantize \
    --config "$CONFIG" \
    --checkpoint "$OUT_DIR/checkpoints/best.h5" \
    --output "$OUT_DIR/${MODEL_NAME}.tflite"

# Tiny metadata file so the firmware can print which model it loaded.
python - <<PY
import json, hashlib, os
path = "$OUT_DIR/${MODEL_NAME}.tflite"
with open(path, "rb") as f:
    data = f.read()
meta = {
    "model_name": "${MODEL_NAME}",
    "phrase": "${PHRASE}",
    "size_bytes": len(data),
    "sha256": hashlib.sha256(data).hexdigest(),
    "feature_type": "log_mel_40",
    "sample_rate": 16000,
    "frame_ms": 20,
}
with open(os.path.join("$OUT_DIR", "${MODEL_NAME}.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote", os.path.join("$OUT_DIR", "${MODEL_NAME}.json"))
PY

log "Done. Artifacts:"
find "$OUT_DIR" -maxdepth 1 \( -name '*.tflite' -o -name '*.json' \) -print
