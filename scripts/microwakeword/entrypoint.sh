#!/usr/bin/env bash
# Runs inside the microWakeWord training container.
#
# Stages datasets, generates synthetic positives via Piper, builds spectrogram
# features, trains the streaming MixedNet classifier via the upstream CLI, and
# copies out the int8 TFLite Micro model.
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
        "$PHRASE" \
        --max-samples "$N_POS" \
        --batch-size 16 \
        --output-dir "$POS_DIR" \
        --model /opt/piper-sample-generator/models/en-us-libritts-high.pt
else
    log "Positives already present at $POS_DIR, skipping"
fi

# ---------------------------------------------------------------------------
# 4a. Build positive spectrogram features (RaggedMmap) for training,
#     validation, and testing splits. Mirrors microWakeWord's
#     basic_training_notebook augmentation + spectrogram pipeline.
# ---------------------------------------------------------------------------
POS_FEAT_DIR="$DATA_DIR/features/${MODEL_NAME}"
if [ ! -d "$POS_FEAT_DIR/training/wakeword_mmap" ]; then
    log "Generating positive spectrogram features for $MODEL_NAME"
    mkdir -p "$POS_FEAT_DIR"
    POS_DIR="$POS_DIR" RIR_DIR="$DATA_DIR/mit_rirs" BG_DIR="$BG_DIR" \
    OUT="$POS_FEAT_DIR" python - <<'PY'
import os
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

POS_DIR = os.environ["POS_DIR"]
RIR_DIR = os.environ["RIR_DIR"]
BG_DIR  = os.environ["BG_DIR"]
OUT     = os.environ["OUT"]

clips = Clips(
    input_directory=POS_DIR,
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=[RIR_DIR],
    background_paths=[BG_DIR],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

# (split, split_name, repetition, slide_frames). The testing split uses
# slide_frames=1 because the streaming evaluation doesn't need artificial
# repetition — the model will see real shifted windows at inference time.
splits = [
    ("training",   "train",      2, 10),
    ("validation", "validation", 1, 10),
    ("testing",    "test",       1,  1),
]
for split, split_name, repetition, slide in splits:
    out_dir = os.path.join(OUT, split, "wakeword_mmap")
    if os.path.exists(out_dir):
        print(f"skip {split}: exists at {out_dir}")
        continue
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    specs = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter,
        slide_frames=slide,
        step_ms=10,
    )
    RaggedMmap.from_generator(
        out_dir=out_dir,
        sample_generator=specs.spectrogram_generator(split=split_name, repeat=repetition),
        batch_size=100,
        verbose=True,
    )
PY
else
    log "Positive features already present at $POS_FEAT_DIR, skipping"
fi

# ---------------------------------------------------------------------------
# 4b. Pre-generated negative spectrogram features. These come from upstream
#     microWakeWord's HF dataset and are required for sane training — the
#     positive-only/background pipeline doesn't produce speech-like negatives
#     that the streaming model needs to suppress.
# ---------------------------------------------------------------------------
NEG_BASE="$DATA_DIR/negative_datasets"
mkdir -p "$NEG_BASE"
for name in dinner_party dinner_party_eval no_speech speech; do
    if [ -d "$NEG_BASE/$name" ] && [ -n "$(ls -A "$NEG_BASE/$name" 2>/dev/null)" ]; then
        continue
    fi
    zip="$NEG_BASE/$name.zip"
    # Skip the redownload if a previous run already pulled the zip down but
    # failed to extract it (e.g. unzip missing). 423 MB per zip is worth saving.
    if [ ! -s "$zip" ]; then
        log "Downloading negative spectrogram dataset: $name"
        curl -fL --retry 3 \
            -o "$zip" \
            "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/$name.zip"
    fi
    log "Extracting $name"
    python -c "import zipfile; zipfile.ZipFile('$zip').extractall('$NEG_BASE')"
    rm -f "$zip"
done

# ---------------------------------------------------------------------------
# 4c. Synthesize the upstream-format training_parameters.yaml from the
#     user-friendly per-phrase YAML. Saved into $OUT_DIR so `make eval` can
#     reuse it with --train 0.
# ---------------------------------------------------------------------------
TRAIN_DIR="$OUT_DIR/checkpoints"
TRAINING_YAML="$OUT_DIR/training_parameters.yaml"
mkdir -p "$TRAIN_DIR"

log "Writing $TRAINING_YAML"
CONFIG="$CONFIG" POS_FEAT_DIR="$POS_FEAT_DIR" NEG_BASE="$NEG_BASE" \
TRAIN_DIR="$TRAIN_DIR" TRAINING_YAML="$TRAINING_YAML" python - <<'PY'
import os, yaml

with open(os.environ["CONFIG"]) as f:
    user = yaml.safe_load(f)

POS_FEAT_DIR = os.environ["POS_FEAT_DIR"]
NEG_BASE     = os.environ["NEG_BASE"]
TRAIN_DIR    = os.environ["TRAIN_DIR"]

cfg = {
    "train_dir": TRAIN_DIR,
    "window_step_ms": 10,
    "features": [
        {"features_dir": POS_FEAT_DIR,
         "sampling_weight": 2.0,  "penalty_weight": 1.0, "truth": True,
         "truncation_strategy": "truncate_start", "type": "mmap"},
        {"features_dir": f"{NEG_BASE}/speech",
         "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
         "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": f"{NEG_BASE}/dinner_party",
         "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
         "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": f"{NEG_BASE}/no_speech",
         "sampling_weight": 5.0,  "penalty_weight": 1.0, "truth": False,
         "truncation_strategy": "random", "type": "mmap"},
        # Used only for ambient FAR estimation during validation/testing.
        {"features_dir": f"{NEG_BASE}/dinner_party_eval",
         "sampling_weight": 0.0,  "penalty_weight": 1.0, "truth": False,
         "truncation_strategy": "split", "type": "mmap"},
    ],
    "training_steps":        user.get("training_steps", [10000]),
    "positive_class_weight": user.get("positive_class_weight", [1]),
    "negative_class_weight": user.get("negative_class_weight", [20]),
    "learning_rates":        user.get("learning_rates",
                                       [user.get("learning_rate", 0.001)]),
    "batch_size":            user.get("batch_size", 128),
    # SpecAugment disabled by default; matches the notebook starting point.
    "time_mask_max_size":  [0],
    "time_mask_count":     [0],
    "freq_mask_max_size":  [0],
    "freq_mask_count":     [0],
    "eval_step_interval":  500,
    "clip_duration_ms":    1500,
    "target_minimization": user.get("target_minimization", 0.9),
    "minimization_metric": user.get("minimization_metric", None),
    "maximization_metric": user.get("maximization_metric",
                                    "average_viable_recall"),
}
with open(os.environ["TRAINING_YAML"], "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
print("wrote", os.environ["TRAINING_YAML"])
PY

# ---------------------------------------------------------------------------
# 4d. Train + quantize + convert to streaming TFLite Micro int8. MixedNet
#     architecture args mirror the upstream basic_training_notebook defaults;
#     start here, tune via hey_*.yml later.
# ---------------------------------------------------------------------------
log "Training mixednet and quantizing to int8 streaming TFLite"
python -m microwakeword.model_train_eval \
    --training_config "$TRAINING_YAML" \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tflite_streaming_quantized 1 \
    --use_weights best_weights \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block "1,1,1,1" \
    --mixconv_kernel_sizes "[5],[7,11],[9,15],[23]" \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3

# ---------------------------------------------------------------------------
# 4e. Copy out the streaming int8 model to a stable name the firmware tree
#     and `make install` can find.
# ---------------------------------------------------------------------------
SRC_TFLITE="$TRAIN_DIR/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
DST_TFLITE="$OUT_DIR/${MODEL_NAME}.tflite"
cp -v "$SRC_TFLITE" "$DST_TFLITE"

# Tiny metadata file so the firmware can print which model it loaded.
python - <<PY
import json, hashlib, os
path = "$DST_TFLITE"
with open(path, "rb") as f:
    data = f.read()
meta = {
    "model_name": "${MODEL_NAME}",
    "phrase": "${PHRASE}",
    "size_bytes": len(data),
    "sha256": hashlib.sha256(data).hexdigest(),
    "feature_type": "micro_frontend_40",
    "sample_rate": 16000,
    "window_step_ms": 10,
    "architecture": "mixednet_streaming",
}
with open(os.path.join("$OUT_DIR", "${MODEL_NAME}.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote", os.path.join("$OUT_DIR", "${MODEL_NAME}.json"))
PY

log "Done. Artifacts:"
find "$OUT_DIR" -maxdepth 1 \( -name '*.tflite' -o -name '*.json' -o -name 'training_parameters.yaml' \) -print
