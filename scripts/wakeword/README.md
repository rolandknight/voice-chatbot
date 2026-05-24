# `hey babel` wake-word training

Builds an [openWakeWord](https://github.com/dscripka/openWakeWord) model
for the phrase **"hey babel"** and emits `hey_babel.onnx` + `hey_babel.tflite`.
**Training only — no app integration.** Wiring the model into `app.py` is
deferred.

## Prerequisites

- Linux host with an NVIDIA GPU (designed for an RTX 2060 / 6 GB; will run on
  anything CUDA-capable with as much or more VRAM).
- Recent NVIDIA driver + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Verify with `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`.
- Docker.
- ~80 GB free disk under `scripts/wakeword/_work` (datasets are the bulk).
- A few hours wall-clock for the first run — most of it is dataset download +
  synthetic clip generation.

## Run

```sh
cd voice-chatbot/scripts/wakeword
./train.sh
```

That's it. The script:

1. Builds the `hey-babel-trainer:latest` image (CUDA 12.1 + openWakeWord +
   piper-sample-generator + libritts voice model).
2. Bind-mounts `./_work` into the container at `/work`.
3. Inside the container, `entrypoint.sh` downloads MIT RIRs, FMA-small,
   AudioSet balanced shard, and the pre-computed ACAV100M / validation
   features, then runs `python -m openwakeword.train` through all four
   phases (generate → augment → train → convert).
4. Final artifacts land in `./_work/output/hey_babel/`.

Re-running picks up where it left off — every download and every training
phase is skipped if its output already exists.

## After training

```sh
mkdir -p ../../models/wakeword
cp _work/output/hey_babel/hey_babel.onnx  ../../models/wakeword/
cp _work/output/hey_babel/hey_babel.tflite ../../models/wakeword/
```

A quick host-side sanity check before shipping to the Mac:

```sh
docker run --rm -it --gpus all -v "$PWD/_work:/work" hey-babel-trainer:latest \
    python -c "
from openwakeword.model import Model
import numpy as np
m = Model(wakeword_models=['/work/output/hey_babel/hey_babel.tflite'])
# feed silence: scores should be near zero
silence = np.zeros(1280, dtype=np.int16)
print('silence:', m.predict(silence))
"
```

## Tuning knobs (`hey_babel.yml`)

- `n_samples` — bump from 20 000 to 50 000+ once you've validated end-to-end.
  Lifts wall-clock by an hour or two but improves robustness noticeably.
- `tts_batch_size` / `augmentation_batch_size` — drop to 8 / 4 if Piper OOMs on
  the 2060 during clip generation.
- `batch_n_per_class.ACAV100M_sample` — drop from 1024 to 512 if training OOMs.
  Hit rates barely change; mostly affects throughput.
- `custom_negative_phrases` — add anything you observe the model misfiring on.
  This is the single highest-leverage knob for cleaning up false positives.
- `target_false_positives_per_hour` — lower = stricter model; raise if recall
  is suffering.

## Troubleshooting

**"nvidia container runtime not detected"** — install `nvidia-container-toolkit`
and restart docker (`sudo systemctl restart docker`).

**Out of memory during `--generate_clips`** — the Piper TTS step is the
memory peak. Halve `tts_batch_size` in `hey_babel.yml` and re-run; the script
resumes from the partial generation.

**Out of memory during `--train_model`** — halve `batch_n_per_class.ACAV100M_sample`.

**Dataset download stalls** — Hugging Face downloads cache to `./_work/data/hf_cache`
via `HF_HOME`. Delete the half-downloaded file there and re-run; it will resume.

## File map

- `Dockerfile` — CUDA + Python + openWakeWord + piper-sample-generator image.
- `entrypoint.sh` — in-container: stage data, run training, emit artifacts.
- `train.sh` — host-side launcher; builds the image and runs the container.
- `hey_babel.yml` — training config (the only file with phrase-specific tuning).
- `_work/` — generated; datasets, intermediate clips, and final models.
