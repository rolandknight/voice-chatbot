# Wake-word training

Builds [openWakeWord](https://github.com/dscripka/openWakeWord) models for
the phrases this project ships — currently **"hey babel"** and
**"hey marvin"** — and emits `<phrase>.onnx` + `<phrase>.tflite` per phrase.
The pipeline is phrase-agnostic: add a new wake word by dropping a
`<model_name>.yml` next to the existing configs and appending the model
name to `PHRASES` in the `Makefile`. **Training only — no app integration.**
Wiring models into `app.py` is deferred.

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
make train               # trains every phrase in Makefile's PHRASES, in order
make train-hey-babel     # train just one phrase
make train-hey-marvin
```

That's it. For each phrase the script:

1. Builds the `wakeword-trainer:latest` image (CUDA 12.1 + openWakeWord +
   piper-sample-generator + libritts voice model). One image serves every
   phrase — the YAML config picks which one to train.
2. Bind-mounts `./_work` into the container at `/work`.
3. Inside the container, `entrypoint.sh` downloads MIT RIRs, FMA-small,
   AudioSet balanced shard, and the pre-computed ACAV100M / validation
   features (once — shared across phrases), then runs
   `python -m openwakeword.train` through all four phases
   (generate → augment → train → convert).
4. Final artifacts land in `./_work/output/<phrase>/`.

Re-running picks up where it left off — every download and every training
phase is skipped if its output already exists. `make train` runs phrases
sequentially, so a partial second phrase doesn't lose progress on the first.

## After training

```sh
make install                          # copies every trained phrase's .onnx + .tflite into ../../models/wakeword/
make install-hey-babel                # just one phrase
make install DEST=/some/other/dir     # override destination
```

A quick host-side sanity check before shipping to the Mac:

```sh
docker run --rm -it --gpus all -v "$PWD/_work:/work" wakeword-trainer:latest \
    python -c "
from openwakeword.model import Model
import numpy as np
m = Model(wakeword_models=['/work/output/hey_babel/hey_babel.tflite'])
# feed silence: scores should be near zero
silence = np.zeros(1280, dtype=np.int16)
print('silence:', m.predict(silence))
"
```

## Tuning knobs (per-phrase `<model_name>.yml`)

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

## Re-training after a config change

```sh
make clean-hey-babel   # drop the trained model + .npy features, keep TTS clips
make train-hey-babel
```

`make clean-<phrase>` is safe when you've only tuned hyperparameters that
affect training/augmentation. If you change `target_phrase`,
`custom_negative_phrases`, or `n_samples`, the synthesized clips are stale
too — wipe everything:

```sh
make clean-all-hey-babel   # rm -rf _work/output/hey_babel/
make train-hey-babel
```

The unsuffixed `make clean` / `make clean-all` / `make train` apply the
operation to every phrase in `PHRASES`. Use the suffixed forms when
iterating on one phrase — the other phrase's hours of synthesized clips
shouldn't be collateral damage.

## Per-phrase tuning history

Tuning is phrase-specific — phonetic distinctiveness, length, and the
relevant hard-negative space all change with the target phrase, so the same
hyperparameters that land one phrase at recall 0.8 / FP 0.3 can give another
0.5 / 2.5. The phrase-agnostic infrastructure (Dockerfile, scripts, Makefile,
and this README) is reusable; the per-phrase YAML and iteration log are not.

Log each phrase's iterations in its own `<phrase>-training.md`:

- [`hey-babel-training.md`](hey-babel-training.md)
- [`hey-marvin-training.md`](hey-marvin-training.md)
- [`hey-one-one-training.md`](hey-one-one-training.md)

## Troubleshooting

**"nvidia container runtime not detected"** — install `nvidia-container-toolkit`
and restart docker (`sudo systemctl restart docker`).

**Training falls back to CPU even though `nvidia-smi` works** — symptom is a
`CUDA driver initialization failed` warning from torch and a `cuInit` that
returns error 3 (`CUDA_ERROR_NOT_INITIALIZED`). NVML works (so `nvidia-smi`
sees the GPU) but the CUDA driver API can't open a context because the
`/dev/nvidia-caps/*` device nodes are root-only and CUDA can't fix them up.
The setuid helper that does that fix-up is `nvidia-modprobe`, which Pop!_OS's
`nvidia-driver-580-open` metapackage does not pull in. Install it:

```sh
sudo apt-get install -y nvidia-modprobe
python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))'   # 0 == fixed
```

No reboot needed; the next CUDA call invokes the new binary and the cap
nodes get created with the right permissions.

**Out of memory during `--generate_clips`** — the Piper TTS step is the
memory peak. Halve `tts_batch_size` in the affected `<phrase>.yml` and re-run; the script
resumes from the partial generation.

**Out of memory during `--train_model`** — halve `batch_n_per_class.ACAV100M_sample`.

**Dataset download stalls** — Hugging Face downloads cache to `./_work/data/hf_cache`
via `HF_HOME`. Delete the half-downloaded file there and re-run; it will resume.

## File map

- `Dockerfile` — CUDA + Python + openWakeWord + piper-sample-generator image.
- `entrypoint.sh` — in-container: stage data, run training, emit artifacts.
  Phrase-agnostic — derives the model name from `$CONFIG` or `$MODEL_NAME`.
- `train.sh` — host-side launcher; builds the image and runs the container.
  Drives one phrase per invocation (selected via `WAKEWORD_CONFIG`).
- `Makefile` — per-phrase + aggregate `train`/`clean`/`clean-all`/`install`
  wrappers around the above. The `PHRASES` list is the source of truth.
- `hey_babel.yml`, `hey_marvin.yml`, `hey_one_one.yml` — training config (one per phrase).
- `hey-babel-training.md`, `hey-marvin-training.md`, `hey-one-one-training.md` — iteration log (one per phrase).
- `_work/` — generated; datasets (shared), intermediate clips, and final models.
