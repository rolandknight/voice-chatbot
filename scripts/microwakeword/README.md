# `hey babel` / `hey marvin` wake-word training (microWakeWord)

Builds [microWakeWord](https://github.com/kahrendt/microWakeWord) models for the phrases **"hey babel"** and **"hey marvin"** and emits `.tflite` artifacts sized for the ESP32-S3 (int8, ~50 KB each). These are the on-device wake-word models used by `firmware/box3/`.

This sits parallel to `scripts/wakeword/`, which trains the same phrase for [openWakeWord](https://github.com/dscripka/openWakeWord). The two pipelines share datasets but produce incompatible models — openWakeWord targets desktop Python, microWakeWord targets TFLite Micro on MCUs.

See `docs/web-rtc.md` for how these models fit into the broader Box-3 ↔ backend design.

## Prerequisites

- Linux host with an NVIDIA GPU (designed for an RTX 2060 / 6 GB; anything CUDA-capable with as much VRAM or more will work).
- Recent NVIDIA driver + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Verify with `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`.
- Docker.
- ~80 GB free disk under `scripts/microwakeword/_work` (datasets are the bulk). If `scripts/wakeword/_work/data/` already exists, `train.sh` symlinks `mit_rirs` and `background_clips` from there to avoid re-downloading.

## Run

```sh
cd voice-chatbot/scripts/microwakeword
make all              # trains both phrases
# or:
make babel            # just 'hey babel'
make marvin           # just 'hey marvin'
```

The first run takes several hours (most of it is dataset download + Piper TTS generating synthetic positives). Re-runs reuse caches and resume from the last completed phase.

Final artifacts:

```
_work/output/hey_babel/hey_babel.tflite     # int8 TFLite Micro
_work/output/hey_babel/hey_babel.json       # metadata (size, sha256, sample rate)
_work/output/hey_marvin/hey_marvin.tflite
_work/output/hey_marvin/hey_marvin.json
```

## Plug the models into the firmware

```sh
make install            # copies both .tflite into firmware/box3/main/models/
# or just one:
make install-babel
make install-marvin
```

Then rebuild the firmware (`cd ../../firmware/box3 && make build`). The `.tflite` files are embedded via `EMBED_FILES` — no flash partition changes needed.

The firmware tree has its own mirror target (`make install` under `firmware/box3/`) that pulls from `_work/output/`; either side works.

## Sanity check the model on the host

```sh
make eval
```

Runs the int8 `.tflite` against a held-out test set inside the same Docker image; reports FAR (false-accepts per hour of negative audio) and FRR (% missed positives).

Targets:

- FAR < 0.5 / hr — comparable to the openWakeWord version
- FRR < 5%

If FAR is high, add the misfiring phrases to `custom_negative_phrases` in the YAML and re-train. If FRR is high, raise `n_samples` (20 k → 50 k) or add accents / noise to augmentation.

## Tuning knobs

`hey_babel.yml` / `hey_marvin.yml`:

- `n_samples` — number of Piper-synthesized positive utterances. Bump from 20 000 to 50 000+ once the pipeline is validated end-to-end.
- `training_steps` / `learning_rates` — list-of-stages, one entry per training phase. e.g. `[8000, 4000]` with `[0.001, 0.0001]` does an 8 k step warmup at 1e-3 then a 4 k step refine at 1e-4. Lists must be the same length; entries get padded to match if not.
- `negative_class_weight` — penalty multiplier on false positives during training. Default `[20]` is aggressive; lower (e.g. `[10]`) if recall is suffering.
- `positive_class_weight` — usually leave at `[1]`.
- `batch_size` — 128 fits a 6 GB RTX 2060. Drop to 64 if OOM.

`custom_negative_phrases` is **read but currently unused** — kept around so that future negative-spectrogram synthesis (per phrase, via Piper) can pick it up. The current pipeline relies on upstream microWakeWord's pre-generated `dinner_party` / `no_speech` / `speech` negative spectrograms, which already cover the failure modes most phrases hit. If you still see FP on a specific phonetic neighbour after `negative_class_weight` tuning, the next step is generating phrase-specific TTS negatives and adding them as a feature set in entrypoint.sh's `features` list.

Feature-extractor params (`n_mels`, `window_size_ms`, `window_stride_ms`) are fixed by upstream's `micro_frontend` preprocessor (40 features, 30 ms window, 10 ms step) and ignored if set in the YAML — the firmware-side preprocessor must match these constants.

## Troubleshooting

**`nvidia container runtime not detected`** — install `nvidia-container-toolkit` and restart Docker (`sudo systemctl restart docker`).

**OOM during feature extraction** — drop `batch_size` in the YAML from 128 to 64.

**OOM during Piper TTS positive generation** — drop `--batch-size` in `entrypoint.sh` from 16 to 8.

**Dataset download stalls** — Hugging Face downloads cache to `./_work/data/hf_cache` via `HF_HOME`. Delete the half-downloaded file there and re-run; it will resume.

**`hey_marvin` and `hey_babel` both fire on the same utterance** — expected at low SNR. The firmware breaks ties by picking the higher-confidence model; if that's not enough, raise `target_false_positives_per_hour` on the loser.

## File map

- `Dockerfile` — CUDA + TF + microWakeWord + Piper image.
- `entrypoint.sh` — in-container: stage data, generate positives, extract features, train, quantize.
- `train.sh` — host-side launcher; builds the image and runs the container.
- `Makefile` — `babel` / `marvin` / `eval` / `install` shortcuts.
- `hey_babel.yml`, `hey_marvin.yml` — per-phrase training configs.
- `_work/` — generated; datasets, intermediate features, final `.tflite` + `.json`.
