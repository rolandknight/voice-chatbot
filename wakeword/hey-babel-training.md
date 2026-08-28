# `hey babel` training iterations

Living log of training runs for the "hey babel" wake word: hyperparameters
used, metrics produced, and the reasoning behind each change. Add a new
section per run; preserve prior ones — knowing what *didn't* work is as
load-bearing as knowing what did.

The config under tuning lives in `hey_babel.yml`. Background on what each
knob does is in `README.md` → "Tuning knobs". This file is the per-phrase
log; the README is the phrase-agnostic reference.

**Target:** recall ≥ 0.70 with FP/hour ≤ 0.5.

## Summary

| Run | `max_neg_weight` | pos batch | `layer_size` | aug rounds | `fp_target` | `n_samples` | recall | FP/hr | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 1500 | 50  | 32 | 1 | 0.2 | 20 000 | 0.461 | 0.00 | 0.730 |
| v2 | 750  | 100 | 64 | 2 | 0.5 | 20 000 | 0.663 | 2.74 | 0.830 |
| v3 | 1000 | 100 | 64 | 2 | 0.5 | 20 000 | 0.636 | 0.62 | 0.816 |
| v4 (planned) | 1000 | 100 | 64 | 2 | 0.5 | **50 000** | — | — | — |

`steps=50000`, `model_type=dnn`, `custom_negative_phrases` list unchanged
across all runs so far.

## v1 — initial (2026-05-24)

### Config

- `max_negative_weight: 1500`
- `batch_n_per_class.positive: 50`
- `layer_size: 32`
- `augmentation_rounds: 1`
- `target_false_positives_per_hour: 0.2`

### Result

- Accuracy: **0.7295**
- Recall: **0.4605**
- FP / hour: **0.0**

### Reading

Severe precision over-tune: 0 false positives but missing >50% of true wake
words. The model effectively learned "always reject" because:

- `max_negative_weight: 1500` ramps the negative-class loss weight linearly
  from 1 → 1500 over 50 000 steps; 1500 is at the upper end of openWakeWord
  stock-model values (typically 1000–1200).
- Per-batch ratio of 1024 ACAV negatives vs. 50 positives = 20:1. Multiplied
  by the weight cap, the cumulative negative-class loss dwarfs positive-class
  loss by ~4 orders of magnitude.
- `layer_size: 32` is well under openWakeWord's `layer_dim` default of 128 —
  limited capacity to find subtle positive-class structure.
- `target_false_positives_per_hour: 0.2` is post-training threshold selection
  only; tightening it can't recover recall the model never learned.

## v2 — rebalance the loss (2026-05-25)

### Changes from v1

- `max_negative_weight` 1500 → **750** (halve the runaway FP penalty)
- `batch_n_per_class.positive` 50 → **100** (neg:pos batch ratio 20:1 → 10:1)
- `layer_size` 32 → **64** (more representational capacity)
- `augmentation_rounds` 1 → **2** (more positive variety after RIR + bg mix)
- `target_false_positives_per_hour` 0.2 → **0.5** (looser checkpoint selection)

### Result

- Accuracy: **0.830**
- Recall: **0.663**
- FP / hour: **2.74**

### Reading

Direction correct, magnitude overshot. Recall climbed ~0.20, accuracy +0.10
— the model is now actually trying to fire on positives. But FP/hour blew
past the 0.5 target by 5.5×.

Notable: achieved FP/hour (2.74) > the threshold-selection target (0.5),
which means *no* checkpoint qualified and openWakeWord fell back to a relaxed
selection. So the **training objective** is what needs to tighten, not the
selection target.

Of the v2 changes, the weight cut is the dominant FP driver. `layer_size` +
`augmentation_rounds` primarily improve discrimination (net wins); the
weight + batch-ratio pair is what produces the imbalance.

(Also of note: this run hit a Dockerfile/`onnx_tf` conversion bug — the
ONNX exported correctly but TFLite conversion crashed with
`KeyError: 'onnx::Flatten_0'` because PyTorch 2.x's auto-named tensors
contain `::` and `onnx_tf 1.10.0` doesn't sanitize names symmetrically.
Worked around with an `onnx.save` rename pass between train and convert
phases in `entrypoint.sh`. Not a tuning issue.)

## v3 — claw back FP rate (2026-05-25)

### Change from v2

- `max_negative_weight` 750 → **1000**

Single-lever change so the contribution stays legible.

### Result

- Accuracy: **0.816**
- Recall: **0.636**
- FP / hour: **0.62**

### Reading

Almost exactly on the target boundary. Bumping the weight back by 250 cost
~0.03 recall and gained ~2.12 FP/hour reduction — much steeper FP slope than
recall slope, which means the knob is well-behaved in this range.

Linear extrapolation between v2 and v3 predicts that `max_negative_weight ≈
1015` would land FP at exactly 0.5 with recall ≈ 0.635. So v4 should either
nudge to 1050 (safe margin) or accept v3.

What to do next is a product decision more than a tuning one:

- **Ship v3.** FP/hour 0.62 means a spurious wake every ~97 minutes. Mildly
  annoying but tolerable for a daily-driver. Recall 0.636 means ~36% of
  legitimate "hey babel" attempts fail — that's the bigger UX problem.
- **v4 fine-tune** for FP target: `max_negative_weight: 1000 → 1050`. Buys
  ~0.1 FP/hour reduction at cost of ~0.005 recall.
- **v4 push recall instead:** bump `n_samples: 20000 → 50000`. Adds 1–2 h
  wall-clock; more positive variety usually lifts recall by 5–10 pts without
  meaningfully changing FP rate. Best lever if recall is the real blocker.

### Conversion-side incident (not a tuning issue)

This run also re-tripped the `onnx_tf` KeyError on `onnx::Flatten_0` that we
thought was fixed after v2. Diagnosis was wrong: openwakeword 0.6.0's
`--train_model` flag runs convert *inline* (there's no separate
`--convert_to_tflite` flag, despite the README/CLI suggesting otherwise), so
the prior fix's sanitization-then-`--convert_to_tflite` sequence never ran —
training crashed inside the `--train_model` invocation before reaching it.
Refixed by patching `openwakeword/train.py` on disk in the container to wrap
`convert_onnx_to_tflite` with ONNX-name sanitization, plus a skip-guard so
re-runs with an existing `.onnx` don't redo training.

## v4 (planned) — push recall via more positive variety

### Change from v3

- `n_samples` 20 000 → **50 000**

Picked over the alternatives (ship v3, or nudge `max_negative_weight` to
1050) because recall is the bigger UX gap: v3's 0.636 means ~36% of true
"hey babel" attempts silently fail, while v3's FP rate of 0.62/hour is one
spurious wake every ~97 min — annoying but tolerable. Per openWakeWord's own
guidance, doubling-plus the positive count is the recall lever; it expands
the model's exposure to phonetic / prosodic variation in the synthesized
positives, which usually lifts recall by 5–10 pts without much FP movement.

### Expected landing zone

Recall ~0.70–0.74, FP/hour ~0.5–0.8, accuracy ~0.83–0.85.

### Caveat — full retrain required

`n_samples` controls how many clips Piper TTS synthesizes during
`--generate_clips`. Changing it invalidates the existing
`_work/output/hey_babel/hey_babel/{positive,negative}_{train,test}/`
directories. Before `make train`, wipe the full output dir:

```sh
rm -rf _work/output/hey_babel/
make train
```

Wall-clock estimate: ~3–5 h on the 2060 (clips ~1.5–2.5 h, augment ~30 min,
train ~1–2 h, convert seconds).

### Decision tree from the v4 result

- **Recall ≥ 0.70 and FP/hour ≤ 0.7** → ship v4.
- **Recall ≥ 0.70 but FP/hour > 0.7** → v5: nudge `max_negative_weight: 1000
  → 1100` to claw FPs back, expecting modest recall cost (the more positive
  variety, the smaller that cost).
- **Recall improvement < 0.04** → diminishing returns from `n_samples`;
  next axis is either `layer_size: 64 → 96` for more capacity, or inspecting
  what's *actually* being missed (dump misclassified clips, look for a
  consistent failure mode like a specific accent or speed).
- **FP/hour rises significantly** → unexpected; means more positives are
  somehow leaking into the negative class. Re-examine
  `custom_negative_phrases` for accidental phonetic overlap.

## Background knobs on the bench

Deliberately held constant during current iteration so the single-lever
signal stays clean. Each is the right next move under specific conditions:

- **`n_samples: 20000`** — the README's "validate-the-pipeline" value. Bump to
  50000+ once the loss balance is right; gives diminishing returns on FP rate.
- **`custom_negative_phrases`** — the single highest-leverage FP-cleanup knob,
  but only useful when you know what's *actually* triggering. Right now we
  only have an aggregate FP rate. To investigate, dump the falsely-triggering
  clips from openWakeWord's validation set and listen — the words that
  recur are exactly what to add.
- **`model_type` / `layer_size: 64`** — DNN is openWakeWord's documented
  default; larger `layer_size` (96, 128) is the next axis to explore *only*
  if discrimination plateaus. Other backbones are larger experiments.
- **`steps: 50000`** — fine for the current config; revisit only if loss
  curves are still descending at the end of training (currently not).
