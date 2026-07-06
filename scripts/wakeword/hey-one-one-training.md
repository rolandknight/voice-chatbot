# `hey one one` training iterations

Living log of training runs for the "hey one one" wake word: hyperparameters
used, metrics produced, and the reasoning behind each change. Add a new
section per run; preserve prior ones — knowing what *didn't* work is as
load-bearing as knowing what did.

The config under tuning lives in `hey_one_one.yml`. Background on what each
knob does is in `README.md` → "Tuning knobs". This file is the per-phrase
log; the README is the phrase-agnostic reference. Iteration history for the
sibling phrases lives in `hey-babel-training.md` and `hey-marvin-training.md`.

**Target:** recall ≥ 0.70 with FP/hour ≤ 0.5.

## Summary

| Run | `max_neg_weight` | pos batch | `layer_size` | aug rounds | `fp_target` | `n_samples` | recall | FP/hr | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 1000 | 100 | 64 | 2 | 0.5 | 50000 | 0.563 | 0.088 | 0.781 |

## v1 — hey_babel-v3 starting hyperparameters

Config as committed in `hey_one_one.yml`: `max_negative_weight: 1000`,
`batch_n_per_class.positive: 100`, `layer_size: 64`, `augmentation_rounds: 2`,
`target_false_positives_per_hour: 0.5`, `n_samples: 50000`, `steps: 50000`.

**Result:** recall **0.563**, FP/hour **0.088**, accuracy **0.781**.

FP/hour is comfortably under the 0.5 target (0.088), but recall is well
short of the ≥ 0.70 goal — exactly the phonetically-thin-phrase concern
noted when the config was seeded. The loss balance is currently far on the
precision side: there's ~0.41 FP/hour of headroom to trade for recall.

**Next to try (in rough priority order):**
- Lower `max_negative_weight` (e.g. 1000 → 500) to relax the precision
  pressure now that FP/hour is 6× under target — the most direct recall lever.
- Raise `batch_n_per_class.positive` (100 → 150/200) to weight the positive
  class harder per step.
- Reconsider the hard-negatives list: "hey one" as a negative directly
  competes with the "hey one one" positive and may be suppressing recall on
  the leading two words. Consider dropping it and leaning on the longer
  embedding negatives instead.
