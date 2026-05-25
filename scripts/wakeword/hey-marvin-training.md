# `hey marvin` training iterations

Living log of training runs for the "hey marvin" wake word: hyperparameters
used, metrics produced, and the reasoning behind each change. Add a new
section per run; preserve prior ones — knowing what *didn't* work is as
load-bearing as knowing what did.

The config under tuning lives in `hey_marvin.yml`. Background on what each
knob does is in `README.md` → "Tuning knobs". This file is the per-phrase
log; the README is the phrase-agnostic reference. Iteration history for the
sibling phrase lives in `hey-babel-training.md`.

**Target:** recall ≥ 0.70 with FP/hour ≤ 0.5.

## Summary

| Run | `max_neg_weight` | pos batch | `layer_size` | aug rounds | `fp_target` | `n_samples` | recall | FP/hr | accuracy |
|---|---|---|---|---|---|---|---|---|---|

(No runs yet. v1 starts from the hey_babel v3 hyperparameters since they're
the best-validated config for this pipeline; expect to retune negatives
once we see real false-positive behavior.)
