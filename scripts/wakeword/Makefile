# Convenience targets for the openWakeWord training pipeline. Run from this
# directory (scripts/wakeword/).
#
# One Docker image (wakeword-trainer:latest) handles every phrase; per-phrase
# state lives in _work/output/<model_name>/ keyed off the YAML's `model_name`
# field. Add a new wakeword by dropping a `<model_name>.yml` next to the
# existing configs and appending it to the PHRASES list below.
#
# `make train` — trains every phrase in PHRASES sequentially (currently
#   hey_babel, hey_marvin, then hey_one_one). Equivalent to running
#   `train-hey-babel`, `train-hey-marvin`, and `train-hey-one-one` back to
#   back.
#
# `make train-hey-babel` / `make train-hey-marvin` / `make train-hey-one-one`
#   — train a single phrase.
#
# Two clean variants per phrase because clip regeneration is expensive (1–2 h
# of Piper TTS) and most config changes don't need it:
#
# `make clean-<phrase>` — surgical. Drops the trained model + the four
#   augmented-feature .npy files but keeps the ~1.3 GB of synthesized Piper
#   clips under _work/output/<phrase>/<phrase>/{positive,negative}_{train,
#   test}/. Use when you've only tuned training hyperparameters
#   (max_negative_weight, layer_size, batch_n_per_class, augmentation_rounds,
#   ...). The clips are still valid because they only depend on
#   `target_phrase`, `custom_negative_phrases`, and `n_samples`.
#
# `make clean-all-<phrase>` — full wipe of _work/output/<phrase>/. Use when
#   you've changed `target_phrase`, `custom_negative_phrases`, or `n_samples`
#   — the clips are stale and need to be regenerated.
#
# `make clean` / `make clean-all` — apply the above to every phrase in PHRASES.
#
# All clean targets deliberately leave _work/data/ (MIT RIRs, ACAV features,
# background clips, HF cache — tens of GB of phrase-independent downloads)
# intact.
#
# `make install` — copy every trained <phrase>.onnx + <phrase>.tflite to
#   $(DEST), defaulting to ../../models/wakeword/ (the voice-chatbot app's
#   model dir). Override with `make install DEST=/path/to/somewhere`.
#   `make install-<phrase>` copies just one phrase's artifacts.

DEST ?= ../../models/wakeword

# Underscore-named model identifiers — must match each YAML's `model_name`
# field and the .onnx/.tflite filenames openWakeWord emits. User-facing
# target names dash-ify these (train-hey-babel, not train-hey_babel) since
# make targets conventionally use dashes.
PHRASES := hey_babel hey_marvin hey_one_one
TARGETS := $(subst _,-,$(PHRASES))

.PHONY: train clean clean-all install \
        $(addprefix train-,$(TARGETS)) \
        $(addprefix clean-,$(TARGETS)) \
        $(addprefix clean-all-,$(TARGETS)) \
        $(addprefix install-,$(TARGETS))

train: $(addprefix train-,$(TARGETS))
clean: $(addprefix clean-,$(TARGETS))
clean-all: $(addprefix clean-all-,$(TARGETS))
install: $(addprefix install-,$(TARGETS))

# Per-phrase rules. $(1) = underscore model name (hey_babel); $(2) = dash
# target name (hey-babel). One `define` + `eval` per phrase keeps the rule
# list a single source of truth as PHRASES grows.
define PHRASE_RULES
train-$(2):
	WAKEWORD_CONFIG=$(1).yml ./train.sh

clean-$(2):
	sudo rm -f _work/output/$(1)/$(1).onnx
	sudo rm -f _work/output/$(1)/$(1).tflite
	sudo rm -f _work/output/$(1)/$(1)/*_features_*.npy

clean-all-$(2):
	sudo rm -rf _work/output/$(1)/

install-$(2):
	mkdir -p $$(DEST)
	cp _work/output/$(1)/$(1).onnx  $$(DEST)/
	cp _work/output/$(1)/$(1).tflite $$(DEST)/
endef

$(foreach phrase,$(PHRASES),$(eval $(call PHRASE_RULES,$(phrase),$(subst _,-,$(phrase)))))
