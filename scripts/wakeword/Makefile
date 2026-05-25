# Convenience targets for the openWakeWord training pipeline. Run from this
# directory (scripts/wakeword/).
#
# Two clean variants because clip regeneration is expensive (1–2 h of Piper
# TTS) and most config changes don't need it:
#
# `make clean` — surgical. Drops the trained model + the four augmented-feature
#   .npy files but keeps the ~1.3 GB of synthesized Piper clips under
#   _work/output/hey_babel/hey_babel/{positive,negative}_{train,test}/. Use
#   when you've only tuned training hyperparameters (max_negative_weight,
#   layer_size, batch_n_per_class, augmentation_rounds, ...). The clips are
#   still valid because they only depend on `target_phrase`,
#   `custom_negative_phrases`, and `n_samples`.
#
# `make clean-all` — full wipe of _work/output/hey_babel/. Use when you've
#   changed `target_phrase`, `custom_negative_phrases`, or `n_samples` — the
#   clips are stale and need to be regenerated.
#
# Both clean targets deliberately leave _work/data/ (MIT RIRs, ACAV features,
# background clips, HF cache — tens of GB of phrase-independent downloads)
# intact.
#
# `make install` — copy the trained hey_babel.onnx + hey_babel.tflite to
#   $(DEST), defaulting to ../../models/wakeword/ (the voice-chatbot app's
#   model dir). Override with `make install DEST=/path/to/somewhere`.

DEST ?= ../../models/wakeword

.PHONY: train clean clean-all install

train:
	./train.sh

clean:
	rm -f _work/output/hey_babel/hey_babel.onnx
	rm -f _work/output/hey_babel/hey_babel.tflite
	rm -f _work/output/hey_babel/hey_babel/*_features_*.npy

clean-all:
	rm -rf _work/output/hey_babel/

install:
	mkdir -p $(DEST)
	cp _work/output/hey_babel/hey_babel.onnx  $(DEST)/
	cp _work/output/hey_babel/hey_babel.tflite $(DEST)/
