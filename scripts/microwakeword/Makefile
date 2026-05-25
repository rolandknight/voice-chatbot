# Convenience wrapper around train.sh for both phrases + an eval target.
# All real work is in train.sh / entrypoint.sh; this is just shortcuts.

SHELL := /usr/bin/env bash

.PHONY: all babel marvin eval clean help

help:
	@echo "Targets:"
	@echo "  make babel   - train the 'hey babel' model"
	@echo "  make marvin  - train the 'hey marvin' model"
	@echo "  make all     - train both"
	@echo "  make eval    - run the streaming FAR/FRR test on the latest .tflite"
	@echo "  make clean   - remove _work/output (keeps _work/data caches)"

all: babel marvin

babel:
	MWW_CONFIG=hey_babel.yml ./train.sh

marvin:
	MWW_CONFIG=hey_marvin.yml ./train.sh

# Streaming evaluation. Runs the int8 .tflite against a held-out test set and
# reports false-accepts per hour and false-reject rate. Same image as training.
eval:
	@for phrase in hey_babel hey_marvin; do \
	    model=_work/output/$$phrase/$$phrase.tflite; \
	    if [ -f "$$model" ]; then \
	        echo "==> Evaluating $$phrase"; \
	        docker run --rm --gpus all \
	            -v "$$PWD/_work:/work" \
	            microwakeword-trainer:latest \
	            python -m microwakeword.eval \
	                --model "/work/output/$$phrase/$$phrase.tflite" \
	                --config "/work/config/$$phrase.yml"; \
	    fi; \
	done

clean:
	rm -rf _work/output
