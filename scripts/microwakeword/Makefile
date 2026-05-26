# Convenience wrapper around train.sh for both phrases + eval + firmware install.
# All real work is in train.sh / entrypoint.sh; this is just shortcuts.

SHELL := /usr/bin/env bash

WORK_DIR  := _work
OUT_DIR   := $(WORK_DIR)/output
FW_MODELS := ../../firmware/box3/main/models

# Sibling openWakeWord pipeline. mit_rirs and background_clips are byte-identical
# between the two pipelines, so we can copy them over instead of re-downloading.
SIBLING_DATA    := ../wakeword/_work/data
SHARED_DATASETS := mit_rirs background_clips

BABEL_SRC    := $(OUT_DIR)/hey_babel/hey_babel.tflite
MARVIN_SRC   := $(OUT_DIR)/hey_marvin/hey_marvin.tflite
BABEL_DEST   := $(FW_MODELS)/hey_babel.tflite
MARVIN_DEST  := $(FW_MODELS)/hey_marvin.tflite

.PHONY: all babel marvin eval install install-babel install-marvin copy-from-wakeword clean help

help:
	@echo "Targets:"
	@echo "  make babel              - train the 'hey babel' model"
	@echo "  make marvin             - train the 'hey marvin' model"
	@echo "  make all                - train both"
	@echo "  make copy-from-wakeword - copy mit_rirs + background_clips from scripts/wakeword/_work/data/"
	@echo "  make eval               - run the streaming FAR/FRR test on the latest .tflite"
	@echo "  make install            - copy both .tflite into firmware/box3/main/models/"
	@echo "  make install-babel      - copy just hey_babel.tflite"
	@echo "  make install-marvin     - copy just hey_marvin.tflite"
	@echo "  make clean              - remove _work/output (keeps _work/data caches)"

all: babel marvin

babel:
	MWW_CONFIG=hey_babel.yml ./train.sh

marvin:
	MWW_CONFIG=hey_marvin.yml ./train.sh

# Streaming evaluation. Reuses model_train_eval with --train 0 to re-test the
# already-trained model against the test + ambient sets and write a FAR/FRR
# ROC report into the train_dir. Same image as training.
#
# We --entrypoint=python because the image's default entrypoint is the training
# script; we want to invoke the eval CLI directly with the saved YAML.
eval:
	@for phrase in hey_babel hey_marvin; do \
	    yaml=$(OUT_DIR)/$$phrase/training_parameters.yaml; \
	    model=$(OUT_DIR)/$$phrase/$$phrase.tflite; \
	    if [ -f "$$yaml" ] && [ -f "$$model" ]; then \
	        echo "==> Evaluating $$phrase"; \
	        docker run --rm --gpus all \
	            -v "$$PWD/_work:/work" \
	            --entrypoint python \
	            microwakeword-trainer:latest \
	            -m microwakeword.model_train_eval \
	                --training_config "/work/output/$$phrase/training_parameters.yaml" \
	                --train 0 \
	                --test_tflite_streaming_quantized 1 \
	                mixednet \
	                --pointwise_filters "64,64,64,64" \
	                --repeat_in_block "1,1,1,1" \
	                --mixconv_kernel_sizes "[5],[7,11],[9,15],[23]" \
	                --residual_connection "0,0,0,0" \
	                --first_conv_filters 32 \
	                --first_conv_kernel_size 5 \
	                --stride 3; \
	        echo "ROC report: $(OUT_DIR)/$$phrase/checkpoints/tflite_stream_state_internal_quant/tflite_streaming_roc.txt"; \
	    fi; \
	done

# Copy the trained int8 models into the firmware tree so `idf.py build` (or
# `make build` under firmware/box3/) can embed them via EMBED_FILES. Same effect
# as `make install` from firmware/box3/, just available on the trainer side too.
install: install-babel install-marvin

install-babel: | $(FW_MODELS)
	@test -f $(BABEL_SRC) || { \
	    echo "error: $(BABEL_SRC) not found."; \
	    echo "Train it first:  make babel"; \
	    exit 1; }
	cp -v $(BABEL_SRC) $(BABEL_DEST)

install-marvin: | $(FW_MODELS)
	@test -f $(MARVIN_SRC) || { \
	    echo "error: $(MARVIN_SRC) not found."; \
	    echo "Train it first:  make marvin"; \
	    exit 1; }
	cp -v $(MARVIN_SRC) $(MARVIN_DEST)

$(FW_MODELS):
	mkdir -p $@

# Copy mit_rirs and background_clips from the openWakeWord pipeline's cache.
# Both corpora are byte-identical between the two trainers (same HF repos, same
# download code), so re-downloading 30+ GB is wasted bandwidth. Skips any
# dataset that's already populated; errors if the sibling cache is missing.
copy-from-wakeword:
	@mkdir -p $(WORK_DIR)/data
	@for shared in $(SHARED_DATASETS); do \
	    src="$(SIBLING_DATA)/$$shared"; \
	    dst="$(WORK_DIR)/data/$$shared"; \
	    if [ -L "$$dst" ]; then \
	        echo "removing dangling symlink $$dst"; \
	        rm -f "$$dst"; \
	    fi; \
	    if [ -d "$$dst" ] && [ -n "$$(ls -A "$$dst" 2>/dev/null)" ]; then \
	        echo "skip $$shared: already populated at $$dst"; \
	        continue; \
	    fi; \
	    if [ ! -d "$$src" ]; then \
	        echo "error: $$src not found. Train the openWakeWord pipeline first (cd ../wakeword && make babel)."; \
	        exit 1; \
	    fi; \
	    echo "copying $$shared from $$src to $$dst (this may take a few minutes)"; \
	    sudo cp -r --preserve=timestamps "$$src" "$$dst"; \
	done

clean:
	rm -rf _work/output
