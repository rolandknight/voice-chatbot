# qwen-tts-tester — the model-testing GUI and bench for crates/qwen-tts. Bare
# `make` builds and serves the UI on :8008. Run from this directory or with -C.

.DEFAULT_GOAL := run

QWEN := ../crates/qwen-tts
export PYO3_PYTHON := $(abspath $(QWEN)/.venv/bin/python)
CARGO := ../bin/cargo
BIN := ../target/release/qwen-tts-tester

DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

.PHONY: run explore setup build info bench e2e clean help

run: build  ## Build if needed, then serve the UI on http://127.0.0.1:8008 (gui.yaml)
	@$(LOAD_ENV) $(BIN) serve

explore: build  ## Serve :8008 with explore.yaml (CustomVoice preloaded, LRU 2) for voice comparisons
	@$(LOAD_ENV) $(BIN) --config explore.yaml serve

setup:  ## The crate's venv (mise python 3.12 + mlx-audio); idempotent
	@$(MAKE) -C $(QWEN) setup

build: setup  ## Release build against the crate's venv Python
	@$(CARGO) build --release -p qwen-tts-tester

info: build  ## Print model_info() from the embedded engine and exit
	@$(LOAD_ENV) $(BIN) info

bench: build  ## Headless TTFA bench, three sentences -> reports/rs_runs.jsonl + reports/bench_*.wav
	@$(LOAD_ENV) $(BIN) bench

e2e: build  ## Real server + real model over /ws (GPU): client-side TTFA
	@$(QWEN)/.venv/bin/python tests/e2e_ws.py

clean:  ## Drop uploads. Build output is the workspace target/; the venv is the crate's.
	rm -rf uploads

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
