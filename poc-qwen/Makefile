# poc-qwen — Qwen3-TTS (mlx-audio) voice-cloning demo.
# Bare `make` installs whatever is missing and starts the Gradio app on :8007.

.DEFAULT_GOAL := run

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp

DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

.PHONY: run setup smoke bench spike test clean help

run: setup  ## Install if needed, then serve the Gradio app on http://127.0.0.1:8007
	@$(LOAD_ENV) $(PY) -m poc_qwen.app

setup: $(STAMP)  ## mise python 3.12, venv, deps, env probe (idempotent)

$(STAMP): requirements.txt mise.toml setup.sh
	@$(LOAD_ENV) ./setup.sh
	@touch $(STAMP)

smoke: setup  ## Go/no-go: clone voices/one-one.mp3 with 0.6B -> reports/smoke.wav
	@$(LOAD_ENV) $(PY) -m poc_qwen.smoke

bench: setup  ## Latency/RTF sweep -> reports/runs.jsonl
	@$(LOAD_ENV) $(PY) -m poc_qwen.bench

spike: setup  ## stream=True spike -> reports/stream_spike.jsonl
	@$(LOAD_ENV) $(PY) -m poc_qwen.spike_stream

test: setup  ## GPU-free unit tests
	@$(LOAD_ENV) $(PY) -m pytest tests -q

clean:  ## Remove the venv and caches. Leaves reports/ and the HF model cache.
	rm -rf $(VENV) .pytest_cache poc_qwen/__pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
