# poc-qwen — Qwen3-TTS (mlx-audio) voice-cloning demo with a Qwen3-TTS-Space-style GUI.
# Bare `make` installs whatever is missing and starts the Gradio app on :8007.

.DEFAULT_GOAL := run

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp

# Load poc-qwen/.env if present (e.g. HOST=0.0.0.0 or POC_QWEN_SERVER_PORT=8010)
# and export its variables to recipes that run Python. Sourced in the shell
# rather than pulled in with make's `include`, because `include` would make
# parse $HOME as a make variable reference ($H followed by literal OME) instead
# of letting the shell expand it. Guarded by $(wildcard .env) so an absent file
# is fine.
DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

MODELS := mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
          mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 \
          mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16 \
          mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
          mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \
          mlx-community/whisper-base.en-mlx

.PHONY: run setup models smoke bench bench-stream test clean help

run: setup  ## Install if needed, then serve the Gradio GUI on http://127.0.0.1:8007
	@$(LOAD_ENV) $(PY) -m poc_qwen.app

setup: $(STAMP)  ## mise python 3.12, venv, deps, MLX env probe (idempotent)

# The stamp records a completed setup. It is newer than requirements.txt and
# mise.toml after a successful run, so repeat `make` invocations skip straight
# to run; editing either file makes setup rerun.
$(STAMP): requirements.txt mise.toml setup.sh
	@$(LOAD_ENV) ./setup.sh
	@touch $(STAMP)

models: setup  ## Pre-download every model the GUI can select into the HF cache (~14 GB)
	@$(LOAD_ENV) $(PY) -c 'from huggingface_hub import snapshot_download; import sys; [print(snapshot_download(m)) for m in sys.argv[1:]]' $(MODELS)

smoke: setup  ## Go/no-go: clone voices/one-one.mp3 with 0.6B -> reports/smoke.wav
	@$(LOAD_ENV) $(PY) -m poc_qwen.smoke

bench: setup  ## Whole-utterance latency/RTF sweep, 0.6B + 1.7B -> reports/runs.jsonl
	@$(LOAD_ENV) $(PY) -m poc_qwen.bench

bench-stream: setup  ## stream=True time-to-first-chunk spike -> reports/stream_spike.jsonl
	@$(LOAD_ENV) $(PY) -m poc_qwen.spike_stream

test: setup  ## GPU-free unit tests
	@$(LOAD_ENV) $(PY) -m pytest tests -v

clean:  ## Remove the venv and caches. Leaves reports/ and the HF model cache.
	rm -rf $(VENV) .pytest_cache poc_qwen/__pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
