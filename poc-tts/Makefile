# poc-tts — Chatterbox Flash PoC.
# Bare `make` installs whatever is missing and starts the app on :8005.

.DEFAULT_GOAL := run

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp

# Load poc-tts/.env if present (e.g. CUDA_HOME=$HOME/cuda-12.4) and export its
# variables to recipes that run Python. Sourced in the shell rather than
# pulled in with make's `include`, because `include` would make parse $HOME as
# a make variable reference ($H followed by literal OME) instead of letting
# the shell expand it. Guarded by $(wildcard .env) so an absent file is fine.
DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

.PHONY: run setup bench test clean help

run: setup  ## Install if needed, then serve the Flash GUI on http://127.0.0.1:8005
	@$(LOAD_ENV) $(PY) -m poc_tts.server

setup: $(STAMP)  ## mise python 3.10, venv, deps, FlashInfer probe (idempotent)

# The stamp records a completed setup. It is newer than requirements.txt and
# mise.toml after a successful run, so repeat `make` invocations skip straight
# to run; editing either file makes setup rerun.
$(STAMP): requirements.txt mise.toml setup.sh
	@$(LOAD_ENV) ./setup.sh
	@touch $(STAMP)

bench: setup  ## Sweep Flash tuning configs -> reports/runs.jsonl
	@$(LOAD_ENV) $(PY) -m poc_tts.bench

test: setup  ## GPU-free unit tests
	@$(LOAD_ENV) $(PY) -m pytest tests -v

clean:  ## Remove the venv and caches. Leaves reports/ and the HF model cache.
	rm -rf $(VENV) .pytest_cache poc_tts/__pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
