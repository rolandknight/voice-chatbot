# poc-gemma4 — does Ollama reuse the prefill (system prompt + tool schemas) across
# turns for gemma4:26b, and does warm TTFT meet the chatbot budget?
# Bare `make` installs what is missing, ensures Ollama is up, and runs the probe.

.DEFAULT_GOAL := run

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp
DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

.PHONY: run setup build serve probe test test-live clean help

run: setup serve  ## Install if needed, start Ollama if needed, run the cache/TTFT probe -> reports/probe.jsonl
	@$(LOAD_ENV) $(PY) -m poc_gemma4.probe

setup: $(STAMP)  ## mise python 3.12, venv, deps (idempotent)

$(STAMP): requirements.txt mise.toml setup.sh
	@$(LOAD_ENV) ./setup.sh
	@touch $(STAMP)

build: setup  ## Pull the model into Ollama
	@$(LOAD_ENV) $(PY) -m poc_gemma4.ollama pull

serve: setup  ## Start `ollama serve` in the background if it is not answering
	@$(LOAD_ENV) $(PY) -m poc_gemma4.ollama ensure

probe: run  ## alias for run

test: setup  ## GPU-free unit tests (Ollama mocked)
	@$(LOAD_ENV) $(PY) -m pytest tests -v -m "not live"

test-live: setup serve  ## Live assertions against Ollama + gemma4:26b (loads the 19 GB model)
	@$(LOAD_ENV) $(PY) -m pytest tests -v -m live

clean:  ## Remove the venv and caches. Leaves reports/.
	rm -rf $(VENV) .pytest_cache poc_gemma4/__pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
