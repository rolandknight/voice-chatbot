# poc-babel — render af_heart (Kokoro, CPU only) reference-clip candidates.
# Bare `make` installs whatever is missing and writes out/*.mp3 + out/*.wav.

.DEFAULT_GOAL := render

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp

.PHONY: render setup test clean help

render: setup  ## Render the three variants -> out/
	@$(PY) render_variants.py

setup: $(STAMP)  ## mise python 3.10, venv, deps (idempotent)

$(STAMP): requirements.txt mise.toml setup.sh
	@./setup.sh
	@touch $(STAMP)

test: setup  ## Model-free unit tests for the audio helpers
	@$(PY) -m pytest tests -q

clean:  ## Remove venv, caches and rendered output. Keeps models/.
	rm -rf $(VENV) out .pytest_cache __pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
