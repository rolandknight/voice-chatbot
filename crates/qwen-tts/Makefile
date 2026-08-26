# crates/qwen-tts — the Python half of the crate. The Rust half builds with the
# root Makefile (make server-build), which links against this venv.

VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp

.PHONY: setup test-py clean help

setup: $(STAMP)  ## mise python 3.12, venv, deps, qwen_tts installed editable (idempotent)

# The stamp is newer than the inputs after a successful run, so repeat
# invocations skip setup; editing any input reruns it.
$(STAMP): requirements.txt mise.toml setup.sh python/pyproject.toml
	@./setup.sh
	@touch $(STAMP)

test-py: setup  ## GPU-free unit tests (fake model behind the engine's loader hook)
	@cd python && ../$(PY) -m pytest -q

clean:  ## Remove the venv and caches
	rm -rf $(VENV) python/.pytest_cache python/qwen_tts/__pycache__ python/tests/__pycache__ python/*.egg-info

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
