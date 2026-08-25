# FlowCat PoC — local voice stack on this machine (docs/poc/flowcat-poc-plan.md).
# Bare `make` installs whatever is missing, builds the Rust embedder, makes sure
# Ollama is serving with the LLM resident, and brings up STT sidecar + stubs +
# TTS shim + FlowCat on http://127.0.0.1:6210. `make client` then connects the
# native mic/speaker client. Run from this directory.
#
# The toolchain is mise-pinned (mise.toml: python 3.12, rust 1.97.1); the rest
# of the repo stays on hermit bin/. All knobs live in the untracked .env
# (cp .env.example .env): POC_STT_BACKEND, OPENROUTER_BASE_URL, POC_LLM_MODEL...

.DEFAULT_GOAL := run

REPO := $(abspath ..)
VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.setup-stamp
# mise-pinned toolchain (mise.toml). platform.sh calls bare `cargo`, so it is
# wrapped in `mise exec` below too.
CARGO := mise exec -- cargo

DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

FLOWCAT_BIN := flowcat/target/debug/flowcat-poc
CLIENT_MANIFEST := flowcat-client/Cargo.toml
CLIENT_PKG_CONFIG := $(abspath .deps/prefix/lib/pkgconfig)
FLOWCAT_URL ?= http://127.0.0.1:6210
LOG_LEVEL ?= info
POC_MARKER ?= smoke
# Ollama base URL for `make status`; the ollama target itself reads .env.
OLLAMA_URL ?= http://127.0.0.1:11434

.PHONY: run setup build ollama up down restart status client client-devices client-build \
        test test-all doctor logs clean help

run: setup build ollama up  ## Install if needed, build, ensure Ollama+model, start the stack on :6210

setup: $(STAMP)  ## mise python 3.12, venv, deps, models, fixtures, STT sidecar runtime (idempotent)

$(STAMP): requirements.txt mise.toml setup.sh $(DOTENV)
	@./setup.sh
	@touch $(STAMP)

build: setup  ## Build flowcat-poc for the STT backend in .env (platform.sh picks Metal/CPU)
	@mise exec -- ./platform.sh build

ollama: setup  ## ollama serve (OLLAMA_KEEP_ALIVE=-1), pull if missing, prewarm the FlowCat prefix via /v1
	@$(LOAD_ENV) $(PY) ollama_ctl.py ensure

up: setup  ## Start STT sidecar (if nemotron) + stubs + kokoro shim + FlowCat; no-op for parts already running
	@./run_poc.sh up

down:  ## Stop everything run_poc.sh started (Ollama and Chatterbox are left alone)
	@./run_poc.sh down

restart: down up  ## Restart the stack, e.g. after `make build` or editing .env

status:  ## Show what is listening and which model Ollama has resident
	@lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -E ':(6210|8790|8880|8178|8004|11434) ' || true
	@curl -fsS -m 2 $(OLLAMA_URL)/api/ps 2>/dev/null | $(PY) -c 'import json,sys; [print("ollama resident:", m["name"], m.get("size_vram",0)//2**20, "MB") for m in json.load(sys.stdin).get("models",[])]' || true
	@curl -fsS -m 2 $(FLOWCAT_URL)/healthz && echo " <- flowcat healthz" || echo "flowcat: not healthy"

client-build:  ## Build the native CPAL/WebRTC client
	@PKG_CONFIG_PATH="$(CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  $(CARGO) build --locked --manifest-path $(CLIENT_MANIFEST)

client-devices:  ## List CoreAudio capture/playback devices
	@PKG_CONFIG_PATH="$(CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  $(CARGO) run --locked --quiet --manifest-path $(CLIENT_MANIFEST) -- devices

client:  ## Connect mic/speaker to the running stack: make client [INPUT_DEVICE=Jabra OUTPUT_DEVICE=Jabra]
	@PKG_CONFIG_PATH="$(CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  $(CARGO) run --locked --quiet --manifest-path $(CLIENT_MANIFEST) -- call \
	    --server-url "$(FLOWCAT_URL)" --log-level "$(LOG_LEVEL)" \
	    $(if $(INPUT_DEVICE),--input-device "$(INPUT_DEVICE)",) \
	    $(if $(OUTPUT_DEVICE),--output-device "$(OUTPUT_DEVICE)",)

test: setup ollama  ## Run one pytest marker against a fresh stack (POC_MARKER=smoke|tools|duplex|wake|voice|latency|soak)
	@./run_poc.sh down >/dev/null 2>&1 || true
	@./run_poc.sh up
	@$(PY) -m pytest harness -m $(POC_MARKER) -q || { ./run_poc.sh down; exit 1; }
	@./run_poc.sh down

test-all: setup ollama  ## Full suite minus wake/voice
	@./run_poc.sh down >/dev/null 2>&1 || true
	@./run_poc.sh up
	@$(PY) -m pytest harness -q -m "not wake and not voice" || { ./run_poc.sh down; exit 1; }
	@./run_poc.sh down

doctor:  ## Verify platform prerequisites for the backend in .env
	@./platform.sh doctor

logs:  ## Tail the stack logs
	@tail -n 20 -F logs/flowcat.log logs/nemotron.log logs/stubs.log logs/kokoro.log logs/ollama.log 2>/dev/null

clean:  ## Remove the venv and caches. Leaves models/, .deps/, logs/, reports/ and cargo targets.
	rm -rf $(VENV) .pytest_cache harness/__pycache__ stubs/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
