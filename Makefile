# voice-chatbot — server + native WebRTC client (Cargo workspace in crates/).
# The PoC targets live on in archive/Makefile.old.
#
# Runtime artifacts live at the repo root (models/, .deps/, logs/; config in
# .env). The PoC trees are archived under archive/ (targets in
# archive/Makefile.old). Skills run in-process (crates/server/src/skills, docs/plans/skills-in-server.md).
# SERVER_FEATURES mirrors the Mac build profile: in-process Nemotron STT
# (.deps/nemo-speech) and Qwen3-TTS via PyO3 against crates/qwen-tts/.venv
# (make -C crates/qwen-tts setup).

.DEFAULT_GOAL := help

CARGO := cargo
SERVER_FEATURES ?= nemotron-native,qwen-tts
SERVER_BIN := target/release/voice-chatbot-server
CLIENT_BIN := target/release/voice-chatbot-client
# Raspberry Pi (64-bit Pi OS) cross-build: `cross` in Docker, not bin/cargo —
# Hermit's rustc carries x86_64 std only and is not rustup-managed, so cross
# runs against the system rustup toolchain (same 1.97.1) with the standard
# CARGO_HOME/RUSTUP_HOME instead of the Hermit ones it would otherwise inherit
# and fail to install a target into. Image setup lives in Cross.toml.
PI_TARGET ?= aarch64-unknown-linux-gnu
PI_TOOLCHAIN ?= 1.97.1
# Its own target dir: the container's host-side build scripts link against the
# image's glibc, so sharing target/ with the native build makes each one
# invalidate the other's artifacts.
PI_TARGET_DIR ?= target/pi
PI_CLIENT_BIN := $(PI_TARGET_DIR)/$(PI_TARGET)/release/voice-chatbot-client
# Deploying that binary to the Pi: `make deploy-pi PI_HOST=pi@raspberrypi.local`,
# or set PI_HOST in .env (read below via env_get, like SERVER_URL); a
# command-line PI_HOST=... still wins.
# PI_STAGE is relative to the Pi user's home -- rsync lands there unprivileged,
# and only the installer it runs needs sudo.
PI_HOST ?= $(call env_get,PI_HOST)
PI_DIR ?= /opt/voice-chatbot
PI_STAGE ?= .cache/voice-chatbot-deploy
PI_SERVICE ?= voice-chatbot-client
PI_CROSS_ENV := CARGO_HOME=$(HOME)/.cargo RUSTUP_HOME=$(HOME)/.rustup \
    PATH=$(HOME)/.cargo/bin:$$PATH CARGO_TARGET_DIR=$(PI_TARGET_DIR) \
    PKG_CONFIG_ALLOW_CROSS=1 PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig
# Installing cross needs the same homes the build runs it under. Hermit sets
# CARGO_HOME into .hermit/rust, so a bare `cargo install` would put the binary
# somewhere PI_CROSS_ENV's PATH never looks.
PI_CARGO := CARGO_HOME=$(HOME)/.cargo RUSTUP_HOME=$(HOME)/.rustup $(HOME)/.cargo/bin/cargo
# Runtime config lives in .env, which both binaries parse for themselves
# (crates/env-file). make never reads it, so `make call` dialled the built-in
# default even with SERVER_URL set there. Take the values the `call` target
# hands the client from .env instead, as *defaults*: an exported SERVER_URL or
# `make call SERVER_URL=...` still wins, mirroring crates/env-file's "variables
# already set are never overridden". The client reads .env for itself now, so
# these scrapes are belt-and-braces: they keep `make call VAR=...` working and
# keep the values visible in the command line make echoes. `include .env` is
# not an option — the file is shared with the Python chatbot (python-dotenv
# grammar), where a single line without an `=` is a fatal makefile syntax
# error. env_get mirrors that lenient parse instead: optional `export ` prefix,
# surrounding quotes stripped, an unquoted value ending at ` #`, first
# occurrence wins, anything unparsable skipped.
ENV_FILE ?= .env
env_get = $(shell [ -f '$(ENV_FILE)' ] && sed -n 's/^[[:space:]]*\(export[[:space:]][[:space:]]*\)\{0,1\}$(1)[[:space:]]*=[[:space:]]*//p' '$(ENV_FILE)' | head -n 1 | sed -e '/^["'\'']/!s/[[:space:]][[:space:]]*\#.*$$//' -e '/^["'\'']/!s/[[:space:]]*$$//' -e 's/^"\(.*\)"$$/\1/' -e "s/^'\(.*\)'$$/\1/")

SERVER_URL ?= $(or $(call env_get,SERVER_URL),http://127.0.0.1:6210)
LOG_LEVEL ?= $(or $(call env_get,LOG_LEVEL),info)
INPUT_DEVICE ?= $(call env_get,INPUT_DEVICE)
OUTPUT_DEVICE ?= $(call env_get,OUTPUT_DEVICE)
QWEN_PYTHON := $(abspath crates/qwen-tts/.venv/bin/python)
NEMO_SPEECH_LIB_DIR := $(abspath .deps/nemo-speech/v0.1.0/lib)
SERVER_BUILD_ENV := PYO3_PYTHON=$(QWEN_PYTHON) NEMO_SPEECH_LIB_DIR=$(NEMO_SPEECH_LIB_DIR)
# Package-qualified features for workspace-wide cargo invocations.
comma := ,
WS_FEATURES := $(subst $(eval) ,$(comma),$(addprefix voice-chatbot-server/,$(subst $(comma), ,$(SERVER_FEATURES))))

.PHONY: build server-build client-build client-build-pi deploy-pi server call devices sfx-up sfx-down sfx-status test check clean help

build: server-build client-build  ## Build server + client (release)

server-build:  ## Build crates/server with SERVER_FEATURES
	$(SERVER_BUILD_ENV) $(CARGO) build --release -p voice-chatbot-server --features "$(SERVER_FEATURES)"

client-build:  ## Build crates/client
	$(CARGO) build --release -p voice-chatbot-client

client-build-pi:  ## Cross-build the client for a Raspberry Pi (aarch64; needs Docker; installs cross on first use)
	@command -v rustup >/dev/null 2>&1 || { echo "rustup not found; cross builds against the rustup toolchain, not Hermit's"; exit 1; }
	@command -v cross >/dev/null 2>&1 || [ -x "$(HOME)/.cargo/bin/cross" ] || { \
	    echo "cross not found; installing it into $(HOME)/.cargo/bin (one-off, a few minutes)"; \
	    $(PI_CARGO) install cross --locked; \
	}
	@docker info >/dev/null 2>&1 || { echo "cross needs a running Docker daemon"; exit 1; }
	$(PI_CROSS_ENV) cross +$(PI_TOOLCHAIN) build --release --target $(PI_TARGET) -p voice-chatbot-client
	@echo "built $(PI_CLIENT_BIN)"
	@file $(PI_CLIENT_BIN) 2>/dev/null || true

deploy-pi: client-build-pi  ## Ship the cross-built client to a Pi and install the autostart service (PI_HOST=pi@host)
	@[ -n "$(PI_HOST)" ] || { echo "set PI_HOST, e.g. make deploy-pi PI_HOST=pi@raspberrypi.local"; exit 1; }
	@ssh $(PI_HOST) 'command -v rsync >/dev/null' || { echo "the Pi has no rsync (sudo apt install rsync)"; exit 1; }
	ssh $(PI_HOST) 'mkdir -p $(PI_STAGE)/models/wakeword'
	rsync -az $(PI_CLIENT_BIN) $(PI_HOST):$(PI_STAGE)/voice-chatbot-client
	rsync -az deploy/rpi/ $(PI_HOST):$(PI_STAGE)/
	rsync -az --delete --include='hey_*.onnx' --exclude='*' \
	    models/wakeword/ $(PI_HOST):$(PI_STAGE)/models/wakeword/
	ssh -t $(PI_HOST) 'sudo env INSTALL_DIR=$(PI_DIR) SERVICE_NAME=$(PI_SERVICE) \
	    "$$HOME/$(PI_STAGE)/install.sh"'

server: server-build  ## Build if needed, then run the server (reads .env)
	./$(SERVER_BIN)

call: client-build   ## Build the client if needed, then call the server (SERVER_URL, defaulted from .env) with native audio
	./$(CLIENT_BIN) --log-level "$(LOG_LEVEL)" call --server-url "$(SERVER_URL)" \
	    $(if $(INPUT_DEVICE),--input-device "$(INPUT_DEVICE)",) \
	    $(if $(OUTPUT_DEVICE),--output-device "$(OUTPUT_DEVICE)",)

devices: client-build  ## List native capture/playback devices
	./$(CLIENT_BIN) devices

# Sound-effect generators for the generate_sound_effect skill: Woosh (Sony)
# on :8005 and Stable Audio Open on :8006, each a separate Python model server
# under vendor/ (first launch clones + installs + downloads weights; see
# scripts/start_woosh.sh and scripts/start_stable_audio.sh). Pid files and logs
# match run.sh's so either launcher can stop what the other started.
WOOSH_PORT ?= 8005
STABLE_AUDIO_PORT ?= 8006

sfx-up:  ## Start the Woosh + Stable Audio Open servers in the background
	@$(call sfx_start,woosh,scripts/start_woosh.sh,$(WOOSH_PORT))
	@$(call sfx_start,stable-audio,scripts/start_stable_audio.sh,$(STABLE_AUDIO_PORT))

sfx-down:  ## Stop the sound-effect servers
	@$(call sfx_stop,woosh,$(WOOSH_PORT))
	@$(call sfx_stop,stable-audio,$(STABLE_AUDIO_PORT))

sfx-status:  ## Show whether the sound-effect servers answer
	@for s in "woosh $(WOOSH_PORT)" "stable-audio $(STABLE_AUDIO_PORT)"; do set -- $$s; \
	  if curl -sS -m 2 "http://127.0.0.1:$$2/docs" >/dev/null 2>&1; then echo "$$1: up on :$$2"; else echo "$$1: down"; fi; done

# $(1)=name $(2)=launcher $(3)=port. WOOSH_PORT/STABLE_AUDIO_PORT are read by the launchers.
define sfx_start
if curl -sS -m 2 "http://127.0.0.1:$(3)/docs" >/dev/null 2>&1; then echo "$(1): already up on :$(3)"; else \
  mkdir -p vendor; WOOSH_PORT=$(WOOSH_PORT) STABLE_AUDIO_PORT=$(STABLE_AUDIO_PORT) nohup ./$(2) >vendor/$(1).log 2>&1 & echo $$! >vendor/$(1).pid; \
  echo "$(1): starting on :$(3) (log: vendor/$(1).log; first launch installs models and can take many minutes)"; fi
endef

# Kill the launcher's process group (the launcher exec's uvicorn, so its pid is the server's).
define sfx_stop
if [ -f vendor/$(1).pid ] && kill -0 "$$(cat vendor/$(1).pid)" 2>/dev/null; then kill "$$(cat vendor/$(1).pid)" && echo "$(1): stopped"; \
elif pid=$$(lsof -nP -tiTCP:$(3) -sTCP:LISTEN 2>/dev/null | head -1); [ -n "$$pid" ]; then kill "$$pid" && echo "$(1): stopped (pid $$pid)"; \
else echo "$(1): not running"; fi; rm -f vendor/$(1).pid
endef

test:  ## Workspace unit tests (Rust, then the qwen-tts Python package)
	$(SERVER_BUILD_ENV) $(CARGO) test --release --workspace --features "$(WS_FEATURES)"
	$(MAKE) -C crates/qwen-tts test-py

check:  ## fmt --check, clippy -D warnings, tests
	$(CARGO) fmt --all -- --check
	$(SERVER_BUILD_ENV) $(CARGO) clippy --release --workspace --features "$(WS_FEATURES)" --all-targets -- -D warnings
	$(MAKE) test

clean:  ## Drop workspace build output
	rm -rf target

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
