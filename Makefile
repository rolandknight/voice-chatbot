# voice-chatbot — server + native WebRTC client (Cargo workspace in crates/).
# The PoC targets live on in Makefile.old.
#
# Runtime artifacts (models, logs) and poc/.env are still read from poc/; skills
# run in-process (crates/server/src/skills, docs/plans/skills-in-server.md).
# SERVER_FEATURES mirrors the Mac build profile: in-process Nemotron STT
# (poc/.deps/nemo-speech) and Qwen3-TTS via PyO3 against crates/qwen-tts/.venv
# (make -C crates/qwen-tts setup).

.DEFAULT_GOAL := help

CARGO := bin/cargo
SERVER_FEATURES ?= nemotron-native,qwen-tts
SERVER_BIN := target/release/voice-chatbot-server
CLIENT_BIN := target/release/voice-chatbot-client
SERVER_URL ?= http://127.0.0.1:6210
LOG_LEVEL ?= info
QWEN_PYTHON := $(abspath crates/qwen-tts/.venv/bin/python)
NEMO_SPEECH_LIB_DIR := $(abspath poc/.deps/nemo-speech/v0.1.0/lib)
SERVER_BUILD_ENV := PYO3_PYTHON=$(QWEN_PYTHON) NEMO_SPEECH_LIB_DIR=$(NEMO_SPEECH_LIB_DIR)
# Package-qualified features for workspace-wide cargo invocations.
comma := ,
WS_FEATURES := $(subst $(eval) ,$(comma),$(addprefix voice-chatbot-server/,$(subst $(comma), ,$(SERVER_FEATURES))))

.PHONY: build server-build client-build server call devices sfx-up sfx-down sfx-status test check clean help

build: server-build client-build  ## Build server + client (release)

server-build:  ## Build crates/server with SERVER_FEATURES
	$(SERVER_BUILD_ENV) $(CARGO) build --release -p voice-chatbot-server --features "$(SERVER_FEATURES)"

client-build:  ## Build crates/client
	$(CARGO) build --release -p voice-chatbot-client

server: server-build  ## Build if needed, then run the server (reads poc/.env)
	./$(SERVER_BIN)

call: client-build   ## Build the client if needed, then call the server (SERVER_URL) with native audio
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
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
