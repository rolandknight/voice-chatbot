# voice-chatbot — server + native WebRTC client (Cargo workspace in crates/).
# The PoC targets live on in Makefile.old.
#
# Runtime artifacts (models, stubs, logs) and poc/.env are still read from poc/;
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

.PHONY: build server-build client-build server stubs call devices test check clean help

build: server-build client-build  ## Build server + client (release)

server-build:  ## Build crates/server with SERVER_FEATURES
	$(SERVER_BUILD_ENV) $(CARGO) build --release -p voice-chatbot-server --features "$(SERVER_FEATURES)"

client-build:  ## Build crates/client
	$(CARGO) build --release -p voice-chatbot-client

server: server-build  ## Build if needed, then run the server (reads poc/.env)
	./$(SERVER_BIN)

stubs:  ## Start the skills stub server on :8790 (tools: time, weather, radio, Spotify)
	@if pgrep -f "uvicorn stub_server:app" >/dev/null; then echo "stubs already running"; else \
	  cd poc/stubs; nohup ../.venv/bin/uvicorn stub_server:app --host 127.0.0.1 --port 8790 \
	    >../logs/stubs.log 2>&1 & echo $$! > ../logs/stubs.pid; echo "stubs started (poc/logs/stubs.log)"; fi

call: build   ## Build if needed, then call the server with native audio
	./$(CLIENT_BIN) --log-level "$(LOG_LEVEL)" call --server-url "$(SERVER_URL)" \
	    $(if $(INPUT_DEVICE),--input-device "$(INPUT_DEVICE)",) \
	    $(if $(OUTPUT_DEVICE),--output-device "$(OUTPUT_DEVICE)",)

devices: client-build  ## List native capture/playback devices
	./$(CLIENT_BIN) devices

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
