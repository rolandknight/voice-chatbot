.PHONY: flowcat-client-build flowcat-client-devices flowcat-client-run flowcat-client-test flowcat-client-check poc-doctor poc-setup poc-moonshine-setup poc-nemotron-setup poc-build poc-chatterbox poc-up poc-down poc-test poc-test-all poc-results poc-tts-setup poc-tts poc-tts-bench poc-tts-test poc-qwen poc-qwen-setup poc-qwen-bench poc-qwen-test poc-qwen-streaming poc-qwen-streaming-build poc-qwen-streaming-bench poc-qwen-streaming-test poc-gemma4 poc-gemma4-test poc-gemma4-test-live help install-server install-server-os install-client install-client-os install-service run run-webrtc-smoke run-webrtc-smoke-lan run-server run-server-lan run-server-local run-server-lan-local run-webrtc-client run-rpi-client-local run-jabra run-wake-test run-wake-client

# Homebrew packages the server needs (macOS). Keep in sync with install_mac.sh.
BREW_PKGS := portaudio ffmpeg mpv librespot git cmake pkg-config ollama corelocationcli

# apt packages the on-device (RPi 5) wake client needs. sounddevice binds to
# the system PortAudio library, which is not a pip dep. Keep in sync with the
# precheck in devices/rpi5/install_rpi.sh.
APT_PKGS := libportaudio2 portaudio19-dev

OFFER_URL ?= http://localhost:8080/api/offer

# install-service defaults: the RPi 5 voice-server IP, and whether to install a
# PipeWire user service (1) or a bare-ALSA system service (0).
SERVER_IP ?= 192.168.0.245
USER_SERVICE ?= 1

CERT_DIR := .certs
CERT := $(CERT_DIR)/cert.pem
KEY := $(CERT_DIR)/key.pem

help:
	@echo "Targets:"
	@echo "  install-server-os         - install the server's OS packages (Homebrew)"
	@echo "  install-server            - install the server's Python packages (pip -> Hermit env)"
	@echo "  install-client-os         - install the RPi 5 client's OS packages (apt: PortAudio)"
	@echo "  install-client            - install the RPi 5 client's Python packages (devices/rpi5/install_rpi.sh)"
	@echo "  install-service           - install the RPi 5 client as an auto-start service (SERVER_IP=$(SERVER_IP), USER_SERVICE=$(USER_SERVICE))"
	@echo "  run                       - legacy local-audio backend (./run.sh)"
	@echo "  run-server                - WebRTC backend on http://localhost:8080"
	@echo "  run-server-lan            - WebRTC backend on HTTPS, reachable from LAN"
	@echo "  run-server-local          - WebRTC + always-on Jabra (LocalAudio, wake mode)"
	@echo "  run-server-lan-local      - run-server-lan + always-on Jabra"
	@echo "  run-webrtc-smoke          - smoke loopback on http://localhost:8080"
	@echo "  run-webrtc-smoke-lan      - smoke loopback on HTTPS, reachable from LAN"
	@echo "  run-webrtc-client         - run the WebRTC client (OS-aware; OFFER_URL=... to target a server)"
	@echo "  run-rpi-client-local      - run-webrtc-client against localhost (macOS dev loop)"
	@echo "  run-jabra                 - macOS dev loop, mic+speaker on the Jabra, server-side wake"
	@echo "  run-wake-test             - on-device openWakeWord test (mic only, no server)"
	@echo "  run-wake-client           - full on-device-wake loop HERE (connects only after wake)"
	@echo "  poc-doctor                - show detected PoC platform, STT backend, Opus, and Chatterbox"
	@echo "  poc-setup                 - create PoC venv and download test models"
	@echo "  poc-moonshine-setup       - install pinned native Moonshine + streaming model"
	@echo "  poc-nemotron-setup        - install pinned NVIDIA Nemotron streaming runtime + model"
	@echo "  poc-build                 - build FlowCat for selected local STT backend"
	@echo "  poc-chatterbox            - run cloned-voice server (macOS/Linux auto-detected)"
	@echo "  poc-test                  - run one PoC marker (POC_MARKER=smoke by default)"
	@echo "  flowcat-client-build      - build the native Rust audio/WebRTC client"
	@echo "  flowcat-client-devices    - list native input/output devices and stable IDs"
	@echo "  flowcat-client-run        - connect selected native devices to FlowCat"
	@echo "  flowcat-client-check      - fmt, clippy, and test the native client"

install-server-os:
	brew install $(BREW_PKGS)

# Installs into the active Python env — run `. bin/activate-hermit` first so
# pip targets the Hermit toolchain.
install-server:
	python -m pip install -r requirements.txt

# On-device (RPi 5) wake client. install-client-os installs the system
# PortAudio that sounddevice binds to; install-client installs the Python
# packages (delegating to install_rpi.sh, which also handles openWakeWord's
# --no-deps quirk). Run `. bin/activate-hermit` first so pip targets the
# Hermit toolchain, same as install-server.
install-client-os:
	sudo apt install -y $(APT_PKGS)

install-client:
	./devices/rpi5/install_rpi.sh

# Install the RPi 5 voice client as an auto-start systemd service (delegates to
# devices/rpi5/install_service.sh). Defaults to a PipeWire user service
# (USER_SERVICE=1) pointed at SERVER_IP=192.168.0.245. Override the server with
# `make install-service SERVER_IP=10.0.0.5`, force a bare-ALSA system service
# with USER_SERVICE=0, or pass INPUT_DEVICE=/OUTPUT_DEVICE=/AUTH_TOKEN=. Run on
# the Pi as your normal user (not sudo).
install-service:
	SERVER_IP=$(SERVER_IP) USER_SERVICE=$(USER_SERVICE) \
	  $(if $(INPUT_DEVICE),INPUT_DEVICE=$(INPUT_DEVICE),) \
	  $(if $(OUTPUT_DEVICE),OUTPUT_DEVICE=$(OUTPUT_DEVICE),) \
	  $(if $(AUTH_TOKEN),AUTH_TOKEN=$(AUTH_TOKEN),) \
	  ./devices/rpi5/install_service.sh

run:
	./run.sh

run-webrtc-smoke:
	python webrtc_smoke/server.py

run-webrtc-smoke-lan: $(CERT)
	@echo ""
	@echo "First visit from another machine: accept the self-signed cert warning."
	@echo "If the OS firewall prompts, allow incoming connections for Python."
	@echo ""
	WEBRTC_SSL_CERT=$(CERT) WEBRTC_SSL_KEY=$(KEY) python webrtc_smoke/server.py

run-server:
	python server.py

run-server-lan: $(CERT)
	@echo ""
	@echo "First visit from another machine: accept the self-signed cert warning."
	@echo "If the OS firewall prompts, allow incoming connections for Python."
	@echo ""
	WEBRTC_SSL_CERT=$(CERT) WEBRTC_SSL_KEY=$(KEY) python server.py

run-server-local:
	python server.py --local-audio

run-server-lan-local: $(CERT)
	@echo ""
	@echo "First visit from another machine: accept the self-signed cert warning."
	@echo "If the OS firewall prompts, allow incoming connections for Python."
	@echo ""
	WEBRTC_SSL_CERT=$(CERT) WEBRTC_SSL_KEY=$(KEY) python server.py --local-audio

# General WebRTC client. Audio backend auto-selects per OS (alsa on Linux /
# the Pi, avfoundation on macOS); override with AUDIO_FORMAT=. Point at any
# server with OFFER_URL=https://host:8080/api/offer. Pick devices with
# INPUT_DEVICE=/OUTPUT_DEVICE= (list mics on macOS:
# ffmpeg -f avfoundation -list_devices true -i "").
run-webrtc-client:
	@echo ""
	@echo "WebRTC client -> $(OFFER_URL)"
	@echo ""
	python devices/rpi5/rpi_webrtc_voice.py \
	  --offer-url $(OFFER_URL) \
	  $(if $(AUDIO_FORMAT),--audio-format $(AUDIO_FORMAT),) \
	  $(if $(INPUT_DEVICE),--input-device $(INPUT_DEVICE),) \
	  $(if $(OUTPUT_DEVICE),--output-device $(OUTPUT_DEVICE),) \
	  $(if $(MODE),--mode $(MODE),) \
	  $(if $(PERSONA),--persona $(PERSONA),) \
	  $(if $(BACKEND),--backend $(BACKEND),)

# Known-good macOS dev loop. Both capture AND playback go through the Jabra
# Speak2 40 so its hardware AEC cancels the TTS out of the mic — without this,
# the Mac's speaker output is picked up by the mic and (a) interrupts the bot
# mid-sentence and (b) transcribes as a new turn, looping forever. Server-side
# wake ('hey babel' / 'hey marvin') gates turns and picks the persona. Every
# value is a default you can override, e.g. `make run-jabra MODE=push PERSONA=marvin`.
run-jabra: OFFER_URL := http://localhost:8080/api/offer
run-jabra: INPUT_DEVICE := :0
run-jabra: OUTPUT_DEVICE := Jabra
run-jabra: MODE := wake
run-jabra:
	@echo ""
	@echo "Jabra dev loop -> $(OFFER_URL)  (mode=$(MODE) in=$(INPUT_DEVICE) out=$(OUTPUT_DEVICE))"
	@echo "Pair with 'make run-server'. Say 'hey babel' or 'hey marvin' to wake + pick the voice."
	@echo ""
	@$(MAKE) run-webrtc-client \
	  OFFER_URL=$(OFFER_URL) INPUT_DEVICE=$(INPUT_DEVICE) OUTPUT_DEVICE=$(OUTPUT_DEVICE) \
	  MODE=$(MODE) $(if $(PERSONA),PERSONA=$(PERSONA),) $(if $(BACKEND),BACKEND=$(BACKEND),)

# Convenience: the client on this machine against a local `make run-server`.
run-rpi-client-local: OFFER_URL := http://localhost:8080/api/offer
run-rpi-client-local:
	@echo ""
	@echo "Pair with 'make run-server' (which does NOT bind local audio)."
	@echo "Override the speaker with OUTPUT_DEVICE='Jabra'; mic with INPUT_DEVICE=':2'."
	@$(MAKE) run-webrtc-client OFFER_URL=$(OFFER_URL) \
	  $(if $(INPUT_DEVICE),INPUT_DEVICE=$(INPUT_DEVICE),) \
	  $(if $(OUTPUT_DEVICE),OUTPUT_DEVICE=$(OUTPUT_DEVICE),) \
	  $(if $(MODE),MODE=$(MODE),) \
	  $(if $(PERSONA),PERSONA=$(PERSONA),) \
	  $(if $(BACKEND),BACKEND=$(BACKEND),)

run-wake-test:
	python devices/rpi5/wake_test.py \
	  $(if $(INPUT_DEVICE),--device $(INPUT_DEVICE),) \
	  $(if $(THRESHOLD),--threshold $(THRESHOLD),)

# Full on-device-wake loop HERE: the client runs openWakeWord and only connects
# after "hey babel"/"hey marvin". Capture+playback on the Jabra (its AEC keeps
# TTS out of the mic). Pair with `make run-server`. Override with THRESHOLD=,
# SESSION_TIMEOUT=, INPUT_DEVICE=, OUTPUT_DEVICE=.
run-wake-client: OFFER_URL := http://192.168.0.245:8080/api/offer
run-wake-client: INPUT_DEVICE ?= Jabra
run-wake-client: OUTPUT_DEVICE ?= Jabra
run-wake-client:
	@echo ""
	@echo "On-device wake loop -> $(OFFER_URL)  (in=$(INPUT_DEVICE) out=$(OUTPUT_DEVICE))"
	@echo "Say 'hey babel' or 'hey marvin'; it connects only after wake."
	@echo ""
	python devices/rpi5/rpi_webrtc_voice.py --local-wake \
	  --offer-url $(OFFER_URL) --input-device $(INPUT_DEVICE) --output-device $(OUTPUT_DEVICE) \
	  $(if $(THRESHOLD),--threshold $(THRESHOLD),) \
	  $(if $(SESSION_TIMEOUT),--session-timeout $(SESSION_TIMEOUT),)

$(CERT):
	@mkdir -p $(CERT_DIR)
	@SAN="DNS:localhost,IP:127.0.0.1"; \
	for iface in en0 en1 en2 en3; do \
	  ip=$$(ipconfig getifaddr $$iface 2>/dev/null); \
	  if [ -n "$$ip" ]; then SAN="$$SAN,IP:$$ip"; fi; \
	done; \
	echo "Generating self-signed cert (SAN: $$SAN)"; \
	openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
	  -keyout $(KEY) -out $(CERT) \
	  -subj "/CN=voice-chatbot-dev" \
	  -addext "subjectAltName=$$SAN" >/dev/null 2>&1

# ---- FlowCat PoC (docs/poc/flowcat-poc-plan.md; branch poc-python-harness) ----
# Mac quickstart:  brew install cmake pkg-config opus
#                  echo 'OPENROUTER_API_KEY=sk-or-...' > poc/.env
#                  make poc-setup poc-build poc-test
POC_PY := poc/.venv/bin/python
POC_MARKER ?= smoke
FLOWCAT_CLIENT_MANIFEST := poc/flowcat-client/Cargo.toml
FLOWCAT_URL ?= http://127.0.0.1:6210
LOG_LEVEL ?= info
FLOWCAT_CLIENT_PKG_CONFIG := $(abspath poc/.deps/prefix/lib/pkgconfig)

.PHONY: flowcat-client-build flowcat-client-devices flowcat-client-run flowcat-client-test flowcat-client-check poc-doctor poc-setup poc-moonshine-setup poc-nemotron-setup poc-build poc-chatterbox poc-up poc-down poc-test poc-test-all poc-results poc-tts-setup poc-tts poc-tts-bench poc-tts-test poc-tts-streaming-setup poc-tts-streaming poc-tts-streaming-bench poc-tts-streaming-test

flowcat-client-build:  ## Build the native Rust CPAL/WebRTC client
	@PKG_CONFIG_PATH="$(FLOWCAT_CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  cargo build --locked --manifest-path $(FLOWCAT_CLIENT_MANIFEST)

flowcat-client-devices:  ## List native capture/playback devices
	@PKG_CONFIG_PATH="$(FLOWCAT_CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  cargo run --locked --quiet --manifest-path $(FLOWCAT_CLIENT_MANIFEST) -- devices

flowcat-client-run:  ## Run native audio against the FlowCat PoC server
	@PKG_CONFIG_PATH="$(FLOWCAT_CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  cargo run --locked --quiet --manifest-path $(FLOWCAT_CLIENT_MANIFEST) -- call \
	    --server-url "$(FLOWCAT_URL)" --log-level "$(LOG_LEVEL)" \
	    $(if $(INPUT_DEVICE),--input-device "$(INPUT_DEVICE)",) \
	    $(if $(OUTPUT_DEVICE),--output-device "$(OUTPUT_DEVICE)",)

flowcat-client-test:  ## Run native-client unit tests
	@PKG_CONFIG_PATH="$(FLOWCAT_CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  cargo test --locked --manifest-path $(FLOWCAT_CLIENT_MANIFEST)

flowcat-client-check:  ## Format, lint, and test the native client
	@cargo fmt --manifest-path $(FLOWCAT_CLIENT_MANIFEST) -- --check
	@PKG_CONFIG_PATH="$(FLOWCAT_CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  cargo clippy --locked --manifest-path $(FLOWCAT_CLIENT_MANIFEST) --all-targets -- -D warnings
	@PKG_CONFIG_PATH="$(FLOWCAT_CLIENT_PKG_CONFIG):$${PKG_CONFIG_PATH:-}" \
	  cargo test --locked --manifest-path $(FLOWCAT_CLIENT_MANIFEST)

poc-doctor:  ## PoC: verify platform-specific build/runtime prerequisites
	@./poc/platform.sh doctor
	@if grep -Eq '^POC_TTS_BACKEND=chatterbox$$' poc/.env 2>/dev/null; then \
	  ./scripts/start_chatterbox.sh --doctor; \
	fi

poc-setup:  ## PoC: python venv, deps, fixtures, models (idempotent)
	@test -f poc/.env || { echo "ERROR: poc/.env missing — needs OPENROUTER_API_KEY (see poc/.env.example)"; exit 1; }
	@mkdir -p poc/models poc/logs
	@test -f poc/models/silero_vad.onnx || curl -sL -o poc/models/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
	@test -f poc/models/ggml-base.en.bin || curl -sL -o poc/models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
	@test -d poc/.venv || python3 -m venv poc/.venv
	@$(POC_PY) -m pip install -q -r poc/requirements.txt
	@if grep -Eq '^POC_STT_BACKEND=moonshine$$' poc/.env 2>/dev/null; then ./scripts/setup_moonshine.sh; fi
	@if grep -Eq '^POC_STT_BACKEND=(nemotron|nvidia)$$' poc/.env 2>/dev/null; then ./scripts/setup_nemotron.sh; fi
	@cd poc && .venv/bin/python -m harness.make_fixtures
	@echo "poc-setup done"

poc-moonshine-setup:  ## PoC: install pinned Moonshine native runtime + streaming model
	@./scripts/setup_moonshine.sh

poc-nemotron-setup:  ## PoC: install pinned NVIDIA Nemotron runtime + English streaming model
	@./scripts/setup_nemotron.sh

poc-build:  ## PoC: build FlowCat with OS-aware STT acceleration
	@./poc/platform.sh build

poc-chatterbox:  ## PoC: run OS-aware cloned-voice TTS sidecar in the foreground
	@./scripts/start_chatterbox.sh

poc-up:     ## PoC: start selected local STT/TTS sidecars + stubs + FlowCat
	cd poc && ./run_poc.sh up

poc-down:   ## PoC: stop the stack
	cd poc && ./run_poc.sh down

poc-test:   ## PoC: bring the stack up, run one marker (POC_MARKER=smoke|tools|duplex|wake|voice|latency|soak), tear down
	cd poc && ./run_poc.sh down >/dev/null 2>&1 || true
	cd poc && ./run_poc.sh up
	cd poc && .venv/bin/python -m pytest harness -m $(POC_MARKER) -q || { ./run_poc.sh down; exit 1; }
	cd poc && ./run_poc.sh down

poc-test-all:  ## PoC: full T1-T12 suite (wake/voice need their own env, see plan)
	cd poc && ./run_poc.sh down >/dev/null 2>&1 || true
	cd poc && ./run_poc.sh up
	cd poc && .venv/bin/python -m pytest harness -q -m "not wake and not voice" || { ./run_poc.sh down; exit 1; }
	cd poc && ./run_poc.sh down
	@echo "NOTE: wake test:  POC_WAKE_MODEL=\$$PWD/models/wakeword/hey_babel.onnx make poc-up && cd poc && .venv/bin/python -m pytest harness -m wake"
	@echo "NOTE: voice test: needs Chatterbox on :8004 and POC_TTS_BACKEND=chatterbox (docs/poc/flowcat-poc-plan.md Phase 1b)"

poc-results:  ## PoC: show recorded performance results (poc/reports/runs.jsonl)
	@test -f poc/reports/runs.jsonl || { echo "no results yet — run make poc-test first"; exit 0; }
	@poc/.venv/bin/python -c "import json,sys; \
	rows=[json.loads(l) for l in open('poc/reports/runs.jsonl')]; \
	[print(f\"{r['ts']}  {r['host']:12.12s} {r['os']:6.6s} {r['test']:28.28s} llm={r['llm_model'].split('/')[-1]:24.24s} stt={r.get('stt_backend','whisper'):9.9s}:{r.get('stt_model',r.get('whisper','?')):16.16s}/{r.get('stt_accelerator','?'):5.5s} tts={r['tts_backend']:10.10s}/{r.get('chatterbox_device','-'):5.5s} \" + ' '.join(f'{k}={v}' for k,v in r['results'].items())) for r in rows]"

poc-tts-setup:  ## poc-tts: mise python 3.10, venv, deps, flashinfer probe (idempotent)
	@$(MAKE) -C poc-tts setup

poc-qwen:  ## poc-qwen: Qwen3-TTS (mlx-audio) Gradio demo on :8007
	@$(MAKE) -C poc-qwen run

poc-qwen-setup:  ## poc-qwen: mise python 3.12, venv, deps, env probe (idempotent)
	@$(MAKE) -C poc-qwen setup

poc-qwen-bench:  ## poc-qwen: latency/RTF sweep -> poc-qwen/reports/runs.jsonl
	@$(MAKE) -C poc-qwen bench

poc-qwen-test:  ## poc-qwen: GPU-free unit tests
	@$(MAKE) -C poc-qwen test

poc-qwen-streaming:  ## poc-qwen-streaming: Rust+PyO3 server streaming Qwen3-TTS over WebSocket on :8008
	@$(MAKE) -C poc-qwen-streaming run

poc-qwen-streaming-build:  ## poc-qwen-streaming: release build against poc-qwen's venv Python
	@$(MAKE) -C poc-qwen-streaming build

poc-qwen-streaming-bench:  ## poc-qwen-streaming: headless TTFA bench -> poc-qwen-streaming/reports/rs_runs.jsonl
	@$(MAKE) -C poc-qwen-streaming bench

poc-qwen-streaming-test:  ## poc-qwen-streaming: GPU-free bridge + Rust unit tests
	@$(MAKE) -C poc-qwen-streaming test

poc-gemma4:  ## poc-gemma4: prefix-cache + TTFT probe of gemma4:26b on Ollama -> poc-gemma4/reports/probe.jsonl
	@$(MAKE) -C poc-gemma4 run

poc-gemma4-test:  ## poc-gemma4: GPU-free unit tests
	@$(MAKE) -C poc-gemma4 test

poc-gemma4-test-live:  ## poc-gemma4: live assertions against Ollama + gemma4:26b
	@$(MAKE) -C poc-gemma4 test-live

poc-tts:    ## poc-tts: run the Chatterbox Flash server + GUI on :8005
	@$(MAKE) -C poc-tts run

poc-tts-bench:  ## poc-tts: sweep Flash tuning configs, append poc-tts/reports/runs.jsonl
	@$(MAKE) -C poc-tts bench

poc-tts-test:  ## poc-tts: GPU-free unit tests
	@$(MAKE) -C poc-tts test

poc-tts-streaming-setup:  ## poc-tts-streaming: mise python 3.10, venv, deps, aiortc probe (idempotent)
	@$(MAKE) -C poc-tts-streaming setup

poc-tts-streaming:    ## poc-tts-streaming: Flash streamed over Realtime/WebRTC on :8006
	@$(MAKE) -C poc-tts-streaming run

poc-tts-streaming-test:  ## poc-tts-streaming: GPU-free unit + loopback tests
	@$(MAKE) -C poc-tts-streaming test

poc-tts-streaming-bench:  ## poc-tts-streaming: streaming TTFA bench -> poc-tts-streaming/reports/stream_runs.jsonl
	@$(MAKE) -C poc-tts-streaming bench-stream
