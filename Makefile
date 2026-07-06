.PHONY: help run run-webrtc-smoke run-webrtc-smoke-lan run-server run-server-lan run-server-local run-server-lan-local run-webrtc-client run-rpi-client-local run-jabra run-wake-test run-wake-client

OFFER_URL ?= http://localhost:8080/api/offer

CERT_DIR := .certs
CERT := $(CERT_DIR)/cert.pem
KEY := $(CERT_DIR)/key.pem

help:
	@echo "Targets:"
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

run:
	./run.sh

run-webrtc-smoke:
	.venv/bin/python webrtc_smoke/server.py

run-webrtc-smoke-lan: $(CERT)
	@echo ""
	@echo "First visit from another machine: accept the self-signed cert warning."
	@echo "If the OS firewall prompts, allow incoming connections for Python."
	@echo ""
	WEBRTC_SSL_CERT=$(CERT) WEBRTC_SSL_KEY=$(KEY) .venv/bin/python webrtc_smoke/server.py

run-server:
	.venv/bin/python server.py

run-server-lan: $(CERT)
	@echo ""
	@echo "First visit from another machine: accept the self-signed cert warning."
	@echo "If the OS firewall prompts, allow incoming connections for Python."
	@echo ""
	WEBRTC_SSL_CERT=$(CERT) WEBRTC_SSL_KEY=$(KEY) .venv/bin/python server.py

run-server-local:
	.venv/bin/python server.py --local-audio

run-server-lan-local: $(CERT)
	@echo ""
	@echo "First visit from another machine: accept the self-signed cert warning."
	@echo "If the OS firewall prompts, allow incoming connections for Python."
	@echo ""
	WEBRTC_SSL_CERT=$(CERT) WEBRTC_SSL_KEY=$(KEY) .venv/bin/python server.py --local-audio

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
	.venv/bin/python devices/rpi5/wake_test.py \
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
