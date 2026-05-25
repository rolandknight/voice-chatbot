.PHONY: help run run-webrtc-smoke run-webrtc-smoke-lan run-server run-server-lan

CERT_DIR := .certs
CERT := $(CERT_DIR)/cert.pem
KEY := $(CERT_DIR)/key.pem

help:
	@echo "Targets:"
	@echo "  run                    - legacy local-audio backend (./run.sh)"
	@echo "  run-server             - WebRTC backend on http://localhost:8080"
	@echo "  run-server-lan         - WebRTC backend on HTTPS, reachable from LAN"
	@echo "  run-webrtc-smoke       - smoke loopback on http://localhost:8080"
	@echo "  run-webrtc-smoke-lan   - smoke loopback on HTTPS, reachable from LAN"

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
