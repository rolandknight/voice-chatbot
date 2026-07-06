#!/usr/bin/env bash
# Install + enable a systemd service that auto-starts the RPi 5 wake client on
# boot (and restarts it if it crashes). Run as your NORMAL user — the script
# uses sudo only for the privileged steps:
#
#   ./install_service.sh
#
# Toolchain is managed by Hermit: the service sources ./bin/activate-hermit to
# put `python` (and everything else) on PATH, then execs the client.
#
# Override any default with an env var, e.g.:
#   SERVER_IP=192.168.0.245 INPUT_DEVICE=Jabra OUTPUT_DEVICE=Jabra ./install_service.sh
#   AUTH_TOKEN=secret ./install_service.sh
set -euo pipefail

SERVER_IP="${SERVER_IP:-192.168.0.245}"
SERVER_PORT="${SERVER_PORT:-8080}"
INPUT_DEVICE="${INPUT_DEVICE:-Jabra}"
OUTPUT_DEVICE="${OUTPUT_DEVICE:-Jabra}"
SERVICE_NAME="${SERVICE_NAME:-rpi-voice}"

# Repo root = two levels up from devices/rpi5/.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

# Hermit provides the toolchain; the service activates it before running.
if [ ! -f "$REPO/bin/activate-hermit" ]; then
  echo "ERROR: Hermit not found at $REPO/bin/activate-hermit" >&2
  echo "Set up Hermit for this repo first (it manages python, etc.)." >&2
  exit 1
fi

# Run the service as the invoking user (works whether or not sudo'd).
RUN_USER="${SUDO_USER:-$USER}"
OFFER_URL="http://${SERVER_IP}:${SERVER_PORT}/api/offer"
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

# Optional auth token (Step F). Only added if AUTH_TOKEN is set.
EXTRA=""
[ -n "${AUTH_TOKEN:-}" ] && EXTRA=" --auth-token ${AUTH_TOKEN}"

# The client command, run under an activated Hermit env. `exec` replaces bash so
# systemd's main PID is python (clean stop/restart signalling).
CLIENT_CMD="python devices/rpi5/rpi_webrtc_voice.py --local-wake --offer-url ${OFFER_URL} --input-device ${INPUT_DEVICE} --output-device ${OUTPUT_DEVICE}${EXTRA}"

echo "Installing ${SERVICE_NAME}.service:"
echo "  repo:    $REPO"
echo "  runtime: Hermit (source ./bin/activate-hermit)"
echo "  user:    $RUN_USER"
echo "  server:  $OFFER_URL"
echo "  devices: in=$INPUT_DEVICE out=$OUTPUT_DEVICE"

sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=voice-chatbot Raspberry Pi 5 WebRTC wake client
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStart=/bin/bash -c 'source ./bin/activate-hermit && exec $CLIENT_CMD'
Restart=on-failure
RestartSec=3
User=$RUN_USER
SupplementaryGroups=audio

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo
echo "Started. Useful commands:"
echo "  journalctl -u $SERVICE_NAME -f      # follow logs"
echo "  sudo systemctl restart $SERVICE_NAME"
echo "  sudo systemctl stop $SERVICE_NAME"
echo "  sudo systemctl disable $SERVICE_NAME  # stop auto-start on boot"
