#!/usr/bin/env bash
# Install + enable a systemd service that auto-starts the RPi 5 wake client on
# boot (and restarts it if it crashes). Run as your NORMAL user.
#
#   ./install_service.sh                 # system service (bare-ALSA setups)
#   USER_SERVICE=1 ./install_service.sh  # user service, routed through PipeWire
#
# USER_SERVICE=1 installs a `systemctl --user` service and enables linger so it
# starts at boot without a login. Use this when PipeWire owns the sound card
# (default on Raspberry Pi OS Bookworm): the client then shares the card with
# librespot (see install_librespot.sh) via PipeWire, so the bot voice and
# Spotify mix. It defaults the audio devices to `pulse` (the ALSA→PipeWire PCM).
#
# Toolchain is managed by Hermit: the service sources ./bin/activate-hermit to
# put `python` (and everything else) on PATH, then execs the client.
#
# Override any default with an env var, e.g.:
#   SERVER_IP=192.168.0.245 INPUT_DEVICE=pulse OUTPUT_DEVICE=pulse USER_SERVICE=1 ./install_service.sh
#   AUTH_TOKEN=secret ./install_service.sh
set -euo pipefail

USER_SERVICE="${USER_SERVICE:-0}"
SERVER_IP="${SERVER_IP:-192.168.0.245}"
SERVER_PORT="${SERVER_PORT:-8080}"
SERVICE_NAME="${SERVICE_NAME:-rpi-voice}"

# Under PipeWire (user service) default to the `pulse` PCM so the client mixes
# with librespot; bare-ALSA (system service) defaults to the Jabra directly.
if [ "$USER_SERVICE" = "1" ]; then
  _DEFAULT_DEV="pulse"
else
  _DEFAULT_DEV="Jabra"
fi
INPUT_DEVICE="${INPUT_DEVICE:-$_DEFAULT_DEV}"
OUTPUT_DEVICE="${OUTPUT_DEVICE:-$_DEFAULT_DEV}"

# Repo root = two levels up from devices/rpi5/.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

# Hermit provides the toolchain; the service activates it before running.
if [ ! -f "$REPO/bin/activate-hermit" ]; then
  echo "ERROR: Hermit not found at $REPO/bin/activate-hermit" >&2
  echo "Set up Hermit for this repo first (it manages python, etc.)." >&2
  exit 1
fi

OFFER_URL="http://${SERVER_IP}:${SERVER_PORT}/api/offer"

# Optional auth token. Only added if AUTH_TOKEN is set.
EXTRA=""
[ -n "${AUTH_TOKEN:-}" ] && EXTRA=" --auth-token ${AUTH_TOKEN}"

# The client command, run under an activated Hermit env. `exec` replaces bash so
# systemd's main PID is python (clean stop/restart signalling).
CLIENT_CMD="python devices/rpi5/rpi_webrtc_voice.py --local-wake --offer-url ${OFFER_URL} --input-device ${INPUT_DEVICE} --output-device ${OUTPUT_DEVICE}${EXTRA}"

if [ "$USER_SERVICE" = "1" ]; then
  # ---- User service (PipeWire) -------------------------------------------
  if [ "$(id -u)" = "0" ]; then
    echo "ERROR: run WITHOUT sudo for USER_SERVICE=1 — it must live in your" >&2
    echo "       PipeWire session. It sudo's only for the linger step." >&2
    exit 1
  fi
  UNIT_DIR="$HOME/.config/systemd/user"
  UNIT="$UNIT_DIR/${SERVICE_NAME}.service"
  mkdir -p "$UNIT_DIR"

  echo "Installing ${SERVICE_NAME}.service (user, via PipeWire):"
  echo "  repo:    $REPO"
  echo "  user:    $USER"
  echo "  server:  $OFFER_URL"
  echo "  devices: in=$INPUT_DEVICE out=$OUTPUT_DEVICE"

  cat > "$UNIT" <<EOF
[Unit]
Description=voice-chatbot Raspberry Pi 5 WebRTC wake client (user, PipeWire)
# Needs PipeWire (and its pulse bridge) up so the alsa "pulse" PCM resolves.
After=pipewire.service pipewire-pulse.service wireplumber.service
Wants=pipewire-pulse.service

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStart=/bin/bash -c 'source ./bin/activate-hermit && exec $CLIENT_CMD'
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE_NAME"
  # Linger so the user manager (and PipeWire + this service) starts at boot
  # without an interactive login.
  sudo loginctl enable-linger "$USER"

  echo
  echo "Started. Useful commands:"
  echo "  journalctl --user -u $SERVICE_NAME -f     # follow logs"
  echo "  systemctl --user restart $SERVICE_NAME"
  echo "  systemctl --user stop $SERVICE_NAME"
  echo "  systemctl --user disable $SERVICE_NAME    # stop auto-start"
else
  # ---- System service (bare ALSA) ----------------------------------------
  RUN_USER="${SUDO_USER:-$USER}"
  UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

  echo "Installing ${SERVICE_NAME}.service (system):"
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
fi
