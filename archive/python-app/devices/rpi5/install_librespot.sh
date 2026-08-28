#!/usr/bin/env bash
# Install librespot as a *user* systemd service so Spotify plays natively out
# the Pi's speaker (e.g. the Jabra), routed through PipeWire — the same audio
# server the voice client uses, so the two mix automatically (bot voice + music
# at once, no ducking). The voice server never streams Spotify audio; it only
# issues Web API control commands to this librespot "Babel" Connect endpoint
# (see scripts/spotify.py).
#
# Why a USER service (not the raspotify system service): PipeWire is per-user
# (runs as you, socket under /run/user/<uid>). A system service running as the
# `raspotify` user can't reach your PipeWire session, so it can't mix with the
# voice client. Running librespot as YOUR user service — with linger so it
# starts at boot without a login — puts it in your PipeWire session.
#
# Usage:  ./install_librespot.sh
#         LIBRESPOT_NAME=Babel LIBRESPOT_DEVICE=pulse LIBRESPOT_BITRATE=320 ./install_librespot.sh
#
# LIBRESPOT_DEVICE is an ALSA PCM. Default "pulse" routes librespot into
# PipeWire (confirmed working via `speaker-test -D pulse`). It plays to
# PipeWire's *default sink*, which this script points at the Jabra.
set -euo pipefail

LIBRESPOT_NAME="${LIBRESPOT_NAME:-Babel}"
LIBRESPOT_DEVICE="${LIBRESPOT_DEVICE:-pulse}"
LIBRESPOT_BITRATE="${LIBRESPOT_BITRATE:-320}"
LIBRESPOT_VOLUME="${LIBRESPOT_INITIAL_VOLUME:-100}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/librespot.service"

if [ "$(id -u)" = "0" ]; then
  echo "ERROR: run this as your normal user (not root/sudo) — the service must" >&2
  echo "       live in your PipeWire session. It sudo's only for the linger step." >&2
  exit 1
fi

# --- Ensure a librespot binary exists -------------------------------------
# raspotify ships /usr/bin/librespot. We install it for the binary but disable
# its system service (which runs as the wrong user for PipeWire).
if ! command -v librespot >/dev/null 2>&1; then
  echo "Installing librespot (via raspotify, for the binary)..."
  curl -sSL https://dtcooper.github.io/raspotify/install.sh | sh
fi
LIBRESPOT_BIN="$(command -v librespot || echo /usr/bin/librespot)"
if [ ! -x "$LIBRESPOT_BIN" ]; then
  echo "ERROR: librespot binary not found after install." >&2
  exit 1
fi
# Stop + disable the raspotify system service so it doesn't grab the card as the
# raspotify user (harmless if it isn't installed).
sudo systemctl disable --now raspotify >/dev/null 2>&1 || true

# Undo a stale dmix /etc/asound.conf from an earlier setup_alsa_sharing.sh run —
# it redefines `default` as a raw-ALSA mixer that fights PipeWire and silences
# the `pulse`/`default` PCMs.
if [ -f /etc/asound.conf ] && grep -q "setup_alsa_sharing.sh" /etc/asound.conf 2>/dev/null; then
  echo "Removing stale dmix /etc/asound.conf (PipeWire mixes natively)..."
  sudo rm -f /etc/asound.conf
fi

# --- Point PipeWire's default sink at the Jabra ---------------------------
if command -v pactl >/dev/null 2>&1; then
  SINK="$(pactl list short sinks 2>/dev/null | awk 'tolower($0) ~ /jabra|speak/ {print $2; exit}')"
  if [ -n "$SINK" ]; then
    echo "Setting default sink -> $SINK"
    pactl set-default-sink "$SINK" || true
  else
    echo "WARNING: no Jabra sink found via pactl. Spotify may play out HDMI." >&2
    echo "         Fix with: pactl set-default-sink <name-from 'pactl list short sinks'>" >&2
  fi
fi

# --- Write + enable the user service --------------------------------------
mkdir -p "$UNIT_DIR"
echo "Writing $UNIT:"
echo "  name:    $LIBRESPOT_NAME"
echo "  device:  $LIBRESPOT_DEVICE (ALSA PCM -> PipeWire)"
echo "  bitrate: $LIBRESPOT_BITRATE"

cat > "$UNIT" <<EOF
[Unit]
Description=librespot Spotify Connect endpoint "$LIBRESPOT_NAME" (via PipeWire)
# Needs the PipeWire pulse bridge up before the alsa "pulse" PCM resolves.
After=pipewire.service pipewire-pulse.service wireplumber.service
Wants=pipewire-pulse.service

[Service]
ExecStart=$LIBRESPOT_BIN --name "$LIBRESPOT_NAME" --bitrate $LIBRESPOT_BITRATE \\
  --backend alsa --device $LIBRESPOT_DEVICE --initial-volume $LIBRESPOT_VOLUME
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now librespot.service

# Linger so the user manager (and thus PipeWire + this service) starts at boot
# without an interactive login.
sudo loginctl enable-linger "$USER"

echo
echo "Started. Useful commands:"
echo "  systemctl --user status librespot"
echo "  journalctl --user -u librespot -f"
echo
echo "Now bind it once: open Spotify on your phone → Connect → pick '$LIBRESPOT_NAME'."
echo "From the server, confirm it's visible: python scripts/spotify.py --list-devices"
