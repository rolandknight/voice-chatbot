#!/usr/bin/env bash
# Install + configure librespot on the Raspberry Pi so Spotify plays *natively*
# out the Pi's speaker (e.g. the Jabra), at full 44.1 kHz stereo quality.
#
# The voice server never streams Spotify audio — it only issues Web API control
# commands targeting this librespot "Babel" Connect endpoint (see
# scripts/spotify.py). That replaces the old pipe-into-WebRTC path, which was
# choppy/staticky because it forced music through a 24 kHz-mono voice channel.
#
# We use raspotify (https://github.com/dtcooper/raspotify): a Debian package
# that bundles librespot with a systemd service. This script installs it (if
# missing), writes /etc/raspotify/conf, and restarts it.
#
# Usage:  ./install_librespot.sh
#         LIBRESPOT_NAME=Babel LIBRESPOT_DEVICE=default LIBRESPOT_BITRATE=320 ./install_librespot.sh
#
# LIBRESPOT_DEVICE is an ALSA PCM name. Default "default" so ALSA can mix
# librespot with the voice client's output — run ./setup_alsa_sharing.sh first
# to make "default" a shared dmix mixer (see devices/rpi5/README.md). To send
# Spotify straight at the Jabra instead, pass e.g.
# LIBRESPOT_DEVICE=plughw:CARD=Speaker,DEV=0 — but then only one process can
# hold the card at a time (no simultaneous bot voice + music).
set -euo pipefail

LIBRESPOT_NAME="${LIBRESPOT_NAME:-Babel}"
LIBRESPOT_DEVICE="${LIBRESPOT_DEVICE:-default}"
LIBRESPOT_BITRATE="${LIBRESPOT_BITRATE:-320}"
LIBRESPOT_VOLUME="${LIBRESPOT_INITIAL_VOLUME:-100}"
CONF="/etc/raspotify/conf"

if ! command -v librespot >/dev/null 2>&1 && ! dpkg -s raspotify >/dev/null 2>&1; then
  echo "Installing raspotify (bundles librespot + a systemd service)..."
  # The upstream one-liner adds the apt repo, key, and installs the package.
  curl -sSL https://dtcooper.github.io/raspotify/install.sh | sh
else
  echo "raspotify/librespot already installed."
fi

if [ ! -f "$CONF" ]; then
  echo "ERROR: $CONF not found — is raspotify installed?" >&2
  exit 1
fi

echo "Configuring $CONF:"
echo "  name:    $LIBRESPOT_NAME"
echo "  device:  $LIBRESPOT_DEVICE (ALSA)"
echo "  bitrate: $LIBRESPOT_BITRATE"
echo "  volume:  $LIBRESPOT_VOLUME"

# raspotify reads LIBRESPOT_* env-style options from /etc/raspotify/conf.
# Replace the managed keys idempotently (uncomment + set), leaving the rest.
sudo tee "$CONF" >/dev/null <<EOF
# Managed by devices/rpi5/install_librespot.sh — re-run it to change these.
LIBRESPOT_NAME="$LIBRESPOT_NAME"
LIBRESPOT_BITRATE="$LIBRESPOT_BITRATE"
LIBRESPOT_BACKEND="alsa"
LIBRESPOT_DEVICE="$LIBRESPOT_DEVICE"
LIBRESPOT_INITIAL_VOLUME="$LIBRESPOT_VOLUME"
# Keep Spotify Connect discovery on so the phone can bind "$LIBRESPOT_NAME".
LIBRESPOT_DISABLE_AUDIO_CACHE=
EOF

sudo systemctl restart raspotify
sudo systemctl enable raspotify >/dev/null 2>&1 || true

echo
echo "Started. Useful commands:"
echo "  systemctl status raspotify"
echo "  journalctl -u raspotify -f"
echo
echo "Now bind it once: open Spotify on your phone → Connect → pick '$LIBRESPOT_NAME'."
echo "From the server, confirm it's visible: python scripts/spotify.py --list-devices"
