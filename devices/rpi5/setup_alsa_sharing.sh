#!/usr/bin/env bash
# Make the Jabra shareable so the bot voice AND Spotify (librespot) can play at
# the same time. There's no ducking — while the bot speaks, music keeps going —
# so two playback streams hit the same USB card at once. A raw ALSA hw:/plughw:
# device is single-owner, so we install an /etc/asound.conf that turns `default`
# into a software mixer (dmix for playback, dsnoop for capture, wrapped in plug
# for automatic rate/format/channel conversion).
#
# After running this, point BOTH the voice client and librespot at `default`:
#   - client:   INPUT_DEVICE=default OUTPUT_DEVICE=default ./install_service.sh
#               (or ALSA_INPUT_DEVICE / ALSA_OUTPUT_DEVICE in /etc/rpi-voice.env)
#   - librespot: LIBRESPOT_DEVICE=default ./install_librespot.sh   (the default)
#
# Usage:  ./setup_alsa_sharing.sh
#         CARD=Speaker ./setup_alsa_sharing.sh          # force the ALSA card id
#         CARD=2 RATE=48000 CHANNELS=2 ./setup_alsa_sharing.sh
#
# CARD is the ALSA card id or index from `aplay -l` (the token right after
# "card N:", e.g. "Speaker", "UC", or the number). Auto-detected from a Jabra /
# Speaker / USB line when unset.
set -euo pipefail

RATE="${RATE:-48000}"
CHANNELS="${CHANNELS:-2}"
CONF="/etc/asound.conf"

command -v aplay >/dev/null 2>&1 || {
  echo "ERROR: aplay not found. Install alsa-utils: sudo apt install -y alsa-utils" >&2
  exit 1
}

# ---- Resolve the card id -------------------------------------------------
if [ -n "${CARD:-}" ]; then
  CARD_ID="$CARD"
else
  # Parse `aplay -l`; prefer a Jabra/Speaker/USB playback device. Grab the card
  # id token after "card N:" (more stable across reboots than the index).
  CARD_ID="$(aplay -l 2>/dev/null | awk '
    tolower($0) ~ /jabra|speak|usb audio/ && $1=="card" {
      id=$3; sub(/:$/, "", id); print id; exit
    }')"
  if [ -z "$CARD_ID" ]; then
    echo "Could not auto-detect a Jabra/USB playback card. Devices seen:" >&2
    aplay -l >&2 || true
    echo >&2
    echo "Re-run with CARD=<id-or-index>, e.g. CARD=Speaker ./setup_alsa_sharing.sh" >&2
    exit 1
  fi
fi

# hw:CARD=Speaker if non-numeric, hw:2 if numeric.
if [[ "$CARD_ID" =~ ^[0-9]+$ ]]; then
  SLAVE="hw:${CARD_ID},0"
else
  SLAVE="hw:CARD=${CARD_ID},DEV=0"
fi

echo "ALSA sharing config:"
echo "  card:     $CARD_ID  -> $SLAVE"
echo "  rate:     $RATE"
echo "  channels: $CHANNELS"

# ---- Sanity-check the slave accepts these params -------------------------
if ! aplay -D "$SLAVE" --dump-hw-params /dev/zero >/dev/null 2>&1; then
  echo "WARNING: couldn't probe $SLAVE hw params. If playback fails, adjust" >&2
  echo "         RATE/CHANNELS to match \`aplay -D $SLAVE --dump-hw-params /dev/zero\`." >&2
fi

# ---- Back up any existing config -----------------------------------------
if [ -f "$CONF" ]; then
  BAK="${CONF}.bak.$(date +%s 2>/dev/null || echo prev)"
  echo "Backing up existing $CONF -> $BAK"
  sudo cp "$CONF" "$BAK"
fi

# ---- Write the shared default --------------------------------------------
# plug -> asym(dmix, dsnoop). plug absorbs rate/format/channel mismatches so a
# 16 kHz wake stream, 24 kHz bot TTS, and 44.1 kHz librespot all mix cleanly.
sudo tee "$CONF" >/dev/null <<EOF
# Managed by devices/rpi5/setup_alsa_sharing.sh — re-run it to regenerate.
# Shared "default" so the voice client and librespot can use the $CARD_ID card
# at the same time (dmix = shared playback, dsnoop = shared capture).

pcm.!default {
    type plug
    slave.pcm "duplex"
}

pcm.duplex {
    type asym
    playback.pcm "dmixer"
    capture.pcm "dsnooper"
}

pcm.dmixer {
    type dmix
    ipc_key 2048
    ipc_perm 0666
    slave {
        pcm "$SLAVE"
        rate $RATE
        channels $CHANNELS
        format S16_LE
    }
}

pcm.dsnooper {
    type dsnoop
    ipc_key 2049
    ipc_perm 0666
    slave {
        pcm "$SLAVE"
        rate $RATE
    }
}

ctl.!default {
    type hw
    card $CARD_ID
}
EOF

echo
echo "Wrote $CONF."
echo "Quick test (should play a tone through the mixer):"
echo "  speaker-test -D default -c $CHANNELS -t sine -f 440 -l 1"
echo
echo "Then point both sides at 'default':"
echo "  librespot: LIBRESPOT_DEVICE=default ./devices/rpi5/install_librespot.sh"
echo "  client:    INPUT_DEVICE=default OUTPUT_DEVICE=default \\"
echo "             ./devices/rpi5/install_service.sh"
echo "  (or set ALSA_INPUT_DEVICE=default / ALSA_OUTPUT_DEVICE=default in /etc/rpi-voice.env)"
