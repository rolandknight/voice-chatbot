#!/usr/bin/env bash
# Install (or update) the systemd unit that autostarts the native client on a
# Raspberry Pi, plus the files it runs on. Run it on the Pi, under sudo:
#
#   sudo ./install.sh
#
# From a dev machine, `make deploy-pi PI_HOST=pi@raspberrypi.local` rsyncs the
# cross-built binary, the wake heads and this directory to the Pi and runs it.
#
# Two source layouts work: the staging directory deploy-pi builds (the binary
# and models/ sit next to this script), and a repo checked out on the Pi
# itself (target/release/ and models/wakeword/ two levels up).
#
# Knobs, all optional:
#   INSTALL_DIR=/opt/voice-chatbot      binary, .env and wake heads
#   RUN_USER=<the sudo'ing user>        account the service runs as
#   SERVICE_NAME=voice-chatbot-client   unit name
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/voice-chatbot}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$USER}}"
SERVICE_NAME="${SERVICE_NAME:-voice-chatbot-client}"
BIN_NAME="voice-chatbot-client"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="$SRC_DIR/$BIN_NAME.service"
UNIT_DEST="/etc/systemd/system/$SERVICE_NAME.service"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[ "$(id -u)" = "0" ] || fail "run under sudo: it writes $UNIT_DEST and $INSTALL_DIR"
# Guarded because the wake-head refresh below is an `rm -rf` under this path.
[ -n "$INSTALL_DIR" ] || fail "INSTALL_DIR is empty"
id "$RUN_USER" >/dev/null 2>&1 || fail "no such user: $RUN_USER (set RUN_USER=)"
# Ownership follows the account's own primary group, which is not always a
# user-private group named after it.
RUN_GROUP="$(id -gn "$RUN_USER")"
[ -f "$UNIT_SRC" ] || fail "missing $UNIT_SRC"
[ -f "$SRC_DIR/env.example" ] || fail "missing $SRC_DIR/env.example"

# Staging layout first (what deploy-pi builds), then a repo checked out here.
REPO_ROOT="$(cd "$SRC_DIR/../.." 2>/dev/null && pwd || echo "$SRC_DIR")"
if [ -f "$SRC_DIR/$BIN_NAME" ]; then
    BIN_SRC="$SRC_DIR/$BIN_NAME"
    WAKE_SRC="$SRC_DIR/models/wakeword"
elif [ -f "$REPO_ROOT/target/release/$BIN_NAME" ]; then
    BIN_SRC="$REPO_ROOT/target/release/$BIN_NAME"
    WAKE_SRC="$REPO_ROOT/models/wakeword"
else
    fail "no $BIN_NAME next to this script or at $REPO_ROOT/target/release/ --
       cross-build it (make client-build-pi) and use make deploy-pi, or build
       it natively on the Pi (make client-build)"
fi
[ -d "$WAKE_SRC" ] || fail "no wake heads at $WAKE_SRC (deploy-pi syncs models/wakeword)"
ls "$WAKE_SRC"/hey_*.onnx >/dev/null 2>&1 || fail "$WAKE_SRC holds no hey_<persona>.onnx heads"

echo "Installing $SERVICE_NAME:"
echo "  from:    $BIN_SRC"
echo "  into:    $INSTALL_DIR"
echo "  as user: $RUN_USER"

# ffmpeg decodes radio, shows and sound effects in-process. Without it the
# client still runs and talks -- those skills just play nothing at all.
command -v ffmpeg >/dev/null 2>&1 \
    || echo "WARNING: ffmpeg is not installed; radio, shows and sound effects will not play (apt install ffmpeg)" >&2

# The unit grants /dev/snd through SupplementaryGroups, so the service does not
# need this. It is for running the client by hand over ssh.
if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx audio; then
    echo "  adding $RUN_USER to the audio group (for manual runs; the unit grants it anyway)"
    usermod -aG audio "$RUN_USER"
fi

# A running service holds the binary open: replacing it in place is ETXTBSY.
WAS_ACTIVE=0
if systemctl is-active --quiet "$SERVICE_NAME"; then
    WAS_ACTIVE=1
    echo "  stopping the running service first"
    systemctl stop "$SERVICE_NAME"
fi

install -d -o "$RUN_USER" -g "$RUN_GROUP" "$INSTALL_DIR" "$INSTALL_DIR/models"
install -o "$RUN_USER" -g "$RUN_GROUP" -m 0755 "$BIN_SRC" "$INSTALL_DIR/$BIN_NAME"
rm -rf "$INSTALL_DIR/models/wakeword"
install -d -o "$RUN_USER" -g "$RUN_GROUP" "$INSTALL_DIR/models/wakeword"
# .onnx only: resolve_heads ignores every other extension, and the .tflite
# twins are dead weight on the wire and on the card.
install -o "$RUN_USER" -g "$RUN_GROUP" -m 0644 "$WAKE_SRC"/hey_*.onnx "$INSTALL_DIR/models/wakeword/"

# Never clobber a configured .env -- an update must not silently reset the
# server address back to the example's.
if [ -f "$INSTALL_DIR/.env" ]; then
    echo "  keeping the existing $INSTALL_DIR/.env"
else
    install -o "$RUN_USER" -g "$RUN_GROUP" -m 0644 "$SRC_DIR/env.example" "$INSTALL_DIR/.env"
    echo "  wrote $INSTALL_DIR/.env from env.example -- set SERVER_URL in it"
fi

# The FLOWCAT_ prefix was retired; the client refuses to start with any left
# set rather than silently running on the defaults. Catch it here, where the
# message can name the file, instead of in a restart loop.
if grep -qE '^[[:space:]]*(export[[:space:]]+)?FLOWCAT_' "$INSTALL_DIR/.env" 2>/dev/null; then
    fail "$INSTALL_DIR/.env still sets FLOWCAT_* names, which the client no longer
       reads (and refuses to start with). Drop the prefix: FLOWCAT_URL is now
       SERVER_URL, FLOWCAT_INPUT_DEVICE is INPUT_DEVICE, and so on."
fi

sed -e "s|@INSTALL_DIR@|$INSTALL_DIR|g" -e "s|@RUN_USER@|$RUN_USER|g" "$UNIT_SRC" > "$UNIT_DEST"
chmod 0644 "$UNIT_DEST"
systemctl daemon-reload

# Smoke test before enabling: catches a wrong-architecture binary, a missing
# shared library and an unparsable .env, with the error in front of you rather
# than in the journal of a service that restarts every 5 s.
if ! (cd "$INSTALL_DIR" && runuser -u "$RUN_USER" -- "./$BIN_NAME" --version >/dev/null); then
    fail "$INSTALL_DIR/$BIN_NAME failed to run (see the error above).
       A cross-build for the wrong target is the usual cause: the Pi needs
       $(uname -m)."
fi

systemctl enable "$SERVICE_NAME" >/dev/null
systemctl restart "$SERVICE_NAME"
sleep 2

echo
if systemctl is-active --quiet "$SERVICE_NAME"; then
    [ "$WAS_ACTIVE" = "1" ] && echo "Updated and restarted." || echo "Installed, enabled at boot, and started."
else
    echo "Installed and enabled, but the service is not running. Its log:" >&2
    journalctl -u "$SERVICE_NAME" -n 20 --no-pager >&2
    exit 1
fi
echo
echo "  journalctl -u $SERVICE_NAME -f       # follow it"
echo "  sudo systemctl restart $SERVICE_NAME"
echo "  sudo systemctl disable --now $SERVICE_NAME   # stop, and stop autostarting"
echo "  \$EDITOR $INSTALL_DIR/.env            # then restart"
