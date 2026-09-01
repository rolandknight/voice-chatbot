#!/usr/bin/env bash
# One-time dev-machine setup for Jabra speakerphone LED control
# (docs/specs/jabra-led.md). The client drives the Jabra's telephony LEDs
# over hidraw, and hidraw nodes are root-only by default. This installs a
# udev rule opening Jabra (GN Audio, USB vendor 0b0e) hidraw nodes to the
# locally logged-in user (systemd uaccess ACL) and to the audio group, so
# the client reaches the LEDs whether it runs from a desktop session or as
# a service.
#
# The Raspberry Pi deploy does NOT need this: deploy/rpi/install.sh ships
# its own rule. Run this once on a desktop/laptop where you run the client
# by hand:
#
#     sudo deploy/jabra-setup.sh

set -euo pipefail

RULE_DEST=/etc/udev/rules.d/99-jabra-hid.rules

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "must run as root: sudo $0"
command -v udevadm >/dev/null 2>&1 \
    || fail "udevadm not found; this script needs a systemd/udev system"

cat > "$RULE_DEST" << 'EOF'
# Jabra (GN Audio) HID interfaces: let the voice-chatbot client drive the
# speakerphone's telephony LEDs via hidraw (docs/specs/jabra-led.md).
# uaccess grants the active local session; the audio group covers services.
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="0b0e", MODE="0660", GROUP="audio", TAG+="uaccess"
EOF
chmod 0644 "$RULE_DEST"
echo "installed $RULE_DEST"

# Re-apply to a speakerphone that is already plugged in, no replug needed.
udevadm control --reload-rules
udevadm trigger --subsystem-match=hidraw
echo "reloaded udev rules and retriggered hidraw"

# Show what the rule now covers, so success (or a missing device) is
# visible immediately. HID_ID's vendor field is the 0B0E in e.g.
# HID_ID=0003:00000B0E:0000AE6D.
found=0
for node in /sys/class/hidraw/hidraw*; do
    [ -e "$node" ] || continue
    grep -qi ':00000b0e:' "$node/device/uevent" 2>/dev/null || continue
    found=1
    dev="/dev/$(basename "$node")"
    echo "jabra hidraw node: $dev"
    ls -l "$dev"
    command -v getfacl >/dev/null 2>&1 && getfacl -p "$dev" 2>/dev/null | grep '^user:.*rw' || true
done
if [ "$found" -eq 0 ]; then
    echo "no Jabra attached right now; the rule applies as soon as one is plugged in"
fi
