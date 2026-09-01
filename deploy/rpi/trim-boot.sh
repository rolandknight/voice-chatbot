#!/usr/bin/env bash
# Disable the boot-time services a chatbot satellite never uses. On a stock
# Pi OS Desktop image these dominate `systemd-analyze blame`: docker.service
# alone accounted for 5min12s on one Pi, rpi-eeprom-update.service 2min41s,
# and the desktop another half minute.
#
# Run from the dev machine (same convention as `make deploy-pi`):
#
#   deploy/rpi/trim-boot.sh pi@raspberrypi.local
#
# One ssh, one sudo prompt. Idempotent: rerunning skips what is already off.
#
# What goes, and why that is safe on a satellite:
#   docker + containerd    nothing on the Pi runs in containers. (The Docker
#                          that client-build-pi needs is on the dev machine.)
#   rpi-eeprom-update      per-boot bootloader check; run
#                          `sudo rpi-eeprom-update -a` by hand now and then
#   bluetooth + hciuart    audio is the USB speakerphone, not BT
#   cups                   nothing prints
#   graphical boot         set-default multi-user.target: no lightdm/X. The
#                          client is a system service with no UI. A desktop
#                          session already running survives until reboot.
#
# Deliberately kept: avahi (deploy-pi reaches the Pi at *.local through it),
# NetworkManager, ssh.
#
# Undo any of it on the Pi:
#   sudo systemctl enable --now <unit>
#   sudo systemctl set-default graphical.target
set -euo pipefail

PI_HOST="${1:-${PI_HOST:-}}"
[ -n "$PI_HOST" ] || {
    echo "usage: ${0##*/} pi@host   (or set PI_HOST)" >&2
    exit 1
}

REMOTE=$(cat <<'EOF'
set -eu
# cups is socket- and path-activated: dropping cups.service alone would just
# see it started again on the next print-shaped poke, so its triggers go too.
for unit in docker.service docker.socket containerd.service \
            rpi-eeprom-update.service \
            bluetooth.service hciuart.service \
            cups.service cups.socket cups.path cups-browsed.service; do
    if systemctl list-unit-files --no-legend "$unit" 2>/dev/null | grep -q .; then
        echo "disabling $unit"
        systemctl disable --now "$unit"
    else
        echo "skipping $unit (not installed)"
    fi
done

DEFAULT_TARGET="$(systemctl get-default)"
if [ "$DEFAULT_TARGET" != multi-user.target ]; then
    echo "boot target: $DEFAULT_TARGET -> multi-user.target (console, no desktop)"
    systemctl set-default multi-user.target
else
    echo "boot target: already multi-user.target"
fi
EOF
)

# -t keeps sudo's password prompt working, so the script has to travel as an
# argument (printf %q): a tty and a stdin pipe cannot both be had. %q emits
# bash quoting -- fine on Pi OS, where the login shell parsing it is bash.
ssh -t "$PI_HOST" "sudo bash -c $(printf '%q' "$REMOTE")"

echo
echo "Done. Takes full effect on the next boot:"
echo "  ssh $PI_HOST sudo reboot"
echo "  ssh $PI_HOST systemd-analyze    # compare against the old numbers"
