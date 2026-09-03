#!/usr/bin/env bash
# Runs ON the Pi (make ws2812-pi ships it next to the binary): make sure SPI0
# is enabled (config.txt + a reboot on first use) and reachable without root,
# then run the PoC with the given arguments. Rerun-safe.
#
#   ./run-on-pi.sh                     # Larson scanner until Ctrl-C
#   ./run-on-pi.sh --pattern wiring    # colour/order check
#   ./run-on-pi.sh --help
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
BIN=./ws2812-poc
SPI_DEV="${SPI_DEV:-/dev/spidev0.0}"
ME="$(id -un)"

[ -x "$BIN" ] || { echo "no $BIN next to this script" >&2; exit 1; }

# SPI is off on a stock Pi OS image. Turn it on in config.txt and reboot --
# never apply it live. `raspi-config nonint do_spi 0` (which runs
# `dtparam spi=on` after editing config.txt) hung the first run on a Pi 5 on
# 2026-09-03: no further output, Ctrl-C dead, no /dev/spidev0.0, though the
# board kept answering. Overlays the firmware applies at boot are the
# reliable path; runtime application is best-effort (Raspberry Pi's own
# guidance), and nothing else here needs it.
if [ ! -e "$SPI_DEV" ]; then
    cfg=/boot/firmware/config.txt
    [ -f "$cfg" ] || cfg=/boot/config.txt
    if grep -qE '^dtparam=spi=on' "$cfg"; then
        echo "$SPI_DEV is missing although $cfg already has dtparam=spi=on: the Pi needs a reboot"
    else
        echo "$SPI_DEV is missing: setting dtparam=spi=on in $cfg (needs sudo); a reboot applies it"
        if grep -qE '^#?dtparam=spi=' "$cfg"; then
            sudo sed -i -E 's/^#?dtparam=spi=.*/dtparam=spi=on/' "$cfg"
        else
            printf 'dtparam=spi=on\n' | sudo tee -a "$cfg" >/dev/null
        fi
    fi
    if [ -t 0 ]; then
        read -r -p "Reboot the Pi now? [y/N] " answer
        case "$answer" in
            y|Y|yes|YES)
                echo "rebooting; rerun make ws2812-pi once it is back"
                sudo reboot
                ;;
        esac
    fi
    echo "reboot the Pi (ssh in and run: sudo reboot), then rerun make ws2812-pi" >&2
    exit 1
fi

# Pi OS puts spidev nodes in the spi group (0660). Join it if needed; the
# membership only reaches new logins, so this run borrows it through sg.
if [ ! -w "$SPI_DEV" ]; then
    if ! id -nG "$ME" | tr ' ' '\n' | grep -qx spi; then
        echo "adding $ME to the spi group (needs sudo; new logins get it, this run uses sg)"
        sudo usermod -aG spi "$ME"
    fi
    exec sg spi -c "$(printf '%q ' "$BIN" "$@")"
fi

exec "$BIN" "$@"
