# ws2812: a WS2812B strip on the Pi, from Rust

The satellite's LED strip: an 8-LED WS2812B on the Raspberry Pi, driven over
SPI. This crate is the library the client uses (the wire encoding and spidev
writer in `strip.rs`, colours in `color.rs`, the Larson scanner in
`larson.rs`) plus `ws2812-poc`, the standalone demo it started as, which
runs the scanner through a palette of colours and doubles as the hardware
check. The decision record, with the alternatives, the timing analysis and
the hardware test gates: `docs/adr/0008-ws2812-strip-over-spi.md`.

## Wiring

| Strip wire | Pi header pin | Notes |
|---|---|---|
| 5V (red) | pin 2 or 4 (5V) | fine for 8 LEDs; over ~30, power the strip separately |
| GND (white) | pin 6 (GND) | shared ground |
| DIN (green) | **pin 19 (GPIO 10, SPI0 MOSI)** | see below |

Mind the arrows on the strip: data enters at the DIN end.

The Core Electronics guide puts the data wire on GPIO 18 (pin 12), where the
`rpi_ws281x` library bit-bangs the signal through the PWM peripheral and DMA.
That peripheral is not reachable on a Raspberry Pi 5 (the guide itself notes
NeoPixels do not work there), so this PoC uses SPI0's MOSI line instead, which
is the guide's own alternative (`LED_PIN = 10` in its `strandtest.py`). It
works unprivileged on every Pi, and cross-compiles with no native library.
**Move the green wire from pin 12 to pin 19.**

## Run it from the dev machine

```sh
make ws2812-pi                                     # PI_HOST from .env
make ws2812-pi WS2812_ARGS="--pattern wiring"      # colour-order check first
make ws2812-pi WS2812_ARGS="--brightness 0.5 --sweep-ms 500 --colors red,white"
```

That cross-builds the binary (same `cross` + Docker setup as the client,
`target/pi/`), rsyncs it with `run-on-pi.sh` into
`~/.cache/voice-chatbot-deploy/ws2812-poc/` on the Pi, and runs it there over
`ssh -t`. Ctrl-C stops it and clears the strip.

On first use `/dev/spidev0.0` will not exist: `run-on-pi.sh` writes
`dtparam=spi=on` into `config.txt` (one sudo), offers to reboot, and exits;
rerun `make ws2812-pi` once the Pi is back. It also adds you to the `spi`
group if the node is not writable. It never applies the SPI overlay live:
`dtparam spi=on` at runtime, which is what `raspi-config nonint do_spi 0`
does, hung the first run on a Pi 5 (no output, Ctrl-C dead, no spidev node,
and the multiplexed ssh session stuck with it). Boot-time application by the
firmware is the reliable path. `/dev/spidev10.0`, which a Pi 5 has out of
the box, is an internal bus, not the header; the strip is `spidev0.0`.

## Options

```
--count 8            LEDs on the strip
--spi /dev/spidev0.0 spidev node (SPI0 = GPIO 10)
--spi-hz 3200000     SPI clock; 2.0-3.8 MHz keeps WS2812 timing in spec
--order grb          colour byte order; WS2812B is grb, some clones rgb
--brightness 0.2     0-1 (full white on 8 LEDs is 480 mA and blinding)
--gamma 1.0          brightness curve; 1 is the kit's linear PWM
--sweep-ms 700       one end-to-end pass of the eye
--colors red,...     one colour per pass, names or #rrggbb
--seconds 0          stop after N seconds (0 = until Ctrl-C)
--pattern larson     or `wiring`: all-red/green/blue/white, then one LED at a time
```

## How it drives the strip

`strip.rs` encodes each colour bit as four SPI bits (`1000` = 0, `1110` = 1):
at 3.2 MHz that is a 1.25 µs bit with a 0.31 µs or 0.94 µs high pulse, inside
the WS2812B windows, followed by 400 µs of zeros as the latch/reset. One frame
is one spidev transfer (8 LEDs: 257 bytes, ~0.6 ms), so nothing on the host
side can stretch a gap into an accidental reset. (On a Pi 4 the SPI clock
follows the core clock, which CPU scaling moves; `core_freq=500` and
`core_freq_min=500` in `config.txt` pin it, per the `rpi_ws281x` README.
The Pi 5's RP1 clocks SPI on its own.) `larson.rs` is a
line-for-line port of the Evil Mad Scientist kit firmware's eye (1:4:2:1
parts, 16 sub-steps per LED, reflected at the ends), generalised to any strip
length; the scanner never touches hardware, so it is unit-tested.

## In the client

`crates/client/src/led/strip.rs` is a second sink behind the phase tracker
that drives the Jabra ring: a thread of its own owns the strip and renders
one pattern per phase — a dim green pixel sweeping slowly while asleep, all
green while listening, this scanner (the palette above, one colour per pass)
while thinking, a soft warm glow while speaking, both ends red over any of
those while the server has the mic gated, and one amber pixel blinking when
there is no server. The full table and the rules behind it are in the ADR's
"Vocabulary" section. Knobs, in the Pi's `.env` or as flags:
`LED_STRIP=auto|off` (auto, the default, drives the strip when
`/dev/spidev0.0` exists), `LED_STRIP_COUNT` (8), `LED_STRIP_BRIGHTNESS`
(0.2, for thinking, speaking and the mute overlay) and
`LED_STRIP_IDLE_BRIGHTNESS` (0.05, for asleep, listening and offline). The
service unit carries `SupplementaryGroups=audio spi` and `install.sh` enables
SPI in `config.txt` (first time: reboot). `voice-chatbot-client led-test`
walks the ring and the strip through every state.

`spidev` is Linux-only and stays behind the `cfg(target_os = "linux")`
target dependency, so the Mac build of the workspace stays green.
