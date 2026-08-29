# Native FlowCat WebRTC client

`flowcat-webrtc-client` is a terminal-only Rust client for the FlowCat PoC. It
captures directly from a selected operating-system audio input, sends 20 ms
Opus frames over WebRTC, decodes the returned Opus stream, and plays it through
a selected output device. The browser and Python audio clients are not involved.

Client and server may be on different machines on the same LAN: the client
binds and advertises the interface that routes to `--server-url`, and the
server advertises the interface that routes back to the caller (host ICE
candidates only, no STUN/TURN). Start the server with `POC_BIND=0.0.0.0:6210`
(and `POC_ADVERTISE_IP` if auto-detection picks the wrong interface), then:

```sh
make client FLOWCAT_URL=http://<server-lan-ip>:6210
```

## Prerequisites

macOS uses CoreAudio through CPAL:

```sh
brew install cmake pkg-config opus
```

The terminal application may need microphone permission in **System Settings →
Privacy & Security → Microphone**.

Debian, Ubuntu, and Pop!_OS use ALSA through CPAL:

```sh
sudo apt install build-essential cmake pkg-config libasound2-dev libopus-dev
```

If `poc/.deps/prefix` contains the PoC-local Opus build, the Make targets use it
automatically instead of requiring the system Opus development package.

### Raspberry Pi (aarch64)

`make client-build-pi` cross-builds the client for a Pi running 64-bit
Raspberry Pi OS. It needs Docker and [`cross`](https://github.com/cross-rs/cross):

```sh
cargo install cross --locked
make client-build-pi
scp target/pi/aarch64-unknown-linux-gnu/release/voice-chatbot-client pi@<host>:
```

The build runs in the pinned image configured in `Cross.toml`, which supplies
the arm64 ALSA and Opus development packages; the resulting binary needs only
glibc 2.18, so it runs on every 64-bit Pi OS release. Hermit's `rustc` carries
x86_64 std only, so this target uses the system rustup toolchain rather than
`bin/cargo`, and keeps its artifacts in `target/pi/` so the two never
invalidate each other.

On the Pi, `libasound2` and `libopus0` must be installed (`sudo apt install
libasound2 libopus0`). The binary carries debug info and is ~270 MB; strip it
first if the copy matters:

```sh
docker run --rm -v "$PWD:/w" -w /w ghcr.io/cross-rs/aarch64-unknown-linux-gnu:0.2.5 \
    aarch64-linux-gnu-strip target/pi/aarch64-unknown-linux-gnu/release/voice-chatbot-client
```

## Use

Start the FlowCat stack, then list the native devices:

```sh
make poc-up
make flowcat-client-devices
```

Omitting selectors auto-selects the Jabra speakerphone when one is plugged in,
and the operating-system defaults otherwise. Pass `default` to ask for the
system default even with the Jabra attached. Any other selector may be the
displayed 1-based index, exact stable device ID, exact name, or a
case-insensitive part of the name:

```sh
make flowcat-client-run
make flowcat-client-run INPUT_DEVICE='Jabra' OUTPUT_DEVICE='Jabra'
```

You can invoke the binary directly for its full help:

```sh
cargo run --manifest-path poc/flowcat-client/Cargo.toml -- call --help
```

Press Ctrl-C to close the WebRTC peer and both device streams. User
transcriptions, assistant text, and tool activity are printed to the terminal.

On Linux, ALSA may print warnings about unavailable JACK, OSS, `dsnoop`, or
`dmix` devices while CPAL probes its plugin list. Those messages are harmless
when the client subsequently prints the intended `input:` and `output:` lines.
They do not mean the selected PipeWire/default device failed to open.

The same Jabra has to serve as input *and* output so its hardware echo
cancellation receives the far-end playback reference -- which is why an
unspecified device auto-selects it on both sides. CPAL and str0m do not provide
software acoustic echo cancellation; using separate laptop speakers and
microphone can cause the assistant to hear and interrupt itself.

On ALSA one speakerphone is listed many times, once per PCM alias of its card,
all under the same name. A selector that hits several of those is not treated
as ambiguous -- they are one piece of hardware reachable by several paths, so
the client ranks them and takes the best.

That ranking prefers `plughw:`, the plug layer over the card's hardware PCM.
The card's `default:`/`sysdefault:` aliases are ranked *below* the raw paths
even though they are the ones that mix with other raw-ALSA clients: they
resolve to `dmix`/`dsnoop`, which pin the playback period at 1024 frames.
CPAL's `BufferSize::Default` double-buffers that into a 42.7 ms ring while
capture on the same card runs 64 ms periods, so playback underruns roughly once
per capture period. Measured on a Jabra Speak2 40: 93 XRUNs in 6 s of duplex on
`sysdefault:` (intermittently -- two runs in three), none on `plughw:`. The
period is not negotiable; CPAL reports the supported range as exactly
`1024..=1024`, so the ring cannot be widened to cover it.

The client also asks each device for a 20 ms period rather than letting CPAL
choose. Left to itself CPAL takes about 241 frames on a USB `plughw:` PCM -- a
5 ms period double-buffered into a 10 ms ring -- and the thread refilling that
ring is not a real-time one (CPAL's `realtime` feature is off, and promoting it
needs an `rtprio` allowance a stock desktop does not grant: `ulimit -r` is 0
with rtkit inactive). 20 ms is one Opus frame, so it costs a frame of latency
and gives four times the slack. Devices that pin their period, such as `dmix:`
at `1024..=1024`, refuse the request and keep their own size.

Either way the call owns the speakerphone for its duration: a sound server
cannot route other desktop audio to a card that CPAL holds directly. Media is
therefore not a second process at all -- `ffmpeg` decodes it to raw PCM and the
output callback sums it with the call's voice, so it reaches the same device.

## Checks

```sh
make flowcat-client-check
```
