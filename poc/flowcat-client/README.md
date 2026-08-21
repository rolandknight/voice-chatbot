# Native FlowCat WebRTC client

`flowcat-webrtc-client` is a terminal-only Rust client for the FlowCat PoC. It
captures directly from a selected operating-system audio input, sends 20 ms
Opus frames over WebRTC, decodes the returned Opus stream, and plays it through
a selected output device. The browser and Python audio clients are not involved.

The current FlowCat server advertises loopback ICE only, so client and server
must run on the same machine.

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

## Use

Start the FlowCat stack, then list the native devices:

```sh
make poc-up
make flowcat-client-devices
```

Omitting selectors uses the operating-system defaults. A selector may be the
displayed 1-based index, exact stable device ID, exact name, or a unique
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

For speakerphone use, select the same Jabra for input and output so its hardware
echo cancellation receives the far-end playback reference. CPAL and str0m do
not provide software acoustic echo cancellation; using separate laptop speakers
and microphone can cause the assistant to hear and interrupt itself.

## Checks

```sh
make flowcat-client-check
```
