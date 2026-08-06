# Raspberry Pi 5 WebRTC Voice Client

This directory contains a Raspberry Pi 5 voice client and a local WebRTC
loopback server for testing with a Jabra Speak2 40 USB conference phone.

The client uses:

- `aiortc` for WebRTC.
- FFmpeg ALSA input/output through `aiortc.contrib.media`.
- HTTP `POST /api/offer` signaling compatible with the repo's WebRTC smoke
  test shape.

## Pi setup

On Raspberry Pi OS:

```bash
sudo apt update
# libportaudio2 is REQUIRED: sounddevice (used for on-device wake capture +
# playback) is only a binding to the system PortAudio library.
sudo apt install -y ffmpeg alsa-utils libavdevice-dev libportaudio2 portaudio19-dev
# From the repo root: Hermit provides python; then install the Pi deps.
. bin/activate-hermit
./devices/rpi5/install_rpi.sh   # requirements.txt + openWakeWord (see note below)
```

### Python 3.12 note (openWakeWord)

`install_rpi.sh` works on **Python 3.11 and 3.12**. openWakeWord 0.6.0's
packaging pins a Linux-only `tflite-runtime`, which has **no wheels for Python
>= 3.12** — so a plain `pip install openwakeword` fails there with "requires
Python < 3.12". We use the ONNX inference path (the tflite import is lazy and
unused), so the script installs openWakeWord with `--no-deps` after its real
runtime deps. Plain Python **3.11** (the Pi OS default) also works with a normal
`pip install` — it just pulls an unused `tflite-runtime`.

Plug in the Jabra Speak2 40, then list ALSA names:

```bash
python list_alsa_devices.py
# or
python rpi_webrtc_voice.py --list-devices
```

Look for a capture and playback device such as
`plughw:CARD=Speaker,DEV=0`, `hw:CARD=Speaker,DEV=0`, or another Jabra-specific
name. Prefer `plughw:` because ALSA can adapt sample format/rate when needed.
If `plughw:CARD=UC,DEV=0` fails with `Unknown PCM`, the card name is not `UC`
on that Pi. Use the exact card name from `arecord -l` / `aplay -l`, or use the
numeric form, for example `plughw:2,0`.

## Auto-start on boot (systemd)

The easiest way to run the wake client as a background service that starts on
boot and restarts on failure:

```bash
# From the repo root on the Pi (server IP defaults to 192.168.0.245):
SERVER_IP=192.168.0.245 ./devices/rpi5/install_service.sh
journalctl -u rpi-voice -f          # follow logs
```

Override `SERVER_IP`, `INPUT_DEVICE`, `OUTPUT_DEVICE`, `AUTH_TOKEN`, or
`SERVICE_NAME` via env vars. The generated service sources
`./bin/activate-hermit` (Hermit-managed toolchain) before running, and runs as
your user (in the `audio` group). Manage it with
`sudo systemctl {restart,stop,disable} rpi-voice`.

## Run the local loopback server

On this machine or on the Pi:

```bash
. bin/activate-hermit          # from the repo root (puts python on PATH)
cd devices/rpi5
python webrtc_loopback_server.py --host 0.0.0.0 --port 8080
```

## Run the Pi voice client

On the Pi, with the loopback server running:

```bash
. bin/activate-hermit          # from the repo root (puts python on PATH)
cd devices/rpi5
python rpi_webrtc_voice.py \
  --offer-url http://127.0.0.1:8080/api/offer \
  --input-device plughw:2,0 \
  --output-device plughw:2,0
```

If the server is running on another host on the LAN, replace the URL:

```bash
python rpi_webrtc_voice.py \
  --offer-url http://192.168.1.50:8080/api/offer \
  --input-device plughw:2,0 \
  --output-device plughw:2,0
```

Successful loopback means audio captured from the Jabra is sent over WebRTC and
played back through the Jabra speaker.

## Useful environment variables

```bash
export WEBRTC_OFFER_URL=http://127.0.0.1:8080/api/offer
export ALSA_INPUT_DEVICE=plughw:2,0
export ALSA_OUTPUT_DEVICE=plughw:2,0
export AUDIO_SAMPLE_RATE=48000
export AUDIO_CHANNELS=1
export DEVICE_ID=rpi5-jabra
```

## Spotify (librespot on the Pi)

Spotify plays **natively on the Pi** — the server only sends Web API control
commands to a librespot "Babel" Connect endpoint running here. (Music does not
go through WebRTC; that path was choppy/staticky because it forced 44.1 kHz
stereo through a 24 kHz-mono voice channel.)

Raspberry Pi OS Bookworm runs **PipeWire**, which owns the sound card and mixes
multiple streams natively — so there's no dmix/ALSA-sharing to set up. The catch
is that PipeWire is **per-user**: to share the Jabra, both librespot and the
voice client must run as *your* user (so they join your PipeWire session), as
**user services** with linger enabled (starts at boot, no login needed).

Install librespot as a user service (routes through PipeWire, points the default
sink at the Jabra, enables linger):

```bash
# From the repo root on the Pi, as your normal user (NOT sudo):
./devices/rpi5/install_librespot.sh
```

Bind it once: open Spotify on your phone → **Connect** → pick **Babel**. From the
server, confirm it's visible:

```bash
python scripts/spotify.py --list-devices
```

Then (re)install the voice client as a **user service** too, so it shares the
card with librespot via PipeWire (defaults its audio devices to `pulse`):

```bash
SERVER_IP=<your-server-ip> USER_SERVICE=1 ./devices/rpi5/install_service.sh
```

Now the bot voice and Spotify mix automatically — while the bot speaks, music
keeps playing (there's no ducking).

Server-side prerequisites: `skills.spotify.enabled: true`, `SPOTIPY_CLIENT_ID`
set, and the one-time OAuth done (`python scripts/spotify.py --bootstrap`). See
the repo README's Spotify section.

### Quick sanity checks

```bash
pactl get-default-sink                       # should name the Jabra, not HDMI
speaker-test -D pulse -c 2 -t sine -f 440 -l 1   # tone via PipeWire
systemctl --user status librespot rpi-voice  # both running as your user
```

If Spotify comes out of HDMI instead of the Jabra, set the default sink by hand:
`pactl set-default-sink <name>` (list them with `pactl list short sinks`).

> **Bare-ALSA Pis (no PipeWire, e.g. a Lite image):** run librespot and the
> client against a `dmix`/`dsnoop` shared `default` in `/etc/asound.conf`
> instead, and install the client with the plain (system) service:
> `./devices/rpi5/install_service.sh`.

## Notes

- The loopback server is for transport testing only. It does not perform STT,
  LLM, or TTS.
- If you hear feedback, lower speaker volume or move the Jabra away from nearby
  reflective surfaces. The Jabra hardware performs echo cancellation, but
  loopback is intentionally unforgiving.
- If `default` works for capture and playback, you can omit the ALSA device
  flags. Pinning the explicit `plughw:` device is more stable for unattended
  boots.
