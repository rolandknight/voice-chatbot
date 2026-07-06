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
sudo apt install -y python3-venv ffmpeg alsa-utils libavdevice-dev libportaudio2 portaudio19-dev
cd devices/rpi5
python3 -m venv .venv
. .venv/bin/activate
./install_rpi.sh          # requirements.txt + openWakeWord (see note below)
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
```

Look for a capture and playback device such as
`plughw:CARD=Speaker,DEV=0`, `hw:CARD=Speaker,DEV=0`, or another Jabra-specific
name. Prefer `plughw:` because ALSA can adapt sample format/rate when needed.

## Auto-start on boot (systemd)

The easiest way to run the wake client as a background service that starts on
boot and restarts on failure:

```bash
# From the repo root on the Pi (server IP defaults to 192.168.0.245):
SERVER_IP=192.168.0.245 ./devices/rpi5/install_service.sh
journalctl -u rpi-voice -f          # follow logs
```

Override `SERVER_IP`, `INPUT_DEVICE`, `OUTPUT_DEVICE`, `AUTH_TOKEN`, or
`SERVICE_NAME` via env vars. The script auto-detects the repo path and venv and
runs the service as your user (in the `audio` group). Manage it with
`sudo systemctl {restart,stop,disable} rpi-voice`.

## Run the local loopback server

On this machine or on the Pi:

```bash
cd devices/rpi5
. .venv/bin/activate
python webrtc_loopback_server.py --host 0.0.0.0 --port 8080
```

## Run the Pi voice client

On the Pi, with the loopback server running:

```bash
cd devices/rpi5
. .venv/bin/activate
python rpi_webrtc_voice.py \
  --offer-url http://127.0.0.1:8080/api/offer \
  --input-device plughw:CARD=Speaker,DEV=0 \
  --output-device plughw:CARD=Speaker,DEV=0
```

If the server is running on another host on the LAN, replace the URL:

```bash
python rpi_webrtc_voice.py \
  --offer-url http://192.168.1.50:8080/api/offer \
  --input-device plughw:CARD=Speaker,DEV=0 \
  --output-device plughw:CARD=Speaker,DEV=0
```

Successful loopback means audio captured from the Jabra is sent over WebRTC and
played back through the Jabra speaker.

## Useful environment variables

```bash
export WEBRTC_OFFER_URL=http://127.0.0.1:8080/api/offer
export ALSA_INPUT_DEVICE=plughw:CARD=Speaker,DEV=0
export ALSA_OUTPUT_DEVICE=plughw:CARD=Speaker,DEV=0
export AUDIO_SAMPLE_RATE=48000
export AUDIO_CHANNELS=1
export DEVICE_ID=rpi5-jabra
```

Then run:

```bash
python rpi_webrtc_voice.py
```

## Notes

- The loopback server is for transport testing only. It does not perform STT,
  LLM, or TTS.
- If you hear feedback, lower speaker volume or move the Jabra away from nearby
  reflective surfaces. The Jabra hardware performs echo cancellation, but
  loopback is intentionally unforgiving.
- If `default` works for capture and playback, you can omit the ALSA device
  flags. Pinning the explicit `plughw:` device is more stable for unattended
  boots.
