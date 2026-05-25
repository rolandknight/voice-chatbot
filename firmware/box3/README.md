# `firmware/box3/` — ESP32-S3-BOX-3 voice client

On-device wake-word ("hey babel" / "hey marvin") + WebRTC streaming client for the `voice-chatbot` backend. See `docs/web-rtc.md` for the architectural design and how this piece connects to the rest of the system.

Status: **scaffolding**. The code structure, audio pipeline shape, signaling protocol, and module boundaries are all in place. Specific SDK API names (esp-sr v1 vs v2, esp-webrtc-solution 1.x release symbols) may need light adjustment when first built against a pinned ESP-IDF release — see the "Building for the first time" section.

## Hardware

- ESP32-S3-BOX-3 (16 MB flash, 8 MB octal PSRAM, ES7210 dual-mic ADC, ES8311 speaker DAC, 320×240 LCD)

## Toolchain

- ESP-IDF v5.1+ (`. ~/esp/esp-idf/export.sh`)
- `idf.py` on PATH
- USB-C cable for flashing + monitor (`/dev/ttyACM0` on Linux, `/dev/cu.usbmodem*` on macOS)

## Configure

Edit `main/config.h`:

- `CONFIG_BOX3_WIFI_SSID`, `CONFIG_BOX3_WIFI_PASSWORD` — your WiFi
- `BACKEND_HOST`, `BACKEND_PORT` — the Mac running `voice-chatbot` (must be on the same LAN; default port is 8080 per `.env` `WEBRTC_PORT`)

Until the backend WebRTC changes land (see `docs/web-rtc.md`, "Backend changes — deferred"), the device will boot, hear the wake phrase, and fail at the HTTP `POST /api/offer` step with no answer. The LCD will show `error: no answer`. That's the expected pre-backend state.

## Drop in the wake-word models

```sh
# After running scripts/microwakeword/make all:
make install
```

That copies `hey_babel.tflite` and `hey_marvin.tflite` from `scripts/microwakeword/_work/output/` into `main/models/`. Or use a stock microWakeWord model first to validate the firmware/transport without waiting for a custom train:

```sh
make install-stock   # downloads okay_nabu.tflite into both slots
```

See `main/models/README.md` for details.

## Build, flash, monitor

```sh
make set-target      # one-time, picks esp32s3
make flash-monitor   # builds, flashes, opens serial monitor
```

Or run individual stages: `make build`, `make flash`, `make monitor`. The serial port is autodetected (`/dev/ttyACM0` on Linux, `/dev/cu.usbmodem*` on macOS) — override with `make flash-monitor PORT=/dev/ttyUSB0`. Run `make help` for the full target list, or `make menuconfig` to tweak sdkconfig.

Under the hood these targets wrap `idf.py` — same as running:

```sh
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Expected boot log:

```
I (xxx) box3: ready — say 'hey babel' or 'hey marvin'
I (xxx) audio: audio pipeline up — 16000 Hz, AEC=on
I (xxx) wake: loaded hey babel (NNNN bytes)
I (xxx) wake: loaded hey marvin (NNNN bytes)
I (xxx) ui: ui ready
```

Say "hey babel"; expected next log lines:

```
I (xxx) box3: WAKE: hey babel (conf=0.84)
I (xxx) webrtc: peer connected      # once the backend exists
```

## Architecture (firmware-side)

```
main.c
  ├── wifi_start            # STA, blocks on first IP
  ├── ui_init               # LVGL display, esp-bsp display + backlight
  ├── audio_pipeline_init   # ES7210/ES8311 + esp-sr AFE + ring buffer
  ├── wakeword_init(on_wake)
  │      └── task: read AFE frames → invoke both .tflite models → fire cb
  └── webrtc_client_init
         ├── task: session FSM (idle → offer → connected → closed)
         └── task: uplink pump (read AFE frame → esp_peer_send_audio)

on_wake(evt)
  → ui_set_state(LISTENING, evt->phrase)
  → webrtc_client_open({.backend = evt->backend, ...})
        (asynchronously goes through CMD_OPEN → SDP offer → POST → answer)
```

Five FreeRTOS tasks total, all pinned to defined cores:

| Task          | Core | Stack | Job |
|---------------|------|-------|-----|
| `afe_feed`    | 1    | 4 KB  | I2S read → AFE feed |
| `afe_fetch`   | 0    | 4 KB  | AFE fetch → ring + queue |
| `wakeword`    | 1    | 8 KB  | Pop frame, invoke TFLM models |
| `rtc_sess`    | 0    | 8 KB  | Session FSM + signaling HTTP |
| `rtc_up`      | 0    | 4 KB  | Uplink frame pump |
| `ui`          | any  | 4 KB  | LVGL state apply on queue msgs |

## Building for the first time — likely sharp edges

The scaffold targets the following API generations:

- esp-sr **v1.4.x** (`esp_afe_sr_iface.h`, `AFE_CONFIG_DEFAULT`, `ESP_AFE_SR_HANDLE`)
- esp-webrtc-solution **v1.0** (`esp_peer.h`, `esp_peer_default.h`, `esp_peer_create`, `esp_peer_send_audio`)
- esp-tflite-micro **v1.3.x** (`MicroInterpreter`, `MicroMutableOpResolver`)
- esp-bsp **v1.4.x** for the BOX-3 board variant

If the registry ships a newer major version when you build, expect symbol renames in `audio_pipeline.c` (AFE struct field names) and `webrtc_client.c` (peer API). The data flow is the same; only the call sites need touching. Each module is small (~200 lines) so fixups are localized.

The TFLM op resolver in `wakeword_tflm.cc` lists the ops microWakeWord's streaming DS-CNN actually uses. If you swap in a model that needs a different op (e.g., a transformer-based one), `MicroMutableOpResolver` will fail to allocate tensors at startup — add the missing op there.

## Testing without the backend

You can iterate on UI, wake-word detection, and audio I/O without the backend up:

1. Leave `BACKEND_HOST` pointed at a non-existent address.
2. Flash and monitor.
3. Speak the wake phrase. LCD goes `IDLE → LISTENING → ERROR: no answer` in ~5 s.
4. Confirm the wake-word detector logs the phrase + confidence, and the LCD transitions look right.

Once `voice-chatbot` is patched to expose `/api/offer` (see deferred work in `docs/web-rtc.md`), point `BACKEND_HOST` at the Mac's LAN IP and try the full loop.

## Layout

```
firmware/box3/
  CMakeLists.txt              # ESP-IDF project file
  Makefile                    # idf.py wrapper: install / build / flash / monitor
  sdkconfig.defaults          # PSRAM, flash 16 MB, esp-sr AFE on, WakeNet off
  partitions.csv              # OTA layout + 128 KB models partition (unused for now)
  README.md                   # this file
  main/
    CMakeLists.txt            # component definition; EMBED_FILES the .tflite
    idf_component.yml         # managed deps: esp-bsp, esp-sr, esp_webrtc, ...
    config.h                  # WiFi + backend URL + audio params
    main.c                    # boot, wifi, glue
    audio_pipeline.{c,h}      # codec + AFE + ring buffer
    wakeword.{c,h}            # frame loop, model selection
    wakeword_tflm.cc          # C++ TFLM interpreter glue
    webrtc_client.{c,h}       # signaling + peer connection lifecycle
    ui.{c,h}                  # LVGL state display
    models/
      README.md               # how to drop in the .tflite files (git-ignored)
```
