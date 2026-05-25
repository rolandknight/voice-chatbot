# WebRTC + on-device wake-word architecture

Reference design for streaming audio between an **ESP32-S3-BOX-3** (and, later, a Raspberry Pi) and the `voice-chatbot` backend. The wake-word runs on the device; audio flows over WebRTC.

The firmware (`firmware/box3/`) and the microWakeWord training pipeline (`scripts/microwakeword/`) implement the device side of this design today. The backend changes described below are **not yet implemented** — they are recorded here so the integration can be picked up later without re-deriving the design.

---

## Why this design

`voice-chatbot` is a Mac-local Pipecat prototype: `LocalAudioTransport` reads from a Jabra USB speakerphone and writes back to it (`app.py:456–474`), and a `WakePhraseUserTurnStartStrategy` (regex over Whisper transcripts) gates turn detection (`app.py:705–745`). The trained `hey_babel` openWakeWord model under `scripts/wakeword/_work/output/hey_babel/` is not yet wired in.

The goal is to make the Box-3 a first-class voice client:

- **Wake-word on the device** (microWakeWord, "hey babel" / "hey claude"). Saves radio, saves power, removes the false-positive-prone regex-on-transcript gate.
- **WebRTC transport**. Bi-directional, low-latency, codec-agnostic. The same protocol works for the Box-3 today and a Raspberry Pi later.
- **Backend pipeline unchanged in spirit** — Whisper → Ollama/Claude → Kokoro/Chatterbox. Only the transport changes.
- **`LocalAudioTransport` removed** in favor of WebRTC everywhere. If the Mac is used as a client during dev, it speaks WebRTC via a browser-based test page.

---

## Topology

```
┌──────────────────────────────┐                        ┌──────────────────────────────┐
│   ESP32-S3-BOX-3 firmware    │ ── HTTP POST /api/offer→│   voice-chatbot backend (Mac)│
│                              │ ←──────── SDP answer ───│                              │
│  esp-sr AFE (AEC/AGC/NS)     │                        │   FastAPI signaling          │
│       ↓                      │                        │       ↓                      │
│  microWakeWord ("hey babel") │ ══ WebRTC (SRTP+DTLS) ══│   SmallWebRTCTransport       │
│       ↓ (on detect)          │   Opus 16 kHz mono up  │       ↓                      │
│  esp-webrtc-solution         │   Opus 24 kHz mono dn  │   Pipecat pipeline:          │
│       ↓                      │   DataChannel control  │     VAD → Whisper MLX        │
│  ES7210 mics / ES8311 spkr   │                        │     → BackendRouter          │
│  LVGL status on LCD          │                        │     → Ollama/Claude          │
│                              │                        │     → Persona TTS            │
└──────────────────────────────┘                        └──────────────────────────────┘
```

### Wake → talk → respond

1. Box-3 idle, mic feeding AFE → microWakeWord. WebRTC peer not yet open.
2. "hey babel" fires on-device.
3. Box-3 POSTs SDP offer to `POST /api/offer`. Backend answers; peer connection up in <500 ms.
4. Box-3 immediately streams from the **pre-roll ring buffer** (~500 ms before the wake) so the first phoneme after wake isn't clipped, then live mic.
5. Server pipeline runs Silero VAD → Whisper → LLM → TTS. TTS audio returns over the same peer connection.
6. Server VAD detects end-of-turn → LLM responds → TTS streams back → device plays via ES8311.
7. After `IDLE_TIMEOUT_SEC` of no speech, server closes the peer connection. Device tears down and returns to wake-only mode.

There is **no server-side wake regex**. Wake is purely device-side; the act of opening the WebRTC peer connection is the turn-start signal.

---

## Backend changes (deferred — to implement later)

### Files to modify

#### `app.py`

- Replace `LocalAudioTransport` (line 51 import, lines 456–474 setup) with `SmallWebRTCTransport` from `pipecat.transports.network.small_webrtc`.
- Remove the `WakePhraseUserTurnStartStrategy` block (lines 705–745). Use the default Silero-VAD turn strategy.
- Keep `BackendRouter` (`app.py:127`), `PersonaTagRouter`, `SkillFilterProcessor`, and all TTS/LLM wiring — they're transport-agnostic.

#### `server.py` (new)

Bootstraps a FastAPI app alongside the Pipecat pipeline. Exposes:

- `POST /api/offer` — accepts an SDP offer, returns an SDP answer. Wraps `SmallWebRTCConnection`.
- `GET /api/health` — liveness probe.
- (Optional) static serve of a Pipecat web client demo for browser-based dev testing.

Reference: Pipecat ships a ~40-line FastAPI example for `SmallWebRTCTransport`. Copy that pattern.

#### `run.sh`

Start `uvicorn server:app --host 0.0.0.0 --port 8080` (or `python server.py`) instead of `python app.py`. Ollama, Chatterbox, and Woosh launches unchanged.

#### `.env.example`

Add:

```
WEBRTC_HOST=0.0.0.0
WEBRTC_PORT=8080
WEBRTC_ICE_SERVERS=stun:stun.l.google.com:19302
IDLE_TIMEOUT_SEC=20
```

Remove (or comment out): `INPUT_DEVICE_INDEX`, `OUTPUT_DEVICE_INDEX`, `WAKE_PHRASES`, `CLAUDE_WAKE_PHRASES`.

### Backend selection without wake-phrase regex

`BackendRouter` currently picks Ollama vs Claude based on which wake phrase matched. With device-side wake, the server no longer sees the wake phrase. Two options:

- **(a) DataChannel control message.** Device opens a DataChannel at connection time and sends `{"backend":"ollama"}` or `{"backend":"claude"}` based on which microWakeWord model fired. Box-3 has both "hey babel" and "hey claude" models loaded simultaneously.
- **(b) Query param.** `POST /api/offer?backend=ollama` baked into firmware build.

**Recommend (a)** — preserves the existing two-wake-phrase UX with one device.

### Why `SmallWebRTCTransport` over Daily / LiveKit

- No external SFU, no cloud account, no extra process.
- HTTP SDP exchange is trivial on ESP-IDF.
- Mac-only deployment; no need for multi-party media.
- Pipecat handles the aiortc lifecycle, codec negotiation, and frame conversion to `InputAudioRawFrame`.

### Reuse as-is

- `chatterbox_tts.py` — TTS protocol unchanged.
- `personas.yaml` + persona router code (`persona_router.py`).
- `BackendRouter`, `SkillFilterProcessor`, all skills (`skills/`).
- Whisper MLX STT block in `app.py`.

---

## ESP32-S3-BOX-3 firmware

Implemented at `firmware/box3/`. See `firmware/box3/README.md` for build / flash / monitor steps.

### Framework choice: ESP-IDF + esp-webrtc-solution

Picked over ESPHome and libpeer for these reasons:

- **AEC matters.** When TTS plays through the Box-3 speaker it cannot bleed into the mic and re-trigger the LLM. Espressif's `esp-sr` AFE provides AEC + AGC + NS tuned for the ES7210/ES8311 codec pair on this exact board.
- **esp-webrtc-solution is the reference.** Espressif published a `box3-openai-realtime` demo using this stack, so the topology is proven on this board.
- **ESPHome would force dropping WebRTC** — its audio transport is custom TCP/UDP or Wyoming, not WebRTC, which breaks the "uniform with the Pi" goal.
- **libpeer** is smaller and MIT, but you'd reinvent codec setup and lose AFE integration — not worth the savings.

Tradeoff: steeper learning curve and a larger flash footprint (~2 MB), but the Box-3 has 16 MB flash + 8 MB PSRAM, so headroom is fine.

### Audio pipeline (firmware side)

1. **Capture.** `esp_codec_dev` opens ES7210, 16 kHz / 16-bit / 2-channel (Box-3 has two dmics for beamforming).
2. **AFE** (`esp-sr`). Beamforming → AEC (with TTS reference from ES8311 loopback) → NS → AGC → 16 kHz mono int16.
3. **Tap point.** Feed AFE output to microWakeWord continuously, and push into a ~1 s ring buffer.
4. **On wake.** Connect WebRTC and start streaming from `(ring_buffer_tail − 500 ms)` so the first phoneme of "hey babel" and the immediately following speech are captured.
5. **Encode.** Opus at 16 kHz / 20 ms frames / 24–32 kbps. esp-webrtc-solution bundles libopus.
6. **Downlink.** Incoming Opus → decode → ES8311 at 24 kHz (Kokoro's output rate; resample if Chatterbox sends 16 kHz).

### LCD UI (minimal)

Five states, each a full-screen LVGL label + colored background:

| State | Trigger | Display |
|---|---|---|
| `IDLE` | boot, after disconnect | "say 'hey babel'" |
| `LISTENING` | wake fired, mic streaming | "listening…" + level meter |
| `THINKING` | server ack, awaiting TTS | "…" pulse |
| `SPEAKING` | TTS frames arriving | active persona name |
| `ERROR` | wifi / signaling / RTC failure | red, brief message |

No touch interaction in v1.

---

## Wake-word model — retrain "hey babel" for microWakeWord

The existing `scripts/wakeword/` pipeline trains openWakeWord (Python/TFLite). microWakeWord is a separate framework (Kevin Ahrendt) with a different feature pipeline (40-band log-mel, streaming inference, INT8 quantization for TFLite Micro). The model file format and inference graph are not interchangeable.

A parallel pipeline lives at `scripts/microwakeword/` — same dataset bootstrap, different trainer + quantizer. Output artifact: `scripts/microwakeword/_work/output/hey_babel/hey_babel.tflite` (~50 KB int8). The firmware build copies it into `firmware/box3/main/models/`.

Train `hey_claude` from the same pipeline so the dual-wake backend-selection design works.

### Validation before custom retrain

Before retraining, flash the firmware with a **stock microWakeWord model** (e.g., "okay nabu" or "hey jarvis", available pretrained) to validate the device → WebRTC → backend → response loop. Only swap to the custom "hey babel" model once the transport is proven. Decouples firmware/transport debugging from model-quality debugging.

---

## Suggested build order

1. **Backend, no device yet.** Replace `LocalAudioTransport` with `SmallWebRTCTransport`, stand up `server.py` + `/api/offer`, test with the Pipecat Small WebRTC web client demo in a browser. Confirm Whisper → Ollama → Kokoro round-trip over WebRTC.
2. **DataChannel backend selection.** Add the `{"backend": "..."}` message handling; verify Claude routing still works.
3. **Box-3 hello-world WebRTC.** Build the firmware with a stock pretrained wake model, point it at the local backend, confirm audio flows both ways.
4. **Audio pipeline polish.** Add the pre-roll ring buffer, tune AFE parameters, confirm TTS doesn't re-trigger the mic via AEC tests.
5. **Custom "hey babel" model.** Build `scripts/microwakeword/` Docker pipeline, train, quantize, flash, verify FAR/FRR roughly matches the openWakeWord version.
6. **LCD UI.** LVGL state machine.
7. **Cleanup.** Remove `LocalAudioTransport` code path, dead env vars, and the `WakePhraseUserTurnStartStrategy` lines from `app.py`.

---

## Verification

**Backend alone (after deferred work lands):**

- `./run.sh`; confirm `GET /api/health` returns 200.
- Open the Pipecat Small WebRTC sample web client at `http://localhost:8080`, connect, say "what time is it" — confirm Whisper transcript, Ollama response, Kokoro audio play back in browser.
- Set `BACKEND=claude` via DataChannel test → confirm Claude path.

**Firmware alone:**

- `idf.py -p /dev/ttyACM0 flash monitor`, watch logs.
- Stock wake model: say the trigger phrase; expect log lines `WAKE detected → opening peer connection → ICE connected → audio in`.
- LCD cycles IDLE → LISTENING → THINKING → SPEAKING → IDLE.

**End-to-end:**

- Box-3 on the same LAN as the Mac; `BACKEND_URL=http://<mac-ip>:8080` in `firmware/box3/main/config.h`.
- "Hey babel, what time is it?" → first phoneme captured (no clipping), <1.5 s to TTS start.
- "Hey claude, summarize this" → routes through Claude path; expected persona TTS plays.
- Play loud music near the Box-3 while TTS speaks: AEC should keep the LLM from re-triggering on its own output.
- Pull WiFi: LCD shows ERROR, device retries gracefully.

**Model quality (after retrain):**

- `make -C scripts/microwakeword eval` runs the standard test set, reports FAR (per hour of negative audio) and FRR (% missed positives). Target: FAR <0.5/hr, FRR <5%, comparable to the openWakeWord version.
