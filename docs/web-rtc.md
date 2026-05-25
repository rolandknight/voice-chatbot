# WebRTC transport + generic client protocol

Reference design for streaming audio (and, later, control / sensor / output data) between **any WebRTC-capable client** and the `voice-chatbot` backend.

The backend is **client-agnostic**. It accepts a WebRTC peer connection and a JSON-over-DataChannel control protocol. Two reference clients are anticipated:

- **Simple client.** A WebRTC-only browser (or other thin client) with no wake word. The user explicitly opens a session — clicking a button, opening a tab. The connection itself is the turn-start signal.
- **Smart client.** An embedded device — initially the **ESP32-S3-BOX-3** (firmware at `firmware/box3/`), later a Raspberry Pi — that runs an on-device wake word (microWakeWord) and may, in future, expose sensors (motion, ambient light, temperature) and outputs (LCD, LEDs, relays, servos).

The firmware (`firmware/box3/`) and the microWakeWord training pipeline (`scripts/microwakeword/`) implement the smart-client side of this design today. The backend changes described below are **not yet implemented** — they are recorded here so the integration can be picked up later without re-deriving the design.

---

## Design principles

1. **Backend knows nothing about the client class.** Same FastAPI signaling, same Pipecat pipeline, same `SmallWebRTCTransport`. A browser tab and a Box-3 look identical to the pipeline once the peer connection is up.
2. **The connection is the turn-start signal.** Wake detection — if any — happens on the client. The backend's default turn strategy is Silero-VAD on the inbound audio track. There is no server-side wake regex.
3. **DataChannel is the control bus.** A single `control` DataChannel carries JSON messages in both directions: client capability advertisement, backend selection, persona hints, future sensor events, future output commands. Audio stays on RTP; everything else on DataChannel.
4. **Capabilities are advertised, not assumed.** The client tells the server what it is and what it can do at connection time. The server adapts (e.g., enables sensor handlers, exposes output skills) based on the advertised set.
5. **Forward-compatible.** Adding a new capability (a new sensor type, a new output action) is a new message kind on the existing DataChannel — no new endpoint, no protocol bump.

---

## Topology

```
┌─────────────────────────────┐                       ┌──────────────────────────────┐
│  Simple client (browser)    │ ─ HTTP POST /api/offer →                              │
│                             │ ←──────── SDP answer ──                               │
│  getUserMedia mic → Opus    │                       │                              │
│  ↔ Opus → <audio> playback  │ ══ WebRTC (SRTP/DTLS) ═│                              │
│  ↔ DataChannel "control"    │                       │                              │
└─────────────────────────────┘                       │   voice-chatbot backend      │
                                                      │                              │
┌─────────────────────────────┐                       │   FastAPI signaling          │
│  Smart client (Box-3)       │ ─ HTTP POST /api/offer →     ↓                       │
│                             │ ←──────── SDP answer ──   SmallWebRTCTransport       │
│  esp-sr AFE (AEC/AGC/NS)    │                       │      ↓                       │
│      ↓                      │                       │   Pipecat pipeline:          │
│  microWakeWord on-device    │ ══ WebRTC (SRTP/DTLS) ═│     VAD → Whisper MLX        │
│      ↓ (on detect)          │                       │     → BackendRouter          │
│  esp-webrtc-solution        │                       │     → Ollama / Claude        │
│      ↓                      │ ↔ DataChannel "control"│     → Persona TTS            │
│  ES7210 mics / ES8311 spkr  │   (caps, backend,     │      ↓                       │
│  LVGL status on LCD         │    sensors, outputs)  │   ControlChannel adapter     │
│  [future: sensors, GPIO]    │                       │                              │
└─────────────────────────────┘                       └──────────────────────────────┘
```

### Lifecycle — simple client (browser)

1. User opens the web client and clicks **Talk**.
2. Browser `getUserMedia` opens the mic; WebRTC offer POSTed to `/api/offer`.
3. Server answers; peer connection establishes.
4. Browser sends `hello` over the `control` DataChannel advertising `{ kind: "simple", capabilities: ["audio"] }`.
5. Audio streams continuously; server Silero-VAD segments turns. Whisper → LLM → TTS round-trip plays back in `<audio>`.
6. User clicks **Stop** (or closes the tab); peer connection torn down. Or after `IDLE_TIMEOUT_SEC` of silence, server closes.

### Lifecycle — smart client (Box-3)

1. Box-3 idle, mic feeding AFE → microWakeWord. WebRTC peer not yet open.
2. "hey babel" (or "hey claude") fires on-device.
3. Box-3 POSTs SDP offer to `/api/offer`; server answers; peer up in <500 ms.
4. Box-3 sends `hello` advertising `{ kind: "smart", capabilities: ["audio", "wakeword", "display"], wake: "hey_babel" }`.
5. Box-3 immediately streams from its **pre-roll ring buffer** (~500 ms before wake) so the first phoneme after wake isn't clipped, then live mic.
6. Pipeline runs as above. TTS returns over the same peer connection.
7. After `IDLE_TIMEOUT_SEC` of no speech, server closes the peer connection. Device tears down and returns to wake-only mode.

---

## Control DataChannel protocol

One DataChannel labelled `control`. JSON messages, one per channel message. All messages have a `type` field. Unknown messages are ignored — both ends must be forward-compatible.

### Client → server

| `type` | Purpose | Fields |
|---|---|---|
| `hello` | Capability advertisement. Sent once, immediately after channel opens. | `kind` (`"simple"` \| `"smart"`), `capabilities` (string array), optional `wake`, `device_id`, `firmware_version` |
| `backend` | Request a specific LLM backend for this session. | `name` (`"ollama"` \| `"claude"`) |
| `persona` | Hint a persona by name (overrides router default for the session). | `name` |
| `sensor` | (smart clients) Sensor event. | `sensor` (e.g. `"motion"`, `"lux"`, `"temp"`), `value`, optional `ts` |
| `bye` | Graceful disconnect intent. | — |

### Server → client

| `type` | Purpose | Fields |
|---|---|---|
| `ready` | Sent after `hello` is processed; pipeline ready for audio. | `session_id` |
| `state` | Pipeline state change for UI display. | `state` (`"listening"` \| `"thinking"` \| `"speaking"` \| `"idle"`), optional `persona` |
| `transcript` | Partial / final ASR transcript (optional for UI). | `text`, `final` (bool) |
| `output` | (smart clients) Command an output the client advertised. | `output` (e.g. `"lcd_text"`, `"led"`, `"relay"`), `value` |
| `error` | Recoverable error notice. | `code`, `message` |

Capability strings are open-ended; only those both sides understand take effect. A simple client that never sends `sensor` events and ignores `output` messages is fully conformant.

---

## WebRTC smoke-test server (build step 0)

Before any Pipecat wiring, stand up a **minimal standalone WebRTC server** whose only job is to prove the transport works end-to-end against a real browser. This decouples three categories of debugging that otherwise overlap:

- "Did the SDP exchange actually negotiate?"
- "Is audio flowing in both directions over RTP?"
- "Is the control DataChannel reachable and parsing JSON?"

…from the much larger surface area of the Pipecat pipeline, STT, LLM, and TTS. If the smoke test passes, every later failure is in the pipeline or downstream services, not in the transport.

### Layout

```
webrtc_smoke/
  server.py        # ~80 lines: FastAPI + aiortc, no Pipecat
  static/
    index.html     # Talk / Stop buttons, log pane
    client.js      # RTCPeerConnection + DataChannel, same code as web/ later
```

Kept under `webrtc_smoke/` (not `tests/`) so it can run independently and isn't pulled into pytest. Once `server.py` and `web/` (the real ones) land in build step 1, `webrtc_smoke/` stays in the repo as a regression harness — useful any time the transport is suspected.

### Server behavior

A single FastAPI app exposing exactly the same surface the real backend will:

- `POST /api/offer` — accept SDP offer, create an `RTCPeerConnection`, return SDP answer. No Pipecat, just raw aiortc.
- `GET /api/health` → `200 OK`.
- `GET /` → serve `static/index.html`; `GET /static/*` → serve assets.

On the peer connection:

- **Audio loopback.** The inbound audio track is added back to the same peer connection as an outbound track (aiortc `MediaRelay`). Speaking into the mic plays back through the browser speakers after one RTT — proves bi-directional Opus over SRTP works without needing any STT/LLM/TTS.
- **DataChannel echo + introspection.** On `control` channel open, log every received message to stderr and echo it back with `{"type":"echo","original":<msg>}`. Also synthesize a `ready` reply to any `hello`, so the same browser client code that will talk to the real backend works unchanged.

### Browser client

The `static/` page is the same code that will become `web/` in build step 1 — Talk/Stop buttons, `getUserMedia`, `RTCPeerConnection`, control DataChannel, transcript/log pane. Building it here first means step 1 is mostly "wire this existing browser client to the Pipecat pipeline" rather than "write a browser client from scratch while also refactoring the backend."

### What it does NOT do

- No Whisper, no LLM, no TTS.
- No persona / backend routing — `backend` and `persona` messages are echoed but otherwise ignored.
- No idle timeout, no auth, no TLS. Localhost / LAN dev only.

### Verification

- `python webrtc_smoke/server.py`; open `http://localhost:8080`.
- Click Talk, speak — own voice loops back through the speakers within ~100 ms. Confirms RTP + Opus + ICE.
- Browser DevTools console shows `ready` arriving and any sent `hello` / `backend` / `persona` being echoed. Confirms DataChannel.
- `chrome://webrtc-internals` shows inbound + outbound audio packets > 0 and no ICE failures.
- Smart client (Box-3, once firmware exists) pointed at the smoke server should reach the same loopback — your own voice plays back through the ES8311 speaker. Confirms the firmware's WebRTC stack independently of the pipeline.

---

## Backend changes (deferred — to implement later)

### Files to modify

#### `app.py`

- Replace `LocalAudioTransport` (line 51 import, lines 456–474 setup) with `SmallWebRTCTransport` from `pipecat.transports.network.small_webrtc`.
- Remove the `WakePhraseUserTurnStartStrategy` block (lines 705–745). Use the default Silero-VAD turn strategy.
- Keep `BackendRouter` (`app.py:127`), `PersonaTagRouter`, `SkillFilterProcessor`, and all TTS/LLM wiring — they're transport-agnostic.
- Add a `ControlChannel` adapter that owns the DataChannel and exposes a small event API (`on_hello`, `on_backend`, `on_persona`, `on_sensor`, `send_state`, `send_output`, …). The Pipecat pipeline subscribes to relevant events; e.g., `BackendRouter` listens for `backend` messages, persona router listens for `persona`, a (future) sensor-skills layer listens for `sensor`.

#### `server.py` (new)

Bootstraps a FastAPI app alongside the Pipecat pipeline. Exposes:

- `POST /api/offer` — accepts an SDP offer, returns an SDP answer. Wraps `SmallWebRTCConnection`. Per-connection, instantiates a `ControlChannel`, attaches it to the pipeline, and waits for `hello`.
- `GET /api/health` — liveness probe.
- Static serve of `web/` — the simple-client reference page (see below).

Reference: Pipecat ships a ~40-line FastAPI example for `SmallWebRTCTransport`. Copy that pattern.

#### `web/` (new)

A minimal browser reference client: one HTML page, vanilla JS, no build step.

- `index.html` — Talk / Stop buttons, persona/backend dropdowns, current-state indicator, transcript pane.
- `client.js` — `RTCPeerConnection` setup, `getUserMedia({ audio: true })`, `/api/offer` exchange, DataChannel send (`hello` with `kind:"simple"`, plus user-driven `backend`/`persona`), DataChannel receive (`state`, `transcript`).
- Served at `GET /` from `server.py`. Doubles as the dev/test harness for backend changes before any embedded hardware is involved.

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

`BackendRouter` currently picks Ollama vs Claude based on which wake phrase matched. With device-side wake (and zero wake on simple clients), the server never sees a wake phrase. The DataChannel `backend` message handles both client classes:

- **Smart client:** firmware sends `{"type":"backend","name":"claude"}` immediately after `hello` based on which microWakeWord model fired (Box-3 has both "hey babel" and "hey claude" models loaded).
- **Simple client:** browser sends `{"type":"backend","name":"…"}` based on a UI dropdown (or skips it to take the server default).

The pre-existing wake-phrase → backend logic is dropped; `BackendRouter` becomes a small adapter that listens for `backend` messages and falls back to a default backend when none is set.

### Why `SmallWebRTCTransport` over Daily / LiveKit

- No external SFU, no cloud account, no extra process.
- HTTP SDP exchange is trivial on both ESP-IDF and `RTCPeerConnection`.
- Single-host deployment; no need for multi-party media.
- Pipecat handles the aiortc lifecycle, codec negotiation, and frame conversion to `InputAudioRawFrame`.

### Reuse as-is

- `chatterbox_tts.py` — TTS protocol unchanged.
- `personas.yaml` + persona router code (`persona_router.py`).
- `BackendRouter`, `SkillFilterProcessor`, all skills (`skills/`).
- Whisper MLX STT block in `app.py`.

---

## Simple client reference (browser)

The browser client exists primarily to:

1. **Be the dev harness** for the backend before any embedded hardware is in the loop — every backend change can be exercised from a laptop with a headset.
2. **Be a real first-class client.** Same protocol, same endpoints, same lifecycle as the smart client. No "test-only" shortcuts.

Out of scope for v1: push-to-talk vs always-on UX modes (always-on is fine — Silero VAD handles segmentation), mobile-optimized UI, auth.

---

## Smart client — ESP32-S3-BOX-3 firmware

Implemented at `firmware/box3/`. See `firmware/box3/README.md` for build / flash / monitor steps.

### Framework choice: ESP-IDF + esp-webrtc-solution

Picked over ESPHome and libpeer for these reasons:

- **AEC matters.** When TTS plays through the Box-3 speaker it cannot bleed into the mic and re-trigger the LLM. Espressif's `esp-sr` AFE provides AEC + AGC + NS tuned for the ES7210/ES8311 codec pair on this exact board.
- **esp-webrtc-solution is the reference.** Espressif published a `box3-openai-realtime` demo using this stack, so the topology is proven on this board.
- **ESPHome would force dropping WebRTC** — its audio transport is custom TCP/UDP or Wyoming, not WebRTC, which breaks the uniform-client goal.
- **libpeer** is smaller and MIT, but you'd reinvent codec setup and lose AFE integration — not worth the savings.

Tradeoff: steeper learning curve and a larger flash footprint (~2 MB), but the Box-3 has 16 MB flash + 8 MB PSRAM, so headroom is fine.

### Audio pipeline (firmware side)

1. **Capture.** `esp_codec_dev` opens ES7210, 16 kHz / 16-bit / 2-channel (Box-3 has two dmics for beamforming).
2. **AFE** (`esp-sr`). Beamforming → AEC (with TTS reference from ES8311 loopback) → NS → AGC → 16 kHz mono int16.
3. **Tap point.** Feed AFE output to microWakeWord continuously, and push into a ~1 s ring buffer.
4. **On wake.** Connect WebRTC and start streaming from `(ring_buffer_tail − 500 ms)` so the first phoneme of the wake phrase and the immediately following speech are captured.
5. **Encode.** Opus at 16 kHz / 20 ms frames / 24–32 kbps. esp-webrtc-solution bundles libopus.
6. **Downlink.** Incoming Opus → decode → ES8311 at 24 kHz (Kokoro's output rate; resample if Chatterbox sends 16 kHz).

### LCD UI (minimal)

Five states, each a full-screen LVGL label + colored background. State changes are driven by `state` messages on the control DataChannel — not inferred locally — so the display can't disagree with the server.

| State | Trigger (`state` value) | Display |
|---|---|---|
| `IDLE` | boot, after disconnect, `idle` | "say 'hey babel'" |
| `LISTENING` | wake fired (local) and post-connect `listening` | "listening…" + level meter |
| `THINKING` | `thinking` | "…" pulse |
| `SPEAKING` | `speaking` | active persona name (from message) |
| `ERROR` | wifi / signaling / RTC failure | red, brief message |

No touch interaction in v1.

### Future smart-client capabilities

Out of scope for v1, but the control protocol is designed to absorb them without new endpoints or breaking changes:

- **Sensors (input).** Motion (PIR), ambient light, temperature, humidity, button presses. Firmware emits `{"type":"sensor","sensor":"motion","value":true}` on edge; server-side skills can use these (e.g., "ask the room is dark, so don't speak loudly").
- **Outputs (server-commanded).** LCD strings (`output: "lcd_text"`), LEDs / RGB (`output: "led"`), relays, servos. Server emits `{"type":"output","output":"led","value":{"r":0,"g":255,"b":0}}`. Smart client advertises which outputs it has in `hello.capabilities`; server only sends commands the client advertised.

The pipeline-side machinery for these arrives only when a concrete use case lands. The protocol shape is fixed now so firmware can stub it.

---

## Wake-word model — retrain "hey babel" for microWakeWord

The existing `scripts/wakeword/` pipeline trains openWakeWord (Python/TFLite). microWakeWord is a separate framework (Kevin Ahrendt) with a different feature pipeline (40-band log-mel, streaming inference, INT8 quantization for TFLite Micro). The model file format and inference graph are not interchangeable.

A parallel pipeline lives at `scripts/microwakeword/` — same dataset bootstrap, different trainer + quantizer. Output artifact: `scripts/microwakeword/_work/output/hey_babel/hey_babel.tflite` (~50 KB int8). The firmware build copies it into `firmware/box3/main/models/`.

Train `hey_claude` from the same pipeline so the dual-wake DataChannel-`backend`-selection design works.

### Validation before custom retrain

Before retraining, flash the firmware with a **stock microWakeWord model** (e.g., "okay nabu" or "hey jarvis", available pretrained) to validate the device → WebRTC → backend → response loop. Only swap to the custom "hey babel" model once the transport is proven. Decouples firmware/transport debugging from model-quality debugging.

---

## Suggested build order

0. **WebRTC smoke-test server.** Build `webrtc_smoke/` (FastAPI + aiortc, no Pipecat) with audio loopback and DataChannel echo. Validate browser → server → browser audio works and `hello` / `ready` exchange parses. Leave it in the repo as a regression harness.
1. **Backend + simple client.** Replace `LocalAudioTransport` with `SmallWebRTCTransport`, stand up `server.py` + `/api/offer`, promote the smoke-server's browser client to `web/`, wire the `ControlChannel` with at least `hello` / `ready` / `backend` / `state`. Confirm Whisper → Ollama → Kokoro round-trip end-to-end in a browser.
2. **Backend selection + persona via DataChannel.** Add `backend` and `persona` handlers; verify Claude routing and persona override from the browser UI.
3. **Box-3 hello-world WebRTC.** Build the firmware with a stock pretrained wake model, advertise `kind:"smart"`, point it first at `webrtc_smoke/` (own-voice loopback through the speaker proves the firmware's RTC stack), then at the real backend. Reuses every backend change from steps 1–2 unchanged.
4. **Audio pipeline polish.** Add the pre-roll ring buffer, tune AFE parameters, confirm TTS doesn't re-trigger the mic via AEC tests.
5. **Custom "hey babel" model.** Build `scripts/microwakeword/` Docker pipeline, train, quantize, flash, verify FAR/FRR roughly matches the openWakeWord version.
6. **LCD UI driven by `state` messages.** LVGL state machine fed by the control channel.
7. **Cleanup.** Remove `LocalAudioTransport` code path, dead env vars, and the `WakePhraseUserTurnStartStrategy` lines from `app.py`.
8. **(Later) Sensors / outputs.** When a concrete use case lands, add the smart-client capability strings and the corresponding pipeline-side handlers. No backend protocol change required.

---

## Verification

**Backend + simple client (after step 1–2 land):**

- `./run.sh`; confirm `GET /api/health` returns 200.
- Open `http://localhost:8080`, click Talk, say "what time is it" — confirm transcript in the UI, Ollama response, Kokoro audio plays in browser.
- Switch backend dropdown to Claude → confirm `backend` message routes through the Claude path.
- Switch persona dropdown → confirm `persona` override takes effect on the next utterance.

**Firmware alone:**

- `idf.py -p /dev/ttyACM0 flash monitor`, watch logs.
- Stock wake model: say the trigger phrase; expect log lines `WAKE detected → opening peer → ICE connected → hello sent → ready received → audio in`.
- LCD cycles IDLE → LISTENING → THINKING → SPEAKING → IDLE driven by inbound `state` messages.

**End-to-end (smart client):**

- Box-3 on the same LAN as the Mac; `BACKEND_URL=http://<mac-ip>:8080` in `firmware/box3/main/config.h`.
- "Hey babel, what time is it?" → first phoneme captured (no clipping), <1.5 s to TTS start.
- "Hey claude, summarize this" → DataChannel `backend:"claude"` arrives, routes through Claude path; expected persona TTS plays.
- Play loud music near the Box-3 while TTS speaks: AEC should keep the LLM from re-triggering on its own output.
- Pull WiFi: LCD shows ERROR, device retries gracefully.

**Cross-client parity:**

- Browser and Box-3 each connect once; confirm both produce identical pipeline traces aside from the `hello.kind` / capability fields. The backend should have no other client-class branches.

**Model quality (after retrain):**

- `make -C scripts/microwakeword eval` runs the standard test set, reports FAR (per hour of negative audio) and FRR (% missed positives). Target: FAR <0.5/hr, FRR <5%, comparable to the openWakeWord version.
