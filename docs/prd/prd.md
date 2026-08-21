# Product Requirements Document — voice-chatbot ("Babel")

| | |
|---|---|
| **Product** | voice-chatbot (working name "Babel") |
| **Status** | Draft v1.0 |
| **Date** | 2026-08-06 |
| **Author** | Roland Knight (drafted from stated requirements + current implementation review) |
| **Sources** | `README.md`, `docs/web-rtc.md`, `docs/esp32.md`, `docs/telegram.md`, `docs/comparison.md`, `config.yaml`, `personas.yaml`, `server.py`, `app.py`, `skills/`, `devices/rpi5/`, `firmware/box3/`, `todo.md` |

---

## 1. Overview

### 1.1 Product vision

A **private, self-hosted, real-time voice assistant** that replaces Alexa-class devices for a household, built LLM-first rather than intent-first. Common requests (timers, weather, time, media control) are answered with sub-second perceived latency by a local model; complex prompts are routed on demand to a larger cloud model (Claude). The assistant is reachable from many surfaces — a room speakerphone, cheap satellite hardware, a browser, and (future) messaging channels — all speaking the same backend protocol.

### 1.2 Problem statement

Commercial voice assistants are cloud-dependent, privacy-invasive, single-personality, and weak at open-ended conversation. Open-source alternatives (Home Assistant Voice, OVOS, Willow) are intent-first — every capability must be hand-authored — and lack high-quality voices, voice cloning, persona switching, and rich media integration. There is no self-hosted assistant that combines low-latency local tool-calling with escalation to a frontier cloud model.

### 1.3 Architectural identity (from `docs/comparison.md`)

Premium, single-household, Apple-Silicon-hosted, LLM-first, audio-rich. Differentiators vs. the open-source field:

1. **LLM-first design** — skills are LLM tool calls, not intent templates.
2. **Hot-swappable backends** — local Ollama by default; Claude reachable mid-session.
3. **Per-persona cloned voices** (Chatterbox zero-shot cloning) — rare in OSS.
4. **Integrated media** — live BBC radio, BBC Sounds on-demand, Spotify Connect.
5. **Sound design** — generative foley/SFX (Woosh, Stable Audio Open); no one else does this.
6. **Apple Silicon performance** — Whisper MLX, MPS-accelerated TTS.

### 1.4 Target users

- **Primary:** the household operator (technical; owns the Mac Studio server, Spotify Premium account, API keys).
- **Secondary:** household members and guests who interact purely by voice or via the web page (non-technical; must need zero setup).
- **Future:** multiple identified speakers with per-person context (see §4.9).

---

## 2. Goals and non-goals

### 2.1 Goals

1. Perceived end-to-end latency for common skills (timer, weather, time, media control) under ~1.5 s from end of speech to start of spoken reply; sub-3 s for general local-LLM turns.
2. Answer complex prompts well by calling out to a larger, slower cloud model (Claude with web search/fetch) without making the fast path slower.
3. Support multiple client surfaces against one backend: local speakerphone, WebRTC browser client, WebRTC smart clients (Raspberry Pi 5 now; ESP32-S3-BOX-3 in future), and later a ChatGPT-style web UI and Telegram/SMS.
4. Keep the default path fully local/private: local STT, local LLM, local TTS. Cloud is opt-in per session.
5. Be extensible in minutes: drop-in skill folders, YAML-declared personas, trainable wake words.

### 2.2 Non-goals (current scope)

- Smart-home device control (lights, locks, thermostats). Possible later via Home Assistant bridge (`docs/comparison.md` §6.7).
- Multilingual support (English-only for now).
- Multi-party media (SFU/conference calling); WebRTC is 1:1 client↔backend.
- Public multi-tenant SaaS deployment; this is a personal/household system (single-tenant, LAN-first, token-auth for remote).
- Native mobile apps — Telegram is the deliberate zero-iOS-development path (`docs/telegram.md`).

---

## 3. System context

```
                         ┌──────────────────────── Backend host (Mac Studio, Apple Silicon) ────────────────────────┐
 Clients                 │                                                                                          │
 ────────                │  FastAPI signaling (POST /api/offer, /api/health, /api/options, /api/sessions)           │
 Browser (web/)  ── WebRTC ─►  SmallWebRTCTransport → Pipecat pipeline per connection:                              │
 RPi 5 + Jabra   ── WebRTC ─►    VAD (Silero) → Wake (openWakeWord, server-side optional) → Whisper MLX STT         │
 ESP32-S3-BOX-3  ── WebRTC ─►    → BackendRouter (Ollama ⇄ Claude) → Skills (tool calls) → Persona router           │
 Jabra (direct)  ── local audio ─►  → TTS dispatch (Kokoro | Chatterbox) → transport out                            │
 Box-3 sensors   ── MQTT ───►  Mosquitto → sensor cache → query_sensors skill        [backend side deferred]        │
 Telegram (future) ─ Bot API ►  aiogram bridge → same services                        [planned]                     │
 SMS (future)    ── gateway ─►                                                        [planned]                     │
                         │                                                                                          │
                         │  Sidecar services (localhost, auto-launched & health-polled by run.sh):                  │
                         │    Ollama (gemma4:26b) · Chatterbox-TTS-Server (:8004) · Woosh SFX (:8005)               │
                         │    Stable Audio Open (:8006) · librespot Spotify Connect · mpv media player              │
                         └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Functional requirements

Status legend: ✅ Implemented · ◐ Partial / built-not-verified · 📋 Designed (doc exists, not built) · ❌ Not started.

### 4.1 Conversation pipeline & latency tiers — P0

| ID | Requirement | Status |
|---|---|---|
| CONV-1 | Real-time voice-to-voice loop: VAD-segmented turns → local STT (Whisper MLX) → LLM → streaming TTS. | ✅ |
| CONV-2 | **Fast tier:** local LLM (default `gemma4:26b` MoE) answers common requests and fires tools without chain-of-thought delay. Warm LLM TTFB ≤ 0.4 s; STT final transcript ≤ 0.5 s after speech end. | ✅ |
| CONV-3 | **Slow tier:** cloud Claude backend (`claude-sonnet-4-6`, with server-side web search and web fetch tools) reachable mid-session via the `ask_claude` skill ("ask Claude…", "switch to Claude"), a client `backend` control message, or a dedicated wake word. Switch is session-scoped and reverts on idle. | ✅ |
| CONV-4 | Latency engineering: Ollama model pinned resident (`keep_alive: -1`), pre-warm of Whisper/LLM/TTS at startup, configurable model fallbacks for low-RAM hosts (documented trade-offs in `config.yaml`). | ✅ |
| CONV-5 | Conversation session model: after `idle_timeout_secs` (default 10 s) of silence, LLM context is wiped, wake state returns to IDLE, and any session-scoped backend flip reverts. | ✅ |
| CONV-6 | An audible/visible cue when the slow tier is engaged (`ClaudeCueEmitter`), so users know why a reply is slower. | ✅ |
| CONV-7 | MCP client support so the LLM can use external MCP tool servers (open question in `todo.md`: local-model MCP competence). | ❌ |

**Latency targets (P0 acceptance):**

| Segment | Target | Measured today |
|---|---|---|
| STT (speech end → final transcript, p99) | ≤ 0.5 s | ~0.28–0.5 s (whisper-tiny.en MLX) |
| Local LLM warm TTFB (with tools in context) | ≤ 0.4 s | ~0.37 s (gemma4:26b, M4 Max) |
| Wake → first TTS audio (smart client, common skill) | ≤ 1.5 s | verified target in `docs/web-rtc.md` |
| General local turn, end-to-end | ≤ 3 s | ~sub-3 s |
| Cloud (Claude) turn | best-effort; cue emitted | n/a |

### 4.2 Skills (tool-calling functions) — P0

| ID | Requirement | Status |
|---|---|---|
| SKILL-1 | Drop-in skill framework: one folder per skill (`skills/<category>/<name>/` with `SKILL.md` frontmatter + `handler.py`), auto-discovered at startup. | ✅ |
| SKILL-2 | Per-turn tool filtering (`SkillFilterProcessor`): only top-K (default 15) relevant tools are exposed to the LLM per turn, so the registry can grow without hurting latency or reliability. | ✅ |
| SKILL-3 | Core skills: current time, current date, `set_timer(minutes, label?)` with spoken alert, `get_weather(location)` (Open-Meteo, no key; default-location precedence config → CoreLocation → IP geo → ask), `web_search(query)` (DuckDuckGo default; Brave/Tavily via key). | ✅ |
| SKILL-4 | Per-skill enable/disable flags in `config.yaml`; skills gated on required credentials auto-hide from the LLM (e.g., Spotify requires `SPOTIPY_CLIENT_ID`). | ✅ |
| SKILL-5 | Sound-effect generation skill (`generate_sound_effect`): dual backends — Woosh (Sony text-to-foley) and Stable Audio Open 1.0 — with keyword-based auto-routing between them. | ✅ |
| SKILL-6 | **[Future]** Device sensor query skill (`query_sensors`: temperature, humidity, presence) backed by the MQTT sensor cache (Box-3 dependent, see §4.10). | 📋 (`docs/esp32.md`) |
| SKILL-7 | "Command memory" (Willow-Auto-Correct-style): log successful transcription→tool pairs and use fuzzy matches as hints/shortcuts. | ❌ (idea, `docs/comparison.md` §6.2) |

### 4.3 Audio input/output & devices — P0

| ID | Requirement | Status |
|---|---|---|
| AUD-1 | Default system audio in/out (leave device names/indexes null in `config.yaml`). | ✅ |
| AUD-2 | Configurable device selection by **name substring** (e.g. "Jabra", stable across reboots) or index; `./run.sh --devices` lists devices. | ✅ |
| AUD-3 | WebRTC audio transport (Opus over SRTP/DTLS) as a first-class, client-agnostic path (see §4.5). | ✅ |
| AUD-4 | Hot-attach/detach of the local audio device: backend starts without the Jabra, a supervisor rescans every 5 s, starts the local pipeline on attach and tears down cleanly on detach; WebRTC endpoint never depends on local audio. | ✅ |
| AUD-5 | Echo/feedback management: rely on hardware AEC (Jabra, ES7210/ES8311 AFE on Box-3); no audio passthrough; documented mitigations. | ✅ |
| AUD-6 | Sample-rate handling: 16 kHz input / 24 kHz output (Kokoro's native rate), with resampling where clients differ. | ✅ |

### 4.4 Wake words — P0

| ID | Requirement | Status |
|---|---|---|
| WAKE-1 | Multiple simultaneous wake-word models (openWakeWord/ONNX) in a single detector; each model binds to a **persona** (e.g. "hey babel" → babel, "hey marvin" → marvin, "hey one one" → one_one). | ✅ |
| WAKE-2 | Custom wake-word training pipelines: `scripts/wakeword/` (openWakeWord, Docker) for server/Pi clients; `scripts/microwakeword/` for ESP32 (INT8 TFLite Micro, ~50 KB). | ✅ |
| WAKE-3 | Tunable detection: per-chunk probability threshold, per-model cooldown, VAD loudness gate and stop time. | ✅ |
| WAKE-4 | Wake placement is client-dependent: server-side wake for the local pipeline and browser "Listen" mode; on-device wake for smart clients (`mode:"push"` skips server wake entirely). | ✅ (Pi live-test pending, step E) |
| WAKE-5 | Wake-model choice can select the backend (e.g. "hey claude" → cloud) via the client's `backend` control message (`--wake-backend-map` on the Pi client). | ◐ built, needs live E2E test |

### 4.5 WebRTC transport & client protocol — P0

| ID | Requirement | Status |
|---|---|---|
| RTC-1 | Client-agnostic backend: `POST /api/offer` SDP exchange, per-connection Pipecat pipeline on `SmallWebRTCTransport`; a browser tab and an embedded device look identical to the pipeline. | ✅ |
| RTC-2 | JSON control DataChannel (`control`): client→server `hello` (capability advertisement), `backend`, `persona`, `sensor`, `bye`; server→client `ready`, `state` (listening/thinking/speaking/idle), `transcript` (partial/final), `output`, `error`. Unknown message types ignored (forward-compatible). | ✅ |
| RTC-3 | Parallel-session isolation: each connection gets its own backend state, persona state, skill registry, and TTS processors; heavyweight models (Whisper, Kokoro ONNX) are shared safely. Verified (browser-on-Claude + client-on-Ollama concurrently). | ✅ |
| RTC-4 | Session lifecycle: smart client owns session end (`bye` / peer close); server idle timeout only resets context (`cancel_on_idle_timeout=False`); absolute stale-session guard (`STALE_SESSION_SECS`, default 300 s) reaps abandoned peers. | ✅ |
| RTC-5 | Remote hardening: HTTPS (`WEBRTC_SSL_CERT/KEY`), constant-time bearer-token auth on `/api/offer` + `/api/sessions`, per-IP rate limiting, configurable STUN/TURN (`WEBRTC_ICE_SERVERS`), `/api/sessions` observability, startup security-posture log line. | ✅ |
| RTC-6 | Smoke-test harness (`webrtc_smoke/`): standalone loopback + DataChannel echo server kept as a permanent transport regression tool. | ✅ |

### 4.6 Client surfaces — P0/P1/Future

| ID | Requirement | Priority | Status |
|---|---|---|---|
| CLI-1 | **Local speakerphone** (Mac + Jabra, `app.py` legacy path and `server.py --local-audio`): always-on wake-word pipeline on the room device. | P0 | ✅ |
| CLI-2 | **Browser client** (`web/`): Talk (push-to-talk/VAD) and Listen (server-side wake) modes, backend & persona pickers, live state indicator, transcript pane, raw control-message console. | P0 | ✅ |
| CLI-3 | **ChatGPT-style web UI**: conversational page with message history bubbles, **typed text input** alongside voice, text and voice output, streaming responses. The current `web/` page is a dev harness (voice-only, raw JSON send box) — this is a redesign plus a backend text-turn path (inject text into the pipeline without STT). | Future | ❌ |
| CLI-4 | Web UI attachments: image, video, and other file types as inputs/outputs. | Future | ❌ |
| CLI-5 | **Raspberry Pi 5 smart client** (`devices/rpi5/`): on-device openWakeWord, pre-roll ring buffer (~500 ms, no clipped first phoneme), full-duplex audio on the Jabra, connect-on-wake (`mode:"push"`), reconnect with exponential backoff, systemd deployment, env-file config. | P0 | ◐ (steps A–D,F done; E built, live E2E pending) |
| CLI-6 | **ESP32-S3-BOX-3 smart client** (`firmware/box3/`): ESP-IDF + esp-webrtc-solution, esp-sr AFE (AEC/AGC/NS/beamforming), microWakeWord on-device, LCD state UI driven by `state` messages, Opus 16 kHz up / 24 kHz down. | Future | ◐ (firmware implemented; custom-model + E2E validation deferred) |
| CLI-7 | **Telegram bridge** (`docs/telegram.md`): aiogram bot, text messages and OGG/Opus voice notes both directions, `/claude` `/ollama` `/persona` commands, user-ID allowlist, wake-word bypass, persona-tag stripping in text replies. Phase 2 option: real-time calls via userbot + pytgcalls. | P1 | 📋 |
| CLI-8 | **SMS bridge**: text in/out via an SMS gateway (e.g. Twilio); same service-layer reuse as Telegram. Requires the `services.py` extraction (§6, TECH-2). | P2 | ❌ |

### 4.7 Personas & output voices — P0

| ID | Requirement | Status |
|---|---|---|
| PERS-1 | Multiple output voices declared in `personas.yaml`; per-persona TTS backend: **Kokoro** (ONNX, default `babel` voice) or **Chatterbox-Turbo** (zero-shot cloning from a 5–15 s reference clip, MPS-accelerated, OpenAI-compatible local server). | ✅ |
| PERS-2 | Declarative persona routing rules, evaluated in order: `voice_command` ("switch to {persona}", "be {persona}", "talk like {persona}"), `llm_tag` (inline `[persona:name]` one-shot switch mid-response), `skill_intent` (`switch_persona` tool for indirect phrasings). | ✅ |
| PERS-3 | Persona selection by wake word (WAKE-1) and by client `persona` control message. | ✅ |
| PERS-4 | Per-persona synthesis settings (Chatterbox `exaggeration`, `cfg_weight`); graceful degradation — Kokoro personas unaffected if the Chatterbox server is down. | ✅ |
| PERS-5 | Adding a voice requires no code: drop a reference clip in `voices/`, add a YAML entry, restart. | ✅ |

### 4.8 External streaming services — P0

| ID | Requirement | Status |
|---|---|---|
| MEDIA-1 | **BBC live radio**: 11 stations via mpv against BBC HLS endpoints, output targeted at the configured device; voice skills `play_bbc_radio` / `stop_bbc_radio`. | ✅ |
| MEDIA-2 | **BBC Sounds on-demand** (`play_bbc_show(show, date?, query?)`): curated RSS-backed shows (fast path), relative-date resolution, keyword episode matching, best-effort fallback via BBC Sounds search + yt-dlp. | ✅ |
| MEDIA-3 | **Spotify** (Premium): headless librespot Connect endpoint ("Babel") controlled via Web API (spotipy, PKCE OAuth); skills for play track/album/artist/playlist (own playlists fuzzy-matched first), pause/resume/skip/what's-playing/stop; device-ID caching and rediscovery on every play. | ✅ |
| MEDIA-4 | **Ducking**: while the assistant is listening or replying, local playback is paused via mpv JSON-IPC (zero-latency, no API call) and auto-resumed, with an 8 s safety timer for stray wakes (`MediaDuckWatcher`). | ✅ |
| MEDIA-5 | **Mutual exclusion**: starting radio stops Spotify and vice versa (one mpv per output device). | ✅ |
| MEDIA-6 | Additional services (internet radio directories, other podcast sources, multi-room targets). | ❌ future |

### 4.9 Multiparty speaker detection — P1 (not implemented)

From `todo.md`, phased:

| ID | Requirement | Status |
|---|---|---|
| SPKR-1 | **Within-session diarization**: distinguish speakers in a session with minimal latency impact (research: streaming diarization / speaker-embedding models alongside VAD). | ❌ |
| SPKR-2 | **Voiceprint enrollment & identification**: store voiceprints, identify known household members across sessions; enables per-person context, personalization, and simple voice-gated permissions. | ❌ |
| SPKR-3 | Latency budget: speaker ID must not add meaningfully to the fast tier (< ~100 ms target, run concurrent with STT). | ❌ |

### 4.10 Device sensors & outputs (smart-client extras) — Future

All items in this section depend on the ESP32-S3-BOX-3 smart client (CLI-6) and are deferred with it.

| ID | Requirement | Status |
|---|---|---|
| SENS-1 | Always-on sensor telemetry over MQTT, independent of WebRTC session state: Box-3 SENSOR add-on (SHT40 temp/humidity, mmWave radar presence, IR rx/tx), retained topics, LWT online/offline, publish-on-change cadence. | ◐ firmware ✅; backend (Mosquitto, `mqtt_bridge.py` cache, skill) 📋 |
| SENS-2 | Server-commanded client outputs via the control channel (`output`: LCD text, LEDs, relays, servos), gated on advertised capabilities. | 📋 protocol fixed; no pipeline handlers yet |
| SENS-3 | Presence-triggered behaviors (greeting on entry) and IR control skill ("turn off the TV") with a YAML codebook. | ❌ future |

---

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Latency** — per the table in §4.1; latency is the product's defining constraint and every feature must state its impact on the fast tier. |
| NFR-2 | **Privacy** — default path 100% local (STT, LLM, TTS on-host). Cloud calls only via explicit `ask_claude`/backend switch. `hub_offline` mode works fully offline after model downloads. |
| NFR-3 | **Security** — secrets in `.env` only (gitignored), non-secret config in committed `config.yaml`; remote access requires bearer token + HTTPS + rate limiting; Telegram gated by user-ID allowlist; OAuth via PKCE (no client secret needed for Spotify). |
| NFR-4 | **Robustness** — config validated at startup via Pydantic (fail loudly before opening audio devices); sidecar services health-polled by `run.sh`; degraded modes: Chatterbox down → Kokoro personas unaffected; Jabra unplugged → WebRTC endpoint unaffected; client Wi-Fi drop → reconnect with backoff; abandoned peers reaped. |
| NFR-5 | **Concurrency** — multiple simultaneous client sessions with isolated routing state; shared services (TTS servers, Whisper, media players) must multiplex without interleaving audio or state. |
| NFR-6 | **Extensibility** — new skill = new folder; new voice = clip + YAML; new wake word = training pipeline run; new client = existing HTTP+WebRTC+DataChannel protocol (no backend changes); new control capability = new message kind (no protocol bump). |
| NFR-7 | **Hardware envelope** — server: Apple Silicon Mac (≥24 GB RAM for the default 26B model; documented fallbacks to E4B/qwen2.5:3b). Clients: RPi 5 + Jabra Speak2 40 (~$150), ESP32-S3-BOX-3 (~$50). |
| NFR-8 | **Observability** — `/api/sessions` (count, per-peer mode/IP/age), startup security-posture log, wake/pipeline event logs, `--print-effective` config dump. |
| NFR-9 | **Licensing** — GPL components (mpv) invoked as subprocesses; yt-dlp path documented as best-effort/fragile; Spotify requires the user's own Premium account and developer app. |

---

## 6. Technical debt / enabling work

| ID | Item | Why it matters |
|---|---|---|
| TECH-1 | Verify RPi 5 step E live end-to-end (spoken turn through on-device wake). | Gates CLI-5 done-ness. |
| TECH-2 | Extract service construction from `app.py`/`server.py` into a shared `services.py` (`build_stt/llm/tts/persona_router`). | Prerequisite for Telegram (CLI-7), SMS (CLI-8), and the future web-UI text-turn path (CLI-3). |
| TECH-3 | Text-turn injection path in the pipeline (text in → LLM → text + optional TTS out, bypassing STT/wake). | Prerequisite for CLI-7, CLI-8, and future CLI-3. |
| TECH-4 | **[Future]** Backend MQTT side: Mosquitto compose, `mqtt_bridge.py`, `query_sensors` skill. | Unlocks SENS-1/SKILL-6 (deferred with Box-3). |
| TECH-5 | **[Future]** Train `hey_claude` microWakeWord model; validate Box-3 with a stock model first. | Unlocks WAKE-5 on the Box-3 (deferred with CLI-6). |
| TECH-6 | Cleanup: remove legacy `LocalAudioTransport` path in `app.py` once `server.py --local-audio` is fully proven (build step 7 in `docs/web-rtc.md`). | Reduces dual-maintenance. |

---

## 7. Release phasing

**Phase 1 — Solidify the core (now):**
RPi 5 live E2E (TECH-1); keep latency targets green.

**Phase 2 — More surfaces:**
Telegram text + voice notes (CLI-7, with TECH-2/3 as prerequisites).

**Phase 3 — Intelligence & people:**
Within-session speaker diarization → voiceprint ID (SPKR-1/2/3); MCP client support (CONV-7); command memory (SKILL-7).

**Phase 4 — Reach:**
SMS bridge (CLI-8); optional Telegram real-time calls.

**Future (deferred, unscheduled):**
ChatGPT-style web UI with text input and attachments (CLI-3/CLI-4); ESP32-S3-BOX-3 smart client completion (CLI-6, TECH-5) and its sensor/output stack (§4.10, SKILL-6, TECH-4); sensor-triggered behaviors and IR control (SENS-3).

---

## 8. Success metrics

- **Latency:** fast-tier targets in §4.1 hold on the reference host, measured per release.
- **Wake quality:** FAR < 0.5/hour, FRR < 5% per custom model (target from `docs/web-rtc.md`).
- **Reliability:** a smart client survives Wi-Fi drops and server restarts without manual intervention; no leaked pipelines (stale-guard verified).
- **Daily-driver test:** the household uses Babel instead of a commercial assistant for timers, weather, radio, and music for a full week without falling back.

## 9. Open questions

1. Is `gemma4` competent as an MCP client, or does MCP support imply routing more turns to Claude? (`todo.md`)
2. Which diarization approach fits the < ~100 ms fast-tier budget for SPKR-1?
3. ChatGPT-style UI: extend the existing DataChannel protocol with text-turn messages, or add a parallel HTTP/WebSocket chat endpoint?
4. SMS provider choice and cost model (Twilio vs. alternatives) for CLI-8.
5. Whether to add a Wyoming-protocol bridge so commodity HA Voice PE satellites can feed the pipeline (`docs/comparison.md` §6.3) — the "whole-house" question.
