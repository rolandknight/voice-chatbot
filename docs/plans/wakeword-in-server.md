# Plan: multi-wake-word Listen mode in the Rust server

Status: implemented 2026-08-26 on branch `wakeword`. Outcome notes at the end.
Scope note added during implementation: server-side wake serves **browser**
clients; the **native Rust WebRTC client detects on-device** and reports the
persona to the server (see "Client-side wake" below). Only the new Rust code
(`crates/`, its vendored `third_party/oww_rs`) was touched — never `poc/`.
Reference implementation: `wakeword_detector.py` + the `wake:` block of
`config.yaml` (Python PoC). Builds on the Phase 1a single-model gate in
`crates/server/src/wake.rs` (`OpenWakeWord` + `WakeGate`).

## Goal

`make server` in Listen mode loads **all** custom heads in
`models/wakeword/` (`hey_babel`, `hey_marvin`, `hey_one_one`), runs them on
every 80 ms chunk, and when one fires it (a) opens the wake session exactly
as today and (b) selects the matching persona voice — babel, marvin or
one-one — for the rest of the session, the way the Python
`WakeWordDetector` calls `apply_skill_persona_switch`. No per-model env
list to maintain: drop a trained `hey_<persona>.onnx` into the directory
and it is live at the next start.

## What exists today

| Piece | Location | State |
|---|---|---|
| Detector | `wake.rs::OpenWakeWord` | one head; wraps `oww_rs::OwwModel` (own melspec+embedding frontend, 12-window smoothing, 2 s refractory) |
| Gate | `wake.rs::WakeGate` | IDLE swallows audio + VAD edges, feeds detector; on fire replays 0.5 s pre-roll, synthesizes `UserStartedSpeaking`, AWAKE for 15 s of silence |
| Wiring | `call.rs` ~L272 | `WAKE_MODEL` non-empty → gate between VAD and SpeechGate; built **before** `CallState` exists |
| Persona → voice | `skills/persona.rs`, `skills/mod.rs::CallState`, `tts_qwen.rs::current_voice` | `set_voice(name)` read per utterance; only the Qwen backend honours it |
| Voice preload | `main.rs::qwen_persona_names` | `QWEN_VOICE` + `QWEN_VOICES` — the only voices that can be selected |
| Python reference | `wakeword_detector.py` | one `openwakeword.Model` with N heads; per-model cooldown; fires → persona switch + `WakeWordDetectedFrame`; ControlChannel sends `{"type":"wake","state":"awake"/"asleep",…}` |
| Heads | `models/wakeword/hey_{babel,marvin,one_one}.onnx` | trained via `scripts/wakeword/` |
| Fixture | `fixtures/t13_wake.wav` (was `poc/harness/fixtures/`) | "Hey babel, what time is it?" (Kokoro) |

The vendored `oww_rs` (`third_party/oww_rs`, local mods listed in
`VENDORED.md`) already exposes what a shared frontend needs:
`AudioFeaturesTract::create_default()` / `get_audio_features(&[f32]) -> [N,96]`
are `pub`, and `OwwModel::detect(features) -> (bool, f32)` is `pub`. The
only private part is `OwwModel.audio`, which each head allocates and we
would leave unused.

## Design

### 1. Configuration — directory, not list

- `WAKE_DIR` (new): directory of head `.onnx` files. Non-empty →
  Listen mode with every `*.onnx` in it. Relative paths resolve against the
  repo root like the other model paths. The documented default in
  `poc/.env.example` is `models/wakeword`.
- `WAKE_MODEL` (existing): kept as a single-file form (the fixture test
  and `poc/README.md` use it). Either variable non-empty enables Listen
  mode; both set → the dir wins and a warning is logged.
- `WAKE_THRESHOLD` (existing, 0.5) applies to every head.
- `WAKE_SESSION_SECS` (new, default 15 — today's hardcoded window;
  Python's `conversation.idle_timeout_secs` is 10).

Persona for a head is derived from the file stem, no mapping table:
strip a leading `hey_`, then match against the loaded Qwen voice names
with the same `-`/`_`-insensitive comparison `SwitchPersona` uses
(`hey_one_one` → `one-one`, `hey_marvin` → `marvin`). A head whose persona
is not a loadable voice is a **startup error** (Python's router also fails
fast on an undeclared persona) — except when the TTS backend is not Qwen,
where every head still gates but none can switch voice; log one warning.

### 2. Voices follow the heads

`qwen_persona_names()` becomes the union of `QWEN_VOICE`,
`QWEN_VOICES` and the personas derived from the wake heads, deduped,
default voice first. With `WAKE_DIR=models/wakeword` that preloads
babel, marvin and one-one without touching `QWEN_VOICES`, and
`switch_persona` is registered whenever more than one voice results (as
now). Startup cost: one extra voice prime each for marvin/one-one
(~seconds, once per process — same as setting `QWEN_VOICES` today).

### 3. `WakeBank` — one frontend, N heads (`wake.rs`)

Replace `OpenWakeWord` with:

```rust
pub struct WakeHead { pub name: String, pub persona: String, model: oww_rs::oww::OwwModel }
pub struct WakeBank { frontend: AudioFeaturesTract, heads: Vec<WakeHead>, pcm: Vec<i16>, threshold: f32 }
pub struct Detection { pub head: usize, pub probability: f32 }
impl WakeBank {
    pub fn load(paths: &[PathBuf], threshold: f32) -> Result<Self, ...>;   // stem → persona here
    pub fn feed(&mut self, samples: &[i16]) -> Option<Detection>;          // best fire in this batch
}
```

Per 1280-sample step: `frontend.get_audio_features(chunk)` once, then
`head.model.detect(features.clone())` for each head; return the highest
probability among heads that fired. Mirrors `OwwModel::detection`'s
warm-up behaviour: while the feature buffer is shorter than 16 frames
(first ~1.3 s) the reshape fails — treat as no detection, as the single
head does today. The vendored crate gets one small addition so the
per-head frontends aren't allocated at all: `OwwModel::head_from_path`
(no `AudioFeaturesTract`), recorded in `VENDORED.md`. If that proves
awkward, fall back to `new_from_path` and simply ignore the unused
frontend — correctness is identical, just ~2 extra tract models per head.

Smoothing and the 2 s refractory stay per head inside `OwwModel`, so
"hey marvin" said twice quickly can't double-fire, and one head firing
does not silence the others.

### 4. `WakeGate` gains persona switching and events

- Constructor takes `Arc<CallState>` and a `CallEvents` handle (the same
  sender `MediaController` publishes on); `call.rs` moves `call_state` /
  `events` creation above the input-processor block.
- On fire in IDLE: `state.set_voice(persona)` **before** the pre-roll
  replay (the reply's TTS must already be in the new voice), then the
  existing `UserStartedSpeaking` + pre-roll + AWAKE transition. Log
  `wake word detected` with `head`, `prob`, `persona`.
- On fire in AWAKE (Python parity — the detector never stops running):
  switch voice, refresh `last_voice`, no synthetic edge (the VAD's real
  edges already pass). Lets "hey marvin" mid-session hand the conversation
  over.
- On session expiry → IDLE: `state.set_backend(LlmBackend::Local)`
  (Python reverts `ask_claude`'s flip on sleep; the Rust server currently
  only reverts at hang-up). Voice is **not** reverted — the next wake word
  sets it anyway, and Python pins the persona.
- Events on the call's WebSocket, mirroring the Python ControlChannel:
  `{"type":"wake","payload":{"state":"awake","model":"hey_marvin","score":0.87,"persona":"marvin"}}`
  and `{"state":"asleep"}` on expiry. `crates/protocol` adds `WAKE_EVENT`
  and a `WakeState` payload enum with a wire-shape test like
  `MediaCommand`'s; `crates/client/src/events.rs::render` prints
  `[awake: marvin 0.87]` / `[asleep]`.

Cross-head cooldown: the gate's existing 2 s `cooldown_until` becomes
global across heads so "hey one one" can't be re-heard as a second fire by
a near-miss head during the same breath.

### 5. Startup validation (`main.rs`)

- Resolve `WAKE_DIR`; error if set and empty of `.onnx` files.
- Derive personas; with `TTS_BACKEND=qwen` each must resolve in the
  engine's `voices/` catalog (the union in §2 makes `qwen_start` do this
  check for free). Log the table `head → persona` once at startup.
- Loading the heads themselves stays per call as today (they are small;
  tract loads in ms), but every path is `fs::read`-checked at startup so a
  bad file fails the process, not the first call.

## Files

| File | Change |
|---|---|
| `crates/server/src/wake.rs` | `WakeBank`, `WakeHead`, stem→persona fn; `WakeGate` with state + events + AWAKE re-fire + backend revert; tests |
| `crates/server/src/call.rs` | reorder `call_state`/`events`; build `WakeBank` from `cfg.wake_paths()`; pass state + events to the gate |
| `crates/server/src/main.rs` | `wake_dir`, `wake_session_secs` in `PocConfig`; `wake_head_paths()`; validation; union in `qwen_persona_names` |
| `crates/protocol/src/lib.rs` | `WAKE_EVENT`, `WakeState` |
| `crates/client/src/events.rs` | render wake events |
| `third_party/oww_rs/crates/oww/src/oww/oww_model.rs`, `VENDORED.md` | `head_from_path` (frontend-less head) |
| `poc/.env.example`, `README.md` | `WAKE_DIR`, `WAKE_SESSION_SECS`, persona-by-filename convention |
| `docs/plans/wakeword-in-server.md` | this file; outcome notes on completion |

## Tests

Unit (`make test`):
- `persona_for_head`: `hey_babel`→`babel`, `hey_one_one`→`one-one` (given
  loaded voices `[babel, marvin, one-one]`), `hey_jeeves`→error listing
  the loaded voices, stem without `hey_` prefix still matches.
- `WakeBank` on `poc/harness/fixtures/t13_wake.wav` with all three heads
  (run when `models/wakeword` exists, skipped otherwise like the current
  fixture test): babel fires ≥0.4; marvin and one_one **never** fire —
  the cross-talk check three heads make necessary.
- Gate state machine with a fake bank (feature-gated test hook or a
  `WakeBank` built from pre-recorded per-chunk probabilities): IDLE fire
  sets voice + pushes `UserStartedSpeaking` then pre-roll then the chunk;
  AWAKE fire switches voice without a synthetic edge; expiry reverts the
  backend and emits `asleep`.
- Protocol wire-shape test for `WakeState`.

Fixtures: generate `hey_marvin` / `hey_one_one` wake WAVs the same way
`poc/harness/make_fixtures.py` made `t13_wake.wav` ("Hey Marvin, what time
is it?", "Hey one one, what time is it?") and commit them under
`poc/harness/fixtures/`; the bank test then asserts each fires only its
own head.

Live (`make server` with `WAKE_DIR=models/wakeword`, `make call`):
1. Startup log shows the three heads and their personas; Qwen primes
   three voices.
2. Silence + unrelated speech → no turn, no events.
3. "Hey babel, what time is it" → `[awake: babel …]`, reply in babel.
4. After the asleep event, "Hey marvin, tell me a joke" → reply in
   marvin's voice; "hey one one …" likewise.
5. Mid-session "hey one one" without waiting for sleep → next reply in
   one-one's voice.
6. "Ask Claude …" then let the session expire → the following wake turn
   goes to Ollama (backend reverted on sleep).

## Out of scope

- Kokoro/Chatterbox voice switching (only Qwen reads `CallState.voice`);
  wake still gates on those backends, persona is logged only.
- The on-device (RPi/ESP32) clients — they do their own wake detection
  and connect in push mode.
- Retraining heads or changing thresholds per head (single global
  threshold; per-head values can come later as `hey_x.threshold` sidecars
  if false accepts differ).

## Client-side wake (native Rust client)

The Python design had two wake paths (server-side for the browser / Jabra
pipeline, on-device for the RPi satellite); the Rust stack keeps both:

| Client | Where the detector runs | How the persona reaches TTS |
|---|---|---|
| Browser / playground | server `WakeGate` (`WAKE_DIR`) | the gate calls `CallState::set_voice` itself |
| Native client (`make call`; wake is on by default, `--no-wake` for push mode) | `crates/client/src/wake.rs::ClientWakeGate` on the capture channel; audio is sent only while a session is open (pre-roll replayed on the opening fire) | the client sends `{"type":"wake","payload":WakeState}` over the events WebSocket; `main.rs::apply_client_wake` sets the voice (and reverts `ask_claude` on `asleep`) |

Both share `crates/wake` (`WakeBank`, `GateCore`, `resolve_heads`,
`persona_for_head`). The client re-arms its session window on the server's
own events (final transcriptions, bot turn boundaries) since it has no VAD.
Running both gates at once is harmless (the client's pre-roll carries the
wake word, so the server gate fires too) but pointless; set `WAKE_DIR`
for browser sessions; the native client wakes on-device by default.

## Outcome notes (2026-08-26)

- `crates/wake` (new): `WakeBank` = one `oww_rs::AudioFeaturesTract`
  frontend + N frontend-less heads (`OwwModel::head_from_path`, added to the
  vendored crate; `audio` became an `Option`). Probabilities are bit-identical
  to the single-model path (checked chunk-by-chunk on `t13_wake.wav`).
- Found while testing: `OwwModel` initialises `last_detection_time` at
  construction and enforces its 2 s refractory from there, so a freshly
  loaded head could not report `detected` for the first 2 s of a call.
  `head_from_path` now starts outside the window, and the bank gates on the
  smoothed probability (0 unless ≥2 of the last 12 windows crossed the
  threshold) with the gate's own 2 s cross-head cooldown — the same shape
  the old `OpenWakeWord::feed` had.
- `GateCore` is pure (time passed in) and unit-tested: open, cooldown,
  hand-over to another head while awake, lazy expiry, re-open.
- Server: `WAKE_DIR` / `WAKE_MODEL` / `WAKE_THRESHOLD` /
  `WAKE_SESSION_SECS`; wake personas are unioned into the Qwen preload
  list, so a head without a matching `voices/` preset fails `start_qwen`.
  `wake` events (`voice_chatbot_protocol::WakeState`) go to the client;
  the native client renders `[awake: marvin 0.87]` / `[asleep]`.
- Fixtures for marvin / one-one were **not** added (they would live under
  `poc/harness/fixtures/`, which is frozen for this work); the bank test
  asserts on the existing babel fixture that only the babel head fires.
- Pre-existing, unrelated: `stt_backend_parser_accepts_only_supported_local_engines`
  fails without the `moonshine` feature on `main` as well.
- Follow-up (same day): persona prompts. `crates/server/prompt.<persona>.txt`
  files are loaded at startup (`main.rs::load_persona_prompts`);
  `CallState::set_voice` selects the matching prompt (None when the persona
  has no file), and `SwitchingLlm::run_llm` swaps it in as the system
  message for that run. Every persona entry point — server gate, client
  wake frame, `switch_persona` — goes through `set_voice`, so all three
  switch prompt and voice together. Switching prompts changes the prefix,
  so the first turn after a persona change pays a full prompt eval on
  Ollama (the keep-warm task only warms `prompt.txt`).
