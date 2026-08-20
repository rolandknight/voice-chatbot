# PoC integration contract (Phase 1)

Fixed interfaces between the four processes so the Python (harness/stubs) and
Rust (FlowCat embedder) sides can be built independently. See
`docs/poc/flowcat-poc-plan.md` for the why; this file is the what.

## Processes & ports (all 127.0.0.1)

| Process | Port | Source |
|---|---|---|
| `flowcat-poc` server (Rust embedder) | **6210** | `poc/flowcat/` |
| Stub skill services | **8790** | `poc/stubs/stub_server.py` |
| Kokoro TTS shim (OpenAI-speech protocol) | **8880** | `poc/stubs/kokoro_shim.py` |
| Chatterbox cloned-voice TTS (selected profile) | **8004** | external `vendor/*Chatterbox-TTS-Server/` sidecar |
| NeMo-Speech.cpp / Nemotron STT (optional profile) | **8178** | local NVIDIA sidecar |

Secrets/config: `poc/.env` (`OPENROUTER_API_KEY`, `POC_LLM_MODEL`, the selected
TTS backend settings, `POC_STT_BACKEND=whisper|moonshine|nemotron`, and
backend-specific STT tuning such as `POC_WHISPER_THREADS`,
`POC_MOONSHINE_UPDATE_INTERVAL_MS`, or `POC_NEMOTRON_RIGHT_CONTEXT`).
`POC_VAD_STOP_SECS` defaults to `0.2`, exactly matching the Python chatbot's
`wake.vad_stop_secs`. The cross-platform laptop/Mac validation
profile uses `anthropic/claude-haiku-4.5` through OpenRouter and local
Chatterbox TTS. This is a cloud-LLM framework validation run, not evidence for
the separate all-local model phase.

## Stub server API (`:8790`)

- `POST /tool/{name}` — body: the tool's JSON arguments object. Returns
  `200 {"result": <json>}`. Unknown tool → `404`. This is the single entry
  point the Rust session relays every workflow tool call to.
- `GET /calls` → `{"calls": [{"tool": str, "args": obj, "ts": float}]}` in
  call order. `DELETE /calls` → clears.
- `POST /admin/latency` — `{"tool": str, "seconds": float}`; next calls to
  that tool sleep first (T9). `POST /admin/fail` — `{"tool": str,
  "status": int}`; that tool returns the HTTP error until cleared with
  `{"tool": ..., "status": null}` (T11 support).
- `GET /health` → `{"ok": true}`.

Tool results (canned): time/date return fixed-format current values;
`get_weather` returns a canned forecast ("18 degrees and cloudy");
`set_timer` echoes `{minutes, label, status: "set"}`; BBC/Spotify return
`{status: "playing"|"stopped"|"paused", ...}` echoes.

## Skill schemas — `poc/stubs/skills.json`

Single source of truth, loaded by BOTH the stub server (dispatch + arg
validation) and the Rust embedder (advertised to the LLM as tools). Array of
`{name, description, parameters}` (JSON-Schema), 8 skills:
`get_current_time`, `get_current_date`, `set_timer(minutes:int, label?:str)`,
`get_weather(location?:str)`, `play_bbc_radio(station:str)`,
`stop_bbc_radio`, `play_spotify(query:str, kind?:track|album|artist)`,
`pause_spotify`. Descriptions adapted from `skills/` SKILL.md frontmatter.

## Kokoro shim (`:8880`)

`POST /v1/audio/speech` — body `{"model": "kokoro", "voice": "af_heart",
"input": str, "response_format": "pcm"}` → `200`, body = raw little-endian
s16 mono PCM at **24000 Hz** (no WAV header). This matches FlowCat's
`tts-kokoro` client contract. `GET /health` → `{"ok": true}`.
Implementation: `kokoro-onnx` pip package. The same package generates the
harness WAV fixtures (deterministic voice `af_heart`).

## FlowCat server surface (`:6210`, consumed by `FlowCatAdapter`)

- `POST /webrtc/offer` — browser-style SDP offer in, SDP answer out (JSON;
  exact field names per upstream `flowcat-server/src/webrtc.rs` — adapter
  confirms `pc_id` key at implementation time).
- `GET /webrtc/events/{pc_id}` — WebSocket; JSON events (transcript lines,
  state). The adapter maps these to normalized harness events; unknown
  types are ignored (forward-compatible, mirrors `docs/web-rtc.md`).
- Audio: Opus over WebRTC, carrier per SDP (adapter resamples 16 kHz
  fixtures to the negotiated rate via aiortc's track machinery).

## Harness event normalization

`Event = {kind: "transcript-user" | "transcript-bot" | "state" | "error",
text?: str, raw: obj, ts: float}`. Adapters produce this; tests consume only
this plus captured PCM and the stub call log.

## Fixtures (`poc/harness/fixtures/`)

16 kHz mono s16 WAV, ~300 ms leading + ~1.2 s trailing silence, spoken via
Kokoro `af_heart`. Regenerate with `python -m harness.make_fixtures`.

| File | Utterance | Used by |
|---|---|---|
| `t1_time.wav` | "What time is it?" | T1, T10 |
| `t2_timer.wav` | "Set a timer for five minutes." | T2 |
| `t3_music.wav` | "Put some music on." | T3 |
| `t3_news.wav` | "I'd like to listen to the news." | T3 |
| `t4_bbc.wav` | "Play BBC Radio 4." | T4 |
| `t4_stop.wav` | "Stop the radio." | T4 |
| `t4_spotify.wav` | "Play Purple Rain by Prince." | T4 |
| `t8_recall.wav` | "What did I just ask you about?" | T8 |

## Run orchestration

`poc/run_poc.sh up|down|test` — brings up stubs + the selected local TTS
dependency + flowcat-poc
(reads `poc/.env`), waits on the three `/health` endpoints (flowcat:
`GET /healthz`), runs `pytest poc/harness -m smoke`. Chatterbox is launched
separately with `make poc-chatterbox`; `up` verifies it with a real synthesis
and `down` deliberately leaves it warm.
Every process logs to `poc/logs/<name>.log`.

## Known FlowCat facts the harness must respect (from source reading)

1. **The original cascaded path was half-duplex.** FlowCat PR #61 added the
   duplex builder and batch-STT `flush()` seam; this integration is pinned to
   its merge commit `4ff03f3`. T5 is now a required barge-in regression test.
2. **Greeting on connect** (`CascadedKickoffProcessor`): the bot speaks
   first on `ClientConnected`. The harness must consume/await the greeting
   before sending the first fixture. The PoC emits the prompt-mandated
   `Ready.` locally (without an OpenRouter request) and, for Chatterbox, replays
   the validated health-check WAV rather than synthesizing it again.
3. Turn boundary = Silero falling edge after the configured silence interval.
   Whisper then performs one whole-utterance decode. Moonshine and Nemotron can
   publish display-only interim hypotheses while the person speaks, but their
   internal boundaries never reach the LLM; the same external edge flushes
   exactly one authoritative final transcript to Haiku/tools.
