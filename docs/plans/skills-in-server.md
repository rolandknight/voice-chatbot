# Plan: move the skills into the Rust server

Status: proposed 2026-08-26. Replaces the PoC stub relay
(`poc/stubs/stub_server.py`, port 8790) with in-process skills in
`crates/server`, ported from the legacy Python `skills/` package.

## Where things are today

| Piece | Location | Notes |
|---|---|---|
| Tool schemas | `poc/stubs/skills.json` | 8 tools; adapted from `skills/*/SKILL.md` |
| Dispatch | `crates/server/src/session.rs` (`StubSession::tool_call`) | `POST {stubs}/tool/{name}`; result JSON stringified back to the LLM |
| Real logic | `skills/core/*`, `skills/radio/*`, `skills/spotify/*` | pipecat handlers |
| Radio backend | `scripts/radio.py` | BBC HLS URLs, alias matching, mpv subprocess + JSON IPC, Jabra device pick, pause/resume ducking |
| Spotify backend | `scripts/spotify.py` | PKCE token in `~/.config/babel/spotify_token.json`, device id cache, Web API targeting librespot "Babel" |
| Weather | `skills/core/get_weather/handler.py` | Open-Meteo geocode + forecast, CoreLocationCLI → ipwho.is fallback, WMO phrasing |
| Timer | `skills/core/set_timer/handler.py` | sleeps, then pushes `TTSSpeakFrame` |

Only the 8 tools in `skills.json` are in scope. Legacy extras (web_search,
play_bbc_show, resume/skip/whats_playing/playlist, sfx, persona switch,
ask_claude) are deliberately deferred — add them the same way once the
runtime below exists.

## Target layout

```
crates/server/src/
  skills/
    mod.rs          # Skill trait, Registry (schemas + dispatch), SkillContext
    time.rs         # get_current_time, get_current_date (spoken-word rendering)
    timer.rs        # set_timer: tokio::time::sleep -> Frame::TtsSpeak into the call
    weather.rs      # get_weather: open-meteo + location resolver
    radio.rs        # play_bbc_radio, stop_bbc_radio: station table + MediaController
    spotify.rs      # play_spotify, pause_spotify: Web API client
  media/
    mod.rs          # MediaController trait: play_stream / stop / pause / resume / is_playing
    client.rs       # sends {"type":"media", ...} over the call's events WS  (default)
    local.rs        # mpv subprocess on this host (dev loop, MEDIA_TARGET=server)
  session.rs        # SkillSession replaces StubSession; node_tools/tool_call -> Registry
crates/server/skills.json   # schemas move out of poc/stubs (unchanged content)
crates/protocol/   # (new, small) shared serde types for the media command + rtf events
crates/client/src/media.rs  # mpv runner driven by "media" events
```

The schemas stay data (`skills.json`) rather than Rust literals so the LLM
prompt prefix is byte-identical to today's — the Ollama prefix-cache
discipline (ADR-0003) depends on the rendered tool list not changing.

## Skill runtime

```rust
#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    async fn call(&self, args: &Value, ctx: &CallCtx) -> String;  // spoken-friendly string, never Err
}
pub struct CallCtx { pub run_id: i64, pub speak: mpsc::UnboundedSender<Frame>, pub media: Arc<dyn MediaController> }
```

- `Registry::new(skills_json, Vec<Box<dyn Skill>>)` cross-checks every schema
  has an implementation and vice versa (startup error otherwise).
- `tool_call` results keep the stub's contract: a plain string the LLM
  summarises; errors fold into "The X service is temporarily unavailable."
- Per-call context: `SessionSource::tool_call` only receives `run_id`, so the
  session keeps a `DashMap<run_id, CallCtx>` that `call.rs` registers after
  `build_cascaded_call_duplex` returns (`task.task.queue_sender()`) and removes
  when the call ends. That is what lets the timer speak into the right call.
- Timer: `tokio::spawn(sleep(d).then(send Frame::TtsSpeak{..}))`. If the call
  ended before it fires, log and drop (same as the Python behaviour when the
  pipeline is gone). Timers are per-call, not persisted.
- Weather: `reqwest` to Open-Meteo; location precedence stays
  `config.default_location` → `CoreLocationCLI` (macOS, 5 s cap) → ipwho.is →
  "ask again with a city". Cache the resolved current location for the process.
- Spotify: port the Web API subset used (`devices`, `search`, `start_playback`
  with `device_id`, `pause_playback`, artist top-tracks). Reuse the existing
  PKCE token file (`~/.config/babel/spotify_token.json`) so `python
  scripts/spotify.py --bootstrap` remains the one-time auth step; implement
  refresh-token exchange only. Device-id cache file reused too.
- Radio: station table + alias matching ported verbatim (longest-alias-first
  ordering is load-bearing), unit-tested against the Python alias cases.

## Media playback target

Spotify already plays on the client (librespot). Radio should too:

- Server: `MediaController::play_stream(url, display)` → events WS message
  `{"type":"media","payload":{"action":"play","url":...,"title":...}}`, plus
  `stop`/`pause`/`resume`. Server keeps `is_playing` state for cross-stop.
- Client: `crates/client/src/media.rs` spawns `mpv --no-video
  --input-ipc-server=…` (same flags as `scripts/radio.py`) and applies the
  IPC pause/resume; picks the same output device as the call's playback
  device. Requires `mpv` on the Pi (add to `devices/rpi5/install_rpi.sh`).
- Ducking: legacy paused radio while the bot spoke. Client already sees
  `rtf-bot-started/stopped-speaking`; it pauses/resumes mpv on those. No
  server change needed.
- Dev loop on the Mac (server and Jabra on one box, browser client): keep a
  `local.rs` controller that runs mpv server-side (`MEDIA_TARGET=server`).

## Steps

1. **Runtime + easy skills** — `skills/mod.rs`, `time.rs`, `weather.rs`;
   `SkillSession` replaces `StubSession`; `skills.json` moves to
   `crates/server/`; `POC_STUBS_URL`/`POC_SKILLS` env removed; `make stubs`
   target dropped. Test: `make call`, ask the time/date/weather.
2. **Timer** — per-call `CallCtx` registry in `call.rs`; `timer.rs`; verify a
   30-second timer speaks mid-call and a timer for an ended call is dropped.
3. **Media protocol + client mpv** — `crates/protocol`, server
   `media/client.rs`, client `media.rs` with ducking; `radio.rs`. Test on the
   Mac browser + Pi.
4. **Spotify** — `spotify.rs` against the existing token cache; play/pause;
   cross-stop with radio both ways.
5. **Cleanup** — `poc/stubs` untouched (PoC harness still uses it), but the
   server no longer references it; README/Makefile help updated.

Each step is independently shippable; 1 alone fixes "I can't check the time".

## Risks / open points

- Spotify PKCE refresh in Rust must read spotipy's JSON cache format
  (`access_token`, `refresh_token`, `expires_at`, `scope`); if the format
  drifts, re-bootstrap with the Python script — acceptable for now.
- BBC HLS pool URLs rot; keep them in one table with the refresh-source
  comment carried over.
- `Frame::TtsSpeak` interrupt semantics: confirm the speech gate doesn't
  drop an injected utterance while the user is mid-turn (test in step 2).
- Weather calls out to the internet from the tool path; keep the 6 s cap so
  a dead network yields a spoken failure, not a hung turn.
