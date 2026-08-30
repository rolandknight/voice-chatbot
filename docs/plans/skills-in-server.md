# Plan: move all skills into the Rust server

Status: implemented 2026-08-26 on branch `skill-cleanup` (v2 — full scope,
rewrite-not-embed). Outcome notes at the end.
Replaces the PoC stub relay (`poc/stubs/stub_server.py`, port 8790) with
in-process skills in `crates/server`. No separate skills server of any kind.

## Ground rules

1. **Rewrite in Rust, don't embed.** The whole skill surface is ~1.4k lines
   of handlers plus ~1.3k lines of backends (`scripts/radio.py`,
   `scripts/spotify.py`, `scripts/bbc_shows.py`) — small enough to port
   outright. No PyO3 (unlike `crates/qwen-tts`), no `python` subprocesses,
   no importing the legacy package. At runtime the server needs no Python
   for skills.
2. **The legacy Python is frozen.** `skills/`, `scripts/`, `app.py`,
   `poc/stubs/` are not edited; they remain the reference implementation and
   the PoC harness keeps using the stub. The Rust port copies behaviour
   (station tables, alias matching, spoken phrasing, error strings) and
   pins it with unit tests.
3. **One process.** Skills run inside `voice-chatbot-server`. Anything that
   plays audio does so on the *client* via a message on the existing events
   WebSocket (Spotify already works this way through librespot). External
   *model* servers the skills merely call over HTTP (Woosh/SAO for sound
   effects, Open-Meteo, Spotify Web API) are not "skill servers" and stay.
4. **Static tool list.** The Python `SkillFilterProcessor` swaps the tool
   list per turn; on Ollama that defeats the prompt prefix cache (measured
   2026-08-24, see `docs/adr/0003`). The Rust server advertises one fixed,
   name-sorted list for the whole process. Gating is at startup only
   (config/credentials), never per turn.

## Where things are today

| Piece | Location | Notes |
|---|---|---|
| Tool schemas | `poc/stubs/skills.json` | 8 tools; adapted from `skills/*/SKILL.md` |
| Dispatch | `crates/server/src/session.rs` (`StubSession::tool_call`) | `POST {stubs}/tool/{name}`; result JSON stringified back to the LLM |
| Real logic | `skills/<category>/<name>/handler.py` | 18 tools, pipecat handlers |
| Radio backend | `scripts/radio.py` | BBC HLS URLs, longest-alias-first matching, mpv subprocess + JSON IPC, Jabra device pick, pause/resume ducking |
| Shows backend | `scripts/bbc_shows.py` | curated RSS feeds → episode pick by date/query; BBC Sounds search + `yt-dlp` fallback |
| Spotify backend | `scripts/spotify.py` | PKCE token in `~/.config/babel/spotify_token.json`, device-id cache, Web API targeting librespot "Babel" |
| Weather | `skills/core/get_weather/handler.py` | Open-Meteo geocode + forecast, CoreLocationCLI → ipwho.is fallback, WMO phrasing |
| Timer | `skills/core/set_timer/handler.py` | sleeps, then pushes `TTSSpeakFrame` |
| SFX | `skills/sfx/generate_sound_effect/handler.py` | keyword-routes to Woosh or Stable Audio HTTP, waits for bot silence, plays FLAC via mpv |

## Scope: all 18 tools

| Tier | Tools | What the port needs |
|---|---|---|
| A — pure functions / HTTP | `get_current_time`, `get_current_date`, `get_weather`, `web_search` | `reqwest` + `chrono`; nothing new in the pipeline |
| B — needs the live call | `set_timer` | inject `Frame::TtsSpeak` into the right call later |
| C — needs client media | `play_bbc_radio`, `stop_bbc_radio`, `play_bbc_show`, `generate_sound_effect` | a `media` message over the events WS + an mpv runner in the client |
| D — Spotify Web API | `play_spotify`, `play_spotify_playlist`, `pause_spotify`, `resume_spotify`, `skip_spotify`, `stop_spotify`, `whats_playing` | PKCE refresh + a 7-endpoint Web API client |
| E — session state | `switch_persona`, `ask_claude` | per-call mutable state that the TTS / LLM stage reads |

Tier E is the only part that touches pipeline behaviour beyond tools.
`switch_persona` maps onto Qwen TTS's per-utterance voice choice; the
Rust server currently fixes one voice per process, so the skill sets a
per-call `voice` that `tts_qwen.rs` reads on each `run_tts`. `ask_claude`
flips a per-call `backend` flag that `PocLlm` consults on each `run_llm`
(both providers are constructed up front; the flag chooses which streams).
Both revert when the call ends. They are ported last and are the first to
cut if they turn out to need more than a flag (see Risks).

Not ported: the `_filter.py` trigger scoring (rule 4), the `SKILL.md`
loader (schemas become checked-in JSON), `_tracker.py` (replaced by the
client-side `rtf-bot-*-speaking` events).

## Target layout

```
crates/server/
  skills.json                 # all 18 schemas; the 8 existing entries byte-identical
  src/
    session.rs                # SkillSession replaces StubSession; node_tools/tool_call -> Registry
    skills/
      mod.rs                  # Skill trait, Registry, CallCtx, CallRegistry
      time.rs                 # get_current_time, get_current_date (spoken-word rendering)
      timer.rs                # set_timer
      weather.rs              # get_weather (+ location resolver)
      web_search.rs           # web_search: duckduckgo | brave | tavily
      radio.rs                # station table, alias matching, play/stop
      shows.rs                # curated RSS + BBC Sounds search + optional yt-dlp
      spotify.rs              # 7 tools over SpotifyClient
      spotify_client.rs       # PKCE refresh, token/device caches, Web API subset
      sfx.rs                  # generate_sound_effect: route, generate, hand to media
      persona.rs              # switch_persona
      claude.rs               # ask_claude
    media.rs                  # MediaController: publishes {"type":"media",...} on the call's events
crates/protocol/              # new, small: serde types for the media command (shared by server+client)
crates/client/src/media.rs    # mpv runner: play url / play file / stop / pause / resume, ducking
```

## Skill runtime

```rust
#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    /// Spoken-friendly string, never Err: failures fold into the text.
    async fn call(&self, args: &Value, ctx: &CallCtx) -> String;
}

pub struct CallCtx {
    pub run_id: i64,
    pub frames: mpsc::UnboundedSender<Frame>,   // task.queue_sender(): timer speech
    pub media: Arc<MediaController>,            // this call's events publisher
    pub state: Arc<CallState>,                  // Mutex<{voice, backend}> for tier E
}
```

- `Registry::new(skills_json, Vec<Box<dyn Skill>>)` fails startup unless the
  schema set and the implementation set match exactly (after gating).
- Gating at startup, mirroring `enabled_when`/`requires`:
  `SKILLS_RADIO/SHOWS/SPOTIFY/SFX=off`, Spotify also needs
  `SPOTIPY_CLIENT_ID` + a token file, SFX needs at least one backend URL,
  `switch_persona` needs >1 configured voice, `ask_claude` needs
  `ANTHROPIC_API_KEY`/OpenRouter creds. A gated-off tool is absent from
  `skills.json`'s advertised subset and from the registry.
- `SessionSource::tool_call` only receives `run_id`, so `SkillSession` keeps
  `CallRegistry: DashMap<i64, CallCtx>`. `call.rs` inserts after
  `build_cascaded_call_duplex` returns (`task.task.queue_sender()`, the
  `CallEvents` handle it already has) and removes when the call ends.
- Results keep the stub contract: plain text the LLM summarises. The
  existing 8 keep their exact strings (`"Playing Radio 4."`, `"Timer set for
  5 minutes for tea."`, …) — the prompt tells the model to summarise, and
  the phrasing was tuned against it.
- Every call logs `tool invoke -> name(args)` / `tool return <-` at info,
  the one chokepoint `_loader._bind_ctx` provided.

## Per-skill notes

- **time/date**: port `_spoken_time` word rendering and the `%A, %B %-d, %Y`
  date; local timezone via `chrono::Local`.
- **timer**: `tokio::spawn(sleep → frames.send(Frame::TtsSpeak{..}))`. If
  the call is gone the send fails; log and drop (Python did the same).
  Timers are per-call, not persisted. Duration phrasing ported verbatim.
- **weather**: Open-Meteo geocode → forecast; location precedence
  `WEATHER_DEFAULT_LOCATION` → `CoreLocationCLI` (macOS only, 5 s cap,
  `tokio::process`) → `ipwho.is` (3 s) → "ask again with a city". Cache the
  resolved current location for the process. WMO table and imperial-if-US
  rule ported.
- **web_search**: three providers behind one `match`; snippet extraction
  rules copied. Keys from `.env` (`BRAVE_API_KEY`, `TAVILY_API_KEY`),
  provider from `WEB_SEARCH_PROVIDER` (default duckduckgo).
- **radio**: station table + `build_alias_table`/`match_alias`
  (longest-alias-first is load-bearing) ported and unit-tested against the
  Python alias cases. HLS URL builders `_ww`/`_uk` kept with the refresh
  source comment. `play` cross-stops Spotify; `stop_bbc_radio` also stops a
  show.
- **shows**: curated `CURATED_SHOWS` table, RSS parse (`quick-xml`),
  `_pick_item` by date/query, `_pretty_pub_date`. Fallback: BBC Sounds
  `rms.api` search → `yt-dlp` run from Rust as a subprocess
  (`tokio::process::Command`, `yt-dlp -j --no-warnings -f bestaudio/best
  --playlist-items 1 <url>`; if the JSON has no `url` but a `webpage_url`,
  re-run on that — the brand-page→episode step in `_ytdlp_resolve_sync`).
  15 s timeout; a missing binary disables the fallback with a spoken
  "couldn't find … on BBC Sounds". `yt-dlp` is a system binary like `mpv`,
  not embedded Python.
- **spotify**: `SpotifyClient` with PKCE `refresh_token` exchange reading
  spotipy's cache format (`access_token`, `refresh_token`, `expires_at`,
  `scope`), writing it back the same way. Endpoints: `me/player/devices`,
  `search`, `me/player/play` (`device_id`, `uris`/`context_uri`),
  `pause`, `next`, `previous`, `me/player`, `artists/{id}/top-tracks`,
  `me/playlists` + the Jaccard `_match_playlist`. Device-id cache file and
  the 5 s "Babel not visible" poll reused; 404 → "Spotify lost the Babel
  device…" string kept. Add `voice-chatbot-server spotify-login`
  (PKCE with a one-shot listener on the configured redirect URI, ~100
  lines) so first-time auth also needs no Python; the existing
  `scripts/spotify.py --bootstrap` token stays valid.
- **sfx**: route by the same keyword regex; POST to Woosh/SAO with the
  identical bodies; write the FLAC under the server's artifact dir and send
  `media play_file` to the client, which starts it on the next
  `rtf-bot-stopped-speaking` (20 s cap, then play anyway). Result text
  "Playing a {desc}." is returned *before* generation, as today.
  The generators stay what they are — separate Python model servers
  (`scripts/start_woosh.sh` on :8005, `scripts/start_stable_audio.sh` on
  :8006) — started/stopped by new top-level make targets `sfx-up` /
  `sfx-down` / `sfx-status` (pid files under `vendor/`, `/docs` readiness
  probe, same shape as `run.sh`'s blocks). The tool is always advertised
  when `SFX_ENABLED` is on; on each call it probes the routed backend
  (`GET /docs`, 1 s) and, if it is down, returns "The sound effect server
  isn't running — start it with make sfx-up." instead of "Playing …", so
  the LLM tells the user rather than promising a sound that never comes.
- **persona / claude**: flags in `CallState`, read by `tts_qwen.rs` /
  `PocLlm`; reset on call end. Spoken strings ported.

## Media on the client

Server side, `MediaController` is a thin wrapper over `CallEvents::publish`:

```json
{"type":"media","payload":{"action":"play","url":"…","title":"Radio 4"}}
{"type":"media","payload":{"action":"play_file","path":"…","after_speech":true}}
{"type":"media","payload":{"action":"stop"|"pause"|"resume"}}
```

It also tracks `is_playing` so cross-stop logic (radio ⇄ Spotify ⇄ show)
works without asking the client. Client side, `media.rs` spawns
`mpv --no-video --no-terminal --input-ipc-server=…` with the same flags as
`scripts/radio.py`, targeting the call's output device (`--audio-device`
matched the way `_mpv_jabra_device` did), and ducks on
`rtf-bot-started/stopped-speaking` (pause/resume over IPC). Unknown event
types are already ignored by the client, so the message is additive.
`play_file` for SFX needs the file reachable by the client: for a remote
Pi the server serves it from `/artifacts/{name}` and sends a URL instead.
`mpv` is added to `devices/rpi5/install_rpi.sh`.

mpv runs on the client only; the server never spawns it. On the Mac the
dev loop is `make call` (native client), which gets media for free. The
browser playground gets no media playback.

## Steps

1. **Runtime + tier A** — `skills/mod.rs`, `time.rs`, `weather.rs`,
   `web_search.rs`; `SkillSession` replaces `StubSession`; `skills.json`
   moves to `crates/server/` (8 entries byte-identical, new entries
   appended); `STUBS_URL`/`SKILLS` removed; `make stubs` dropped
   from the top-level Makefile (PoC `Makefile` keeps it). Verify the warm
   prefix still hits (`prompt_eval_duration` gate) with the bigger list.
2. **Timer (tier B)** — `CallRegistry` in `call.rs`; `timer.rs`. Verify a
   30 s timer speaks mid-call and one for an ended call is dropped.
3. **Media + radio/shows (tier C)** — `crates/protocol`, server
   `media.rs`, **client `crates/client/src/media.rs`** (mpv runner driven
   by `media` events, ducking on `rtf-bot-*-speaking`, output device
   matched to the call's), `radio.rs`, `shows.rs` incl. the yt-dlp
   subprocess. The client change ships in the same step as the server
   side. Test on the Mac (native client) and the Pi; `mpv` and `yt-dlp`
   added to `devices/rpi5/install_rpi.sh`.
4. **Spotify (tier D)** — `spotify_client.rs` + 7 tools + `spotify-login`.
   Cross-stop both ways.
5. **SFX** — `sfx.rs` + `play_file`/artifact URL; `make sfx-up/sfx-down/sfx-status`.
6. **Tier E** — `CallState` flags, `switch_persona`, `ask_claude`.
7. **Cleanup** — README skills section, Makefile help, `.env.example`
   keys, `prompt.txt` tool sentence extended to shows/search/sfx.

Each step ships on its own; step 1 alone fixes "I can't check the time".

## Risks / open points

- Tool-list size: 18 schemas roughly doubles the prefix. Prefix caching
  makes that a one-off cost per model load, but measure step 1 before
  step 3 adds more; if TTFT regresses, gate rarely-used tools off by
  default rather than reintroducing per-turn filtering.
- Spotify token format drift → re-run either bootstrap; acceptable.
- BBC HLS pool URLs and podcast PIDs rot; one table each, source comments
  carried over.
- `Frame::TtsSpeak` injected while the user is mid-turn: confirm the speech
  gate does not drop it (test in step 2).
- Tier E: if `PocLlm`/`tts_qwen.rs` can't switch per utterance with a flag
  alone, ship steps 1–5 and revisit with a design note rather than
  widening the pipeline change.
- `yt-dlp`'s BBC extractor tracks BBC site changes; a stale `yt-dlp` fails
  the fallback only, never the curated RSS path.

## Outcome (2026-08-26)

All seven steps landed; 107 unit tests plus ignored live tests
(`cargo test -p voice-chatbot-server -- --ignored network`,
`cargo test -p voice-chatbot-client -- --ignored live`) that hit Open-Meteo,
DuckDuckGo, BBC RSS, Spotify (token refresh + devices), the Messages API, and
play three seconds of Radio 4 through mpv. Deviations from the plan above:

- The Pi installer (`devices/rpi5/install_rpi.sh`) belongs to the legacy
  Python wake client and was left alone; `mpv` is documented as a client
  requirement in the README instead.
- `yt-dlp`'s BBC extractor is broken upstream as of 2026.03.17 ("Unable to
  extract playlist data"), so the BBC Sounds search fallback currently
  fails for every show — the same failure the Python path has. Curated RSS
  shows work. Re-test after a `yt-dlp` upgrade.
- `dotenvy` was replaced by `env_file.rs`: the shared root `.env` contains a
  line dotenvy rejects (`WAKE_PHRASES=hey babel,hey babe,hey baby`), which made
  it drop the whole file — and with it every secret — silently. The server
  now reads `poc/.env` then `.env`, line by line, never overriding set vars.
- `ask_claude` talks to `/v1/messages` directly (`llm_claude.rs`) rather
  than through an OpenAI-compatible shim; default model `claude-opus-5`
  (`CLAUDE_MODEL`). The legacy Python used `claude-sonnet-4-6`.
- Not verified live: an end-to-end spoken turn (Ollama wasn't running and
  gemma4:26b needs ~17 GB), the timer firing mid-conversation through the
  speech gate, Spotify playback (no "Babel" librespot device was online),
  and sound-effect generation (model servers not started). Everything up to
  the pipeline boundary is covered by tests.

