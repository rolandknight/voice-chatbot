# Pipecat Jabra Mac Prototype

Local voice-to-voice prototype for Apple Silicon Mac Studio + Jabra USB speakerphone.

## Install

```bash
chmod +x install_mac.sh run.sh
./install_mac.sh
```

## Configuration

Non-secret config lives in `config.yaml` at the repo root — edit values
there. Secrets (API keys) live in `.env` (gitignored); see `.env.example`
for the keys that are read. Config is validated at startup via Pydantic, so
a malformed `config.yaml` fails loudly before any audio device is opened.

Dump the resolved tree:

```bash
python -m config.loader --print-effective
```

## Pick the Jabra device

Plug in the Jabra, set it as the macOS default input/output, then:

```bash
./run.sh --devices
```

If needed, set the indexes in `config.yaml` under `audio:`:

```yaml
audio:
  input_device_index: 3
  output_device_index: 3
```

## Run

```bash
./run.sh
```

## Stack

- Pipecat local audio transport
- Whisper MLX local STT
- Ollama local LLM (Claude optional via API key)
- Local TTS, per-persona: **Kokoro** for the default `babel` voice, **Chatterbox-Turbo** (via a local OpenAI-compatible server) for cloned voices
- Jabra USB speakerphone via PyAudio

## Personas & cloned voices

Voices and routing live in `personas.yaml`. The default install ships
with one persona, `babel`, using Kokoro `af_heart` — identical to the
original single-voice setup. To add cloned voices on a Mac Studio:

1. **One-time**: `./scripts/setup_chatterbox.sh` (clones the
   Chatterbox-TTS-Server, builds its venv, downloads the model on
   first launch).
2. Drop a 5–15s mono WAV per voice into `voices/` (see
   `voices/README.md`).
3. Add a persona entry in `personas.yaml`:

   ```yaml
   personas:
     marvin:
       backend: chatterbox
       voice: marvin
       ref_audio: voices/marvin.wav
   ```
4. Run `./scripts/start_chatterbox.sh` (leave it running in another
   terminal). The server speaks the OpenAI `/v1/audio/speech` protocol
   on `http://127.0.0.1:8004` and Chatterbox-Turbo uses MPS / Metal
   acceleration on Apple Silicon.
5. Run `./run.sh` and either set `tts.default_persona: marvin` in
   `config.yaml` for boot or say *"switch to marvin"* mid-session.

Routing rules in `personas.yaml` are fully declarative:

- `voice_command` — spoken phrases like "switch to {persona}".
- `llm_tag` — inline `[persona:name]` tags the LLM can emit to
  one-shot-switch voice for a single utterance.
- `skill_intent` — exposes the `switch_persona` tool to the LLM when
  more than one persona is declared, so indirect phrasings ("use the
  butler voice") also work.

The `babel` persona stays on Kokoro regardless of what Chatterbox is
doing; if the Chatterbox server is down, `babel` is unaffected.

## Babel skills

Babel (the local Ollama backend, wake "hey babel") has tool/function calling
wired up via Pipecat's standard schema. Each skill is one folder under
`skills/<category>/<name>/` containing a `SKILL.md` (Claude-style frontmatter
with description, parameters, and trigger keywords) and a `handler.py`. To add
a new skill, drop in a new folder — the loader picks it up at startup. The
`SkillFilterProcessor` swaps the LLM's tool list per turn so the model only
sees ~15 relevant tools no matter how many are registered. Skills shipped today:

- `get_current_time`, `get_current_date` — local clock.
- `set_timer(minutes, label?)` — counts down and speaks the alert out loud.
- `get_weather(location)` — Open-Meteo, no API key. Honors `skills.weather.default_location` in `config.yaml`.
- `web_search(query)` — DuckDuckGo by default. Switch with `skills.web_search.provider: brave` or `tavily` plus the matching API key in `.env`.
- `play_bbc_radio(station)` / `stop_bbc_radio` — live BBC streams via `mpv`, targeted at the Jabra CoreAudio device. See "BBC radio" below. Disable with `skills.radio.enabled: false`.
- `play_bbc_show(show, date?, query?)` — on-demand BBC Sounds programmes and podcast episodes. See "BBC shows" below. Disable with `skills.shows.enabled: false`.
- `play_spotify(query, kind?)` / `play_spotify_playlist(name)` / `pause_spotify` / `resume_spotify` / `skip_spotify(direction?)` / `whats_playing` / `stop_spotify` — Spotify Premium playback via a local `librespot` Connect endpoint. See "Spotify" below. Disable with `skills.spotify.enabled: false`.

Default model is `gemma4:26b` (MoE, ~4B active per token, ~17 GB resident).
It scores 85.5% on the τ²-bench agentic tool-use benchmark (vs Gemma 3 27B's
6.6%) and fires tools without emitting chain-of-thought tokens first, so
warm TTFB stays under ~0.4s on M-series. On lower-RAM machines (<24 GB
free), fall back to `llm.ollama_model: gemma4:latest` (E4B, ~9.6 GB) —
same warm TTFB on direct queries but can spike to several seconds on
indirect phrasings because E4B uses thinking mode. Disable the whole skill
feature with `skills.enabled: false`.

`config.yaml` sets `llm.ollama_keep_alive: "-1"` so the model stays
resident across idle stretches; `app.py` pre-warms it at startup so the
first wake-phrase turn doesn't pay the cold-load cost (~9s for 26B,
~6s for E4B).

## BBC radio

`scripts/radio.py` spawns `mpv` against BBC's public HLS endpoints and points
its CoreAudio output at the Jabra (auto-detected via `mpv --audio-device=help`).
The babel skill exposes `play_bbc_radio` and `stop_bbc_radio` to the LLM, so
ordinary turns trigger playback:

- *"Hey babel, play BBC Radio 1"*
- *"Hey babel, put on 6 Music"*
- *"Hey babel, tune to Radio 4 Extra"*
- *"Hey babel, switch to the World Service"*
- *"Hey babel, stop"*

Supported stations: Radio 1, 1Xtra, Radio 2, Radio 3, Radio 4, Radio 4 Extra,
Radio 5 Live, Radio 5 Sports Extra, 6 Music, Asian Network, World Service.

While radio is playing, any wake-phrase utterance pauses the stream via mpv's
JSON-IPC socket; it resumes automatically once babel finishes its reply (with
an 8-second safety timer for stray noises that never trigger a turn). `mpv` is
GPL and installed by `install_mac.sh`.

## BBC shows

`scripts/bbc_shows.py` resolves on-demand BBC Sounds programmes and feeds the
result through the same `mpv` engine as live radio, so pause/resume ducking
works the same. The babel skill is `play_bbc_show(show, date?, query?)`:

- *"Hey babel, play the Archers omnibus"* — latest episode of a curated show.
- *"Hey babel, play yesterday's Today programme"* — the LLM resolves the
  relative date into an ISO date that's matched against the RSS feed.
- *"Hey babel, play the In Our Time about Spinoza"* — keyword-matches episode
  titles and descriptions.
- *"Hey babel, play Kermode and Mayo's Take"* — for shows not in the curated
  list, falls back to a BBC Sounds search plus `yt-dlp` to resolve the play
  URL.

Curated shows (RSS-backed, fastest path): The Archers Omnibus, The Archers,
In Our Time, Desert Island Discs, Front Row, Thinking Allowed, Just A Minute,
Friday Night Comedy. Extend the list in `scripts/bbc_shows.py:CURATED_SHOWS` —
verify a PID's RSS feed is live with
`curl -I https://podcasts.files.bbci.co.uk/<pid>.rss` before adding.

Anything not in the curated list falls through to a BBC Sounds search plus
`yt-dlp`. This path is best-effort: the BBC Sounds endpoints yt-dlp depends
on shift periodically, and shows without a published podcast feed (some
talk strands, most music programmes) may fail to resolve. `yt-dlp` is
installed by `install_mac.sh`. Disable the whole feature with
`skills.shows.enabled: false`.

## Spotify

`scripts/spotify.py` runs a headless `librespot` Spotify Connect endpoint
named **Babel** and pipes its raw PCM into `mpv`, targeting the same Jabra
CoreAudio device as BBC radio. Playback is controlled via the Web API
through `spotipy`, always against that device.

Voice commands:

- *"Hey babel, play Purple Rain"* — searches and plays a track.
- *"Hey babel, play the album OK Computer"* — `kind=album`.
- *"Hey babel, play Radiohead"* — top tracks from an artist (`kind=artist`).
- *"Hey babel, play my Discover Weekly"* — fuzzy-matches your own
  playlists first, falls back to public playlist search.
- *"Hey babel, pause"* / *"resume"* / *"skip"* / *"go back"*.
- *"Hey babel, what's playing?"* — names the current track.
- *"Hey babel, stop Spotify"* — pauses via the API and tears down the
  local sink.

Ducking works the same way as radio: while babel is listening or
replying, the local `mpv` is paused via its JSON-IPC socket (zero-latency,
no API call), then resumed with the same 8-second safety timer. The
Connect session on Spotify's side keeps running, so the song clock stays
accurate.

If you start radio while Spotify is playing (or vice versa) the other
backend is automatically stopped — two `mpv` processes pushing to the
same CoreAudio device would garble the output.

### Requirements

- **Spotify Premium** — the Web API playback endpoints (`start_playback`,
  `pause_playback`, etc.) are Premium-only.
- `librespot` (installed by `install_mac.sh`).
- `spotipy` (installed by `install_mac.sh`).

### Setup

1. **Create a Spotify app** at
   [https://developer.spotify.com/dashboard](https://developer.spotify.com/dashboard).
   In the app's settings, add a Redirect URI matching
   `skills.spotify.redirect_uri` in `config.yaml` exactly. The default is
   `http://127.0.0.1:8765/callback`. Loopback URIs require PKCE — which
   is what `spotipy` uses here, so no client secret is strictly required.

2. **Put your credentials in `.env`** (the variable names match the
   `spotipy` library convention):

   ```bash
   SPOTIPY_CLIENT_ID=<your client id>
   SPOTIPY_CLIENT_SECRET=   # optional with PKCE; leave blank
   ```

   Leave `skills.spotify.enabled: true` in `config.yaml` (the default).

3. **One-time OAuth** — opens Safari for consent, caches the token at
   `~/.config/babel/spotify_token.json` (refresh tokens don't expire
   unless you revoke them):

   ```bash
   python scripts/spotify.py --bootstrap
   ```

4. **Run librespot on the machine with the speaker.** Playback is native —
   the bot only sends Web API control commands, it never streams audio
   through the voice pipeline. Run a librespot "Babel" endpoint wherever you
   want the music to come out:

   - **Raspberry Pi client:** see [`devices/rpi5/README.md`](devices/rpi5/README.md)
     for the systemd librespot service (ALSA → Jabra).
   - **Mac (server `--local-audio`):** `brew install librespot`, then run
     `librespot --name Babel --bitrate 320` (leave it running; it plays out
     the default CoreAudio output).

   Then, on your phone or any Spotify client, open the Now Playing bar →
   Connect to a device → pick **Babel** once. The device id gets cached to
   `~/.config/babel/spotify_device.txt` so subsequent runs find it without
   re-binding, as long as the device name stays stable. Confirm it's visible
   with `python scripts/spotify.py --list-devices`.

5. Run `./run.sh`. Spotify tools register only when both
   `skills.spotify.enabled: true` and `SPOTIPY_CLIENT_ID` are set, so the
   LLM won't see them otherwise.

### Troubleshooting

- *"Spotify can't see the Babel device yet."* — librespot isn't running on
  the client, or no Spotify client has selected it. Start librespot on the
  speaker machine, then open Spotify on your phone, pick Connect, choose
  Babel. `python scripts/spotify.py --list-devices` shows what's visible.
  The bot retries device discovery on every play command.
- *"Spotify isn't authorised yet."* — token cache is missing or revoked.
  Re-run `python scripts/spotify.py --bootstrap`.
- *Playback is choppy or staticky* — this was the symptom of the old
  pipe-into-WebRTC path and shouldn't happen with native librespot playback.
  Make sure librespot is running on the client itself, not piped through the
  bot.

## Latency optimization knobs

Local LLM choices (set in `config.yaml` under `llm.ollama_model`).
Measured warm TTFB on M4 Max with the babel system prompt + tool schemas,
after the startup pre-warm:

```yaml
llm:
  # Default. Gemma 4 26B MoE, ~4B active per token. ~17 GB resident.
  # Warm TTFB ~0.37s. Best tool-call reliability; no chain-of-thought tax.
  ollama_model: gemma4:26b

  # Gemma 4 E4B edge. ~9.6 GB. Warm TTFB ~0.35s on direct queries BUT can
  # spike to 4+s on indirect phrasings because it emits reasoning tokens
  # before firing the tool. Pick if RAM is tight.
  # ollama_model: gemma4:latest

  # Smallest fallback for low-RAM machines.
  # ollama_model: qwen2.5:3b
```

Whisper trade-offs (`stt.whisper_model`):

```yaml
stt:
  # Fastest:
  whisper_model: mlx-community/whisper-tiny.en-mlx
  # Better accuracy:
  # whisper_model: mlx-community/whisper-small-mlx
```

After first model download, set `huggingface.hub_offline: true` to skip
the HF metadata check.

## Common fixes

### PyAudio install fails

```bash
brew install portaudio
. bin/activate-hermit
python -m pip install --no-cache-dir pyaudio
```

### No sound / wrong mic

Run:

```bash
./run.sh --devices
```

Then set `audio.input_device_index` and `audio.output_device_index` in `config.yaml`.

### Feedback / echo

Keep `audio_in_passthrough=False` in `app.py`.
Lower the Jabra speaker volume.
Move the speakerphone away from walls/reflective surfaces.
