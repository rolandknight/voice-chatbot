# poc-tts-streaming — Chatterbox Flash over the OpenAI Realtime API (WebRTC)

Copy of `poc-tts/` that streams audio sentence-by-sentence over WebRTC on
:8006, speaking the OpenAI Realtime API (`POST /v1/realtime/calls`,
`oai-events` data channel). poc-tts keeps :8005 so both run side by side.

    make              # install anything missing, then serve on :8006
    make test         # GPU-free unit + loopback WebRTC tests
    make bench-stream # TTFA per baseline sentence -> reports/stream_runs.jsonl
    make clean

Design: `docs/superpowers/specs/2026-08-23-poc-tts-streaming-design.md`
Measured results: `results-rtx-2060.md`

## Defaults and knobs

`config.yaml`'s `generation:` block:

- `chunk_size: 300` — sentence-packing target for `chunk_text`. 120 optimised
  TTFA for short replies but fragments literary text into clause chunks; 300
  keeps prosody while block streaming keeps TTFA low.
- `temperature: 0.5` — lowered from the paper default of 0.6 to reduce
  per-chunk variance / over-generation.
- `split_on_clauses: true` — lets `chunk_text` split long sentences on clause
  punctuation (`, ; :`) when a sentence alone would exceed `chunk_size`,
  rather than only ever splitting on sentence boundaries. Exposed as a
  toggle in the UI (`ui/index.html`'s "Split on clauses" checkbox); the
  toggle and every generation knob initialise from `config.yaml`'s
  `generation:` block (`generation_defaults`, wired through
  `_ui_shaped_config` in `poc_tts_streaming/server.py`) rather than
  hardcoded slider defaults.

`config.yaml`'s `engine:` block:

- `block_streaming: true` — vocodes each finished T3 block rather than each
  finished sentence, for a TTFA that no longer scales with sentence length.
  Effective only when the resolved engine is CUDA + torch backend
  (`engine_flash.block_streaming_effective()`); every other resolved
  device/backend (cpu, mlx, flashinfer) falls back to sentence streaming
  with a logged notice, so this stays on across machines without
  per-machine overrides. See `results-rtx-2060.md` for the measurements
  behind the default.

## Voices

`voices.paths: [../voices]` — the curated list lives in the repo-tracked
`../voices/` directory: `babel.mp3`, `marvin.mp3`, `one-one.mp3`. The vendor
clone's `reference_audio/` carries its own bundled voices, which are
deliberately left out of this list (`tests/test_voices.py` pins it).

## Realtime API surface

Four routes, plus the WebRTC data channel:

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/realtime/client_secrets` | mint an ephemeral key (`ek_...`), optionally with a `session` patch |
| `POST` | `/v1/realtime/calls` | SDP offer in, SDP answer out (`application/sdp` or `multipart/form-data`); requires a bearer ephemeral key |
| `DELETE` | `/v1/realtime/calls/{call_id}` | hang up a call |
| `POST` | `/v1/audio/speech` | non-realtime chunked-PCM/WAV synthesis (`response_format: pcm \| wav`) |

Once the call connects, events flow over a WebRTC data channel named
`oai-events` (`poc_tts_streaming/realtime/webrtc.py:EVENTS_CHANNEL`); audio
flows on the associated media track.

## Manual OpenAI swap check

`ui/realtime-client.js`'s `RealtimeTtsClient` takes the same shape the real
OpenAI Realtime API expects, so pointing it at `api.openai.com` from the
DevTools console is a quick sanity check that nothing here has drifted from
the spec it's implementing:

```js
new RealtimeTtsClient({baseUrl: "https://api.openai.com", apiKey, model: "gpt-realtime"})
```

Everything should flow the same way it does against `:8006`, with one
expected exception: this server's Chatterbox-specific knobs travel under an
`x_chatterbox` key (`SpeechRequest.x_chatterbox`, `ChatterboxKnobs.merged`),
which `api.openai.com` doesn't know about and will reject.

Two known fidelity gaps to keep in mind while poking at this: the
client-secret's `session` patch is not carried into the call -- this server
applies session config via `session.update` instead, which the UI sends
before every speak (`ui/script.js`'s `updateSession(sessionPatchFromControls())`);
and `RealtimeTtsClient.disconnect()` assumes the `Location` header returned
by `/v1/realtime/calls` is a relative path, so it would break against a
server that returns an absolute URL there.

## aiortc

`setup.sh` pins `aiortc>=1.9,<2` (`requirements.txt`) and resolves to
**aiortc 1.15.0** (pulling **av 17.1.0**) in this venv. `setup.sh` prints
the resolved versions on every run so a wheel mismatch with torch surfaces
immediately rather than as a confusing failure on the first `/calls`.
