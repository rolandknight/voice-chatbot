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
