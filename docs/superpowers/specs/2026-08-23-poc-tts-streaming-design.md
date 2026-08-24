# poc-tts-streaming: Chatterbox Flash streamed over the OpenAI Realtime API (WebRTC)

**Date:** 2026-08-23
**Status:** Draft — written autonomously; decisions marked **[assumption]** need the requester's confirmation before implementation starts.
**Parent:** `docs/superpowers/specs/2026-08-23-poc-tts-flash-design.md` (poc-tts)

## Purpose

Stand up `poc-tts-streaming/`, a copy of `poc-tts/` adjusted so that synthesized
audio reaches the browser **while it is still being generated**, instead of
after the whole utterance is encoded as one WAV. The client-facing contract is
the **OpenAI Realtime API over WebRTC** — the same `POST /v1/realtime/calls`
SDP exchange, `oai-events` data channel, and event vocabulary a browser uses
against `api.openai.com` — so that any Realtime-capable client can drive the
local TTS and the browser test page is one URL swap away from the real thing.

The number this PoC exists to produce is **time-to-first-audio (TTFA)**: how
long after `response.create` the first speech sample plays in the browser, and
how that compares with poc-tts's whole-utterance turnaround (0.59 s / 1.03 s /
3.38 s for the short / medium / long baseline sentences on the RTX 2060,
tuned config). Everything else is in service of measuring and hearing that.

## What streaming can and cannot mean here

`chatterbox_flash.ChatterboxFlashTTS.generate()` is not a streaming API. It
runs the T3 block-diffusion decoder to completion for the whole utterance,
then vocodes every speech token in one `S3Gen.inference()` call
(`tts.py:229-292`). Nothing in the public surface yields partial audio.

So streaming in this PoC has two levels:

1. **Sentence pipelining (in scope, required).** `chunk_text()` already splits
   input on sentence boundaries. Instead of concatenating the per-chunk
   waveforms and encoding once, each chunk's audio is pushed onto the WebRTC
   track the moment its `generate()` returns, while the next chunk is being
   generated. TTFA becomes "time to synthesize the first sentence" — about
   0.6 s for a 30-character sentence on the 2060 with the tuned config —
   regardless of how long the paragraph is. Tuned Flash runs at RTF ≈ 0.2-0.3
   on this card, so once the first chunk is playing, generation stays ahead of
   playback and the stream is gapless.

2. **Intra-sentence streaming (out of scope; optional spike at the end).**
   The T3 loop emits speech tokens block by block (16 or 32 tokens ≈ 0.64 s
   or 1.28 s of audio at 25 Hz), and the base `chatterbox` package's S3Gen
   carries an incremental vocoder path (`flow_inference(finalize=False)` +
   `hift_inference(cache_source=...)`, the CosyVoice-style token-window
   streaming the docstring calls `S3GenStreamer`). Hooking the block loop and
   vocoding windows as they land would cut TTFA below the first-sentence cost,
   but it means patching a copy of `ChatterboxFlashT3.generate` and validating
   that windowed vocoding does not introduce seams. That is a separate
   investigation with its own go/no-go; the plan ends with a bounded spike
   task, gated on the level-1 numbers.

### Why the unit is a sentence, not a word

Every `generate()` call is an independent draw: fresh noise, fresh sampling,
its own `_speech_len_for_text_tokens()` length budget, its own EOS and
trailing silence. Nothing carries acoustic state from one call to the next.
That is why the chunk cannot be a word:

- **Prosody needs the phrase.** Pitch contour, stress, and pause placement
  are decided from the text the model can see. A word synthesized alone gets
  list-reading intonation, and consecutive words disagree with each other on
  pace and pitch — the output is audibly a sequence of separate utterances.
- **The seams are the artifacts.** Each call ends with a decay to silence and
  the `trim_fade` of the reference spillover (`s3gen.py:294`); word-level
  chunks stack those seams every few hundred milliseconds.
- **The model's cost is per block, not per character.** T3 emits speech in
  blocks of 16/32 tokens (0.64/1.28 s of audio); a one-word chunk pays a full
  block and a full prefix forward for a fraction of a second of speech, and
  the sweep already showed Flash's RTF *improving* with length (0.79 → 0.58)
  — short inputs are its worst case. It also over-generates on short text
  (the `blk16/steps4/cfm1` row produced 10 s of audio for a 104-char
  sentence).

This is why pipecat and FlowCat both aggregate LLM tokens into sentences
before TTS (`SimpleTextAggregator` is a port of pipecat's), and why
`chunk_text()` packs whole sentences up to `chunk_size`. Two knobs remain:

- **Clause splitting** (in scope, small): when a single sentence exceeds
  `chunk_size`, split on `, ; :` before giving up and emitting it whole.
  Exposed as `x_chatterbox.split_on_clauses` (default on) so its effect on
  TTFA and on prosody can be judged by ear in the same UI.
- **Level-2 intra-sentence streaming** (the spike) is the real answer to
  "word-level latency": T3 conditions on the *whole* sentence but produces
  its audio block by block, so vocoding blocks as they land gives
  sub-sentence TTFA with full-sentence prosody. That is strictly better than
  word-level chunking and is why the spike is worth a bounded attempt.

## Non-goals

- Speech input / STT. The server ignores an inbound audio track if the client
  adds one, and answers `input_audio_buffer.*` events with an `error`.
- An LLM. `response.create` speaks the text of user message items verbatim.
  There is no model choosing what to say.
- The WebSocket and SIP Realtime transports. **WebRTC only**, per the request.
- Multi-response concurrency on one GPU. One synthesis runs at a time; further
  `response.create`s queue behind it.
- Auth beyond the ephemeral-key shape, TLS, TURN, remote access. Localhost.
- Touching `poc-tts/`. It stays frozen on :8005 so the two can be A/B'd in two
  tabs, the same way poc-tts was judged against Turbo.

## Decisions

1. **Copy, don't import.** [assumption] `poc-tts-streaming/` is a full copy of
   `poc-tts/` (own venv, own `mise.toml`, own `Makefile`, port **8006**) with
   the package renamed `poc_tts_streaming`. The hardware-resolution code
   (`resolve_device/dtype/backend`, the sm_75 traps, OOM reporting) is copied
   verbatim with its tests. Duplicating ~600 lines is accepted: the two PoCs
   are disposable, must run side by side, and a cross-directory import across
   two venvs is exactly the kind of fragility a PoC should not carry.

2. **OpenAI Realtime GA vocabulary**, not the 2024 beta. Event names are the
   GA ones (`response.output_audio_transcript.delta`,
   `conversation.item.added`/`.done`, session object with `type: "realtime"`,
   `output_modalities`, `audio.output.voice`). Verified against the current
   API reference on 2026-08-23.

3. **Audio travels on the media track, never as events.** Matches OpenAI's
   WebRTC behaviour ("you don't have to handle audio events from the model").
   No `response.output_audio.delta` is emitted; the transcript deltas carry
   the text as each sentence starts.

4. **Flash knobs are a namespaced extension.** `session.update` carries the
   standard fields; Chatterbox-specific parameters live under
   `session.x_chatterbox`. A strictly conformant client that never sends it
   gets `config.yaml` defaults. Nothing standard is overloaded.

5. **Voice = reference-clip filename.** `session.audio.output.voice` is
   `one-one.mp3`, `marvin.wav`, etc., validated against the same voice search
   paths as poc-tts. An unknown voice is an `error` event, not a fallback.

6. **The engine boundary is transport-agnostic.** `StreamingEngine.
   synthesize_stream()` is a plain generator of PCM chunks. The Realtime/WebRTC
   layer is one consumer; a chunked-HTTP `POST /v1/audio/speech` PCM endpoint
   is a second, ten-line consumer that exists for tests, for `curl`, and as
   the integration seam for the Rust PoC (see the FlowCat section). This is
   what keeps the Rust-hosted-protocol option open without re-doing engine
   work.

## Architecture

```
poc-tts-streaming/
  mise.toml, setup.sh, requirements.txt      as poc-tts + aiortc
  config.yaml                                 port 8006, + realtime: section
  Makefile                                    run / test / bench / bench-stream / clean
  poc_tts_streaming/
    config.py                                 copied
    engine_flash.py                           copied + synthesize_stream()
    audio.py                                  float32 -> s16 20 ms frames, silence, resample-free
    track.py                                  PcmQueueTrack (aiortc MediaStreamTrack)
    realtime/
      ids.py                                  sess_/conv_/item_/resp_/call_/ek_/event_
      events.py                               pydantic models: client events in, server events out
      session.py                              RealtimeSession state machine (no aiortc, no torch)
      webrtc.py                               aiortc glue: /calls, oai-events, track, teardown
    server.py                                 FastAPI: UI routes, /v1/realtime/*, /v1/audio/speech
    bench_stream.py                           TTFA / total / audio_s per baseline sentence
  ui/                                         copied from poc-tts + realtime-client.js
  tests/
  reports/stream_runs.jsonl
```

### Component boundaries

| unit | does | depends on | never imports |
|---|---|---|---|
| `engine_flash.py` | device/dtype/backend, load once, `synthesize_stream()` yielding `(chunk_text, pcm_float32)` per sentence, cancellation between chunks | `chatterbox_flash`, torch | FastAPI, aiortc |
| `audio.py` | float32 → int16, 480-sample framing, silence frame | numpy | everything else |
| `track.py` | `PcmQueueTrack`: asyncio queue of 20 ms frames → `av.AudioFrame` with monotonic pts; silence on underrun; `clear()`; `drained` future | aiortc, av | engine, realtime |
| `realtime/events.py` | validate inbound JSON, build outbound events | pydantic | aiortc, torch |
| `realtime/session.py` | the protocol: session/conversation/response objects, event sequencing, cancel, errors. Takes a `synthesize(text, voice, knobs) -> AsyncIterator[np.ndarray]` and a `sink` with `push(pcm)`, `clear()`, `drained()` | events | aiortc, torch |
| `realtime/webrtc.py` | `POST /calls` → `RTCPeerConnection`, wait for `oai-events`, wire `RealtimeSession` to `PcmQueueTrack`, teardown | aiortc, session, track | torch |
| `server.py` | routes, static UI, client secrets, `/v1/audio/speech` | all of the above via injection | `chatterbox_flash` |

`session.py` is the piece worth keeping pure: it is the part that would be
ported to Rust if the protocol surface ever moves into FlowCat, and it is the
part the GPU-free tests exercise most.

### Generation threading

Torch calls block. `synthesize_stream()` runs in a single dedicated worker
thread (one per engine — one GPU, one generation at a time); chunks are handed
to the event loop through an `asyncio.Queue`. Cancellation is a flag checked
between chunks; a chunk already inside `generate()` finishes (≤ ~1 s tuned)
and is discarded. `response.create`s from other sessions queue behind the
running one, in arrival order.

## Realtime protocol surface

### HTTP

| route | request | response |
|---|---|---|
| `POST /v1/realtime/client_secrets` | JSON `{ "session": {...}?, "expires_after": {...}? }` | `200` `{ "value": "ek_…", "expires_at": <unix>, "session": <effective session> }`. Tokens are in-memory, default TTL 600 s. |
| `POST /v1/realtime/calls` | `Content-Type: application/sdp`, body = offer SDP; `Authorization: Bearer ek_…` (also accepts `multipart/form-data` with `sdp` and optional `session` JSON fields, as the GA API does) | `201`, `Content-Type: application/sdp`, body = answer SDP, `Location: /v1/realtime/calls/{call_id}` |
| `DELETE /v1/realtime/calls/{call_id}` | — | `200`; closes the peer |
| `POST /v1/audio/speech` | JSON `{ "input", "voice", "response_format": "pcm", "x_chatterbox"?: {...} }` | chunked `audio/pcm` s16le mono 24 kHz, first bytes after the first sentence; `wav` also accepted and returns a whole file (poc-tts behaviour) |

A missing or unknown bearer on `/calls` is `401` with the OpenAI error JSON
shape. [assumption] The bearer check is kept even though it is cosmetic on
localhost, because it is what makes the browser client's code path identical
to the real one.

### Data channel

The client creates `oai-events`. On open the server sends `session.created`
then `conversation.created`. Every message is one JSON event with `type` and
`event_id`; unknown client event types get an `error` with the offending
`event_id` echoed.

**Client → server**

| type | behaviour |
|---|---|
| `session.update` | merge `session` (only `audio.output.voice`, `instructions`, `output_modalities`, `x_chatterbox` are meaningful; others accepted and echoed) → `session.updated`. Unknown voice → `error` (`invalid_request_error`, code `invalid_value`, param `session.audio.output.voice`) and the session is unchanged. |
| `conversation.item.create` | user `message` with `input_text` content → stored; emits `conversation.item.added` then `conversation.item.done`. Other roles/content types → `error`. |
| `conversation.item.delete` | removes; `conversation.item.deleted`. |
| `response.create` | text = `input_text` of user items in `response.input` if given, else the user items added since the last response. Empty → `error` (`invalid_request_error`, "nothing to speak"). Otherwise the response sequence below. A second `response.create` while one is active → `error` (`conversation_already_has_active_response`), as OpenAI does. |
| `response.cancel` | flag the worker; `response.done` arrives with `status: "cancelled"`. Buffered audio keeps playing unless cleared. |
| `output_audio_buffer.clear` | drop queued frames → `output_audio_buffer.cleared`. |
| `input_audio_buffer.*`, `conversation.item.truncate/retrieve` | `error` (`invalid_request_error`, "not supported by this server"). |

**Server → client, per response** (in this order)

```
response.created                       status "in_progress"
response.output_item.added             assistant message item, status "in_progress"
response.content_part.added            part { type: "audio", transcript: "" }
  -- per sentence chunk, as its audio is pushed:
  response.output_audio_transcript.delta   delta = chunk text (+ " ")
  output_audio_buffer.started              once, on the first pushed frame
response.output_audio_transcript.done
response.output_audio.done
response.content_part.done
response.output_item.done
response.done                          status "completed" | "cancelled" | "failed", usage.output_tokens = 0
output_audio_buffer.stopped            when the track has drained the last frame of this response
```

`error` events carry `{ type, code, message, param, event_id }` under `error`.
Session object as reported in `session.created`/`.updated`:

```json
{ "type": "realtime", "id": "sess_…", "object": "realtime.session",
  "model": "chatterbox-flash", "output_modalities": ["audio"],
  "instructions": "",
  "audio": { "input":  { "format": { "type": "audio/pcm", "rate": 24000 }, "turn_detection": null },
             "output": { "format": { "type": "audio/pcm", "rate": 24000 }, "voice": "one-one.mp3", "speed": 1.0 } },
  "x_chatterbox": { "temperature": 0.6, "exaggeration": 0.5, "cfg_scale": 1.0,
                    "num_steps": 10, "n_cfm_timesteps": 2, "chunk_size": 120, "split_text": true,
                    "split_on_clauses": true } }
```

Default `x_chatterbox` values come from `config.yaml:generation`, so the
tuned `blk32 / steps4 / cfm1` config from the sweep can be the default here.

### WebRTC media

- Outbound: one audio track, `PcmQueueTrack`, 24 kHz mono s16, 480 samples
  per frame with monotonically increasing `pts`. aiortc's Opus encoder
  resamples to 48 kHz itself. Exactly 20 ms per frame — the aiortc
  same-timestamp-per-packet bug recorded in `docs/web-rtc.md` bites anything
  larger.
- Underrun policy: silence frames, never a stall — the RTP clock must keep
  ticking or the browser's jitter buffer resets.
- Inbound: a client that offers a mic track gets it accepted and drained.
  The test UI offers `recvonly` audio and no mic.
- ICE: host candidates only on the server (loopback use). The browser side
  keeps a public STUN entry as `webrtc_smoke` does; harmless when it fails.

## Browser test interface

`ui/` is the poc-tts copy with these adjustments:

- **`realtime-client.js`** — a standalone `RealtimeTtsClient` class:
  `connect({baseUrl, session})` (client secret → offer → `/calls` → answer),
  `speak(text)` (item.create + response.create), `cancel()`, `clear()`,
  `on(eventType, fn)`. It is written so that changing `baseUrl` to
  `https://api.openai.com` and supplying a real key is the only edit needed —
  that swap is the manual conformance check.
- **Generate** connects lazily on first use and keeps the peer for the page
  lifetime; it sends the textarea through `speak()`. **Stop** sends
  `response.cancel` + `output_audio_buffer.clear`.
- Playback is an `<audio autoplay>` bound to the remote stream. The WaveSurfer
  player is replaced by a **stream panel**: connection pills (pc / ice / dc,
  as `webrtc_smoke`), a live level meter (WebAudio `AnalyserNode` on the
  remote stream), and metrics: **TTFA** (`response.create` sent → first
  non-silent sample seen by the analyser, client-measured), server TTFA
  (`output_audio_buffer.started` − `response.created`), total
  (`response.done`), audio duration (`stopped` − `started`).
- An **events pane** logging every `oai-events` message both ways, raw JSON.
- The voice select, the sliders (temperature, exaggeration, CFG, num_steps,
  n_cfm_timesteps, chunk size, split text) feed `session.update` on change.
  Output-format select and seed/language controls are removed.
- Optional, last: a `MediaRecorder` on the remote stream so the finished
  utterance can be downloaded/A/B'd offline as before.

## Bench

`make bench-stream` runs the three baseline sentences through
`synthesize_stream()` with the tuned config and appends to
`reports/stream_runs.jsonl`: `ttfa_s` (first chunk ready), `gen_s` (all
chunks), `audio_s`, `n_chunks`, `first_chunk_chars`, resolved dtype/backend,
VRAM peak. Same sentences as every other baseline in the repo, so the row is
comparable to `poc-tts/bench-rtx-2060.md`. Browser-measured TTFA is recorded
by hand from the stream panel into the results doc, since it includes Opus
encode, jitter buffer, and decode.

## Testing

GPU-free, engine mocked, following poc-tts's pattern:

- `RealtimeSession` with a fake synthesizer: the exact server-event sequence
  for a two-sentence response; transcript deltas match chunk text; cancel
  after chunk 1 → `cancelled` and no further deltas; unknown voice → error and
  unchanged session; empty conversation → error; `x_chatterbox` knobs reach
  the synthesizer; second `response.create` while active → error.
- `PcmQueueTrack`: 1001 samples → 480/480/41-padded frames; pts advance by
  480; silence when empty; `clear()` drops pending; `drained()` resolves.
- `synthesize_stream` with a mocked model: yields in order, one `generate`
  call per chunk, cancel stops after the current chunk.
- HTTP: client secret shape and TTL; `/calls` rejects wrong content type and
  missing/unknown bearer with the OpenAI error shape; `Location` header set.
- **Loopback end-to-end**: an in-process aiortc `RTCPeerConnection` posts an
  offer to the TestClient, opens `oai-events`, drives item.create +
  response.create, and asserts the event sequence and that audio frames
  arrive on the remote track — with the fake synthesizer, no GPU.
- `/v1/audio/speech` `pcm`: chunked body, first chunk before the second is
  generated (fake synthesizer with an event gate).
- Copied poc-tts tests (resolution, chunking, voices, config overrides) are
  kept as-is.

GPU generation is covered by `bench_stream.py` and the browser, not pytest.

## Error handling

- CUDA OOM during a chunk → `response.done` with `status: "failed"` and
  `status_details.error` carrying the VRAM report; the session survives.
- Model not loaded → `/calls` returns `503`; the UI shows it.
- Peer disconnect mid-response → cancel flag set, worker result discarded,
  session torn down; nothing logged at warning level per frame (the
  `MediaStreamError` spin from ADR-0002 is the failure to avoid).
- Bad JSON on the channel → `error` (`invalid_request_error`, "invalid JSON").

## Integration with the Rust streaming PoC (`poc/flowcat`)

What is there today, from the source:

- FlowCat's TTS contract is `TtsService::run_tts(&mut self, text) -> Result<Vec<Frame>>`
  (`poc/vendor/flowcat-core/src/service/mod.rs:64`) — **batch per call**.
  Its `SimpleTextAggregator` already hands TTS one sentence at a time, so
  FlowCat's own granularity is sentence-level.
- `poc/flowcat/src/tts_chatterbox.rs` calls the vendored Chatterbox server's
  `/v1/audio/speech` with `response_format: "wav"` and strips the RIFF
  header (the 120-line workaround counted in the T14 finding).
- FlowCat's str0m transport carries **no data channel**; events go over a
  side WebSocket (`GET /webrtc/events/{pc_id}`). str0m 0.21 itself has SCTP
  data channels built in (`ChannelOpen`/`ChannelData` in `str0m/src/lib.rs`);
  `flowcat-transports` simply does not surface them.

Three ways to connect the two, with a recommendation:

**A. Chunked-PCM sidecar (recommended, cheap).** poc-tts-streaming's
`POST /v1/audio/speech` with `response_format: "pcm"` streams 24 kHz s16le as
each sentence lands. `ChatterboxTts` in `poc/flowcat` drops `strip_wav` and
requests `pcm` — a net deletion. Because `run_tts` returns a `Vec`, FlowCat
still collects the body before emitting frames; with sentence-level chunking
on both sides that costs nothing today (the first sentence is one chunk
either way). The moment level-2 intra-sentence streaming exists, the same
endpoint delivers it, and FlowCat then needs either an upstream
`run_tts_stream() -> BoxStream<Frame>` on the trait or a `poc/flowcat`
processor that reads the HTTP stream and emits `TtsAudio` as bytes arrive.
That Rust change is follow-on work, not part of this plan.

**B. FlowCat as a Realtime WebRTC client of poc-tts-streaming.** Rejected:
FlowCat would need a client-side str0m dial plus data channels, and audio
would be Opus-encoded twice (TTS→FlowCat, FlowCat→browser) on one machine.
No latency win, real complexity.

**C. Host the Realtime surface in FlowCat (the "pure Rust" question).** Split
the answer in two:

*Protocol and transport — yes, and it is a good fit.* `flowcat-server`
already exposes `handle_offer` (`webrtc.rs:119`) and str0m already speaks
SCTP, so a `POST /v1/realtime/calls` route that takes `application/sdp` is a
thin wrapper, and the `oai-events` channel is plumbing str0m's
`ChannelOpen`/`ChannelData` through `flowcat-transports` (a few hundred lines
plus a `Frame` variant for channel messages). The Realtime state machine is a
port of `realtime/session.py`, which is why that module is kept free of
aiortc and torch in this design — it is the spec for the Rust port.

*Inference — not in pure Rust today.* Chatterbox Flash is a PyTorch model
with a custom block-diffusion decode loop (masking schedule, PMI top-k
position ranking, CFG combination — `chatterbox_flash/model.py:525+`), a
flow-matching mel decoder, and an iSTFT HiFT-GAN vocoder. The realistic
Rust-runtime paths are, in order of plausibility: (1) ONNX export of the T3
denoise step, S3Gen and HiFT from Python, with the sampling loop as Rust
control code over `ort` (risks: KV-cache dynamic shapes, iSTFT ops,
FlashInfer fast path lost — though the 2060 never had it); (2) a `candle`
or `tch-rs` port of the three networks (weeks; the upstream MLX port proves
it is *possible*, and also how much work it was). Neither belongs inside a
TTS PoC whose question is "does streaming help TTFA".

**D. Embed the Python engine in the FlowCat binary with PyO3.** The other
answer to "in the Rust framework": link CPython into `flowcat-poc`, import
`poc_tts_streaming.engine_flash` from the PoC venv, and call
`synthesize_stream()` from a `spawn_blocking` thread under
`Python::with_gil` / `allow_threads`. A Rust `TtsService` (or a streaming
processor) iterates the Python generator and emits `TtsAudio` frames per
chunk — the same generator the HTTP sidecar wraps, which is the point of
keeping the engine transport-agnostic. What it buys and costs, honestly:

- *Buys:* no HTTP hop and no second process; the per-call overhead is
  microseconds against a ~600 ms synthesis, so the win is operational
  (one binary to start, one log) rather than latency. Streaming maps
  naturally: PyO3 can drive a Python generator or accept a Rust callback,
  so level-1 and level-2 chunks arrive the same way they do over HTTP.
- *Costs:* it is not pure Rust — the binary needs `libpython3.10`, the venv's
  `site-packages` (torch + CUDA) on `PYTHONPATH`, and `PYO3_PYTHON` pointed
  at the mise interpreter at build time; on the Mac the MLX backend rides
  along the same way. The GIL is released inside torch kernels but held
  between them in the Python-level block loop, so synthesis must run on a
  dedicated blocking thread, never on a tokio worker. A CUDA OOM or a torch
  segfault now takes FlowCat down with it (the sidecar isolates that). And
  the dev loop changes: `poc/README.md` already says to keep Chatterbox
  warm across FlowCat rebuilds because a cold load takes tens of seconds —
  embedded, every `cargo build` restart reloads the model.

**Recommendation:** build this PoC in Python as specified and treat the
engine as a localhost **inference sidecar** whose seam is the chunked-PCM
endpoint (A). When FlowCat becomes the host that matters, the choice is
between A (process boundary, warm model across rebuilds, crash isolation)
and D (single process, PyO3): both consume `engine_flash.synthesize_stream()`
unchanged, so nothing in this plan forecloses either. The Realtime surface in
Rust (C, protocol half) is a follow-on that ports `realtime/session.py` and
adds data-channel plumbing to `flowcat-transports`. Pure-Rust inference is a
separate research spike with its own go/no-go and is not scheduled here.

## Risks

- **aiortc + PyAV in the Flash venv.** `av` ships its own ffmpeg with libopus
  in the wheel; torch 2.6 and `av` have coexisted in this repo's root venv
  (pipecat's `webrtc` extra), so no known conflict — verify in `setup.sh` by
  importing both.
- **Gaps between sentences** if a chunk generates slower than its predecessor
  plays. Not expected with the tuned config (RTF ≈ 0.2-0.3) but a real
  possibility with `num_steps=10` defaults on a long sentence. The stream
  panel's level meter makes it audible and visible; `bench_stream.py`
  records per-chunk `gen_s` vs `audio_s` so it is measurable.
- **Realtime API drift.** The GA vocabulary was verified today; OpenAI
  still lists some legacy names (`conversation.item.created`). The events
  module is the single place names live.
- **Browser autoplay policy.** The `<audio>` element only plays after a user
  gesture; Generate is a click, so the first stream is fine, but a page
  reload followed by a programmatic call would be muted. Documented in the UI.
- **Intra-sentence spike may not pan out.** It is last and optional for that
  reason.
