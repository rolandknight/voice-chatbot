# ADR-0006: Realtime-voice runtime for the next build — FlowCat (Rust), embedded as a library, full-duplex

| | |
|---|---|
| **Status** | Accepted — for the new Rust server build. Supersedes the framework choice in ADR-0002 for that build; the existing Pipecat server keeps running until the Rust server reaches feature parity and passes the Phase 2 latency gates on the reference host (listed under "Test strategy — open gates"). |
| **Date** | 2026-08-24 |
| **Decision** | Build the next server on **FlowCat** (native-Rust, Pipecat-compatible Frame/FrameProcessor runtime, Apache-2.0), **pinned to the upstream merge commit `4ff03f3` (PR #61, 2026-08-17)**, used as a *library* from a small embedder binary rather than through its config-file factory. Pipeline: WebRTC (str0m) → Silero VAD → optional wake-word gate → speech gate → whole-utterance/streaming STT flushed at the VAD edge → single-state brain with skills relayed as workflow tools → LLM (OpenAI-compatible) → TTS → carrier sink, **full-duplex** with an out-of-band interrupt reactor. Local audio is a separate native CPAL/WebRTC client, not an in-process transport. The LLM, TTS and STT engines are those of ADR-0003/0004/0005. |
| **Related** | ADR-0002 (kept Pipecat; named FlowCat the watch item with re-evaluation trigger "FlowCat reaches maturity with proven local paths" — this ADR records that trigger firing); ADR-0003 (LLM), ADR-0004 (TTS), ADR-0005 (STT); PRD conversation/latency/transport requirements. |

---

## Context

ADR-0002 (2026-08-06) kept Pipecat and quantified why a rewrite was not
justified *then*: ~45 % of the Python server (≈ 1,400 of 3,043 lines) was
framework adaptation — sandwich filters around each LLM, a silent-turn
detector for dropped tool-call JSON, an in-flight/idle clock unifier, a
per-connection TTS factory plus a Kokoro monkeypatch, a two-phase shutdown
for a spinning transport read-loop, a 139-line Chatterbox subclass, a
326-line audio wake-word detector — but those fixes were sunk, stable and
addressed defect classes every framework in the space fights. The ADR
named FlowCat the watch item: a ~7-week-old Rust runtime that mirrors
Pipecat's architecture, promises no GIL, bounded per-processor
backpressure, a priority system-frame channel and flat p99 under load, but
whose cascaded local pipeline was "wire-ready but unproven" and whose
stock builder was half-duplex.

The PoC that followed was designed to be that first real use, with a
black-box harness that would give the same verdicts for any implementation.
This ADR records what it found and the decision that follows.

### What the runtime must provide (unchanged from ADR-0002)

1. Realtime voice loop with **barge-in**: VAD-segmented turns, streaming
   STT → LLM → TTS, interruption mid-reply.
2. **Fully local inference first-class**: the LLM (ADR-0003), TTS (ADR-0004)
   and STT (ADR-0005) are on-host services behind localhost sockets; cloud
   is opt-in.
3. **Two client surfaces, one pipeline**: a USB speakerphone on the host and
   per-connection WebRTC (browser, Raspberry Pi 5, future ESP32-S3), with
   session isolation.
4. **Provider-neutral tool calling** for a file-based skill registry, and a
   future MCP client path.
5. **Custom turn machinery**: audio wake words bound to personas, a session
   idle model, mid-stream text rewriting, tool-set control.
6. Household scale: a handful of concurrent sessions, not thousands.

## The PoC (2026-08-06 → 2026-08-20)

### Harness

Black-box only: the system under test is driven exclusively over the
network as a real client would — SDP offer over HTTP, audio as Opus over
WebRTC, events over a side WebSocket. Fixtures are 16 kHz mono WAV
utterances synthesised once with a deterministic voice, padded with
≈ 300 ms leading and ≈ 1.2 s trailing silence so the real VAD segments
turns. Assertions, strongest first: (1) tool calls and arguments in the
**stub skill services' call log** (eight tool schemas adapted from the
production skill registry: `get_current_time`, `get_current_date`,
`set_timer(minutes, label?)`, `get_weather(location?)`,
`play_bbc_radio(station)`, `stop_bbc_radio`, `play_spotify(query, kind?)`,
`pause_spotify`, always all in context and byte-stable); (2) captured bot
audio: non-silence within budget, duration sanity, and content by
re-transcription; (3) timestamps at speech end and first bot audio. The
stubs can inject per-tool latency or HTTP failures. Test IDs map to PRD
requirements and to ADR-0002's defect classes.

### Stack under test

| Component | Phase 1 (Linux dev box, RTX 2060 6 GB) |
| --- | --- |
| Runtime | FlowCat core/services/transports/server crates, pinned by commit; embedder binary ≈ 4 small files |
| Transport | str0m WebRTC, loopback ICE; events on `GET /webrtc/events/{pc_id}` |
| LLM | OpenRouter `google/gemma-4-26b-a4b-it` (paid; the `:free` tier rate-limits on the second request), later Claude Haiku 4.5 for the cross-platform profile — cloud on purpose, to isolate framework behaviour from local inference |
| STT | whisper.cpp `base.en` (CPU; `tiny.en` failed content accuracy through Opus: "Radio 4" → "Radio for"); later Moonshine Medium Streaming (CPU) and Nemotron 0.6B via NeMo-Speech.cpp (CUDA) as runtime-selectable backends |
| TTS | Kokoro shim (OpenAI-speech protocol, raw 24 kHz PCM); Chatterbox-TTS-Server (CUDA) for the cloned-voice test |
| VAD / turn | Silero ONNX via `ort`; explicit parameters (see Settings) |
| Wake | openWakeWord ported to Rust (`oww_rs`, tract-based), production `hey_babel` model |

### Findings before the first run

1. **The stock cascaded builder was half-duplex by assembly, not by engine
   limit** (`TurnMute` muted STT until the reply finished). The VAD,
   interruption broadcast, queue drain and sink clear primitives were all
   public; what was missing was interruption handling in the LLM/TTS
   adapters and context repair.
2. **No VAD in the stock cascaded chain**: turn boundaries came from
   whisper.cpp's fixed ≈ 4 s batch windows, which fired turns on partial
   utterances and hallucinated on silence.
3. **The config-file factory could not build a local stack** (Kokoro base
   URL dropped; keyless local providers rejected by a key check). Services
   are constructed directly — a few lines each.
4. **The declarative brain force-advertised an `endCall` tool.** A 50-line
   custom `AgentBrain` (six-method trait) removed it.
5. **Embedder ergonomics are good**: the server crate works as a library —
   offer plumbing, event registry, browser event mapping and playground page
   were reusable.
6. The OpenRouter client had a fixed request body (no provider routing
   preferences, no `max_tokens` knob).
7. whisper.cpp shipped CPU-only; GPU needs feature unification and a toolchain.
8. The events WebSocket drops user-speaking markers and metrics, so latency
   segmentation relies on harness-side audio timestamps.

### Defects found live, and fixes

| Defect | Fix | Where |
| --- | --- | --- |
| str0m ICE with the media socket bound to `127.0.0.1`: same-host clients (aiortc, Chrome) send checks from non-loopback interfaces → `UDP send error: Invalid argument`, connection never establishes | bind `0.0.0.0`, advertise `127.0.0.1` | embedder; reported upstream |
| `Frame::Interruption` was **never delivered** to `process_frame` (runtime drains queues and forwards); the sinks' interruption arms were dead code — frame-level barge-in had never run on the cascaded path | new `on_interruption()` processor hook; sink flushes (`send_clear`), assistant aggregator repairs context (partial reply retained, open span dropped) | vendored core patch → upstream PR #61 |
| Runtime cannot pre-empt a busy `process_frame`, so an in-flight LLM stream kept speaking after barge-in | cooperative cancel: the LLM adapter polls a VAD-bumped barge-in generation counter between streamed chunks | core patch |
| Frame-path interruption stalls behind any mid-`await` hop: sink delivery measured **14 ms to 2.1 s** depending on TTS activity (FlowCat's analogue of Pipecat's frame-race class, serialised) | **out-of-band interrupt reactor**: a `Notify`-woken task flushes the carrier **≈ 110 µs** after VAD detection, with a stale-audio latch so a late-finishing TTS cannot resurrect the interrupted reply | core patch |
| Whisper fixed windows produced 6–7 bogus user turns per 30 s call once `TurnMute` was removed | **`SpeechGate`**: forwards audio only between VAD edges, 300 ms pre-roll, flush marker at the falling edge; STT services implement `flush()` and produce exactly one final per VAD turn | core patch + embedder STT |
| VAD `min_volume` default 0.6 (Pipecat parity) gated out moderate speech — the same class as the production speakerphone issue | explicit VAD parameters (`min_volume 0.2`) | embedder |
| A pending inbound receive on the shared media-transport facade could block bot-first audio and playback clears | single-owner command actor for the transport facade | vendored core (retained) |

**Upstream outcome.** Issue AreevAI/flowcat#60 (filed 2026-08-06) and PR #61
(the duplex patch) were **merged on 2026-08-17** as `4ff03f3` after a
maintainer follow-up that fixed interruption-hook coverage, STT
endpointing (`SttService::flush()` replaced the PoC's synthetic marker) and
a stale-audio-latch race, with upstream CI green. The vendored core now
carries only two local modifications: an input-processor seam between the
VAD and the speech gate (for the wake gate) and the transport command actor.
Maintainer responsiveness — eleven days from first-contact issue to merged
architectural PR — is itself a data point ADR-0002 asked for.

### Results

Phase 1, Linux dev box, cloud LLM, CPU whisper `base.en`, Kokoro shim (all
timings on one monotonic clock):

| ID | Test | Verdict | Headline |
| --- | --- | --- | --- |
| T1–T4 | basic turn; direct tool; indirect phrasing ("put some music on" → `play_spotify`, "I'd like to hear the news" → `play_bbc_radio`); BBC/Spotify round-trips | **pass 6/6** | tool selection correct in every call once STT accuracy was fixed |
| T5 | **barge-in** during a long reply | **pass** (3/3 consecutive; later re-verified on the merged upstream code with Whisper and with Nemotron) | bot audio stops ≈ 290–320 ms after new speech onset (dominated by VAD onset detection; the flush itself is µs-scale); follow-up turn coherent |
| T6 | two concurrent sessions, interleaved different tools | **pass** | zero cross-talk in audio or tool log |
| T7 | abrupt peer kill mid-reply (no bye, no DTLS close) | **pass** | reconnect **0.16 s**; no log spin (a class the Python server failed publicly) |
| T8 | idle context wipe after 20 s | **xfail as designed** | the cascaded builder hard-disables idle timeout; the product's idle semantics are a migration item |
| T9 | 12 s tool latency vs idle timer | **pass** | reply delivered after 14.4 s; not killed |
| T10 | 20 warm turns | recorded, no gate | e2e p50 5.55 s / p95 7.04 s — cloud LLM TTFT + CPU batch STT + 0.5 s VAD stop; the framework-overhead baseline for Phase 2 |
| T11 | tool HTTP 500 | **pass** | spoken graceful degradation, not a dead turn |
| T12 | 30-turn soak | **pass** | 0 failures; p95 drift +2.5 %; RSS 615 → 719 MB (+104 MB, likely rolling context — bounded once idle wipe exists) |
| T13 | **server-side wake over WebRTC** (Listen mode) | **pass** | wake-less speech swallowed before and after the session window; "hey babel, what time is it" fires the turn |
| T14 | **cloned-voice TTS** (Chatterbox, CUDA) | **pass** | reply in the cloned voice (median F0 82 Hz vs 200 Hz preset); ≈ 120-line service vs the Python server's 139-line subclass — workaround parity |

The load-bearing positive: **the "wire-ready but unproven" cascaded
tool-calling path works** end-to-end (greeting, transcription, tool relayed
to the stub, result fed back, reply synthesised), and after the duplex
patch the interruption layer is *better* than the incumbent's on the
metric that matters (detection-to-flush).

Workaround accounting for the migration-cost question: embedder ≈ 3,600
lines of Rust across 11 files, of which the two STT services (≈ 800 lines
each for Moonshine and Nemotron, mostly protocol/threading), the Chatterbox
TTS service (375), the wake gate (220) and the whole-utterance Whisper
service (300) are *feature* code that the Python server also carries; the
framework-adaptation residue is the brain (51), the LLM greeting policy
(176), the call assembly (232) and the vendored-core seam.

### Why now, against ADR-0002's reasoning

| ADR-0002 argument for staying | What changed |
| --- | --- |
| Local-first is first-class only in Pipecat | The local engines are now **sidecar services behind localhost sockets** (Ollama/llama-server OpenAI API, Qwen3-TTS streaming server, NeMo-Speech.cpp WebSocket) chosen on their own merits (ADR-0003/4/5). The runtime needs HTTP/WebSocket clients, not in-tree Python adapters; Rust is at no disadvantage. |
| The expensive defects are behind us | The Rust duplex work re-encountered the interruption class once — and fixed it upstream with a better mechanism (µs reactor vs frame-path); concurrency (T6), teardown (T7) and in-flight/idle (T9) passed without workarounds. |
| Barge-in is one flag in Pipecat | True, and the flag's implementation is what produced the frame-race bugs. FlowCat's version is now upstream too. |
| FlowCat is pre-1.0 with one proven path | Still pre-1.0. Mitigated by: pinning to an exact commit, a vendored copy of the one crate we patch, an upstream that merged our architectural change in 11 days, and a harness that re-verifies any bump in minutes. |
| Python is where the inference ecosystem lives | Inference stays in Python/C++ processes we do not embed in the runtime (except ADR-0004's TTS server, which embeds Python *inside Rust*). |
| Latency is dominated by inference, not framework | Still true; the Rust choice is not made for framework latency. It is made for the operational properties (single binary, no GIL, per-processor backpressure, clean teardown) and because the same runtime gives a **single-binary satellite client** (wake + WebRTC on a Pi) from shared code. |

## Decision

1. **Runtime.** FlowCat, pinned to commit `4ff03f3` (never a branch or
   tag), used as a library: `flowcat-core` (feature `vad-ort`),
   `flowcat-services` (`llm-openrouter`/OpenAI-compatible client,
   `tts-kokoro` as the OpenAI-speech reference client), `flowcat-transports`
   (`webrtc-str0m`), `flowcat-server` (`webrtc-helper`). Services are
   constructed directly in the embedder; the config-file factory is not
   used.
2. **Vendored core with a cargo `[patch]`** carrying exactly two
   modifications (wake-gate seam; transport command actor), each an upstream
   PR candidate. Any further core change goes through the same path:
   vendored patch first, PR second, drop the patch when merged.
3. **Full-duplex pipeline shape:**
   `TransportInput → VadProcessor (Silero) → [WakeGate] → SpeechGate → STT (flush at VAD edge) → BrainProcessor (skills as workflow tools) → LLM → assistant aggregator → TTS → carrier sink`, with the out-of-band interrupt reactor armed by the sink's bot-speaking notifier and the VAD's barge-in gate.
4. **Turn ownership.** The runtime's VAD owns turn boundaries. Every STT
   service implements `flush()` returning exactly one final per VAD turn;
   interim hypotheses (Moonshine, Nemotron) are display-only. Model-side
   endpointing is disabled everywhere.
5. **Skills as workflow tools.** A single-state brain exposes no tools of its
   own; the session source advertises the skill schemas (from the production
   registry, sorted, byte-stable per session — ADR-0003's cache rule) and
   relays every call as `POST /tool/{name}` to a skills service. Tool
   errors become synthetic tool results the LLM verbalises (T11 behaviour
   is the contract).
6. **Engines, all local sidecars:** LLM per ADR-0003 (OpenAI-compatible
   endpoint, streaming, tools; a `max_tokens` knob must be added to the LLM
   client — it has none today); TTS per ADR-0004 (a FlowCat `TtsService`
   over the Qwen3-TTS WebSocket streaming raw 24 kHz int16 PCM — the
   OpenAI-speech client is kept only for Kokoro/legacy); STT per ADR-0005
   (Nemotron over the NeMo-Speech.cpp realtime socket; Whisper batch as the
   last fallback).
7. **Clients.** WebRTC is the only audio transport of the server. The host's
   speakerphone is driven by the **native Rust client** (CPAL capture and
   playback, 20 ms Opus over WebRTC, device selection by index/id/name,
   hardware echo cancellation by selecting the same speakerphone for in and
   out; no software AEC). The browser playground adds microphone selection
   and a level meter. There is no in-process local-audio transport.
8. **Wake.** Listen mode = the wake gate between VAD and speech gate:
   swallow audio until a wake model fires (threshold 0.5), open a session
   window (pre-roll replay so the command tail is not clipped), return to
   idle after silence. The detector core is framework-free for reuse in the
   satellite client. Push mode (client-side wake) bypasses the gate.
9. **Greeting.** The pipeline asks the LLM for an opening turn on connect;
   the embedder answers it locally with a fixed spoken "Ready." (replaying a
   cached WAV for cloned voices) so reconnects never wait on the LLM.

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| Server | `POST /webrtc/offer`, `GET /webrtc/events/{pc_id}` (WebSocket), `GET /healthz`, `GET /` playground; port 6210, loopback ICE (bind `0.0.0.0`, advertise `127.0.0.1`) | the str0m loopback defect; LAN exposure is a later, deliberate change |
| Skills service | `POST /tool/{name}` → `{"result": …}`; `GET/DELETE /calls`; admin latency/fail injection | the harness's primary signal |
| VAD | Silero ONNX; `min_volume 0.2`; stop **0.2 s** (production parity; Silero's 32 ms windows make the observed edge ≈ 192 ms); 300 ms pre-roll into the speech gate | 0.6 gated out moderate speech; longer stop delays every final |
| Barge-in | out-of-band reactor, stale-audio latch, cooperative LLM cancel | measured 110 µs detection-to-flush vs 14 ms–2.1 s on the frame path |
| Wake | openWakeWord models (`hey_babel`, `hey_marvin`, `hey_one_one`), threshold 0.5, session window 15 s | parity with the Python detector |
| STT | Nemotron right-context 6 (560 ms), endpointing off; fallback Moonshine; fallback Whisper `base.en` (not `tiny.en`) | ADR-0005; `tiny.en` fails content accuracy through Opus |
| LLM request | streaming, tools, all skills sorted, `keep_alive -1`, `num_ctx 8192`, `think false`, temp 0.2 / 0.0 after a tool result, `max_tokens 512` | ADR-0003 |
| TTS | Qwen3-TTS 1.7B-Base streaming, 0.32 s chunks, PCM 24 kHz | ADR-0004 |
| Logging | `ort` INFO silenced (measured ≈ 140 lines/s of allocator chatter during sessions) | masks real log spin |
| Build | `cargo --locked`; macOS `--features metal` for the Whisper fallback; Linux `--features cuda` only with the CUDA toolkit; Opus from a project-local prefix if the system lacks it | reproducible per host |

## Test strategy

**Standing suite (implementation-agnostic, black box).** T1–T14 as above,
runnable per marker (`smoke`, `tools`, `duplex`, `wake`, `voice`,
`latency`, `soak`) against any server that speaks the WebRTC/events
contract; every recorded run appends a row with host, OS, engine
identities and per-test metrics. Runs the same way against the Python
server, which is how the harness was validated.

**Verified so far:** T1–T14 on Linux/CUDA with cloud LLMs (results above);
T5 and T13 re-verified on the merged upstream commit; T5 re-verified with
Nemotron on CUDA (2026-08-20).

**Open gates — Phase 2 on the reference host (Mac Studio M4 Max), all
local engines, before the production switch:**

| Gate | Pass |
| --- | --- |
| T10 warm, 20 turns, common skills | speech-end → first audio **p50 ≤ 1.5 s** (PRD); segment breakdown STT ≤ 0.5 s, LLM first token ≤ 0.4 s, TTS first audio ≤ 0.3 s |
| T13 Listen mode | wake onset → first audio ≤ 1.5 s |
| T5 | bot audio stops ≤ 300 ms after speech onset; no double-speak; context coherent |
| T12 soak, 50 turns across ≥ 5 sessions | 0 failures; RSS growth bounded once idle wipe is implemented (separates leak from context growth) |
| T6/T7/T9 | pass unchanged with local engines |
| T8 | **must pass** (idle wipe implemented, see Consequences) |
| Framework overhead | e2e minus (STT + LLM + TTS) ≤ 1.5× the Python server on identical fixtures |
| Co-residency | LLM + TTS + STT sidecars + server within the 36 GB budget with no swap growth during the soak |

## Consequences

- **Feature work to reach parity** (each is product logic, not framework
  patching): idle-session semantics and context wipe (the builder disables
  idle timeout — implement as a processor, unify with wake state so "reset
  but still awake" cannot recur); persona routing and mid-stream text
  rewriting; per-connection TTS voice selection; media ducking via the
  player's IPC; the control protocol (`hello`/`state`/`transcript`/`persona`
  /`backend`) mapped onto the events WebSocket (**no WebRTC data channels in
  FlowCat** — clients change accordingly); user-transcript events on the
  events socket (none are emitted mid-session today); the Claude backend
  switch; an MCP client path.
- **Two new Rust services** must be written against the ADR-0004/0005
  sidecars (streaming TTS over WebSocket; Nemotron exists from the PoC).
- The runtime is pre-1.0: every bump is a pinned commit change plus a full
  harness run; the vendored patch is kept minimal and upstreamed.
- Operational shape: one server binary + three inference sidecars + the
  skills service, each with a health endpoint; the native client is a
  second binary on the host; satellites reuse the client crate.
- The Python server remains the production system until the Phase 2 gates
  pass; the harness makes the cut-over a measured decision rather than a
  leap.

## Re-evaluation triggers

- A Phase 2 gate fails for a framework reason (not an engine reason
  attributable via the segment breakdown) and cannot be fixed with a bounded,
  upstreamable change: stay on the Python server; record the finding.
- Upstream stalls (no merges for > 3 months) or a breaking redesign of the
  processor model: freeze on the pinned commit and vendor all four crates,
  or revert to ADR-0002's runner-up (LiveKit Agents).
- The adaptation residue in the embedder starts growing past the Python
  server's ratio, or interruption-class defects recur in daily use after
  cut-over.
- A satellite (Pi/ESP32) client is needed: the wake detector and client
  crate are the starting point; if they cannot be made to fit, the
  single-binary rationale weakens.

## Sources

- FlowCat: [repository](https://github.com/AreevAI/flowcat) (pinned `4ff03f3ef8e179d988a20c6f46498dfb9419c1c1`), [FEATURES.md](https://github.com/AreevAI/flowcat/blob/main/FEATURES.md), [PROCESSOR-DESIGN.md](https://github.com/AreevAI/flowcat/blob/main/PROCESSOR-DESIGN.md), [bench/RESULTS.md](https://github.com/AreevAI/flowcat/blob/main/bench/RESULTS.md); [issue #60 — interruption never delivered, VAD gate never armed, factory and loopback-bind defects](https://github.com/AreevAI/flowcat/issues/60); [PR #61 — full-duplex cascaded pipeline (merged 2026-08-17)](https://github.com/AreevAI/flowcat/pull/61).
- Components: [str0m](https://github.com/algesten/str0m); [ort (ONNX Runtime for Rust)](https://github.com/pykeio/ort); [Silero VAD](https://github.com/snakers4/silero-vad); [whisper-rs](https://github.com/tazz4843/whisper-rs); [openWakeWord](https://github.com/dscripka/openWakeWord); [CPAL](https://github.com/RustAudio/cpal); [Moonshine](https://github.com/moonshine-ai/moonshine); [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp); [Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server).
- Comparators from ADR-0002: [Pipecat](https://github.com/pipecat-ai/pipecat) (interruption issues #3986, #3985, #2791); [LiveKit Agents](https://github.com/livekit/agents).
- Cloud LLM used in Phase 1: [OpenRouter gemma-4-26b-a4b-it](https://openrouter.ai/google/gemma-4-26b-a4b-it).
