# ADR-0002: Realtime-voice / LLM orchestration framework

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-06 |
| **Decision** | Keep **Pipecat** (Python); pin to a 1.x release and stop tracking unpinned HEAD. Do **not** reimplement on another framework. LiveKit Agents (Python) is the documented reimplementation target if the re-evaluation triggers fire. |
| **Related** | PRD §4.1 (CONV-1..5), §4.5 (RTC-1..6), §5 (NFR-1/2/4/5), §6 (TECH-2/6); ADR-0001 (core LLM); `docs/comparison.md` §3 Camp C; `docs/web-rtc.md` |

---

## Context

The framework question is: **the current Pipecat implementation works reasonably well, but implementation was defect-heavy. If we were reimplementing today, is there a better choice?** The candidate does not need to be Python.

### What the framework must provide (distilled from the PRD)

1. **Realtime voice loop with barge-in**: VAD-segmented turns, streaming STT→LLM→TTS, user interruption mid-reply (CONV-1, NFR-1).
2. **Fully local inference as a first-class citizen**: Whisper MLX, Ollama, Kokoro/Chatterbox all on-host; cloud only as an opt-in tier (NFR-2). This is the constraint that kills most of the market — the commercial voice-agent frameworks assume cloud STT/TTS/LLM.
3. **Two transports, one pipeline**: local USB speakerphone and per-connection WebRTC (browser, RPi 5, future ESP32-S3), with parallel-session isolation (RTC-1/3, AUD-1..4).
4. **Provider-neutral tool calling** so the drop-in skill system targets Ollama and Anthropic identically (SKILL-1/2), plus a future MCP client path (CONV-7).
5. **Custom turn machinery**: audio-domain wake words bound to personas, session idle semantics, mid-stream text rewriting (persona tags), per-turn tool filtering (WAKE-1..5, PERS-2, SKILL-2).
6. Single-household scale. We need one or a handful of concurrent sessions, not thousands — density and SFU features carry no weight.

### The incumbent, honestly accounted

`pipecat-ai` with 9 extras — **installed unpinned since the first commit**, which matters below.

**What the defect history actually was.** A code audit of `app.py`/`server.py`/git history sorts the implementation defects into recognizable classes:

| Defect class | Example in this repo |
|---|---|
| Frame routing | `ParallelPipeline` branch filters only block *downstream*; the tool-result re-trigger flows upstream through the LLM before the filter sees it → both LLMs answered the same turn. Fix: double "sandwich" filters around every LLM (`app.py:760-790`). |
| Silent failures | Tool-call JSON truncated by `max_tokens` is dropped in Pipecat's `json.loads` with nothing surfaced; turn dies silently. Fix: `max_tokens` 80→200→512 plus a custom silent-turn detector (`app.py:254-306`). |
| Clock/session races | Pipecat's idle timer, the wake strategy's timeout, and in-flight tool work are three independent clocks — the idle timer fired mid-Claude-request; the wake state outlived the wiped LLM context ("reset but still awake"). Fix: `InFlightTracker` + `force_idle()` unification. |
| Concurrency | Pipecat processors can't be linked into two live pipelines; shared persona-TTS singletons corrupted concurrent WebRTC sessions. Fix: per-connection TTS factory + a monkeypatch of `pipecat.services.kokoro.tts` to inject a preloaded model. |
| Lifecycle/teardown | Peer disconnect left the transport read-loop spinning on `MediaStreamError` (hundreds of warnings/sec); asyncio-cancelling a `PipelineTask` leaves frame handlers dangling. Fix: `closed`-handler cancel + two-phase shutdown. |
| Audio-path limits | In-pipeline Spotify audio (`SpotifyMediaInjector`, 162 lines) was choppy/staticky and **abandoned** — Pipecat's output transport is a TTS sink, not a mixer. Music moved out-of-band to librespot+mpv with IPC ducking. |
| Transport bugs below Pipecat | aiortc stamps all RTP packets from one frame with the same timestamp — >20 ms frames delivered 1 of 155 packets. Not Pipecat's code, but our layer to debug. |
| Version drift | Interruption signaling differs across Pipecat versions; the code defensively matches three frame types (`app.py:296-310`). Root cause: **unpinned dependency during Pipecat's pre-1.0 breaking-change era.** |

**Cost:** roughly **45% of `app.py`+`server.py` (~1,400 of 3,043 lines)** is framework adaptation rather than product logic — plus `wakeword_detector.py` (326 lines replacing the transcript-regex wake Pipecat ships), `chatterbox_tts.py` (139 lines patching `OpenAITTSService`), ~230 deleted lines of abandoned workarounds, and two permanent "prove it isn't the framework" harnesses (`webrtc_smoke/`, `scripts/pipeline_audio_test.py`). The three biggest structural gaps: no conditional-routing/mux primitive, no shareable/clonable processors, and no unified session model spanning turn-start, idle, in-flight work, and transport teardown.

**What Pipecat delivered that we never had to build:** barge-in/interruption is literally one constructor flag (`allow_interruptions=True`) — the single hardest realtime-voice feature, and zero custom code in this repo; transport swap (LocalAudio → SmallWebRTC) reused every processor unchanged; Whisper-MLX/Ollama/Anthropic/Kokoro adapters in-tree; provider-neutral `FunctionSchema` tool calling that made the skill system ~50 lines of glue; shared `LLMContext` across two LLM backends; the `FrameProcessor` extension model itself (our ~10 custom processors are each 15–90 productive lines). The complaint is about *what* had to be built, not the ergonomics of building it.

---

## Research: the field (August 2026)

Deep-research pass over releases, issue trackers, and 2026 comparisons. Full sources at the end.

| Candidate | Language | Local-first? | Barge-in/turn-taking | Self-host story | Verdict for Babel |
|---|---|---|---|---|---|
| **Pipecat 1.7.0** (incumbent) | Python | **First-class**: Whisper-MLX, Ollama, Kokoro, LocalAudioTransport, SmallWebRTC all in-tree; Daily publishes a Mac-local reference stack | Free with the pipeline; open **Smart Turn v3.2** (12 ms CPU, weights+training open); known interruption edge-cases concentrated in WebSocket transports we don't use | Serverless — SmallWebRTC is direct P2P, no SFU/relay | **Keep** |
| **LiveKit Agents 1.6.8** | Python (Node port **beta**, "use Python for production") | Supported via OpenAI-compatible base-URL overrides; community-wired, not showcased | Strong — but the best adaptive interruption model is **LiveKit-Cloud-only**; self-hosted gets "v1-mini" | Requires running a LiveKit SFU server (Apache-2.0, self-hostable). ESP32-S3 client SDK exists (Developer Preview) | **Runner-up** |
| **TEN Framework** (Agora) | C++/Go/Python/Node | Works via OpenAI-compatible extensions; reference agents lean on cloud keys | Graph runtime, capable | Docker; heavier conceptual model (graphs, manifests, designer UI); Agora-centric, small Western self-host community | No advantage here |
| **Vocode** | Python | — | — | Last commit Nov 2024 | **Dead**; eliminate |
| **OpenAI Agents SDK / Realtime** | Python/JS | Voice models default to OpenAI cloud; no local STT/TTS adapters, no device-WebRTC transport | Cloud speech-to-speech | — | Wrong fit; only the "expose the Realtime protocol locally" pattern (cf. `huggingface/speech-to-speech`) is interesting |
| **FlowCat** | **Rust** | Whisper/Kokoro/Piper/Ollama behind Cargo features — "wire-ready but unproven" | Pipecat-compatible clean-room architecture (same Frame/FrameProcessor taxonomy), str0m WebRTC | Single binary, air-gap capable | **Watch item**: pre-1.0, 81 stars, only one cloud path proven E2E |
| **Feros** | Rust + Python | Optional self-hosted Whisper/Fish | — | Open-sourced ~Apr 2026, call-center-oriented, early-stage | Too young |
| **Roll-your-own** | any (Rust/tokio, Node, asyncio) | Components now exist: silero-vad 6.2 (MLX port available), Smart Turn v3.2 standalone, str0m/webrtc-rs, FastRTC, sherpa-onnx (STT/TTS/**speaker-ID** toolkit, 11 language bindings) | **You own it** — frame lifecycle, backpressure, barge-in cancellation across STT/LLM/TTS tasks, context repair on interruption | Total | Feasible in 2026, but re-derives exactly the layer where our defects (and Pipecat's issue tracker) live |

Two field observations that decide this ADR:

1. **The defect classes are the domain's, not uniquely Pipecat's.** Pipecat's tracker shows the same families we hit (interruption-before-audio-out #3986, hard TTS cancel #3985, context-not-updated-on-interruption #2791; one production writeup counts "queue recreation, deadlock, frame-drop, and race condition" as the four interrupt/resume bug classes). LiveKit's turn/interruption stack is its top engineering topic too. A rewrite on any framework — or from scratch — re-encounters this layer with **zero** of our accumulated fixes, which are now written, tested, and stable in ~1,400 lines we already own.
2. **Pipecat crossed 1.0 in April 2026.** Post-1.0 releases every ~2–3 weeks with flagged, minor breaking changes and an official pre-1.0 migration guide. A meaningful share of our implementation pain (interruption-frame drift, moved module paths, the phantom `faster-whisper` import) traces to riding **unpinned pre-1.0 HEAD** — a self-inflicted amplifier that no framework choice fixes and a version pin does.

### Why not the non-Python options specifically

The "does not need to be Python" door was checked and is genuinely open — but nothing credible walks through it today. LiveKit's Node port is officially beta; FlowCat/Feros are pre-1.0 with unproven local paths; TEN's polyglot extensions buy nothing for a single-host assistant; roll-your-own in Rust means re-implementing barge-in semantics for a system whose latency budget is dominated by model inference (ADR-0001), not framework overhead — both mainstream frameworks add sub-500 ms and are not the bottleneck. Python also remains where the local-inference ecosystem lives (MLX, Ollama clients, ONNX runtimes, openWakeWord). FlowCat's deliberate Pipecat-compatible architecture is the notable hedge: if a Rust rewrite ever becomes justified, our processor decomposition maps across without an architecture change.

---

## Decision

**Keep Pipecat. Do not reimplement.** The reasoning, in order of weight:

1. **Local-first is first-class only in Pipecat.** Every element of our stack is in-tree and reference-documented by the maintainer. In every alternative, local inference is community-wired, cloud-gated at the top tier (LiveKit's best interruption model), or unproven (Rust options).
2. **The expensive defects are behind us and their fixes are assets.** ~45% adaptation code is a real cost, but it is *sunk and stable* — and the audit shows the worst of it addresses problems (interruption lifecycle, session clocks, concurrent-pipeline state) that every framework in this space is still fighting in public. A reimplementation trades known, patched defects for unknown ones.
3. **What Pipecat provides free is exactly the hardest part.** Barge-in works with zero custom code. That asymmetry — we patched *conveniences* and got the *core* for free — is the opposite of a reimplementation signal.
4. **Post-1.0 stability + a version pin removes the largest historical defect amplifier.**
5. **Scale requirements don't reward the alternatives.** LiveKit's genuine advantages (SFU, SIP telephony, multi-participant, density) are all PRD non-goals.

**Runner-up: LiveKit Agents (Python).** The reimplementation target if we ever need whole-house SFU-scale satellite fleets, phone-callable Babel (SIP), or multi-participant sessions — accepting a self-hosted LiveKit server and the cloud-gated top-tier turn model.

**Watch item: FlowCat** — Pipecat-compatible Rust; re-inspect at 1.0 with proven local providers.

## Consequences

1. **Pin the dependency now.** Replace the unpinned `pipecat-ai[...]` in `requirements.txt` with an exact 1.x pin; upgrade deliberately per release notes. This is the single highest-leverage lesson from the defect history.
2. **De-duplicate the pipeline assembly** (~215-line near-clone of `build_pipeline_task` in `server.py`, plus `app.py` as a third copy) as part of the already-planned TECH-2 `services.py` extraction, and retire the legacy `app.py` path (TECH-6). This shrinks the framework-adaptation surface we carry even while staying on Pipecat.
3. **Upstream the top gaps** as issues/PRs where they're framework bugs rather than design disagreements: silent tool-call JSON drop, `chunk_size==0` in `OpenAITTSService` warm-up, non-injectable Kokoro model (our monkeypatch), no in-flight-aware idle timeout.
4. **Keep the harnesses.** `webrtc_smoke/` and `scripts/pipeline_audio_test.py` stay as permanent transport regression tools — they earn their keep on every upgrade of the newly-pinned dependency.
5. Pipecat's in-tree **MCP client** integration becomes the default path for CONV-7 (model competence remains the open question per ADR-0001, not framework support).
6. We accept continued exposure to Daily.co governance and to Pipecat's known interruption edge-cases (mitigated by staying on WebRTC/local transports, where they are materially rarer than on WebSocket transports).

## Re-evaluation triggers

Re-open this ADR when any of:

1. **Governance/cadence**: Daily pivots Pipecat toward cloud-only, release cadence stalls >6 months, or a 2.0 breaks the FrameProcessor model.
2. **Scope change into LiveKit's strengths**: PRD adopts SIP/phone access, multi-participant sessions, or a satellite fleet large enough to want an SFU (the `docs/comparison.md` "whole-house question").
3. **FlowCat (or a successor) reaches 1.0** with proven local Whisper/Kokoro/Ollama paths — the Pipecat-compatible architecture makes migration cost mostly mechanical.
4. **The adaptation ratio grows instead of shrinking**: a new P0 feature (e.g. in-pipeline audio mixing returning for multi-room, or streaming speaker diarization per SPKR-1..3) again requires patching Pipecat internals rather than extending via `FrameProcessor`.
5. **Interruption-class defects recur in daily-driver use** after pinning to 1.x — that would falsify the "defects were the pre-1.0 era" premise.

## Sources

- **This repo (defect evidence):** `app.py` (sandwich filters `:760-790`, `InFlightTracker` `:227-310`, keep-alive reimplementation `:926-1011`), `server.py` (Kokoro monkeypatch `:666-682`, `ControlChannel`/`PipelineStateEmitter` `:363-584`, two-phase shutdown `:1553+`, `MediaStreamError` cancel `:1900-1910`), `chatterbox_tts.py`, `wakeword_detector.py`, `webrtc_smoke/`, `docs/web-rtc.md` (Step C shared-TTS hazard, aiortc timestamp bug), git: `25a32af`, `02a105c`, `05e4b3a`, `8100e39`, `547832a`, `9c5a8f3`→`fbcb0fb` (SpotifyMediaInjector added/removed).
- [Pipecat releases](https://github.com/pipecat-ai/pipecat/releases) — 1.0.0 (2026-04-14), 1.7.0 (2026-08-01); [1.0 migration guide](https://docs.pipecat.ai/pipecat/migration/migration-1.0); interruption issues [#3986](https://github.com/pipecat-ai/pipecat/issues/3986), [#3985](https://github.com/pipecat-ai/pipecat/issues/3985), [#2791](https://github.com/pipecat-ai/pipecat/issues/2791); [production bug-class writeup](https://luonghongthuan.com/en/blog/pipecat-voice-agent-production-scalable-guide/); [macOS local reference stack](https://github.com/kwindla/macos-local-voice-agents); [Smart Turn v3.2](https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/); [pipecat-mcp-server](https://github.com/pipecat-ai/pipecat-mcp-server).
- [LiveKit Agents releases](https://github.com/livekit/agents/releases) — 1.6.8 (2026-08-03); [turn detection & interruption handling](https://livekit.com/blog/turn-detection-and-interruption-handling) (cloud-gated adaptive model; self-hosted v1-mini); [agents-js](https://github.com/livekit/agents-js) (beta, "use Python for production"); [ESP32 client SDK](https://github.com/livekit/client-sdk-esp32) (Developer Preview).
- [TEN Framework](https://github.com/TEN-framework/ten-framework); [Vocode](https://github.com/vocodedev/vocode-core) (dormant since Nov 2024); [OpenAI Agents SDK voice](https://openai.github.io/openai-agents-python/ref/voice/pipeline/); [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech).
- Rust/roll-your-own: [FlowCat](https://github.com/AreevAI/flowcat); [Feros](https://github.com/ferosai/feros); [sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/index.html); [silero-vad v6.2](https://github.com/snakers4/silero-vad/releases); [FastRTC](https://github.com/gradio-app/fastrtc).
- 2026 comparisons: [Evalgent Pipecat vs LiveKit (2026-07-03)](https://www.evalgent.com/blog/pipecat-vs-livekit), [thinnest.ai framework survey](https://www.thinnest.ai/blog/open-source-voice-ai-frameworks), [Cekura](https://www.cekura.ai/blogs/pipecat-vs-livekit-the-real-difference), [Soniox wiki](https://soniox.com/wiki/voice-agent-frameworks) — consistent split: Pipecat for pipeline control + local/custom models; LiveKit for scale/SIP/multi-participant.
