# PoC Plan — FlowCat as a candidate realtime-voice runtime

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-08-06 |
| **Goal** | Evaluate FlowCat (Rust, Pipecat-compatible) against Babel's core requirements via a **fully automated, implementation-agnostic test harness**, using cloud-hosted Gemma 4 26B-A4B first and local Gemma 4 on the Mac Studio second. |
| **Related** | ADR-0002 (framework decision; FlowCat is its watch item and this PoC informs re-evaluation trigger #3), ADR-0001 (model), PRD §4.1/§4.2/§4.5, `docs/web-rtc.md` (control protocol) |
| **FlowCat pin** | Original evaluation: `37b09ba` (2026-07-26). Current integration: `4ff03f3` (PR #61 merge, 2026-08-17). Pre-1.0 — pin exact commits, not a moving branch or tag. |

---

## 1. Why this PoC, and what "pass" means

ADR-0002 kept Pipecat but named FlowCat the watch item: a native-Rust, single-binary runtime that deliberately mirrors Pipecat's Frame/FrameProcessor architecture (line-referenced ports of the VAD state machine, context aggregators, metrics, RTVI observer). Its promises — no GIL, bounded backpressure per processor, priority system-frame channel, ~0.2 µs frame routing, flat p99 under load — target exactly the defect classes we hit in Pipecat (frame races, clock races, teardown leaks).

But only one FlowCat path is live-verified by its maintainers (Gemini Live + Plivo telephony). The cascaded local pipeline we need — VAD → STT → LLM-with-tools → TTS over WebRTC/WS — is explicitly "wire-ready but unproven," and FlowCat's closed-issue history shows connectors breaking on first real use. **The PoC's job is to be that first real use, cheaply and reproducibly.**

The PoC answers four questions:

1. **Does the cascaded tool-calling pipeline work at all** (correct tool selection/args with our skill schemas, streamed responses, spoken replies)?
2. **Does the framework machinery survive the scenarios that broke Pipecat** (barge-in, concurrent sessions, teardown, in-flight vs idle clocks, silent tool-call failures)?
3. **Is the performance claim real** for our workload (framework overhead, E2E latency vs the Pipecat baseline on identical fixtures)?
4. **What would migration actually cost** (workaround LOC count, missing features, upstream issues filed)?

**Non-goals:** production migration; feature parity for personas/voice cloning, media ducking, wake-word training, RPi/ESP32 clients; any change to the running assistant.

---

## 2. Architecture

```
                        ┌─ SUT A: Pipecat (existing server.py, baseline) ─┐
 poc/harness (pytest) ──┤                                                 ├── LLM endpoint
   WAV fixtures in      └─ SUT B: FlowCat (poc/flowcat Rust crate)  ──────┘   Phase 1: cloud (OpenRouter)
   audio/events out                    │                                      Phase 2: local (llama-server)
   latency probes                      │ tool executions (HTTP)
   assertions                          ▼
                        poc/stubs — stub skill services (BBC, Spotify,
                        weather, timer): call log + latency/failure injection
```

Three phases, strictly ordered:

- **Phase 0 — Harness + baseline.** Build the harness and stub services; run the full matrix against the **existing Pipecat `server.py`**. This validates the harness itself (a test that can't pass against the known-working system is a harness bug) and produces the baseline numbers every FlowCat result is compared to. Deliberate side effect: the repo gains an automated E2E regression suite it currently lacks — useful regardless of the FlowCat outcome, and reusable against **any** future implementation.
- **Phase 1 — FlowCat vs cloud Gemma 4 26B-A4B.** Isolates framework behavior from local-inference variables; runs on the Linux dev box (FlowCat is Linux-primary; macOS is untested by upstream — treat a macOS build quirk as a finding, not a blocker). **The dev box has a 6 GB NVIDIA GPU, which is enough to run both STT and TTS locally with CUDA** — Whisper (whisper.cpp `cuda` feature via feature unification, or a CUDA speaches sidecar) and the Chatterbox-TTS-Server — so Phases 1/1a/1b exercise real GPU-served audio services, not CPU stand-ins.
- **Phase 1a — server-side wake word over WebRTC.** Port the openWakeWord detector as a custom FlowCat `FrameProcessor` (Rust, via the `ort` ONNX runtime FlowCat already uses for Silero/Smart-Turn, reusing the existing `models/wakeword/*.onnx`) and test the Listen-mode path: harness connects over WebRTC, streams wake-prefixed fixtures ("hey babel, what time is it"), and asserts the turn only fires after wake. This doubles as the PoC's probe of FlowCat's custom-processor ergonomics — the dimension Pipecat scored well on (our ~10 custom processors at 15–90 lines each).
- **Phase 1b — custom-voice TTS (Chatterbox).** Route one persona through the Chatterbox-TTS-Server (cloned voice, OpenAI `/v1/audio/speech` protocol, CUDA on the 6 GB GPU) via FlowCat's OpenAI TTS client with a base-URL override — directly testing the path that needed a 139-line workaround subclass in Pipecat (arbitrary voice names, WAV→PCM handling).
- **Phase 1c — full-duplex spike (decisive).** Full duplex is an essential product requirement, and FlowCat's stock cascaded builder is half-duplex (`TurnMute`). The engine primitives exist and are public (VadProcessor barge-in broadcast, interruption queue-drain, sink `send_clear`, turn strategies, exported s2s outer processors), but the LLM/TTS service adapters lack interruption handling and there is no context repair. The spike: assemble a custom full-duplex cascaded pipeline in `poc/flowcat` (Silero via `vad-ort` — model already in `poc/models/` — replacing `TurnMute`), add interruption to the LLM adapter (cancel the in-flight stream, discard the aggregator buffer, truncate context to spoken text), and make **T5 pass**. Gate: if the runtime's frame semantics hold under barge-in (no double-speak, no context corruption), FlowCat remains a candidate and the work becomes an upstream PR; if the spike surfaces architectural resistance, that is disqualifying — record and stop. This phase outranks 1a/1b in decision weight.
- **Phase 2 — FlowCat on the Mac Studio vs local Gemma 4** via llama.cpp `llama-server` (not Ollama — see §5), plus a Rapid-MLX A/B. This is where the latency targets are judged.

## 3. The implementation-agnostic test harness (`poc/harness/`)

**Principle: black-box only.** The harness talks to a System Under Test (SUT) exclusively through network protocols a real client would use. No framework imports, no internal hooks. Everything implementation-specific lives behind a small **SUT adapter** interface:

```python
class SutAdapter(Protocol):
    async def connect(self) -> Session          # negotiate transport, open event stream
    async def send_audio(self, pcm: bytes)      # 16 kHz mono s16 stream
    def audio_out(self) -> AsyncIterator[bytes] # capture bot audio
    def events(self) -> AsyncIterator[Event]    # normalized: state/transcript/error
    async def close(self, graceful: bool)       # bye vs abrupt-kill (for teardown tests)
```

- **`PipecatAdapter`**: `POST /api/offer` + aiortc peer + the control DataChannel protocol from `docs/web-rtc.md` (`hello`/`state`/`transcript`/`bye`).
- **`FlowCatAdapter`**: two variants, both cheap —
  - *WS transport* (debug aid only, never a recorded channel): FlowCat's generic WebSocket transport takes raw LE s16 mono PCM; deterministic, no ICE — useful for isolating pipeline failures from transport failures.
  - *WebRTC* (`POST /webrtc/offer` + str0m peer + `GET /webrtc/events/{pc_id}`): FlowCat has **no data channels** — events arrive on a separate WebSocket. The adapter normalizes both shapes into the same `Event` stream, which is exactly why the adapter layer exists.

**Transport coverage rule: WebRTC is the only audio channel under test.** All recorded verdicts, for both SUTs, run over WebRTC — it is the production transport for every client surface the PoC models, and on the FlowCat side the str0m stack is one of its "wire-ready but unproven" components, so exercising it *is* part of the evaluation. The WS adapter variant is kept solely as a debugging aid (isolating whether a failure is transport-level or pipeline-level); nothing measured over WS counts as a result.

**Wake-word note:** the harness supports both client modes. Default is a **push-mode smart client** (connect-on-wake, speech starts immediately) — faithful to the satellite path, where wake detection is on-device (RPi/ESP32, PRD WAKE-4); T10 additionally records connect→first-bot-audio, the server's share of the "wake → first TTS audio ≤ 1.5 s" budget. **Listen mode** (server-side wake, as used by the browser client and the local Jabra pipeline) is covered by Phase 1a: wake-prefixed fixtures over WebRTC against the openWakeWord processor ported to FlowCat (T13), with wake→first-audio measured end-to-end. The port's cost and ergonomics are themselves an M4 finding.

Any future implementation (LiveKit, roll-your-own) gets a third adapter; the test matrix and fixtures never change.

**Audio in:** pre-recorded WAV fixtures in `poc/harness/fixtures/` — one spoken utterance per test command, synthesized once with Kokoro (deterministic voice, committed to the repo), each padded with ~300 ms leading and ~1.2 s trailing silence so real Silero VAD segments turns naturally. No mock VAD in the SUT: VAD behavior is part of what's being tested.

**Audio out assertions:** capture bot PCM; assert (a) non-silence within the latency budget, (b) utterance duration sanity, and (c) **content** — transcribe the captured audio with faster-whisper inside the harness and fuzzy-match expected phrases ("timer", "five minutes"). Self-referential but effective: STT-of-TTS is stable for short confirmations.

**Tool assertions:** via the stub services' call logs (below) — the strongest signal in the whole harness, immune to TTS/STT fuzz.

**Latency probes:** the harness timestamps `last_audio_sample_sent` (speech end) and `first_bot_audio_received`, plus per-event timestamps; segment metrics (STT final, LLM TTFB) come from SUT event streams where available and are reported per-SUT with a comparability note.

## 4. Stub external services (`poc/stubs/`)

One small FastAPI process exposing fake backends for every skill the PoC registers, so no test touches the real BBC, Spotify, or Open-Meteo:

| Stub endpoint | Replaces | Behavior |
|---|---|---|
| `POST /bbc/play`, `/bbc/stop` | mpv + BBC HLS | records call, returns station ack |
| `POST /spotify/play`, `/pause`, `/skip` | librespot + Web API | records call + parsed args |
| `GET /weather` | Open-Meteo | canned forecast JSON |
| `POST /timer` | in-process timer | records `{minutes, label}` |
| `GET /calls`, `DELETE /calls` | — | harness inspects/clears the call log |
| `POST /admin/latency`, `/admin/fail` | — | inject per-endpoint delay or 500s (for T9/T11) |

Each SUT registers the **same ~8 tool schemas** (a subset of the production skills, copied verbatim from `skills/` frontmatter so tool-selection difficulty is realistic — plus the production `SkillFilterProcessor` top-K behavior is *not* replicated; all 8 tools are always in context, byte-stable for prefix caching). Handlers are trivial HTTP calls to the stubs — in Pipecat via `register_function`, in FlowCat via its provider-agnostic `Tool` type + a function-call processor. *Stretch:* expose the stubs as an MCP server instead and use Pipecat's MCP client + FlowCat's `mcp` feature — doubles as the first CONV-7 experiment.

## 5. LLM configuration

### Phase 1 — cloud (research summary; full provider survey retained in the research notes)

The exact `google/gemma-4-26B-A4B-it` MoE is served serverless by only a handful of providers. Together, Groq, Cerebras, SambaNova either skip it or host the dense 31B (wrong variant); Fireworks/Baseten are dedicated-GPU only.

| | Provider | Why |
|---|---|---|
| **Smoke tests** (harness bring-up, M1 debugging) | **OpenRouter** — [`google/gemma-4-26b-a4b-it:free`](https://openrouter.ai/google/gemma-4-26b-a4b-it:free) at `https://openrouter.ai/api/v1` | Zero cost for the high-iteration phase (wiring FlowCat, debugging the adapter, fixture round-trips). Rate limits and arbitrary host routing are acceptable here — nothing measured on `:free` counts as a result. FlowCat has a first-class `llm-openrouter` wrapper. |
| **Real tests** (T1–T12 verdicts, anything recorded) | OpenRouter **paid**, routed to the **fastest BF16 tool-capable host**: model snapshot `google/gemma-4-26b-a4b-it-20260403` with `provider: {quantizations: ["bf16"], sort: "throughput"}` | BF16 = the released weights, so behavioral gaps vs the Mac are attributable to our local Q4 quantization, not the endpoint; `sort: "throughput"` picks the fastest qualifying host; passing `tools` auto-excludes hosts serving this model without tool support (SiliconFlow, Parasail). The pinned dated snapshot keeps mid-PoC provider updates from moving the target. ~$0.13/$0.40 per Mtok at BF16 hosts. |
| **Cross-check** | Google AI Studio free tier (`gemma-4-26b-a4b-it`) | Zero-cost first-party sanity check of "reference" tool-call behavior; rate limits and 95% uptime rule it out as primary. |

Cloud TTFT (0.7–1.8 s) **will miss the 0.4 s TTFB target by design** — Phase 1 latency assertions test framework overhead (harness-measured E2E minus provider-reported/API-measured LLM time), not absolute targets.

### Phase 2 — local runtime on the Mac Studio (does not need to be Ollama)

| | Runtime | Why |
|---|---|---|
| **Primary** | **llama.cpp `llama-server`**: `llama-server -hf ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M -ngl 99 -c 32768 --jinja` | Gemma 4 tokenizer + tool-template fixes merged upstream (PRs #21326/#21343); `--jinja` tool calling verified E2E on Apple Silicon; model loads once and never unloads (no `keep_alive` dance — the thing we hand-rolled ~85 lines around in Pipecat); per-slot prefix caching makes warm TTFB ≤ 0.4 s achievable; OpenAI-compat + streaming; deepest operational control. |
| **A/B** | **Rapid-MLX** (`brew`/`pip install rapid-mlx`) | Best measured numbers for this exact model (85 tok/s, 0.08 s cached TTFT on M3 Ultra); dedicated Gemma 4 tool parser handling its unquoted-numeric-argument quirk. Young project — A/B, don't depend. |
| **Baseline** | Ollama v0.32 (`gemma4:26b`) | The incumbent, for continuity with ADR-0001 numbers. Note: `/v1` per-request `keep_alive` is *still* open upstream (issue #2963); env-var workaround only. |
| Ruled out | `mlx_lm.server` (Gemma 4 tool parser broken, issue #1125 open), vllm-metal (MoE support unconfirmed, v0.2.x), SGLang (no macOS), mistral.rs (no Gemma 4 GGUF — recheck later; a Rust in-process engine would suit FlowCat long-term) | |

**Prefix-cache discipline (all runtimes):** the ≤ 0.4 s warm-TTFB target only holds if the system prompt + 8 tool schemas are byte-stable across turns — schemas first, session-varying content last. The harness asserts this indirectly via T10's warm-turn TTFB distribution.

### FlowCat-side audio services (both phases)

- **STT:** `stt-whisper-local` (whisper.cpp via whisper-rs — batch ~4 s segments, CPU-only as shipped). Phase 1 (dev box): enable `whisper-rs/cuda` via feature unification against the 6 GB NVIDIA GPU. Phase 2 (Mac): attempt Metal the same way. Either phase: if segmentation latency dominates T10, fall back to a local [speaches](https://github.com/speaches-ai/speaches) sidecar (CUDA on the dev box) through FlowCat's `stt-speaches` wrapper and record the finding.
- **TTS:** `tts-kokoro` → a **kokoro-fastapi sidecar** (FlowCat's Kokoro support is an HTTP client, not in-process ONNX — a real "single binary" caveat to document). Phase 1b adds the **Chatterbox-TTS-Server** (CUDA) for the custom-voice persona. VRAM budget on the 6 GB card: Whisper small (~1 GB) + Chatterbox (~3–4 GB) + Kokoro (CPU-friendly, keep it off-GPU) fit together, but T14 runs are the only ones needing Chatterbox resident — load it per-phase, not permanently.
- **VAD/turn:** `vad-ort` with Silero ONNX + Smart-Turn v3 ONNX files supplied by us (not vendored); `TurnSilenceTracker` fallback.

## 6. Test matrix

Every test runs against **both SUTs** with identical fixtures. IDs map to PRD requirements and ADR-0002 friction points (F#: the numbered defect classes in ADR-0002's context table).

| ID | Test | Asserts | Targets |
|---|---|---|---|
| T1 | Basic turn: "what time is it" WAV → spoken reply | transcript event; `get_current_time` called; audio reply contains a time | CONV-1 |
| T2 | Direct tool call: "set a timer for five minutes" | timer stub called with `{minutes: 5}`; spoken confirmation | SKILL-3 |
| T3 | Indirect phrasing: "put some music on" → `play_spotify`; "I'd like to hear the news" → `play_bbc_radio` | correct tool + plausible args (the gemma4 reliability bar from ADR-0001) | SKILL-2/3 |
| T4 | Stubbed media round-trip: play BBC station → stop; play Spotify track → pause | stub call sequence + args exact-match; confirmations spoken | MEDIA-1/3 |
| T5 | **Barge-in**: trigger a long reply ("count to thirty slowly"), inject speech at t+1 s | bot audio stops ≤ 300 ms after new speech onset; new turn answered; context not corrupted (follow-up "what did I just ask" coherent) | F: interruption; Pipecat's #1 public bug class |
| T6 | **Concurrent sessions**: two harness clients, interleaved turns, different tools | zero cross-talk in transcripts/audio/tool logs | RTC-3, F: shared-processor corruption |
| T7 | **Teardown**: abrupt peer kill mid-reply (no `bye`) | server healthy afterward (next connect succeeds ≤ 2 s); no log spin (harness greps SUT stderr rate) | F: `MediaStreamError` spin, dangling handlers |
| T8 | Idle lifecycle: 15 s silence after a turn | context wiped (probe: "what did I just ask" → doesn't know); session/transport still usable | CONV-5, F: clock races |
| T9 | **In-flight vs idle**: stub injects 12 s tool latency (> idle timeout) | reply still delivered; idle timer does not kill the in-flight call | F: `InFlightTracker` class |
| T10 | **Latency bench**: 20 warm T1-class turns | p50/p95 for speech-end→first-audio, plus segment breakdown; Phase 2 gates: warm LLM TTFB ≤ 0.4 s, E2E ≤ 1.5 s common-skill; Phase 1 gate: framework overhead (E2E minus LLM+STT+TTS time) within 1.5× Pipecat baseline | NFR-1, PRD §4.1 |
| T11 | **Silent-failure probe**: cap `max_tokens` at 40 to force truncated tool-call JSON | *some* observable signal (error event, metric, log) — not a silently dead turn | F: swallowed `json.loads` |
| T12 | Soak: 50-turn scripted session | zero failures; RSS/fd growth bounded; T10 percentiles stable in the last 10 turns | NFR-4 |

| T13 | **Server-side wake over WebRTC** (Phase 1a): wake-prefixed fixture → turn fires; wake-less speech → no turn; wake mid-session re-arms correctly after idle reset | wake gating correct both ways; **wake → first-bot-audio ≤ 1.5 s** (Phase 2); no "reset but still awake" divergence (the Pipecat clock-race bug) | WAKE-1/4, F: clock races |
| T14 | **Custom-voice TTS** (Phase 1b): one turn routed through Chatterbox (cloned voice, CUDA) via FlowCat's OpenAI TTS client + base-URL override | audio reply produced in the cloned voice (assert non-Kokoro sample rate/duration signature + content match); arbitrary voice name accepted; TTS first-audio segment recorded | PERS-1, F: `chatterbox_tts.py` workaround class |

Stretch (time-boxed, skip without guilt): T15 two-LLM conditional routing (`ask_claude`-style backend flip — Pipecat friction #1, the sandwich filters); T16 MCP-served stubs (CONV-7).

## 7. Milestones

| # | Deliverable | Est. | Exit criterion |
|---|---|---|---|
| M0 | `poc/harness/` + `poc/stubs/` + fixtures + `PipecatAdapter`; full matrix vs `server.py` | 2–3 days | **G0:** T1–T12 pass (or documented-fail with cause) against Pipecat; baseline numbers recorded in `poc/reports/baseline-pipecat.md` |
| M1 | `poc/flowcat/` Rust crate: cascaded pipeline builder + tool-executor processor + stub-backed tools + Novita config; `FlowCatAdapter` | 2–4 days (more if first-contact connector bugs bite — budget for filing upstream issues) | **G1:** T1–T4 pass vs cloud |
| M1a | Phase 1a: openWakeWord `FrameProcessor` in Rust (`poc/flowcat/src/wakeword.rs`, `ort` + existing `models/wakeword/*.onnx`); Listen-mode harness variant + wake-prefixed fixtures | 1–2 days | **G1a:** T13 passes vs cloud LLM; port LOC + ergonomics noted for M4 |
| M1b | Phase 1b: Chatterbox-TTS-Server on the dev-box GPU (CUDA); persona routed via FlowCat OpenAI TTS client | 0.5–1 day | **G1b:** T14 passes; any workaround code counted (target: fewer than Pipecat's 139 lines) |
| M2 | Full matrix vs FlowCat/cloud; upstream issues filed for failures | 1–2 days | **G2:** T5–T9, T11, T12 verdicts recorded |
| M3 | Mac Studio: llama-server + Rapid-MLX A/B + Ollama baseline; rerun matrix, T10 judged for real | 1–2 days | **G3:** latency table vs PRD targets, three runtimes |
| M4 | `poc/reports/flowcat-verdict.md` + ADR-0002 amendment (confirm keep, or open a migration ADR) | 0.5 day | Report includes: pass/fail matrix, latency tables, **workaround-LOC count** (the ADR-0002 metric: how much code exists because of FlowCat's shape), issues filed/fixed upstream, migration-cost estimate |

Kill criteria (stop early, write it up): FlowCat cannot complete a streamed tool-call round-trip (G1 fails after ~2 days of debugging), or barge-in (T5) is architecturally broken rather than buggy, or upstream is unresponsive to a blocking defect.

## 8. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Cascaded path is unproven upstream; issue history shows first-use connector bugs | M1 overruns | Pin exact commits (`37b09ba` for the original run, `4ff03f3` after PR #61 merged); time-box; file issues; failures are *findings*, not wasted time |
| **Local mic/speaker transport is a stub** in FlowCat | None for the PoC (all testing is over WebRTC — same as satellite clients); blocks the Jabra path in any real migration | Record as migration cost; upstream explicitly invites a cpal backend contribution |
| No WebRTC data channels — our control protocol (`docs/web-rtc.md`) assumes one | Client changes in a real migration (events on a side WebSocket) | Adapter normalizes it for the PoC; count protocol-port cost in M4 report |
| whisper.cpp batch STT (~4 s segments, no streaming partials, CPU-default) | Could dominate T10 and mask framework latency | CUDA (dev box) / Metal (Mac) via feature unification; speaches sidecar fallback; report STT segment separately |
| Bus factor: ~7-week-old project, one small team, no MSRV, pre-1.0 churn | Strategic, not tactical | Exactly why this is a PoC and not a migration; ADR-0002 trigger #3 requires 1.0 + proven local paths before any commitment |
| Cloud/local model behavior drift (BF16 vs Q4_K_M) confuses tool-call comparisons | Misattributed failures | Real tests run BF16-only routing; rerun any cloud-phase tool-call failure against llama-server before blaming FlowCat |
| OpenRouter routing varies host run-to-run (`:free` routes anywhere; `sort: "throughput"` can flip between BF16 hosts as load shifts) | Non-reproducible results | `:free` is smoke-test-only, never measured; real tests constrain to `quantizations: ["bf16"]` + dated snapshot, and the harness logs the serving host (OpenRouter returns it in the response `provider` field) with every recorded run |
| Harness self-reference (Kokoro fixtures in, whisper assertions out) | False confidence | Tool-call-log assertions (stub-side) are the primary signal; audio-content matching is secondary |

## 9. Repo layout

```
poc/
  harness/        # pytest suite; adapters/{pipecat,flowcat}.py; fixtures/*.wav; conftest with SUT lifecycle
  stubs/          # FastAPI stub services + call-log API
  flowcat/        # Rust crate (pinned flowcat git dep); cascaded pipeline + tool executor + config
  reports/        # baseline-pipecat.md, flowcat-cloud.md, flowcat-local.md, flowcat-verdict.md
```

Everything under `poc/` is isolated from the production tree; the harness and stubs are the explicitly reusable artifacts (SUT-agnostic by construction) and graduate out of `poc/` if adopted as the standing E2E suite.

## 10. Key sources

- FlowCat: [repo](https://github.com/AreevAI/flowcat) (original evaluation `37b09ba`; current integration `4ff03f3`), [FEATURES.md](https://github.com/AreevAI/flowcat/blob/main/FEATURES.md), [PROCESSOR-DESIGN.md](https://github.com/AreevAI/flowcat/blob/main/PROCESSOR-DESIGN.md), [bench/RESULTS.md](https://github.com/AreevAI/flowcat/blob/main/bench/RESULTS.md); maintainer claim: only Gemini Live + Plivo live-verified.
- Cloud providers: [OpenRouter endpoints for gemma-4-26b-a4b-it](https://openrouter.ai/google/gemma-4-26b-a4b-it) (per-provider quant/tools/uptime), [artificialanalysis.ai provider table](https://artificialanalysis.ai/models/gemma-4-26b-a4b/providers), [DeepInfra benchmarks](https://deepinfra.com/blog/gemma-4-26b-a4b-api-benchmarks), [Novita model page](https://novita.ai/models/model-detail/google-gemma-4-26b-a4b-it), [Cloudflare Workers AI model](https://developers.cloudflare.com/workers-ai/models/gemma-4-26b-a4b-it/).
- Local runtimes: [llama.cpp Gemma 4 tokenizer fix PR #21343](https://github.com/ggml-org/llama.cpp/pull/21343), [tool-calling-on-Mac walkthrough](https://gist.github.com/daniel-farina/87dc1c394b94e45bb700d27e9ea03193), [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), [mlx-lm Gemma 4 tool-parser bug #1125](https://github.com/ml-explore/mlx-lm/issues/1125), [Ollama /v1 options issue #2963](https://github.com/ollama/ollama/issues/2963), [vllm-metal](https://github.com/vllm-project/vllm-metal).
