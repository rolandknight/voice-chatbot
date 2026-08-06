# FlowCat PoC — running findings ledger

Facts discovered while implementing Phase 1, accumulated for the M4 verdict
report. FlowCat pinned at `37b09bafd6a50cb65936411b40b09e77386e83e3`.

## Framework findings (pre-first-run)

1. **The *stock* cascaded builder is half-duplex — but this is an assembly
   choice, not an engine limit.** (Corrected 2026-08-06 after a targeted
   re-read; the original version of this finding overstated it.)
   `TurnMute` in `build_cascaded_call_with_observers` mutes STT until the
   reply plays out ("No barge-in" verbatim in source), so T5 fails by design
   against the stock builder. However: `VadProcessor` is public with barge-in
   built in (rising edge while bot speaks → broadcast `Frame::Interruption`),
   the runtime drains queues on interruption (race-regression-tested), the
   cascaded sink already handles `Interruption` → `send_clear`, the public
   turn-strategy module includes Smart-Turn stop, and the s2s outer
   processors (`TransportInput`, `BrainProcessor`) are exported — a custom
   full-duplex cascaded assembly is buildable from outside the crate.
   **The genuine gaps:** the LLM/TTS service adapters have no interruption
   handling (an in-flight LLM stream isn't cancelled; the assistant
   aggregator would speak the full reply post-barge-in), and there is no
   context repair (truncate-to-spoken). Bounded work — two adapters + one
   aggregator — and the priority-channel runtime is a good substrate for it,
   but it must be proven: see **Phase 1c** in the plan. Pipecat comparison
   stands only as "free vs. build-it": `allow_interruptions=True` vs. a
   custom assembly + adapter interruption work (candidate upstream PR).
2. **No VAD in the cascaded chain at all.** Turn boundaries come from
   whisper.cpp's fixed ~4 s batch segmentation, not speech detection. Silero
   (`vad-ort`) exists in flowcat-core but is unused by
   `build_cascaded_call_with_observers`. Latency floor: up to ~4 s of
   buffering before STT even runs — expect T10 Phase 1 numbers to be
   dominated by this, not by the framework hot path.
3. **`factory::cascaded` can't build our stack.** Kokoro's `base_url` option
   is dropped by the factory (must call `KokoroTts::with_base_url` directly),
   and keyless local providers (whisper_local, kokoro) still fail the
   factory's `require_key` check without a dummy `api_key`. Consequence: the
   config-file path (`run_call_with`) is unusable for a local stack; we build
   services directly. Upstream-issue candidates, both trivial fixes.
4. **`DeclarativeBrain` force-advertises an `endCall` tool** (unconditional
   `tools.push(end_call_tool())`) — wrong for an always-on assistant, so we
   wrote a 50-line custom `AgentBrain` instead. The seam made this easy
   (point in FlowCat's favor: the trait is 6 methods).
5. **Embedder ergonomics are genuinely good.** `flowcat-server` works as a
   library (`webrtc-helper` feature): offer plumbing, event registry, RTVI→
   browser event mapping, and the playground page were all reusable. The
   whole embedder is ~4 small files. `handle_offer` couldn't be used only
   because of finding 3 (it routes through the factory).
6. **OpenRouter client has a fixed request body** — no way to pass OpenRouter
   `provider: {quantizations, sort}` routing preferences (needed for the
   plan's BF16/throughput real-test routing). Workarounds: set provider
   preferences on the OpenRouter account/key, or patch `OpenAiLlm`. Smoke
   (`:free`) unaffected. Upstream-issue candidate (extra-body support).
7. **whisper.cpp is CPU-only as shipped** (whisper-rs built without GPU
   features); CUDA can be feature-unified from our crate but needs the CUDA
   toolkit installed. Phase 1 smoke runs CPU tiny.en.
8. **Events WS drops user-speaking markers and metrics** (`map_rtvi_to_rtf`
   returns None for `user-started-speaking`, `metrics`, `bot-tts-text` etc.),
   so harness latency segmentation from server events alone is limited;
   harness-side audio timestamps carry the burden.

## Defects found live

1. **str0m/loopback ICE interop bug (fixed in our embedder, upstream-issue
   candidate).** With the media socket bound to `127.0.0.1` (upstream
   flowcat-server's default `FLOWCAT_WEBRTC_BIND_IP`), ICE checks from a
   same-host client using per-interface sockets (aiortc; Chrome behaves the
   same) arrive from non-loopback sources (docker bridges, LAN IP) and every
   reply fails with `webrtc UDP send error: Invalid argument (os error 22)` —
   the connection never establishes. Fix: bind `0.0.0.0`, advertise
   `127.0.0.1` (`call.rs`). Matches the "first live use breaks a connector"
   pattern in FlowCat's issue history.
2. **OpenRouter `:free` variant is unusable even for smoke** in practice —
   429 "temporarily rate-limited upstream" from the shared pool (both free
   providers) on second requests. Smoke now runs the paid standard variant
   (~fractions of a cent per run).

3. **whisper.cpp fixed 4 s batch windows fire turns on partial transcripts.**
   A long utterance splits across windows and each window's fragment becomes a
   *final* transcription → the LLM acts on partials ("people" → a mangled
   `play_spotify`, then the correct call seconds later). Root cause is
   finding 2 in the pre-first-run section (no VAD-driven turn boundary in the
   cascaded chain). Harness compensates by matching on the *correct* call
   appearing, but a production migration would need a real endpointing story.
4. **tiny.en is below the content-accuracy bar through the Opus path**
   ("Radio 4" → "Radio for", "Purple Rain" → "Purple Vayne"); T4 tests failed
   on argument content (tool selection was still right in every call).
   **Resolved: `ggml-base.en` flips the full suite green** (CPU, ~4 s windows
   unchanged). Latency cost acceptable in Phase 1; Phase 2 measures it.

## Phase 1 smoke outcome (2026-08-06)

**6/6 tests pass** (T1 time, T2 timer, T3 music→spotify, T3 news→bbc,
T4 bbc round-trip, T4 spotify track) with: paid OpenRouter
`google/gemma-4-26b-a4b-it`, whisper.cpp **base.en** CPU, Kokoro shim,
str0m WebRTC, tools relayed to stubs. Full suite wall time 163 s.
Working config is pinned in `poc/.env` (`POC_LLM_MODEL`,
`POC_WHISPER_MODEL`); `:free` remains default-off due to defect 2.

## Phase 1c — full-duplex spike (2026-08-06): **PASS**

Full duplex is achievable on FlowCat, and now works: **T5 (barge-in) passes
3/3 consecutive runs; full suite 7/7** (T1–T5) against the duplex server.
Built as a vendored `flowcat-core` patch (cargo `[patch]` over the pinned git
dep — the diff IS the candidate upstream PR) plus ~160 lines in the embedder:

1. **`on_interruption()` trait hook** (new): the runtime *never* delivers
   `Frame::Interruption` to `process_frame` (it drains queues and forwards) —
   the pre-existing `Interruption` arms in FlowCat's own sinks are unreachable
   dead code, meaning frame-level barge-in had likely never run live (the
   proven Gemini path does barge-in model-side). The hook gives processors a
   real delivery path; the sink flushes (`send_clear`) and the assistant
   aggregator repairs context (partial reply retained, open span dropped).
2. **Cooperative LLM cancel**: the runtime cannot preempt a busy
   `process_frame`, so the LLM adapter polls a barge-in generation counter
   (bumped synchronously by the VAD) between streamed chunks.
3. **`SpeechGate` + whole-utterance STT**: with `TurnMute` gone, whisper's
   fixed 4 s windows produced 6–7 bogus "user turns" per 30 s call (silence
   hallucination + utterance splitting). The gate forwards audio only between
   VAD edges (300 ms pre-roll, flush marker at falling edge); the embedder's
   `BabelStt` transcribes one whole utterance per VAD turn. Turn chaos gone.
4. **Out-of-band interrupt reactor**: the frame-path `Interruption` stalls
   behind any mid-`await` hop — measured sink delivery 14 ms to **2.1 s**
   depending on TTS activity (the priority channel only helps between
   `process_frame` calls; this is FlowCat's analogue of Pipecat's frame-race
   bug class, in serialized form). A `Notify`-woken reactor flushes the
   carrier **~110 µs** after VAD detection, with a stale-audio latch so a
   late-finishing TTS can't resurrect the interrupted reply.
5. **VAD `min_volume` default (0.6, pipecat parity) gated out moderate-volume
   speech** — only the loudest tail of utterances passed (the same failure
   class as our production Jabra issue). Fixed with explicit `VadParams`
   (min_volume 0.2, stop_secs 0.5).

**Upstreamed (2026-08-06):** issue
[AreevAI/flowcat#60](https://github.com/AreevAI/flowcat/issues/60)
(Interruption never delivered to `process_frame`; VAD gate never armed;
frame-path stall measurements; plus the factory base_url/require_key and
loopback-bind EINVAL items noted inline) and PR
[AreevAI/flowcat#61](https://github.com/AreevAI/flowcat/pull/61) (the full
duplex patch from `poc/vendor/flowcat-core`, branch
`rolandknight/flowcat:cascaded-full-duplex`, rebased clean on upstream HEAD
= our pinned rev; 302 upstream tests + clippy -D warnings green).
Maintainer responsiveness to these is itself an ADR-0002 data point.

**Verdict for ADR-0002:** the essential full-duplex requirement does NOT
disqualify FlowCat. The engine's primitives held up; every gap was closable
with bounded, additive, upstreamable changes, and the resulting barge-in
stop latency (µs-scale flush after detection) is excellent. The honest
counterweight: we had to *build* what Pipecat ships as one flag, the
interruption layer had clearly never been exercised on the cascaded path,
and endpointing needed a custom gate + STT service.

## Phase 1 full matrix (2026-08-06): T6–T12 run vs the duplex server

Detail in `poc/reports/flowcat-cloud.md`. Verdicts: **T6 concurrency PASS**
(two sessions, zero cross-talk), **T7 teardown PASS** (abrupt kill →
reconnect 0.16 s, no log spin — a class Pipecat failed publicly), **T8
XFAIL as designed** (no idle context wipe: the cascaded builder
hard-disables idle timeout; Babel CONV-5 must be implemented in a
migration), **T9 in-flight PASS** (12 s tool survives), **T10 recorded**
(e2e p50 5.55 s / p95 7.04 s — dominated by CPU whisper + cloud LLM TTFB +
the 0.5 s VAD stop floor; Phase 2 on the Mac judges this for real), **T11
PASS** (tool 500 → graceful spoken degradation, not a dead turn), **T12
soak PASS** (30/30; watch item: RSS +104 MB over 30 turns, likely rolling
context — CONV-5 wipe would bound it). New oddities: the events WS emits no
mappable transcript events mid-session (needs a Rust-side look), and
`ort` INFO logging is noisy (~140 lines/s during sessions).

## Phase 1a/1b implementation notes (E2E pending)

- **Wake (1a):** hand-rolled openWakeWord DSP chain hit a subtle
  mel-alignment mismatch (p=0.0008 vs reference 0.84 — same model, same
  audio); replaced with the vendored `oww_rs` crate (tract-based) which
  matches reference exactly (0.8501). Lesson recorded: reuse validated DSP.
  oww_rs needed light surgery (mic/cpal strip, custom-model-path
  constructor — upstream-PR candidate to that repo). The detector core is
  framework-free and reusable for a future single-binary Rust satellite
  client, which is now an explicit interest.
- **Custom voice (1b):** Chatterbox-TTS-Server runs CUDA on the 6 GB card
  (3.3 GB used) cloning from a synthesized reference clip. FlowCat's
  KokoroTts client can't drive it (422 on `response_format=pcm`; voice =
  reference-WAV filename) — a ~120-line custom `TtsService` was needed,
  vs Pipecat's 139-line subclass for the same backend: **workaround parity,
  neither framework covers this natively.**

## Positive result (major)

**The "wire-ready but unproven" cascaded tool-calling path WORKS.** First
fully-successful server-side turn (paid OpenRouter `gemma-4-26b-a4b-it`,
whisper.cpp tiny.en CPU, Kokoro shim, str0m WebRTC): greeting spoken on
connect, "what time is it" transcribed, `get_current_time` relayed via
`SessionSource::tool_call` to the stub server (verified in its call log),
tool result fed back, reply synthesized. Call-end usage:
`bot_turns=2, user_turns=1, tool_calls=1, input_tokens=2374, output_tokens=30,
tts_characters=31`. Remaining failures at that point were harness-side
(single-use audio stream in the adapter, being fixed).
