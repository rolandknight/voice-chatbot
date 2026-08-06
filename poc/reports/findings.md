# FlowCat PoC — running findings ledger

Facts discovered while implementing Phase 1, accumulated for the M4 verdict
report. FlowCat pinned at `37b09bafd6a50cb65936411b40b09e77386e83e3`.

## Framework findings (pre-first-run)

1. **The cascaded path is half-duplex by design.** `TurnMute` in
   `flowcat-core/src/pipeline/cascaded.rs` mutes STT from turn start until the
   reply finishes playing ("No barge-in" is verbatim in the source; a 12 s
   safety unmute covers tool-only turns). Barge-in exists only as frame
   plumbing (`Frame::Interruption` → queue drain + `send_clear`) driven by
   `VadProcessor` — which the cascaded builder does **not** include. T5
   (barge-in) will fail by design in Phase 1. Contrast: Pipecat gives cascaded
   barge-in for free (`allow_interruptions=True`). **This is currently the
   single biggest gap vs Pipecat for Babel's use case.**
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
