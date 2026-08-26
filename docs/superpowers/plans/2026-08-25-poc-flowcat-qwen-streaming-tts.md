# FlowCat PoC: Qwen3-TTS streaming backend (Kokoro stays a startup option)

> **Status 2026-08-25: implemented on branch `poc-flowcat-qwen-tts` (design B, in-process PyO3).** Tasks below are the executed record plus the open follow-ups; the sidecar design (A) is kept at the end as the fallback.

**Goal:** `POC_TTS_BACKEND=qwen` streams Qwen3-TTS to the caller chunk-by-chunk (first chunk ~0.14–0.2 s after the sentence is known) instead of after whole-sentence synthesis. `kokoro` stays the default and pays nothing: no libpython, no venv, no model unless `qwen` is selected at build time.

**Architecture (as built):**
1. **poc-qwen-streaming is a lib+bin crate** (`src/lib.rs`); `flowcat-poc` depends on it by path behind Cargo feature `qwen-tts`. Its `Engine` (one Python thread owning the GIL, mlx-audio's worker owning Metal, `StreamEvent::{Start,Audio,Done,Error}` over `mpsc`, cancel-on-drop after the current chunk) runs inside FlowCat. `config.flowcat.yaml` is the clone-only profile (one model, no transcription).
2. **Vendored flowcat-core streaming seam** — `TtsService::run_tts_stream` (default wraps `run_tts`, so Kokoro/Chatterbox are untouched); `TtsProcessor` forwards frames as they arrive and, with the shared barge-in generation (`with_interrupt_flag`, same as the LLM adapter), drops the stream mid-utterance — measured stop 270–340 ms. Two regression tests.
3. **`poc/flowcat/src/tts_qwen.rs`** — `QwenTts` maps events to `TtsStarted/TtsAudio(20 ms)/TtsStopped`; start-up errors surface as `Err` from the first event; `main.rs` starts the engine, waits for preload, resolves the preset voice from the catalog, caches the `Ready.` greeting; `build.rs` adds libpython's rpath (a dependency's link-arg is not transitive); `platform.sh` sets `PYO3_PYTHON` + the feature for `POC_TTS_BACKEND=qwen`; `setup.sh` creates poc-qwen's venv only then; `run_poc.sh` starts no Kokoro shim and waits up to 900 s for FlowCat's health (model load before bind).

**Measured (Mac Studio M4 Max 36 GB, gemma4:26b resident, Nemotron/Metal STT):**

| | Qwen 1.7B | Qwen 0.6B | Kokoro |
|---|---|---|---|
| preload at FlowCat start | 8.6 s, 4.31 GB active / 6.49 peak | 5.5 s, 2.42 GB / 4.6 peak | — |
| first chunk after `TtsSpeak` (normal turn) | 0.8 s (under memory pressure; 0.22 s standalone) | **0.14 s** | ~0.4 s whole sentence |
| T5 `reply_start_latency` (LLM-bound long reply) | 6.3 s | **4.6–4.8 s** | 6.3 s |
| barge-in stop | 268–336 ms | 273–313 ms | 325 ms |
| smoke+tools+duplex | 7/7 | 7/7 (smoke+duplex rerun) | 7/7 |

**Open issue — memory/GPU headroom on this box, not the integration:** with 26b resident (17 GB, `--no-mmap` after Ollama's own reload) plus Qwen plus Nemotron the machine sits at 21 GB wired / 33 GB used with 19.6 GB of swap. Two consequences seen in the logs: (1) Ollama evicted-and-reloaded the *resident* 26b when Metal "available" shrank below its predicted need (fixed by `OLLAMA_CONTEXT_LENGTH=8192` in `ollama_ctl.py` — 32K predicted 18.0 GB, 8K fits; the PoC prompt is ~1.1 K tokens); (2) LLM requests that overlap a long Qwen synthesis on the same GPU stretch from ~0.9 s to 7–14 s (T5's `second_reply_latency` 9–13 s vs Kokoro 3.4 s), and once a 72 s stall under swap. Kokoro is CPU, so it never collides.

**Follow-ups (ordered):**
- [ ] **Pace TTS generation to playout.** The engine renders a 20 s reply in ~7 s of GPU time; the pipeline needs only real time. Throttle the `event_stream` pull (or the bridge's `interval_s`) so Qwen stays ≤ ~2 s ahead of the carrier — cuts GPU contention with the LLM and makes barge-in cancel cheaper (less wasted audio). Measure T5 `second_reply_latency` before/after.
- [ ] Re-measure with memory headroom (quit Chrome/idle apps or a smaller LLM) to separate contention from paging; record in `poc/README.md`.
- [ ] `test_voice.py`: accept `qwen` (same marvin identity, pitch check).
- [ ] Optional: skip `_warm` when preset voices are preloaded (~2 GB startup peak, redundant on the clone-only profile); `mx.clear_cache()` after preload.

## Global Constraints

- **Zero overhead for `kokoro`:** the Qwen server is a separate process started only by `run_poc.sh` when `POC_TTS_BACKEND=qwen`; the Rust client is plain tokio-tungstenite (no PyO3, no mlx in `flowcat-poc`); `make setup` does not build poc-qwen-streaming or create poc-qwen's venv unless `POC_TTS_BACKEND=qwen` (mirror the existing `nemotron` conditional in `poc/setup.sh`).
- **Do not modify `poc-qwen-streaming/` behaviour** except additive: a `/health`-style readiness signal if `/api/info` is not sufficient (it already reports preload progress — prefer using it), and a `--cancel-on-close` no-op check (the socket close already cancels after the current chunk, engine.rs L114).
- **Audio contract unchanged:** 24 kHz s16 mono into the pipeline (same as Kokoro; `CascadedTransportOutput` resamples to the carrier), frames ≤ 20 ms.
- **Memory (measured 2026-08-25, clone-only profile: 1.7B-Base, `max_resident: 1`, 3 preset voices preloaded, transcription off):** **4.31 GB active, 6.43 GB peak**, preload 11 s (load 3.8 s + warm-up 4.2 s + three short clones). Voices are free — the per-voice ICL cache is kilobytes and they are already primed serially on the single worker thread; the peak is load + warm-up transients. Streaming a 313-char sentence (17.4 s of audio, 56 chunks, TTFA 0.22 s) did **not** move the peak: the 9.9 GB in `poc-qwen/bench-m4-max.md` was whole-utterance decode of long text, which the streaming path never does. Budget with gemma4:26b resident (17 GB) + Nemotron (~1 GB): ~23 GB, inside the 36 GB box. Configure `preload.models: [clone_default]`, `max_resident: 1`, `transcribe.enabled: false`. Watch `sysctl vm.swapusage` — the box was already at 16.6/17.4 GB swap during the measurement (other processes), and poc-qwen-streaming's README documents the swap failure mode (TTFA 8 s / RTF 2.9).
- **Reducing the startup peak further (optional):** the ~2 GB over steady state is (a) `_warm` running `WARMUP_TEXT` through the ICL path with a 1 s noise reference, then (b) the first real reference encoding. Options, cheapest first: skip `_warm` when preset voices are preloaded (each voice's `Hi.` clone already exercises the ICL path and compiles the kernels — one extra warm-up generation is redundant on the clone-only profile); `mx.clear_cache()` after preload (already called on evict/unload, not after preload); `mx.set_cache_limit()` to stop MLX holding freed buffers; or the 0.6B model (~1.7 GB weights, TTFC 0.12 s). Peak is transient — steady state is what matters next to the LLM.
- **Voice:** clone from `voices/marvin.mp3` + `voices/marvin.txt` (same identity the Chatterbox `voice` test asserts by pitch), `size` configurable (`POC_QWEN_SIZE=1.7B|0.6B`; 0.6B halves TTFC to ~0.12 s if latency matters more than quality).
- **Keep `run_tts` (whole-utterance) implemented too** on the Qwen service by collecting the stream — so the service is still a valid `TtsService` for any path that calls the non-streaming method, and unit tests can drive it without a socket.
- Branch `poc-flowcat-qwen-tts` off `poc-mac-nemotron-ollama` (PR 1); commit after every task with `poc(qwen-tts): …` prefixes. All `make`/`pytest` commands run from `poc/`.

---

## Original task list (design A, sidecar) — kept as the fallback plan

### Task 1: Sidecar profile for poc-qwen-streaming (no FlowCat changes yet)

**Files:**
- Create: `poc-qwen-streaming/config.flowcat.yaml` (copy of `config.yaml` with `server.port: 8008`, `preload.models: [clone_default]`, `preload.voices: [marvin]`, `models.max_resident: 1`, `transcribe.enabled: false`)
- Modify: `poc-qwen-streaming/src/config.rs` / `main.rs` only if a `--config <path>` flag does not exist (check first; add it additively)
- Create: `scripts/start_qwen_tts.sh` (builds if `target/release/poc-qwen-streaming` is missing via `make -C poc-qwen-streaming build`, then `exec` serve with the flowcat config; mirrors `scripts/start_nemotron.sh`)

**Steps:**
- [ ] Confirm `/api/info` exposes preload state (`preloading: true/false` or equivalent); note the exact JSON key for Task 2's `wait_health`.
- [ ] `scripts/start_qwen_tts.sh` starts the server; verify `curl :8008/api/info` and that `tests/e2e_ws.py --voice marvin` (or a 5-line `websockets` script sending `generate` clone) receives `start` → binary → `done` with `ttfa_s` ≈ 0.2 s warm.
- [ ] Record memory (`/api/info` active/peak) with gemma4:26b resident in the README table.

### Task 2: `run_poc.sh` + Makefile + `.env` wiring

**Files:**
- Modify: `poc/run_poc.sh` (`TTS_BACKEND` case: `qwen` → `ensure_qwen_tts` = start_proc `qwen-tts` with `scripts/start_qwen_tts.sh`, `wait_health` on `/api/info` until preload done, 600 s; skip Kokoro shim), `poc/setup.sh` (build poc-qwen-streaming only when `POC_TTS_BACKEND=qwen`), `poc/.env.example` (`POC_TTS_BACKEND=kokoro|chatterbox|qwen`, `POC_QWEN_URL=http://127.0.0.1:8008`, `POC_QWEN_VOICE=marvin`, `POC_QWEN_SIZE=1.7B`, `POC_QWEN_INTERVAL_S=0.32`), `poc/Makefile` (`make logs` tails `logs/qwen-tts.log`; `status` lists :8008), `poc/README.md`
- Modify: `poc/flowcat/src/main.rs` — accept `"qwen"` in the `tts_backend` match with `require_nonempty` on `POC_QWEN_URL`/`POC_QWEN_VOICE` (still constructs nothing yet)

**Steps:**
- [ ] `POC_TTS_BACKEND=qwen make up` starts nemotron + stubs + qwen-tts (no kokoro), flowcat still fails cleanly with "qwen backend not implemented" until Task 4 — acceptable intermediate state, or gate Task 2's main.rs change behind Task 4.
- [ ] `make down` stops qwen-tts (it owns a pid file; the `down` exit-wait already handles slow shutdown).

### Task 3: Streaming seam in vendored flowcat-core

**Files:**
- Modify: `poc/vendor/flowcat-core/src/service/mod.rs` — `TtsService::run_tts_stream` default impl (`Ok(stream::iter(self.run_tts(text).await?).boxed())`), plus the `Box<dyn TtsService>` blanket forwarding
- Modify: `poc/vendor/flowcat-core/src/service/adapters.rs` — `TtsProcessor::run` consumes `run_tts_stream`, pushes each frame immediately (map `TtsAudio`→`OutputAudio` as today), emits `Metrics::TtsUsage` after the first frame; `TtsProcessor::with_interrupt_flag` (same shape as `LlmProcessor`) — snapshot the generation at run start, check between chunks, drop the stream on change and push a `TtsStopped` so downstream framing closes
- Modify: `poc/vendor/flowcat-core/src/pipeline/cascaded.rs` — `build_cascaded_call_duplex` passes `interrupt_flag` to `TtsProcessor`
- Tests: adapters.rs — a fake streaming service yields 3 frames with a delay; assert frames reach the observer before the stream ends (streaming, not batched) and that bumping the flag mid-stream stops it after ≤ 1 more frame; existing Kokoro-path tests unchanged

**Steps:**
- [ ] `../bin/cargo test --manifest-path vendor/flowcat-core/Cargo.toml --lib` green (currently 320 tests).
- [ ] Rebuild flowcat-poc with `kokoro`; `pytest harness -m "smoke or tools or duplex"` still 7/7 — proves the default path is behaviour-identical.

### Task 4: `QwenStreamingTts` service

**Files:**
- Create: `poc/flowcat/src/tts_qwen.rs` — `QwenStreamingTts::new(url, voice, ref_text, size, interval_s)`; `run_tts_stream` opens the socket, sends `generate`, maps events → frames (24 kHz, re-chunked to 480 samples), `context_id` `qwen-N`; a `Drop`/stream-abort closes the socket (server cancels after the current chunk); `run_tts` = collect; reuse Chatterbox's `with_ready_pcm` for the cached `Ready.` greeting
- Modify: `poc/flowcat/src/call.rs` — `PocTts::Qwen(...)` arm; `poc/flowcat/src/main.rs` — read the `POC_QWEN_*` config, load `voices/<voice>.txt` as `ref_text`, validate the clip exists
- Tests (offline): parse the `start`/binary/`done`/`error` sequence from a canned event list into the expected frame sequence; 24 kHz re-chunking; error → `Err`; size/interval config plumbing

**Steps:**
- [ ] `POC_TTS_BACKEND=qwen make restart`; `make status` shows :8008; a harness smoke turn plays audio; flowcat log shows `TTS produced` frames arriving over time (debug: log first-chunk latency per `run_tts_stream`).
- [ ] Barge-in: `pytest harness -m duplex` passes with `qwen` (T5 interrupts a long reply mid-stream — the stream must stop within the existing 1 s budget).

### Task 5: Measure and record

**Files:**
- Modify: `poc/harness/results.py` (snapshot `tts_backend=qwen`, `qwen_size`), `poc/harness/test_voice.py` (generalize the pitch check to `chatterbox|qwen` — same marvin identity), `poc/README.md` results table
- Optional: `poc/harness/test_matrix.py` latency marker rows for kokoro vs qwen TTFA (the T5 `reply_start_latency` probe already measures end-of-user-speech → first bot audio; the playground TTFA chips from PR 2 show it live)

**Steps:**
- [ ] Run `smoke`, `tools`, `duplex`, `voice`, `latency` with `kokoro`, then with `qwen` (1.7B and 0.6B); record TTFA (reply_start_latency), RTF, and memory in the README.
- [ ] Success bar: qwen first-audio ≤ kokoro's for a medium sentence (kokoro whole-sentence ≈ 0.4 s synth; qwen streaming first chunk ≈ 0.2 s + WS), no swap (`vm.swapusage` flat), duplex green.

## Integration shape: separate process vs PyO3 in-process

Two viable shapes; the plan above is written for (A) but (B) is simpler code and is the recommendation if the build-time coupling is acceptable.

**(A) Sidecar (poc-qwen-streaming as-is, WebSocket).** Pro: `flowcat-poc` stays a plain Rust binary (no libpython link, no venv at run time) so `kokoro` builds/users pay nothing; the TTS model survives `flowcat-poc` rebuilds/restarts (the harness restarts FlowCat per `make test`; Chatterbox and Nemotron are sidecars for exactly this reason); crash isolation. Con: a wire protocol to implement and keep in sync, a second process to orchestrate, one extra localhost hop (measured ~0 ms in poc-qwen-streaming's bench).

**(B) PyO3 in-process (recommended).** Turn `poc-qwen-streaming` into a lib+bin crate and have `flowcat-poc` depend on it by path behind a Cargo feature `qwen-tts`; `engine.rs` (336 lines: one Python worker thread owning the GIL, `Python::attach` + `py.detach` around the blocking mpsc, `StreamEvent::{Start,Audio,Done,Error}` over `mpsc::Receiver`, cancel-on-drop after the current chunk) is already the exact shape `run_tts_stream` needs — the `QwenStreamingTts` service becomes ~80 lines mapping `StreamEvent` → `TtsStarted/TtsAudio/TtsStopped`, no sockets, no JSON. Preload runs at FlowCat start-up only when `POC_TTS_BACKEND=qwen` (11 s, once per process). Costs: `platform.sh build` must set `PYO3_PYTHON=poc-qwen/.venv/bin/python` when the feature is on (and `make setup` creates that venv only then — same conditional as Nemotron); the binary embeds libpython and needs `config.yaml`'s `python.paths` at run time, so **`kokoro` builds must leave the feature off** to keep the zero-overhead promise (feature-gated `PocTts::Qwen` arm, like the existing `moonshine` feature); every FlowCat restart reloads the model (~11 s) — mitigate by keeping `make restart` rare during TTS work, or accept it; MLX Metal state is per-thread, so the engine's dedicated worker thread must stay (it does). GIL: generation runs on the Python worker thread with the GIL released around channel ops, so the tokio runtime is never blocked — poc-qwen-streaming's bench shows client TTFA equals the engine's request→first-chunk time.

Decision rule: pick (B) unless the ~11 s per-restart reload proves annoying in practice, in which case the sidecar is a mechanical extraction later (the `StreamEvent` boundary is the same either way).

## Risks / decisions to confirm

- **Sentence granularity is already in place** (`AssistantContextAggregator` emits one `TtsSpeak` per assembled utterance), so streaming buys the intra-utterance gap; the first-token→first-sentence wait is unchanged and dominated by the LLM.
- **Playout pacing:** the sink extends its playout estimate per frame; 20 ms frames arriving in 0.32 s bursts is the same shape Kokoro produces (one burst), so no pacing changes expected — verify no gaps at chunk seams (poc-qwen-streaming crossfades seams server-side).
- **Memory headroom** is the real risk with the 26b LLM resident; if `vm.swapusage` climbs, drop to 0.6B (~2.5 GB) before changing anything else.
- **Cancel latency** is bounded by the model's chunk interval (0.32 s) since the server cancels after the current chunk; acceptable inside T5's 1 s stop budget because the carrier flush (reactor) happens independently at VAD detection.
