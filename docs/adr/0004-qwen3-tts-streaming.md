# ADR-0004: Streaming voice-clone TTS — Qwen3-TTS on MLX, chunk-streamed from a Rust server

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-24 |
| **Decision** | Voice the cloned personas with **Qwen3-TTS-12Hz 1.7B-Base** (bf16, `mlx-audio` on Apple Silicon), generated in **0.32 s streaming chunks** and pushed to the client as raw **24 kHz int16 PCM** the moment each chunk exists. Serve it from a **Rust (axum) process that embeds the Python engine via PyO3** on one dedicated interpreter thread, with models and reference voices **preloaded and warmed at start-up**. Chatterbox (Turbo/Flash) is retired for cloned voices; Kokoro remains only for the legacy default persona until it is re-voiced. |
| **Related** | ADR-0002 (Pipecat orchestration: this engine is a TTS service behind it); ADR-0003 (LLM serving: shares the 36 GB memory budget — the TTS allowance is 4.3 GB). PRD latency budget: wake → first audio ≤ 1.5 s; the LLM stream must never starve the TTS. |

---

## Context

The chatbot speaks through cloned voices: each persona is a 5–15 s reference
clip. Text arrives as a stream from the LLM (ADR-0003: warm first token ≈
0.35 s, 80+ tokens/s) and the user is waiting in silence until the first
audio plays. The TTS engine therefore has to satisfy, in priority order:

1. **Zero-shot clone quality** from the persona clip — at least the
   Chatterbox-Turbo class that shipped.
2. **Low time-to-first-audio (TTFA) on a streaming path that runs on Apple
   Silicon.** The host is a Mac Studio M4 Max (36 GB unified memory, macOS
   26.4) — there is no CUDA. Whole-utterance synthesis is not acceptable:
   first audio must not wait for the last token of the sentence.
3. **Decode faster than real time with margin**, so playback never
   underruns once it has started (real-time factor RTF ≤ ~0.5).
4. Expressive control (style/emotion), a small memory footprint next to a
   ~19 GB resident LLM, and a commercial-friendly licence.

### What was shipping, and the bars it set

| Path | Whole-utterance latency, 30 / 104 / 317-char sentences | First audio | Notes |
| --- | --- | --- | --- |
| Chatterbox Flash, MLX fp16 on this Mac (tuned `drf_block_size 32, num_steps 4, n_cfm_timesteps 1`) | **0.92 / 1.37 / 4.20 s** (RTF ≈ 0.28) | ≈ 0.9 s | Package exposes no streaming API; its S3Gen vocoder and voice encoder run on PyTorch CPU. No expressiveness knob (`exaggeration` absent in Flash). |
| Chatterbox, block-streaming engine on an RTX 2060 (Linux) | — | **0.44 s engine, 0.68–0.74 s in the browser** over WebRTC/Opus | Flat with sentence length. The best streaming number the project had; a Mac candidate had to beat it at Chatterbox clone quality. |
| Chatterbox on PyTorch MPS | RTF 4.55 | — | Unusable; the flow-matching vocoder has no viable MPS path. |

Research bar adopted for the Mac: **TTFA ≤ 0.9 s** (Chatterbox Flash's
first-chunk time), with 0.44 s the number to beat.

### Why "streamable on this Mac" is the hard constraint

Three layers must all hold; most models fail at the third:

1. The **architecture** emits audio before the utterance ends (autoregressive
   codec-token models do; flow-matching non-autoregressive models such as
   F5-TTS, ZipVoice/LuxTTS and IndexTTS's second stage do not — their
   "streaming" is sentence chunking, first packet = one whole generation).
2. A **runtime exposes** it — the official Qwen3-TTS, Chatterbox Flash and
   IndexTTS repositories all return whole files; streaming lives in
   community runtimes.
3. The **Mac port exposes** it — PyTorch MPS is not viable for the
   flow-matching vocoders (S3Gen, HiFT, BigVGAN); every fast Mac port
   re-implements the vocoder in MLX, CoreML or ggml.

## Research (August 2026)

A web survey (six parallel sweeps, ~550 pages) plus the project's own
measurements produced a ranked shortlist rather than a winner, because
**nobody had published a Mac TTFA for any cloning-capable model**; every
sub-0.9 s Apple-Silicon number belonged to a non-cloning or ~100 M model.

Clone quality (Seed-TTS-eval `test-en`, WavLM speaker-similarity — only
comparable within one table; human reference SIM 0.734):

| Model | WER % ↓ | SIM ↑ | Streams + clones on a Mac? |
| --- | --- | --- | --- |
| dots.tts SOAR 2B | 1.30 | 0.771 | one single-maintainer MLX port streams it |
| VoxCPM2 2B | 1.84 | 0.753 | MLX at or below real time (8-bit 0.85× RT) |
| **Qwen3-TTS-12Hz-1.7B-Base** | **1.23** | 0.717 | **yes — `mlx-audio` documents 0.32 s token-chunk streaming; also speech-swift, audio.cpp** |
| VoxCPM1.5 0.8B | 2.12 | 0.714 | mlx-audio; streaming undocumented |
| Chatterbox Flash 0.5B | 2.04 | 0.704 | no (whole-utterance) |
| Chatterbox 0.5B (Turbo's base) | 2.20 | 0.685 | mlx-audio Turbo port streams token chunks; unmeasured, known seam clicks |
| Higgs Audio v3 4B | (v2: 2.44) | (v2: 0.677) | streams; **non-commercial licence** |
| Kyutai Pocket TTS 100M | n/a | n/a | ~200 ms first chunk on CPU; no speed/emotion control; SIM unpublished |

Shortlist and outcome:

| Rank | Candidate | Why | Outcome |
| --- | --- | --- | --- |
| 1 | **Qwen3-TTS-12Hz 1.7B / 0.6B Base** (Alibaba, Jan 2026, Apache-2.0) | best WER of anything with a Mac streaming path; SIM ≥ Chatterbox; the only model whose maintained MLX port documents chunk streaming; three sibling checkpoints (Base = clone, CustomVoice = presets + style instruct, VoiceDesign = voice from a description) | **Benchmarked, adopted** — results below |
| 2 | Chatterbox Turbo via mlx-audio | the incumbent voice end-to-end on Metal; cheapest experiment | Not run: superseded once Qwen3-TTS beat the bar by ~5× with better WER |
| 3 | VoxCPM 1.5 / 2 | highest SIM of any AR model; style instruct | Second round only; the 2B is ≤ real time on MLX |
| 4 | ZONOS2 (8B MoE, 0.9B active) | explicit emotion vectors and rate buckets | Second round; Q8 is 8.5 GB against the LLM budget |
| 5–7 | dots.tts, Higgs v3, Pocket TTS | SIM / control / latency floor respectively | Licence (Higgs), single-maintainer ports (dots), no control (Pocket) |

Rejected outright for one of: cannot clone a user clip (Kyutai TTS 1.6B,
VibeVoice-Realtime, Voxtral), no Mac streaming (CosyVoice 3, IndexTTS 2.5,
F5-TTS, XTTS, Fish S2 Pro — RTF 3.77 on MLX), or CUDA-only (Dia2, Gepard,
VoXtream2, MOSS-TTS-Realtime).

Rust ports of Qwen3-TTS were also evaluated for a pure-Rust server: the
second-state MLX port runs at RTF 1.5–3.4 on an M4 (slower than real time)
and the candle port is an unmeasured experiment. The Python `mlx-audio`
engine (RTF 0.28–0.38 measured) stays.

## The engine

Qwen3-TTS-12Hz: a Qwen3 "talker" language model emits 12.5 Hz, 16-codebook
audio tokens which a causal convolutional decoder turns into 24 kHz audio —
no diffusion stage, which is why it streams naturally. Cloning is in-context:
the reference clip is tokenised together with its **transcript** and
prepended to the prompt (a transcript-free "x-vector only" mode exists and is
faster, 0.73 s vs 1.08 s for a short sentence, but is a weaker match).

Checkpoints (Hugging Face, `mlx-community`, bf16):

| Role | Model | Size on disk | Resident |
| --- | --- | --- | --- |
| Persona cloning (default) | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | ≈ 3.5 GB | **≈ 4.3 GiB active**, 9.9 GiB peak alone during first warm-up |
| Cloning, small (fallback) | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | ≈ 1.5 GB | 8.0 GiB peak during warm-up |
| Preset speakers + style instruct | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | ≈ 3.5 GB | ≈ 4.3 GiB |
| Voice from a text description | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | ≈ 3.5 GB | ≈ 4.3 GiB |

Runtime pins that produced the results: macOS 26.4, Python 3.12.14 (mise),
`mlx 0.32.1`, `mlx-audio 0.5.0` (constraint `>=0.5.0,<0.6`), `mlx-whisper
0.4.3` (auto-transcription of reference clips), Rust via cargo with `pyo3`
(embedded interpreter), `axum`/`tokio`.

## Serving architecture

```
client ──WebSocket /ws──▶ axum (tokio) ──channel──▶ "python" OS thread (holds the GIL)
       ◀── int16 PCM 24 kHz frames ◀── channel ◀── Bridge.stream() generator
                                                        └─▶ Qwen3Engine ─▶ mlx-worker daemon thread (Metal)
```

Design rules, each of which was learned from a failure:

1. **All MLX calls run on one persistent daemon thread.** MLX keeps
   per-thread Metal state; running it from short-lived worker threads (a web
   framework's thread pool) segfaulted the process when threads were
   recycled between requests. Every generation is queued to that thread and
   results are returned through a queue/future.
2. **One OS thread owns the Python interpreter** in the Rust process. Web
   handlers send commands over a channel and await results; no Python object
   is ever touched from an async worker. Blocking waits inside Python
   release the GIL, so the engine thread and the Metal thread never deadlock.
   Dropping a request's receiver sets a stop event the generator checks per
   chunk (cancellation / client disconnect).
3. **Embedded interpreter wiring.** The Python used at build time must be
   the one whose site-packages hold `mlx-audio` (shared `libpython`; its
   directory goes on the binary's rpath). The binary sets `sys.executable`
   to that interpreter (an embedded interpreter otherwise reports the Rust
   binary, and any library that spawns `sys.executable -c …` would launch a
   second server) and prepends the engine's site-packages and package
   directories to `sys.path` at start-up so nothing needs activating.
4. **Transport is raw PCM over WebSocket, not Opus/WebRTC**, for the demo
   and LAN path: lowest possible TTFA, no codec build. Client-side playback
   schedules each frame back-to-back from the first with Web Audio; the
   client also assembles the full take into a WAV for replay. WebRTC remains
   the transport for remote satellites (ADR-0002) and adds ≈ 0.25–0.3 s
   (measured on the CUDA path) that generation cannot remove.
5. **Sentence chunking + crossfade.** Text is split at sentence boundaries
   into chunks of ≤ 300 characters (never inside a number or abbreviation);
   each chunk is one model call with the same reference, which keeps every
   Metal call well under the GPU watchdog (~40 s of audio / ~500 tokens per
   call). Chunk seams are crossfaded 20 ms; within a chunk, mlx-audio's
   decoder keeps streaming state so its 0.32 s chunks need no crossfade
   (seam sample jumps measured within in-speech range: max |Δ| 0.11).
6. **Preload and warm at start-up, before serving.** Right after the port
   binds: load each configured model, run a short generation to compile
   Metal kernels *through the in-context-cloning path* (a generic warm-up
   left the first real clone at 5.9 s), then run one tiny clone per preset
   voice so mlx-audio's per-model reference cache (keyed on transcript +
   audio; kilobytes per voice) is primed. Cost ≈ 8 s once; without it the
   first request paid ≈ 6 s (4 s model load + kernel compilation +
   reference encoding). A clip that was not preloaded still pays ≈ 0.5–1 s
   on first use.
7. **Model residency is an LRU** with a configurable cap; all three demo
   checkpoints resident cost 12.8 GiB active / 14.3 GiB peak. **Production
   keeps one model — 1.7B-Base — resident (≈ 4.3 GiB), which is the 4.3 GB
   TTS allowance in ADR-0003's memory budget.**

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| `streaming_interval` | **0.32 s** of audio per chunk (4 talker tokens) | measured TTFC 0.18 s (1.7B) / 0.12 s (0.6B); 0.64 s raised it to 0.28 / 0.20 s for slightly lower total time |
| Sampling | temperature 0.9, top_p 0.9 | model-card defaults; produce intelligible, verbatim-transcribing output |
| Language | English (explicit, not auto) | |
| Reference | clip 5–15 s + accurate transcript | the transcript is *required* for in-context cloning quality; a wrong transcript tail leaked into output ("Here you go") in 2 of ~15 runs |
| `max_chunk_chars` | 300 | Metal watchdog; also the unit of LLM-sentence → TTS hand-off |
| `crossfade_ms` | 20 | chunk seams |
| `max_resident` | 1 in production (3 in the demo) | memory budget |
| Preload | models + all preset voices at start-up | first-request TTFA 0.185 s instead of ≈ 6 s |
| Sample format on the wire | int16 PCM, 24 kHz, mono, binary frames | |
| Telemetry | per generation: `ttfa_s` (request → first chunk), `gen_s`, `audio_s`, `rtf`, `chunks`, `cold` | every number below comes from this |

Persona configuration changes shape: Chatterbox's numeric `exaggeration` /
`cfg_weight` knobs do not exist here. Expressiveness is a free-text
**style instruction** (CustomVoice / VoiceDesign checkpoints; e.g. "very
happy and excited", "whispering"), which suits an LLM that can emit a
per-utterance style hint alongside the text. Each persona declares: reference
clip, transcript, model size, optional instruct string.

## Test strategy

| Layer | What | GPU |
| --- | --- | --- |
| Go/no-go smoke | clone the `one-one` reference with 0.6B, write a WAV, print wall time — if mlx-audio's Qwen3-TTS path does not produce intelligible audio on the box, stop | yes |
| Unit (engine) | mlx-audio's loader mocked with a fake model that yields a 24 kHz tone (0.05 s per character): LRU registry, kwarg mapping per tab (clone / custom voice / design), timings with forced evaluation, x-vector path, sentence chunking (900-char paragraph → ≥ 3 chunks; total length = parts − overlaps) | no |
| Unit (bridge + Rust) | streaming generator with the fake model: chunking, crossfade hold-back, stop flag, kwarg mapping; Rust unit tests for PCM conversion and the command protocol | no |
| Whole-utterance bench | the same three sentences as the Chatterbox bench (30 / 104 / 317 chars), `one-one` reference, 3 repeats, first (cold) repeat excluded, 0.6B and 1.7B; peak memory recorded | yes |
| Streaming spike | `stream=True` at 0.32 s and 0.64 s intervals: time-to-first-chunk, chunk cadence, seam continuity (sample-jump statistics), re-transcription with Whisper equal to the whole-utterance output | yes |
| Headless server bench | the Rust binary's `bench` subcommand: the three sentences through the embedded engine, warm, 2 repeats → TTFA / gen / audio / RTF | yes |
| End-to-end over WebSocket | a client connects to the running server, requests a clone, measures TTFA at the client and compares with the server's own timing | yes |
| Quality | listen to the saved WAVs; Whisper re-transcription must match the input text verbatim | yes |

## Results — 2026-08-24, Mac Studio M4 Max, 36 GB

Whole-utterance (warm, `one-one` clone, median of 2 warm repeats):

| Sentence | Qwen3-TTS 0.6B | Qwen3-TTS 1.7B | Chatterbox Flash MLX fp16 |
| --- | --- | --- | --- |
| short (30 chars) | 1.04 s, RTF 0.35 | 1.04 s, RTF 0.49 | 0.92 s |
| medium (104 chars) | 1.82 s, RTF 0.29 | 2.21 s, RTF 0.38 | 1.37 s |
| long (317 chars) | 5.08 s, RTF 0.28 | 6.49 s, RTF 0.36 | 4.20 s |

Qwen3-TTS produces 15–25 % more audio per sentence (slower, more natural
delivery: 6.1 s vs ≈ 5 s for the medium sentence), which is most of the
wall-clock gap; **per second of audio the 0.6B model is the fastest engine
measured on this box.** Whole-utterance latency is not the number that
matters — streaming is:

Streaming, in-process (medium sentence, warm):

| Model | Chunk interval | Time to first chunk | Cadence | Total | Audio |
| --- | --- | --- | --- | --- | --- |
| 0.6B | 0.32 s | **0.12 s** | 83 ms per 320 ms chunk | 1.75 s | 6.5 s |
| 0.6B | 0.64 s | 0.20 s | 160 ms per 640 ms chunk | 1.56 s | 6.0 s |
| 1.7B | 0.32 s | **0.18 s** | 106 ms per 320 ms chunk | 1.96 s | 5.6 s |
| 1.7B | 0.64 s | 0.28 s | 205 ms per 640 ms chunk | 1.70 s | 5.0 s |

Generation runs at 3–4× real time, so playback never starves after the
first chunk. Cold first call (kernel compilation) was 2.6–4.4 s — absorbed by
the start-up warm-up.

Streaming through the Rust server (1.7B-Base, `one-one`, warm):

| Path | TTFA | Generation | Audio | RTF |
| --- | --- | --- | --- | --- |
| Whole-utterance GUI (medium sentence), for reference | 2.21 s | 2.21 s | 5.6 s | 0.38 |
| In-process streaming spike | 0.18 s | 1.96 s | 5.6 s | — |
| **Rust ← PyO3 ← engine, headless bench (short / medium / long)** | **0.178 / 0.184 / 0.187 s** | 0.79 / 1.98 / 6.7 s | 2.1 / 5.6 / 19.4 s | 0.35–0.38 |
| **WebSocket client, localhost** | **0.182 s** (0.124 s with 0.6B) | 1.94 s | 5.5 s | 0.35 |
| First request of a fresh process, with preload | 0.185 s | | | |
| First request of a fresh process, without preload | 6.1 s | | | |

- The PyO3 hop and the WebSocket add nothing measurable: client TTFA equals
  the server's request-to-first-chunk time to the millisecond on localhost.
- **The 0.9 s research bar is beaten ≈ 5×, and the CUDA block-streaming
  engine's 0.44 s ≈ 2.4×.** The 0.18 s floor is the model's own
  time-to-first-chunk at a 0.32 s interval.
- Long input (317 chars) streams as 57–63 chunks; sentence-chunk seams are
  crossfaded; the saved WAVs re-transcribe verbatim.
- Memory: one 1.7B model ≈ 4.3 GiB active; all three ≈ 12.8 GiB active /
  14.3 GiB peak of MLX's 28 GiB recommended working set.
- Failure signature worth knowing: one run showed TTFA 8.1 s / RTF 2.9 on a
  warm model **because the box was swapping** (29 of 30 GB swap in use with
  a stray LLM server holding 5 GiB and the compressor 12 GB); the next runs
  were 0.54 s then 0.19 s as weights paged back in. Check swap before
  blaming the engine.

Clone quality (by ear and by Whisper): the `one-one` clone re-transcribes
verbatim for every bench sentence with both sizes. No blind A/B against
Chatterbox was run; the survey's SIM ordering (Qwen3-TTS 0.717 ≥ Chatterbox
Flash 0.704 ≥ Chatterbox 0.685) is the basis for calling it at least
Chatterbox-class.

## Decision

1. **Engine:** Qwen3-TTS-12Hz **1.7B-Base** (bf16) through `mlx-audio` for
   all cloned personas; 0.6B-Base is the configured fallback if the memory
   budget tightens (saves ≈ 2 GB; TTFA 0.12 s; slightly weaker clone).
   CustomVoice / VoiceDesign checkpoints are loaded only where a persona
   needs a preset speaker or a designed voice.
2. **Streaming:** `stream=True`, `streaming_interval = 0.32 s`; sentence
   chunks ≤ 300 chars; 20 ms crossfade at sentence-chunk seams; raw int16
   PCM 24 kHz frames pushed as generated.
3. **Process model:** Rust (axum/tokio) server embedding the Python engine
   via PyO3; one interpreter thread; one MLX worker thread; command/response
   channels; stop events for cancellation. The engine seam is a trait so a
   Rust engine or an out-of-process sidecar can replace the embedded Python
   later without touching the transport.
4. **Start-up contract:** preload the configured models, warm the
   in-context-cloning path, prime every preset voice's reference cache;
   advertise readiness only afterwards. Budget ≈ 8 s per model.
5. **Residency:** one 1.7B model resident in production (≈ 4.3 GiB); the LRU
   cap is 1. This is the TTS line item in the shared memory budget.
6. **Reference voices:** each persona's clip ships with a verified transcript
   (auto-transcribed with Whisper, then corrected by ear). Cloning without
   the transcript is exposed as an explicit option, not a default.
7. **Chatterbox** (Turbo server, Flash package) is removed from the
   cloned-voice path. **Kokoro** stays only for the legacy default persona
   until that persona is re-voiced with a Qwen3-TTS preset or clone.
8. **LLM hand-off:** the pipeline sends the TTS a sentence as soon as the LLM
   stream closes it (`. ? !`); with LLM decode at 80+ tok/s and TTS
   generating at 3–4× real time, the first spoken word follows the LLM's
   first sentence by ≈ 0.2 s and the stream never underruns.

## Consequences

- Time-to-first-audio on the local/LAN path drops from ≈ 0.9 s (Chatterbox
  Flash) / 2.2 s (whole-utterance Qwen) to **≈ 0.18 s** (+ ≈ 0.25–0.3 s on
  WebRTC satellites), well inside the wake → first audio budget together with
  ADR-0003's 0.35 s LLM first token.
- Memory: ≈ 4.3 GiB instead of Chatterbox Turbo's ≈ 3–4 GB plus the Flash
  process; the three-checkpoint demo configuration (12.8 GiB) is **not** a
  production configuration.
- Persona schema changes: numeric `exaggeration` / `cfg_weight` are replaced
  by an optional style instruction and a required transcript.
- Operational rules that must survive into the main application: the single
  MLX thread, the single Python thread, `sys.executable`, stop events, the
  start-up warm-up, and the swap check when latency looks wrong.
- The Metal watchdog bounds any single generation to ≈ 40 s of audio; the
  300-char chunking keeps every call far below it.
- Dependency exposure: `mlx-audio` releases every 1–2 weeks with ~70 open
  issues; versions are pinned (`>=0.5.0,<0.6`) and re-benchmarked on bump.

## Re-evaluation triggers

- Streaming TTFA at the client > 0.4 s (LAN) or seam artefacts audible after
  an `mlx-audio` upgrade: pin back, re-run the streaming spike and headless
  bench.
- Memory pressure (swap growth with LLM + STT resident): switch the default
  to 0.6B-Base (≈ 2 GB less) before touching the LLM.
- Clone quality complaints on a persona: verify the transcript first (the
  known failure mode), then trim the clip; only then consider VoxCPM 1.5 /
  VoxCPM2 (higher SIM, style instruct) or ZONOS2 (explicit emotion/rate
  control) as the next candidates to bench on the box.
- A Rust-native Qwen3-TTS port reaching RTF ≤ 0.5 on Metal: replace the
  embedded Python behind the engine seam.
- A Chatterbox Nano (110 M) MLX port with streaming: candidate for a
  sub-100 ms "instant filler" voice, not a replacement.

## Sources

- Qwen3-TTS: [Qwen3-TTS-12Hz-1.7B-Base (mlx-community)](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16); [Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts); [Qwen3-TTS demo Space](https://huggingface.co/spaces/Qwen/Qwen3-TTS); [mlx-audio](https://github.com/Blaizzy/mlx-audio) (Qwen3-TTS streaming, ICL cache; server `stream:true` seam bug #898); [speech-swift](https://github.com/soniqo/speech-swift); [audio.cpp](https://github.com/0xShug0/audio.cpp).
- Clone-quality evidence: [VoxCPM2 report (Seed-TTS-eval cross-model table)](https://github.com/OpenBMB/VoxCPM); Chatterbox Flash paper v3 (Resemble AI, 2026-08-21; Seed-TTS-eval table); [Artificial Analysis TTS arena](https://artificialanalysis.ai/text-to-speech/arena).
- Chatterbox: [chatterbox-flash](https://github.com/resemble-ai/chatterbox); [Chatterbox Turbo streaming clicks (HF discussion #18)](https://huggingface.co/ResembleAI/chatterbox-turbo/discussions/18); [silent runaways #531](https://github.com/resemble-ai/chatterbox/issues/531).
- Alternatives: [VoxCPM](https://github.com/OpenBMB/VoxCPM); [ZONOS2 / zonos2.cpp](https://github.com/Zyphra/Zonos); [dots.tts](https://github.com/rednote-hilab/dots.tts); [Higgs Audio v3](https://github.com/boson-ai/higgs-audio); [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts); [Kyutai TTS clone encoder unreleased #404](https://github.com/kyutai-labs/delayed-streams-modeling/issues/404).
- Rust ports considered: second-state's Qwen3-TTS MLX port in Rust (RTF 1.5–3.4 on M4); TrevorS's candle port (unmeasured on Metal).
- MLX: [mlx](https://github.com/ml-explore/mlx); [PyO3](https://pyo3.rs).
