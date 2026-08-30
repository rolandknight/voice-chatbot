# ADR-0005: Streaming speech-to-text — NVIDIA Nemotron Speech Streaming 0.6B via NeMo-Speech.cpp

| | |
|---|---|
| **Status** | Accepted — conditional on the Mac acceptance gates in "Test strategy" (the engine has been validated end-to-end on Linux/CUDA; the Apple-Silicon latency and memory numbers have not yet been measured on the reference host) |
| **Date** | 2026-08-24 |
| **Decision** | Replace batch Whisper (whisper-tiny.en on MLX, one decode per utterance) with **`nvidia/nemotron-speech-streaming-en-0.6b`**, a cache-aware streaming FastConformer-RNNT, served by the **NeMo-Speech.cpp** sidecar (Q8 GGUF, Metal on the Mac Studio, CUDA on the Linux dev box) over a localhost **WebSocket realtime session**. The pipeline's own VAD keeps ownership of turn boundaries: interim hypotheses are display-only; the VAD falling edge commits the buffer and exactly one final transcript reaches the LLM. Operating point: **560 ms right context** (`rnnt_right_context=6`). |
| **Related** | ADR-0002 (Pipecat orchestration: this is the STT service behind it; the FlowCat PoC exercised the same contract); ADR-0003 (LLM: shares the 36 GB budget, STT allowance < 1 GB); ADR-0004 (TTS). PRD: STT final transcript ≤ 0.5 s after speech end (p99); wake → first audio ≤ 1.5 s; fully local by default. |

---

## Context

Speech-to-text sits between the user's last word and the LLM's first token.
Its contribution to perceived latency is the time from **end of speech to
the final transcript**; everything else in the turn (LLM first token ≈
0.35 s, ADR-0003; TTS first audio ≈ 0.18 s, ADR-0004) is downstream of it.

### What was shipping

`mlx-community/whisper-tiny.en-mlx` on Apple Silicon: the VAD (Silero,
0.2 s stop threshold) segments a turn, then Whisper decodes the **whole
utterance in one batch call**. Measured on the reference host: time to
final transcript **≈ 0.28 s** typical, **≤ 0.5 s p99** (the value the
pipeline's timeout is tuned to). Larger Whisper variants (base.en,
small) improve accuracy at 2–4× that latency.

Limitations that motivated the change:

1. **Latency scales with utterance length.** A batch decoder cannot start
   until the VAD closes the turn; the decode then costs a fraction of the
   utterance duration. Long requests ("play yesterday's Today programme
   from BBC Sounds and then set a timer for the pasta") pay the most.
2. **No interim text.** Nothing can be shown, pre-fetched or wake-checked
   while the person is still speaking; the browser/satellite transcript
   pane updates only after the turn.
3. **Accuracy of `tiny.en`** on proper nouns and media titles (stations,
   shows, artists) is the weakest link in tool-argument fidelity; the
   bigger Whisper models that fix it blow the 0.5 s budget.
4. Whisper hallucinates on silence/noise tails and has no native
   punctuation-and-capitalisation guarantee in the tiny variant.

### Host and budget

Reference host: Mac Studio M4 Max, 36 GB unified memory, macOS 26.4;
resident neighbours: LLM ≈ 19 GB (ADR-0003), TTS 4.3 GB (ADR-0004),
pipeline + wake word + VAD + media player ≈ 1.5 GB, OS reserve ≈ 5 GB.
**STT allowance: < 1 GB resident.** Development/validation box: Linux
(Pop!_OS), RTX 2060 6 GB, where the GPU is shared with the Chatterbox TTS
server during PoC runs.

## Candidates (August 2026)

The FlowCat PoC (a Rust cascaded-pipeline validation, ADR-0002) was
built with three runtime-selectable local STT backends so the same
end-to-end harness could compare them:

| Backend | Model | Streaming | Runs on | Notes |
| --- | --- | --- | --- | --- |
| **Whisper** (whisper.cpp) | `ggml-base.en` / `tiny.en` | no — batch after VAD | CPU / Metal / CUDA (needs `nvcc` for CUDA) | the baseline; identical behaviour to the production Whisper-MLX path |
| **Moonshine Medium Streaming** (v0.1.3 native library, `medium-streaming-en` quantised 2026-07-30) | ~medium | yes — interim hypothesis every ≈ 250 ms (native cache floor 200 ms) | **CPU only** | leaves the GPU free; optional `keyterms` spelling bias; English only |
| **Nemotron Speech Streaming EN 0.6B** (NeMo-Speech.cpp v0.1.0, released 2026-08-19) | 600 M FastConformer-RNNT, Q8 GGUF (≈ 0.7 GB) | yes — cache-aware chunked encoder, interim deltas + committed final | CPU / **Metal** / CUDA / Vulkan (prebuilt archives, no toolchain needed) | native punctuation & capitalisation; 4 latency settings; phrase boosting API |

Also considered and excluded:

- **Parakeet TDT/CTC** (same runtime): higher raw accuracy but offline
  models — no streaming partials; would be a batch replacement for Whisper,
  not a change of latency class.
- **Kyutai STT** (delayed-streams): streams, but the Mac MLX path was
  reported choppy and it offers no phrase boosting or punctuation control.
- **Cloud STT** (Deepgram, etc.): violates the local-by-default requirement.
- **Full-duplex speech-to-speech models** (NemotronLabs VoiceChat 11B,
  PersonaPlex 7B): they remove STT entirely, but VoiceChat has a single
  fixed voice (no cloning) and PersonaPlex has no native tool calling; both
  need ~10–12 GB on the Mac. Tracked separately; not an STT decision.

### Why Nemotron over Moonshine

| | Nemotron 0.6B | Moonshine Medium Streaming |
| --- | --- | --- |
| Latency setting | selectable 80 / 160 / 560 / 1120 ms right context; WER published per setting | fixed ≈ 200–250 ms update cadence |
| Accuracy (published, OpenASR-leaderboard datasets) | **6.93 % avg WER** at 1.12 s, 7.07 % at 0.56 s, 8.43 % at 0.08 s; LibriSpeech clean 2.3–2.8 % | no comparable published table |
| Punctuation / capitalisation | native | post-processing |
| Accelerator | Metal on the Mac (the STT should not compete with the LLM/TTS for CPU) | CPU only |
| Runtime | NVIDIA-maintained C++ runtime (Apache-2.0 code), OpenAI-compatible HTTP + realtime WebSocket, Riva gRPC, C API | vendor SDK, C API via Cargo feature |
| Model licence | NVIDIA Open Model License | Moonshine licence |
| Extras in the same runtime | Silero VAD, streaming diarization (Sortformer v2, up to 4 speakers — relevant to the PRD's within-session diarization item), MagpieTTS, translation | — |

Moonshine remains the documented CPU-only fallback (one-line config
change) for hosts without a usable GPU.

## The engine

**Model.** `nvidia/nemotron-speech-streaming-en-0.6b`: 24-layer
cache-aware FastConformer encoder + RNNT decoder, 600 M parameters, English
(en-US), 16 kHz mono input, 80 ms frames, native punctuation and
capitalisation, trained on ≈ 530 k hours. The encoder keeps a cache of
past context and looks ahead a configurable number of **right-context
frames**, which sets the latency/accuracy trade-off:

| `rnnt_right_context` | Window | Avg WER | LibriSpeech clean | AMI (meetings) |
| --- | --- | --- | --- | --- |
| 0 | 80 ms | 8.43 % | 2.80 | 18.29 |
| 1 | 160 ms | 7.67 % | 2.56 | 14.71 |
| **6** | **560 ms** | **7.07 %** | **2.46** | **11.88** |
| 13 | 1120 ms | 6.93 % | 2.32 | 11.73 |

560 ms is the operating point: within 0.14 WER points of the maximum
context, and — because the pipeline's VAD closes the turn 200 ms after
speech ends and the final decode is then forced by a commit — the right
context is mostly *already available* by the time the final is requested.
The 80/160 ms settings buy nothing for a VAD-committed final and cost
1.4–1.5 WER points.

**Runtime.** NeMo-Speech.cpp v0.1.0 (GGML-based; pinned by version and
SHA-256 per platform archive: `macos-aarch64-metal`, `macos-aarch64-cpu`,
`linux-x86_64-cuda`, `linux-x86_64-cpu`). Model artefact
`nemotron-speech-streaming-en-0.6b.q8_0.gguf`, pulled by `nemo-speech pull
nemotron-en` into a project-local model directory. Nothing is added to the
user's PATH; the binary lives under a project-local `.deps` directory.

**Memory.** Q8 GGUF ≈ 0.7 GB on disk. No resident-set figure is published
for NeMo-Speech.cpp on Metal; the closest analogue is the MLX INT8 port of
the same model in speech-swift: **≈ 0.8–1.0 GB peak RSS, median RTF 0.037**
(≈ 27× real time) on Apple Silicon. The < 1 GB allowance in the shared
budget is therefore expected to hold; it is measured, not assumed, in the
acceptance gates below.

## Serving architecture

```
mic / WebRTC ─▶ pipeline (Silero VAD owns turn boundaries)
                   │ 16 kHz PCM16 frames, queued to a worker task
                   ▼
   ws://127.0.0.1:8178/v1/realtime  ──▶  NeMo-Speech.cpp sidecar (nemotron-en Q8, Metal)
                   ▲ interim deltas (display only)
                   ▲ committed final (one per VAD turn) ──▶ LLM
```

Design rules (each carried from the FlowCat validation):

1. **The sidecar owns the resident model; the pipeline owns the turn.** The
   server is started with **endpointing disabled**
   (`--asr.endpointing.enable=false`) and the session sets
   `endpointing_ms: 0`, so the model never decides a turn is over. The
   pipeline's VAD falling edge (0.2 s of silence, the same threshold the
   production chatbot uses) sends `input_audio_buffer.commit`; the
   `…transcription.completed` event that follows is the **single
   authoritative final** for the turn. Interim
   `…transcription.delta` events update the UI transcript pane and are
   never sent to the LLM. Two decoders are never run in shadow.
2. **One realtime socket per call, reused across turns.** Connect once
   (`session.created` → `session.update` → `session.updated`), then stream
   binary little-endian PCM16 frames; on the VAD edge commit; on barge-in or
   cancellation `input_audio_buffer.clear`. Reconnect is lazy on error.
3. **Audio writes go through a queue to a worker task** so decoder time and
   socket back-pressure can never stall the audio input processor (the
   PoC's first version blocked the input path).
4. **Barrier semantics on flush.** A flush while a previous commit is still
   pending is an error, not a queue — it means the VAD fired twice without a
   final in between and the pipeline must surface it rather than silently
   merge turns. Timeouts: 15 s connect, 30 s request.
5. **Readiness gate.** The pipeline polls `GET /ready` (returns device and
   loaded capabilities; 503 until the engine is up) before accepting calls,
   so the first utterance never hits a loading model.
6. **Server flags** (Mac): `nemo-speech serve --asr-model nemotron-en
   --device metal --port 8178 --threads 2
   --asr.streaming.rnnt_right_context=6 --asr.endpointing.enable=false`,
   bound to 127.0.0.1 only, batching disabled to minimise latency.
   `--device cuda:0` on the Linux box; `cpu` for a comparison run.
7. **Session fields** sent by the client: `sample_rate 16000`, `language
   en-US`, `automatic_punctuation true`, `word_timestamps false`,
   `speaker_diarization false`, `endpointing_ms 0`; optional
   `speech_contexts: [{phrases: [...], boost: 3.0}]`.
8. **Phrase boosting is configured but not relied on.** The API accepts
   `speech_contexts` (station names, show titles, artists, persona names),
   but NeMo-Speech.cpp v0.1.0 reports that the published Q8 artefact lacks
   the tokenizer data boosting needs, so it is a no-op today. Accuracy on
   media vocabulary must be met without it; boosting is a future upgrade
   when NVIDIA republishes the artefact.

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| Model | `nemotron-en` → `nemotron-speech-streaming-en-0.6b.q8_0.gguf` | English-only deployment; Q8 is the published quantisation |
| Runtime | NeMo-Speech.cpp **v0.1.0**, archive SHA-256 pinned per platform | early release, "API may evolve" — never float the version |
| Device | `metal` (Mac Studio) / `cuda:0` (Linux) / `cpu` (fallback, comparison) | |
| Right context | **6 frames = 560 ms** (`0`/`1`/`13` selectable; `-1` = model maximum) | accuracy/latency operating point, see table |
| Endpointing | **off** in the server and the session | the pipeline's VAD owns turn boundaries |
| VAD stop threshold | 0.2 s | matches the production chatbot |
| Threads | 2 | the sidecar must not steal CPU from the pipeline; the GPU does the work |
| Port / bind | 8178, 127.0.0.1 | never exposed on the LAN |
| Sample format | 16 kHz mono PCM16 LE, binary frames | the model's native rate; resample once at the transport edge |
| Punctuation | on | the LLM receives natural text; better tool-argument extraction |
| Speech contexts | optional, boost 3.0, kept short | inert until the artefact carries tokenizer data |
| Timeouts | connect 15 s, request 30 s | |

## Test strategy

**Validated (FlowCat PoC harness, Linux/CUDA, 2026-08-20).** The harness
drives the full cascaded pipeline over WebRTC with fixture WAVs (16 kHz
mono, ≈ 300 ms leading / 1.2 s trailing silence), asserts tool calls from
stub-service logs (the primary signal, immune to STT/TTS fuzz), and
transcribes the bot's audio to check content. With
`STT_BACKEND=nemotron` on CUDA:

| Test | Result | Reading |
| --- | --- | --- |
| T5 barge-in (duplex): long reply interrupted by a second utterance | **pass** — barge-in stop 304 ms; reply-start 11.5 s; second reply 4.6 s | The barge-in stop time (audio halts within ~0.3 s of the user speaking) is the STT-relevant number and matches the Whisper runs (288–317 ms). The reply-start figures are dominated by the PoC's cloud LLM (Claude Haiku via OpenRouter) and CUDA Chatterbox on a shared 6 GB card, not by STT; the Whisper-CPU runs on the same day measured 13.4–14.1 s on that metric. |
| Streaming partials | interims appear while speaking; exactly one final per VAD turn reaches the LLM | protocol contract holds |
| Turn ownership | model endpointing off; VAD-committed final; no duplicate decode | |

The harness was **not** instrumented to isolate speech-end → final-transcript
for the STT segment, and **no run has yet been recorded on the Mac Studio
with Metal**. Those are the open gates.

**Acceptance gates on the reference host (to run before switching the
production pipeline):**

| Gate | Pass |
| --- | --- |
| Speech end → final transcript, 20 fixture utterances (3–8 s), Metal, right context 6 | **p50 ≤ 0.3 s, p99 ≤ 0.5 s** (PRD); must not grow with utterance length (long fixture within 1.2× short) |
| Same at right context 13 and 1 | recorded for the trade-off table; 13 must still meet p99 ≤ 0.5 s or 6 stays |
| Accuracy on the household command set (≈ 60 utterances incl. station/show/artist names, timers, times) | WER ≤ whisper-tiny.en's on the same set, and ≥ 95 % of tool-argument entities (station, show, duration) transcribed correctly |
| Interims | first delta ≤ 0.6 s after speech onset; no interim ever forwarded to the LLM |
| Memory | sidecar RSS ≤ 1.0 GB steady state with the model loaded; no growth over a 30-turn soak; measured with LLM (ADR-0003) and TTS (ADR-0004) resident |
| Robustness | 30-turn soak with no reconnects; barge-in `clear` leaves no stale text in the next final; sidecar restart is detected by `/ready` and reconnected without a pipeline restart |
| No hallucination on silence | 10 silence-only VAD turns yield empty finals |

Unit/GPU-free coverage carried from the PoC: URL derivation
(`http://…:8178` → `ws://…:8178/v1/realtime`), event parsing for every
server event type, session-update payload, flush barrier error, interim
delta accumulation, PCM16 conversion.

## Decision

1. **Adopt Nemotron Speech Streaming EN 0.6B via NeMo-Speech.cpp** as the
   pipeline's STT, on Metal on the Mac Studio, at right context 6 (560 ms),
   with model endpointing off and the pipeline's VAD committing turns.
2. **Keep the single-final contract**: interim text is for display only;
   exactly one final transcript per VAD turn reaches the LLM. No shadow
   decoding, no duplicate passes.
3. **Pin the runtime** (v0.1.0, per-platform SHA-256) and the model
   artefact; re-run the acceptance gates on any bump.
4. **Fallback ladder**, selectable by configuration without a rebuild:
   Nemotron on Metal/CUDA → Nemotron on CPU → Moonshine Medium Streaming
   (CPU) → Whisper batch (`tiny.en`), the last preserving today's
   behaviour exactly.
5. **Switch production only after the Mac gates pass**; until then the
   production chatbot stays on Whisper-MLX `tiny.en` with its 0.5 s p99
   timeout, and the Nemotron path is exercised in the PoC pipeline.

## Consequences

- Speech-end → transcript becomes a near-constant ≈ 0.1–0.3 s (right
  context + one committed decode) instead of a per-utterance batch decode;
  the LLM stream can start that much sooner, and long requests stop paying
  a length penalty. Together with ADR-0003/0004: STT ≈ 0.3 + LLM ≈ 0.35 +
  TTS ≈ 0.2 ≈ **0.85 s speech end → first audio** on the local path,
  inside the 1.5 s budget.
- Interim transcripts become available to the UI (browser transcript pane,
  satellite displays) and to future features (command memory, early tool
  pre-fetch, server-side wake confirmation on text).
- A second resident process (the sidecar) with its own health endpoint
  and restart policy; ≈ 1 GB more resident memory than Whisper-tiny.
- English only; multilingual turns would need a different model in the same
  runtime (Nemotron 3.5 ASR / Parakeet) at batch latency.
- The NVIDIA Open Model License governs the weights (permissive for this
  use; not Apache-2.0 like the runtime code).
- Phrase boosting — the feature that would most help media vocabulary — is
  inert until the published artefact includes tokenizer data.

## Re-evaluation triggers

- A Mac acceptance gate fails: first try right context 13 (accuracy) or 1
  (latency); if the memory gate fails, try the runtime's CPU build (the
  model is small enough); then Moonshine; then stay on Whisper.
- NeMo-Speech.cpp ships a Q8 artefact with tokenizer data: enable
  `speech_contexts` with the station/show/artist lists and re-run the
  entity-accuracy gate.
- NeMo-Speech.cpp releases streaming diarization on Metal: candidate for
  the PRD's within-session speaker identification, in the same sidecar.
- A full-duplex model with voice cloning *and* native tool calling becomes
  runnable in ≤ 12 GB on the Mac: the whole STT → LLM → TTS cascade, not
  just STT, is re-evaluated.

## Sources

- Model: [nvidia/nemotron-speech-streaming-en-0.6b (Hugging Face)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) — architecture, latency settings, WER table, licence.
- Runtime: [NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp); [server docs](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/server.md); [API reference — `/v1/realtime` session fields and events](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/api.md); [releases — v0.1.0, 2026-08-19](https://github.com/NVIDIA/NeMo-Speech.cpp/releases).
- Apple-Silicon analogue for memory/RTF: [speech-swift — Nemotron streaming ASR on MLX](https://github.com/soniqo/speech-swift/blob/main/docs/models/nemotron-asr-streaming.md).
- Alternatives: [Moonshine](https://github.com/moonshine-ai/moonshine); [NVIDIA NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B); [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1); [Kyutai STT](https://kyutai.org/stt/).
- Baseline: [mlx-community/whisper-tiny.en-mlx](https://huggingface.co/mlx-community/whisper-tiny.en-mlx).
