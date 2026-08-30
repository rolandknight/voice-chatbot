# Speaker recognition for Babel

Research date: 2026-08-28. Web survey of speaker-embedding models, ONNX/Rust
runtimes and streaming diarization, plus a read of this repo's own pipeline
(`crates/server/src/call.rs`, `wake.rs`, `stt.rs`, `crates/wake/src/lib.rs`) at
`b838771`.

Revised same day after reading `third_party/flowcat-core/src/pipeline/cascaded.rs`
and `crates/client/src/wake.rs`: the integration seam (§5) and the native-client
TD question (§3) are now settled rather than open. Implementation plan:
[`docs/plans/speaker-recognition.md`](../plans/speaker-recognition.md).

**Requirement:** `todo.md` line 1 — "first step is within a session. later, store
voiceprints and identify people. this needs to be done with minimal impact on
latency". Formalised as PRD SPKR-1 (within-session), SPKR-2 (voiceprint
enrolment + ID), SPKR-3 (< ~100 ms, concurrent with STT).

**Target hardware:** Mac Studio M4 Max (NFR-7). The embedder runs on the server,
not on the Pi 5 / Box-3 clients.

---

## Bottom line

1. **Don't add diarization.** Diarization answers "who spoke when" over a
   continuous multi-speaker stream. The `SpeechGate` already hands you exactly
   one endpointed, single-speaker utterance per turn. The question here is only
   *which enrolled person produced this utterance* — a verification/identification
   problem, not a segmentation one. Streaming diarization (Sortformer, DIART,
   EEND-GLA) solves a problem this pipeline doesn't have, at 10–100× the cost.

2. **One embedding per turn, cosine against enrolled centroids.** A 192-dim
   CAM++ or ERes2NetV2 forward pass over a 1–3 s utterance is single-digit to
   low-tens-of-ms on an M4 Max. Run it on the same buffer whisper is about to
   consume, on a separate task. Whisper's whole-utterance decode dominates, so
   speaker ID lands **off the critical path entirely** — SPKR-3 is met by
   construction rather than by optimisation.

3. **The wake phrase is the single biggest accuracy lever, and it's free.**
   Text-dependent verification massively outperforms text-independent at short
   duration (3.62 % vs 8.86 % EER at 3 s in the VoxPhrase study). Every command
   in listen mode is preceded by a *fixed-content* wake phrase, and
   `WakeGate::preroll` (`wake.rs:45`) is already buffering ~0.5 s of it. Enrol
   two vectors per person — one TD on "hey babel", one TI on accumulated
   commands — and score both. This is an architectural advantage most assistants
   don't have.

4. **Integrate through `ort` + `kaldi-native-fbank`, not sherpa-onnx.** One ONNX
   file and a feature extractor, mirroring exactly how `WakeBank` already loads
   oww heads. sherpa-onnx would bolt a second C++ inference stack next to the
   `ort` you already link through `flowcat-core`'s `vad-ort` and `oww_rs`.

5. **Phase 1 needs no enrolment, no storage, and no privacy surface.** Online
   clustering of per-turn embeddings within a call gets you SPKR-1 in ~200 lines.
   Persistence (SPKR-2) is a separate, later decision.

### Model shortlist

| rank | model | params / ONNX size | Vox1-O EER | short-utterance EER | why |
|---|---|---|---|---|---|
| **1** | **ERes2NetV2** (3D-Speaker) | ~17 M | **0.61 %** | **0.98 % @ 3 s, 1.48 % @ 2 s** | the only shortlisted model with *published* short-duration numbers, and they're the best in the field. Explicitly designed for this. Apache-2.0 |
| **2** | **CAM++** (3D-Speaker) | 192-dim, **~27 MB** ONNX | ~0.65 % class | not published | smallest/fastest credible option; ~60 ms/segment measured by a third party on commodity CPU (≈ 2× faster than ResNet34). Good fallback if ERes2NetV2 misses budget |
| 3 | ReDimNet2-B0 / B2 | **1.1 M / 0.33 GMACs** (B0) | 1.04 % (B0), 0.287 % (B6) | not published | best accuracy-per-FLOP published anywhere; B0 is absurdly cheap. **No shipped ONNX export** — you'd export it yourself. Revisit if 1 or 2 misses budget |
| 4 | WeSpeaker ResNet34 / ECAPA-TDNN c1024 | ~6.2 M (ECAPA) | 0.723 % / 0.728 % | ResNet-class: ~6.8 % @ 2 s | the well-trodden baseline; ONNX exports are first-class. Degrades hardest on short segments |
| — | Streaming Sortformer (NVIDIA) | — | — | — | **wrong tool.** GPU-oriented (RTF measured on RTX 6000 Ada), caps at 4 speakers, produces session-local `spk_0..3` labels with *no persistent identity*. A CoreML port exists (~1.04 s buffer latency) but it still can't answer "is this Roland" |

All ERes2NetV2/CAM++ weights are Apache-2.0 via 3D-Speaker. Check per-model
licences at download time — sherpa-onnx's model release explicitly notes each
model carries its own.

---

## 1. Why this is a smaller problem here than in general

The general speaker-recognition literature assumes a continuous stream with
overlap, unknown speaker count, and no endpointing. This pipeline has already
paid for all of that:

- `VadProcessor` (Silero, 512-sample hops, `confidence: 0.7`) marks speech.
- `WakeGate` swallows everything until a wake word fires (`call.rs:392`).
- `SpeechGate` produces one gated speech window per turn; `BabelStt` accumulates
  it and transcribes the **whole utterance** at the falling edge (`stt.rs:5`).

So per turn you get: a contiguous, VAD-trimmed, 16 kHz mono buffer, guaranteed
≥ 300 ms (`MIN_UTTERANCE_SAMPLES = 4800`), capped at 29 s, with silence already
stripped. That is precisely the input a speaker-embedding model wants, and it
arrives for free.

The remaining risk is **overlap** — two people talking over each other inside one
gated window produces a blended embedding that matches neither. In a household
command context this is rare and the correct response is to detect low confidence
and return "unknown", not to add a segmentation model.

## 2. The short-utterance problem — the real risk

Headline EERs are measured on VoxCeleb1-O, where test utterances average ~8 s.
Voice commands are 1–3 s. The degradation is steep and it is the thing that will
actually determine whether this works:

| model | full duration | 3 s | 2 s |
|---|---|---|---|
| ERes2NetV2 | 0.61 % | 0.98 % | 1.48 % |
| ResNet baseline (Res2Net paper) | 1.78 % | 3.78 % | 6.77 % |
| Res2Net | — | 3.06 % | 5.58 % |

A plain ResNet loses ~4× going from full-length to 2 s. ERes2NetV2 loses ~2.4×
and stays under 1.5 %. That gap is the entire reason it's ranked first.

Independent corroboration: a teacher-student study measured a **46 % relative EER
increase** just moving the evaluation set from 3.59 s to 2.05 s (8.72 % → 12.8 %).
And these are *clean YouTube-interview* numbers. Far-field capture through a
Jabra, with AEC residue and possibly Spotify playing in the room, will be worse
by an amount nobody has published. **Assume the VoxCeleb numbers are a
best case and budget 2–5× degradation until measured on the box.**

## 3. The wake-phrase lever

The most useful finding in the survey is from the VoxPhrase / hybrid-enrolment
work (Jun 2026), which separates *text-dependent* (TD) enrolment — fixed phrase —
from *text-independent* (TI):

- TD enrolment: **3.62 % EER** (3.09 % with a neural re-scorer)
- TI enrolment at 3 s: **8.86 %**
- TI improves monotonically with enrolment duration, but stays worse than TD
  below ~2 s of test audio.

Babel is in an unusually good position to exploit this, because **every listen-mode
turn is preceded by the same fixed phrase**. Three things follow:

1. `WakeGate` already holds it. `preroll: VecDeque<i16>` with `preroll_cap = 8000`
   (0.5 s @ 16 kHz, `wake.rs:45–48`) is a rolling buffer of the audio immediately
   before the wake fire — which, as `todo.md` notes at length, *is* the wake
   phrase. The pre-roll leakage that's a nuisance for whisper is an asset here.
2. The wake head already tells you which persona was addressed, so the TD
   comparison is against a phrase-specific centroid, not a generic one.
3. TD and TI scores can be fused. Score the wake phrase against the person's TD
   centroid and the command against their TI centroid; combine (mean of cosines,
   or take TD as primary and TI as tiebreak). This should recover most of the
   short-utterance loss in section 2.

**Native-client path — settled.** The Pi/native client detects on-device and the
server never runs `WakeGate` for it, so the concern was whether TD is available
there at all. Reading `crates/client/src/wake.rs:142–160`: on a wake with
`open`, the client drains its own 0.5 s pre-roll ring (`preroll_cap =
input_rate / 2`) and **prepends it to the outgoing pcm**. The wake phrase is
therefore already in the audio the server receives — this is exactly the leakage
`todo.md` diagnoses as "Hey, one what?" reaching whisper.

What's missing is only the boundary. The wake arrives as a
`{"type":"wake"}` frame on the events WebSocket (`main.rs::apply_client_wake`)
while audio arrives over WebRTC: two transports, no shared offset. The fix is one
additive field — `preroll_ms: u32` on `WakeState::Awake`
(`crates/protocol/src/lib.rs:60–65`), which the client already knows exactly.
`#[serde(default)]` keeps it wire-compatible with older clients, the same pattern
`persona` uses today. Milliseconds rather than samples, because the client's
input rate and the server's 16 kHz carrier differ.

That single field also fixes the `todo.md` bug properly: the server can strip
exactly `preroll_ms` from the STT buffer instead of lowering `--preroll-ms` to
200 and hoping, or asking the LLM to ignore a leading wake phrase. **The TD
design and the leakage fix are the same change.**

Ordering caveat: the WS frame and the audio can arrive out of order, so the TD
window must be resolved lazily at utterance finalization — "was a wake armed
within the last N ms?" — rather than eagerly on the wake frame. `arm_wake_grace`
already has this shape.

## 4. Integration options

**A. `ort` + `kaldi-native-fbank` — recommended.**
The 3D-Speaker/WeSpeaker ONNX exports take Fbank features, not raw audio, so you
need a feature extractor. `kaldi-native-fbank` (crates.io, Jan 2026) is a **pure
Rust port** — FBANK/MFCC on `realfft`/`rustfft`, with an `OnlineFeature` streaming
wrapper — with no C++ dependency. `ort` is already in the build twice over
(`flowcat-core` `vad-ort`, `oww_rs`). Total new surface: two crates, one `.onnx`
in `models/`, one module. This mirrors `WakeBank::load` almost exactly, which is
the strongest argument for it — it's the pattern the codebase already uses.
(`knf-rs` is the alternative feature crate but it's FFI bindings to
kaldi-native-fbank's C++; prefer the native port.)

**B. sherpa-onnx (official `sherpa-onnx` crate, or `sherpa-rs`).**
Batteries included: `SpeakerEmbeddingExtractor`, an embedding manager, and
diarization, with a curated model release covering NeMo/WeSpeaker/3D-Speaker
exports. Genuinely good, actively maintained, and it would work. The cost is
linking a second full C++ inference stack alongside `ort` — build complexity,
binary size, and two onnxruntime versions to keep from colliding. **Take this
only if the feature-extraction details in option A turn out to be fiddly to
match** (Fbank config mismatch against the model's training frontend is the
classic failure mode here, and sherpa handles it for you).

**C. `pyannote-rs` / `native-pyannote-rs`.**
Diarization-shaped: segmentation model + embedding model + clustering. Wrong
granularity, extra model to load, and you'd bypass most of it. `native-pyannote-rs`
is interesting as a pure-Rust Burn-backed inference path with no onnxruntime at
all, but it's a fork with a small user base. Not for this.

## 5. Where it lands in the code

```
InputAudio → VadProcessor → WakeGate → WakeGrace → [SpeakerGate] → SpeechGate → BabelStt
                                                         │
                                                         └→ CallState.speaker + `speaker` event
```

- **New crate or module.** `crates/server/src/speaker.rs` for the FrameProcessor,
  with the model-loading + scoring core in `crates/speaker/` if you want it unit-
  testable the way `crates/wake` is. The `GateCore`-style split (pure state
  machine, no I/O) has paid off for wake and would again.
- **Placement — settled: wrap the `SttService`, don't tap the pipeline.** An
  `input_processors` tap would *not* see the same audio as the STT. `SpeechGate`
  keeps its own 600 ms pre-roll ring (`SPEECH_GATE_PREROLL_MS`,
  `cascaded.rs:451`) and drains it into a single `InputAudio` push at the rising
  edge (`cascaded.rs:526–533`); an upstream tap sees the edge but not that
  re-injected audio, and would have to duplicate the ring and stay in sync with a
  vendored file forever.

  Instead add `SpeakerTap<S: SttService>`, a decorator that forwards every call
  to the inner service and keeps its own copy of the same `Arc<AudioFrame>`
  stream. Wrap where the `stt` box is built (`call.rs:428–455`) and all four
  backends — Whisper, Moonshine, Nemotron, NemotronNative — are covered with no
  change to any of them. `BabelStt.buf` is already the ideal buffer, but living
  inside it would cover only the whisper path.

  Two behaviours it must mirror: `set_muted(true)` clears the buffer
  (`stt.rs:231`), and `flush()` is not the only finalization point —
  `MAX_UTTERANCE_SAMPLES` forces a mid-utterance transcribe at 29 s
  (`stt.rs:215`).
- **Model loading.** Once per process, next to the whisper context in `PocState`,
  shared into each call by `Arc` — same shape as `SharedWhisperContext`.
- **Output.** Two channels, both already precedented by `WakeGate`:
  set a `speaker: Mutex<Option<SpeakerId>>` field on `CallState`
  (`skills/mod.rs:96`) so skills and the prompt builder can read it, and publish
  a `speaker` event on `CallEvents` alongside `WAKE_EVENT` for the client UI.
- **Config.** `SPEAKER_MODEL`, `SPEAKER_THRESHOLD`, `SPEAKER_ENABLED`
  in the root `.env`, matching the existing `WAKE_*` convention. Absent model
  path = feature off, like `cfg.wake_heads.is_empty()` today.
- **Concurrency.** Spawn the embed on a blocking task at the falling edge and let
  it race whisper. Resolve before the LLM call. Do **not** make the turn wait on
  it — on timeout, proceed with `None`.

## 6. Phasing

### Phase 1 — SPKR-1, within-session (no storage)

Per call, keep `Vec<(centroid: [f32; 192], count: u32)>`. For each turn:
embed → cosine against every centroid → if best ≥ threshold, assign and update
that centroid by running mean; else append a new one. Labels are session-scoped
(`speaker 1`, `speaker 2`) and die with the call.

This is genuinely small — no enrolment flow, no persistence, no privacy question,
no cross-session drift. It's also the right place to gather data: log the cosines
to `logs/` and you'll have the household's real score distribution, which is what
you need to pick a threshold for Phase 2.

Useful immediately even without names: the LLM can be told "the previous turn was
a different speaker", which fixes pronoun and context confusion when two people
share a session.

### Phase 2 — SPKR-2, persistent voiceprints

- **Storage.** Centroids are ~768 bytes each at fp32/192-dim. A JSON or SQLite
  file next to `.env` alongside the Spotify token is proportionate; nothing here
  justifies a database. Keep it gitignored — the repo is public.
- **Enrolment.** A skill (`skills/persona.rs` is the closest existing pattern):
  "Babel, I'm Roland" → collect N utterances across a few turns → store TD
  centroid (wake phrase, from pre-roll) + TI centroid (commands). The TI-duration
  curve says more enrolment audio keeps helping well past 10 s, so collect
  generously rather than one-shot.
- **Scoring.** Fuse TD + TI as in section 3. Three-way outcome with a reject
  band: match / **unknown** / ambiguous. Unknown must be a first-class, common
  answer, not an error path.
- **Adaptation.** Update centroids only on high-confidence matches, with a floor
  on how far a centroid may drift per update. Otherwise one bad far-field match
  poisons a voiceprint permanently.

## 7. Latency budget (SPKR-3)

| stage | cost | notes |
|---|---|---|
| Fbank, 2 s @ 16 kHz | ~1–3 ms | pure Rust, single core |
| ERes2NetV2 forward | ~10–30 ms est. | M4 Max CPU; unmeasured — CAM++ measured ~60 ms on commodity CPU, an M4 Max should beat that comfortably |
| Cosine vs ≤ 10 centroids | µs | 192 floats |
| **total, on its own task** | **~15–35 ms est.** | vs. whisper whole-utterance decode, which is far longer |

Because it runs concurrently with STT and STT is the longer leg, **marginal added
latency should be ~0 ms**, not 35. The < 100 ms target has a lot of headroom —
enough that ERes2NetV2 is affordable and there's no reason to reach for
ReDimNet2-B0's 0.33 GMACs unless something surprising shows up in measurement.

All estimates. Nothing here is measured on the M4 Max; that's task 1.

## 8. Risks and things not to do

- **Don't voice-gate anything that matters.** PRD SPKR-2 mentions "simple
  voice-gated permissions". A cosine score on a 2 s far-field command is not
  authentication — it's a hint. It's replayable, it's spoofable by a recording,
  and at a realistic operating point you'll be wrong a few percent of the time.
  Fine for "play *my* playlist". Not fine for anything destructive, anything
  involving spend, or anything with a physical effect. If voice-gating is wanted
  later, it needs anti-spoofing (ASVspoof-style countermeasures), which is a
  separate project.
- **Family members are the hard case.** Speaker verification is hardest on
  genetically similar voices, and children's voices drift fast enough that
  centroids go stale in months. VoxCeleb EERs are computed over strangers and
  will understate household confusion. Plan for periodic re-enrolment.
- **Media playback contaminates embeddings.** The radio/shows/Spotify skills mean
  speech often arrives over background music. Consider suppressing ID (return
  `unknown`) when the client reports media active, rather than recording a
  polluted match.
- **Whispering, illness, and shouting** all move embeddings a long way — one study
  measured ECAPA-TDNN at 13.72 % EER on normal-vs-whispered trials, versus low
  single digits matched-condition. Another reason `unknown` must be cheap.
- **Fbank config mismatch is the silent killer.** If the frame length, mel count,
  dither, or CMVN don't match what the model was trained with, you get plausible-
  looking embeddings that cluster badly. Validate against a reference embedding
  from the Python 3D-Speaker pipeline on a fixture WAV before trusting anything.
  `fixtures/` already has `t1_time.wav` and `t13_wake.wav` to use.

## 9. Open questions to settle on the box

Questions 2 and 3 in the first draft (tap window; native-client TD) are answered
in §5 and §3 above. Remaining:

1. Measure Fbank + ERes2NetV2 and CAM++ on the M4 Max for 1 s / 2 s / 3 s buffers.
   Does ERes2NetV2 fit comfortably enough to make CAM++ moot?
2. Collect ~50 real household utterances per person, plot the cosine ROC, and set
   the threshold from *that* — not from any number in this document.
3. Does the fp32 → int8 quantised export cost meaningful EER at 2 s? Only worth
   asking if (1) says latency is tight.
4. Does TD+TI fusion actually beat TI alone *on this household's audio*? The
   VoxPhrase gap is large but measured on a different corpus and a different
   phrase length. If it doesn't, Task 6 of the plan drops out and the protocol
   change is justified by the `todo.md` leakage fix alone.

---

## Sources

- ERes2NetV2 (Interspeech 2024) — 0.61 / 0.98 / 1.48 % EER at full / 3 s / 2 s:
  https://www.isca-archive.org/interspeech_2024/chen24l_interspeech.html
- Res2Net / ResNeXt for SV — truncated-test-set table (2 s / 3 s / 4 s):
  https://arxiv.org/pdf/2007.02480
- Short-utterance compensation — 46 % relative EER increase, 3.59 s → 2.05 s:
  https://arxiv.org/pdf/1810.10884
- Hybrid-enrolment neural re-scoring / VoxPhrase — TD 3.62 % vs TI-3 s 8.86 %:
  https://arxiv.org/html/2606.16115v1
- WeSpeaker — ResNet34 0.723 % / ECAPA c1024 0.728 % EER on vox1-O-clean:
  https://github.com/wenet-e2e/wespeaker
- 3D-Speaker (CAM++, ERes2NetV2, ONNX export scripts):
  https://github.com/modelscope/3D-Speaker
- CAM++ ONNX, 192-dim, ~27 MB, ~60 ms/segment (third-party measurement):
  https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx
- ReDimNet2 — B0 1.1 M params / 0.33 GMACs / 1.04 % EER; B6 0.287 %:
  https://arxiv.org/pdf/2603.11841
- Whispered-speech SV degradation (ECAPA-TDNN 13.72 % EER):
  https://arxiv.org/pdf/2604.20229
- Streaming Sortformer (Interspeech 2025) — AOSC speaker cache, GPU RTF:
  https://arxiv.org/pdf/2507.18446 ·
  https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
- Sortformer CoreML port (~1.04 s buffer latency, Apple Silicon):
  https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml
- `kaldi-native-fbank` (pure Rust FBANK/MFCC + online wrapper):
  https://crates.io/crates/kaldi-native-fbank
- sherpa-onnx Rust API + speaker-recognition model release:
  https://docs.rs/sherpa-onnx · https://k2-fsa.github.io/sherpa/onnx/speaker-identification/index.html
- `pyannote-rs` / `native-pyannote-rs`:
  https://crates.io/crates/pyannote-rs · https://crates.io/crates/native-pyannote-rs
