# speaker-recognition — per-turn speaker ID on the Babel pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-08-28
**Status:** Not started. Decisions marked **[assumption]** are the author's calls and should be confirmed by the requester.
**Machine:** Mac Studio, Apple M4 Max, 14 cores, 36 GB unified memory.
**Research:** `docs/research/speaker-recognition.md` — read it first; this plan does not repeat it.
**Requirements:** PRD SPKR-1 (within-session), SPKR-2 (voiceprints), SPKR-3 (< ~100 ms). `todo.md` line 1.

**Goal:** Identify who is speaking on each turn, first as session-scoped labels
(SPKR-1) and then as named household members (SPKR-2), with **zero added turn
latency** — the embedder runs concurrently with STT, which is always the longer
leg.

**Non-goal (this iteration):** diarization of overlapping speech, voice-gated
permissions, anti-spoofing, and any speaker signal reaching the *client UI*
beyond one event. Overlap resolves to `Unknown`, by design.

**Architecture:** Same hard boundary as `wake`: a pure, I/O-free core in
`crates/speaker/` (embedding + scoring + clustering, unit-testable with
synthetic vectors) and a thin server adapter in `crates/server/src/speaker.rs`
that owns ONNX loading, the `SttService` decorator, and event publishing. The
core never imports `ort`, `flowcat-*`, or `tokio`.

**Tech stack:** `ort` (already in-tree via `flowcat-core` `vad-ort` and
`oww_rs`), `kaldi-native-fbank` (pure Rust), ERes2NetV2 ONNX from 3D-Speaker.

---

## Why these choices

| Decision | Choice | Why |
| --- | --- | --- |
| Approach | Per-utterance embedding + cosine, **not** diarization | `SpeechGate` already delivers one endpointed single-speaker utterance per turn. Streaming diarization (Sortformer et al.) solves segmentation this pipeline has already paid for, at 10–100× the cost, and yields no persistent identity. |
| Model | **ERes2NetV2** (3D-Speaker, Apache-2.0), CAM++ as fallback | Only shortlisted model with published short-duration numbers, and they're the best available: 0.61 % / 0.98 % / 1.48 % EER at full / 3 s / 2 s. Commands are 1–3 s, so short-duration behaviour *is* the spec. |
| Runtime | `ort` + `kaldi-native-fbank` | `ort` is already linked twice. `kaldi-native-fbank` is a pure-Rust port (no C++). Mirrors `WakeBank::load` — the pattern the codebase already uses. |
| Rejected | sherpa-onnx | Batteries-included and genuinely good, but links a second full C++ inference stack beside `ort`. Hold as the escape hatch if Fbank config matching (Task 2) turns painful. |
| Seam | `SpeakerTap<S: SttService>` decorator | An `input_processors` tap misses `SpeechGate`'s 600 ms re-injected pre-roll (`cascaded.rs:451,526`). The decorator sees byte-identical audio to the STT and covers all four backends with no change to any of them. |
| Text-dependent scoring | Wake phrase as a second, fixed-content sample | TD enrolment measures 3.62 % EER vs 8.86 % for TI at 3 s. Every listen-mode turn is preceded by the same phrase and it's already in the buffer. Biggest accuracy lever available. |
| Phase 1 storage | None | Session-scoped clustering needs no persistence, no enrolment flow, and raises no privacy question. It is also how we collect the data needed to set Phase 2's threshold. |

---

## Global Constraints

- New crate `crates/speaker/`, added to the workspace `members` in the root
  `Cargo.toml`. Do not touch `third_party/` or `archive/`.
- **Feature is off unless configured.** No `POC_SPEAKER_MODEL` → `SpeakerTap` is
  never wrapped and not one byte is copied, exactly as `cfg.wake_heads.is_empty()`
  gates the wake path today.
- **Never block a turn.** The embed runs on `spawn_blocking` and races STT. On
  timeout or error the turn proceeds with `Speaker::Unknown`. There is no code
  path where speaker ID can delay or fail a response.
- Model file in `models/` (gitignored), fetched by a `make models-speaker`
  target next to the existing wake/whisper model targets.
- Config via `.env` following the `POC_WAKE_*` convention: `POC_SPEAKER_MODEL`,
  `POC_SPEAKER_THRESHOLD`, `POC_SPEAKER_MARGIN`, `POC_SPEAKER_STORE`.
- **Nothing in this feature gates a skill, a tool call, or a permission.** The
  speaker is context for the prompt and an event for the UI. Enforcing anything
  on it is out of scope and explicitly discouraged (research §8).
- The repo is public: voiceprint stores, enrolment audio, and `logs/*.jsonl`
  score dumps are all gitignored before Task 5 writes the first one.
- Commit after every task on branch `speaker-recognition`.

---

### Task 1: Crate skeleton and the pure core

**Files:** `crates/speaker/{Cargo.toml,src/lib.rs}`; modify root `Cargo.toml`.

- [ ] `crates/speaker/` with no dependency on `ort`, `flowcat-*`, or `tokio`.
- [ ] `Embedding([f32; 192])` newtype with L2-normalise on construction and
      `cosine(&self, &Self) -> f32`. Normalising at the boundary means scoring is
      a dot product and every downstream comparison is automatically in `[-1, 1]`.
- [ ] `SpeakerId` enum: `Session(u32)` (Phase 1) | `Known(String)` (Phase 2) |
      `Unknown`.
- [ ] `Roster` — the online clusterer. `assign(&mut self, Embedding) -> SpeakerId`:
      cosine against every centroid, take the best; assign if `best >= threshold`
      **and** `best - second_best >= margin`, else open a new session speaker.
      Update the winning centroid by running mean, re-normalised.
- [ ] The two-threshold rule is deliberate: a single threshold produces confident
      wrong answers when two household members sit near each other in the space.
      The margin turns those into `Unknown`, which is the correct output.
- [ ] Cap the roster (**[assumption]** 8 session speakers) and return `Unknown`
      past it rather than growing without bound on a noisy call.
- [ ] Unit tests on synthetic vectors only — no audio, no model. Cover: identical
      vectors assign together; orthogonal vectors split; a vector inside the
      margin band returns `Unknown`; centroid drift stays bounded over 100
      updates; roster cap holds.

**Gate:** `cargo test -p voice-chatbot-speaker` green with no ONNX in the tree yet.

---

### Task 2: Fbank + ONNX embedder, validated against the reference

**Files:** `crates/server/src/speaker.rs` (embedder half); `Makefile`.

This is the **highest-risk task in the plan** and it is deliberately second.
Fbank config mismatch — frame length, mel count, dither, CMVN — produces
plausible-looking embeddings that cluster badly and will silently poison every
later measurement.

- [ ] `make models-speaker` downloads the ERes2NetV2 ONNX export to `models/`,
      checksums it, and is idempotent.
- [ ] `SpeakerEmbedder::load(path) -> Result<Self>` building one `ort::Session`,
      shared across calls behind `Arc` like `SharedWhisperContext` (`stt.rs:29`).
- [ ] `embed(&self, samples: &[f32]) -> Result<Embedding>`: Fbank via
      `kaldi-native-fbank` → ONNX → L2-normalise.
- [ ] **Validation harness, before anything else uses this.** Run the Python
      3D-Speaker pipeline once on `fixtures/t1_time.wav` and `fixtures/t13_wake.wav`,
      commit the two reference embeddings as JSON, and assert cosine ≥ **0.99**
      against the Rust path in a test. Check the Fbank matrix itself against the
      Python frontend first if the embeddings disagree — that isolates frontend
      from model.
- [ ] Bench binary: 1 s / 2 s / 3 s buffers × 20 runs, report p50/p95 for Fbank,
      ONNX, and total. Write `docs/research/speaker-bench-m4-max.md` in the format
      of `poc-tts/bench-m4-max.md`.

**Gate:** cosine ≥ 0.99 vs reference, **and** p95 total < 100 ms at 3 s. If the
cosine check fails and an hour of frontend debugging doesn't fix it, switch to
sherpa-onnx (research §4 option B) rather than grinding — the model is the point,
not the frontend. If the latency gate fails, drop to CAM++ before anything else.

---

### Task 3: `SpeakerTap` — the STT decorator

**Files:** `crates/server/src/speaker.rs`; modify `crates/server/src/call.rs`.

- [ ] `SpeakerTap<S: SttService>` implementing `SttService`, forwarding every
      method to the inner service and returning its frames unchanged.
- [ ] Accumulate `run_stt`'s `Arc<AudioFrame>` into a local `Vec<f32>` with the
      same resample-to-16 kHz logic as `BabelStt::append` (`stt.rs:85`).
- [ ] Mirror the inner service's finalization exactly:
      - `flush()` → finalize
      - `run_stt` past `MAX_UTTERANCE_SAMPLES` → finalize (`stt.rs:215`)
      - `set_muted(true)` → **clear** the buffer, don't finalize (`stt.rs:231`)
      - buffer under `MIN_UTTERANCE_SAMPLES` (300 ms) → drop, no embed
- [ ] On finalize, `tokio::task::spawn_blocking` the embed and store the
      `JoinHandle` on the call. **Do not await it here** — `flush()` is on the
      turn's critical path.
- [ ] Wrap in `call.rs:428–455` only when a model is configured, after the
      backend match, so all four STT backends are covered by one line.
- [ ] Test with a mock inner `SttService` (the `BabelStt::without_context`
      pattern, `stt.rs:75`): assert the inner sees identical frames, that the tap
      buffer equals what the inner accumulated, and that mute clears both.

**Gate:** existing STT tests still pass unmodified; a mute mid-utterance leaves
no orphaned embed task.

---

### Task 4: Resolve, publish, and put it in the prompt (SPKR-1 complete)

**Files:** modify `crates/server/src/{speaker.rs,skills/mod.rs}`, `crates/protocol/src/lib.rs`.

- [ ] `CallState.speaker: Mutex<Option<SpeakerId>>` alongside `voice` and
      `wake_armed_at` (`skills/mod.rs:96`), with a per-call `Roster`.
- [ ] Join the embed handle at the point the transcript is ready and before the
      LLM runs, with a **hard timeout** (**[assumption]** 150 ms — generous,
      since the embed started when the buffer closed and whisper has been
      decoding since). On timeout, log and set `Unknown`.
- [ ] `SPEAKER_EVENT` in `crates/protocol/src/lib.rs` and publish on `CallEvents`,
      mirroring `WAKE_EVENT` (`wake.rs:24`). Payload: label, cosine, margin.
- [ ] Prompt integration: when a turn's speaker differs from the previous turn's,
      prepend a short system line ("this turn is from a different speaker than
      the last"). **[assumption]** Prefix, not a schema change — no skill,
      backend, or tool contract changes.
- [ ] Structured log per turn to `logs/speaker.jsonl`: timestamp, session label,
      best and second-best cosine, buffer duration, whether media was playing.
      **This file is the input to Task 7's threshold** and is why Phase 1 ships
      before enrolment exists.
- [ ] `.gitignore` `logs/speaker.jsonl` and `models/*.onnx` in this task, not later.

**Gate:** SPKR-1 done. Two people alternating on one call get stable distinct
labels across ≥ 10 turns. Measure end-to-end turn latency with the feature on and
off — **the delta must be within noise**, and if it isn't, the join is on the
critical path and Task 3's spawn is wrong.

---

### Task 5: `preroll_ms` on the wire, and the `todo.md` leakage fix

**Files:** modify `crates/protocol/src/lib.rs`, `crates/client/src/wake.rs`, `crates/server/src/main.rs`, `crates/server/src/stt.rs`.

Standalone value independent of speaker ID: this is the real fix for the
"Hey, one what?" bug in `todo.md`.

- [ ] Add `preroll_ms: u32` to `WakeState::Awake` with `#[serde(default)]`
      (`protocol/src/lib.rs:60–65`), matching how `persona` is already optional.
      Milliseconds, not samples — client input rate and the 16 kHz carrier differ.
- [ ] Client populates it from the drained ring length at
      `crates/client/src/wake.rs:152–157`, converted from `input_rate`.
- [ ] `apply_client_wake` (`main.rs:841`) records `(preroll_ms, Instant::now())`
      on `CallState` next to `wake_armed_at`.
- [ ] Server-side `WakeGate` populates the same field from its own
      `preroll` drain (`wake.rs:45`) so both wake paths converge on one
      representation.
- [ ] Strip exactly `preroll_ms` of leading audio from the STT buffer when a wake
      was armed within the grace window — replacing the `--preroll-ms 200` and
      prompt-hack workarounds discussed in `todo.md`.
- [ ] Resolve **lazily at finalization** ("was a wake armed in the last N ms?"),
      never eagerly on the wake frame: the WS frame and the WebRTC audio are on
      different transports and can arrive out of order. `arm_wake_grace` has this
      shape already.
- [ ] Round-trip test for an older client omitting the field → `0` → current
      behaviour unchanged.

**Gate:** the wake phrase no longer appears in transcripts on the native-client
path, and tool selection on "hey babel play X" improves (the failure `todo.md`
traces to this). Verify with a real Pi session, not a unit test.

---

### Task 6: TD + TI fusion

**Files:** modify `crates/speaker/src/lib.rs`, `crates/server/src/speaker.rs`.

- [ ] Split the finalized buffer at `preroll_ms` into a TD window (wake phrase)
      and a TI window (command). Embed both — two forward passes, still well
      inside budget per Task 2's bench.
- [ ] `Roster` carries two centroids per speaker. Fuse by mean of the two cosines
      (**[assumption]** — simplest thing that could work; the alternative is TD as
      primary with TI as tiebreak, and Task 7's data will say which).
- [ ] Fall back to TI-only when `preroll_ms` is 0 or absent (push-mode clients,
      older clients) — TD must never be a hard requirement.
- [ ] Extend `logs/speaker.jsonl` with TD, TI, and fused scores side by side so
      the fusion can be evaluated against TI-alone on the same turns.

**Gate:** on the household set from Task 7, fused EER beats TI-only. **If it
doesn't, delete this task's fusion and keep TI** — Task 5 stands on its own merits
and the research note's 3.62 % vs 8.86 % figure is from a different corpus and
phrase length. This is a hypothesis, not a certainty.

---

### Task 7: Threshold calibration on real household audio

**Files:** `crates/speaker/src/bin/roc.rs`, `docs/research/speaker-bench-m4-max.md`.

- [ ] Collect ≥ 50 utterances per household member from ordinary use — through
      the Jabra, at realistic distance, some with music playing, some with the
      TV on. **Not** clean close-mic recordings; those will produce an optimistic
      threshold that fails in the room.
- [ ] Offline ROC tool over `logs/speaker.jsonl`: same-speaker vs
      different-speaker cosine distributions, EER, and the operating point.
- [ ] Pick threshold and margin from **this** curve. Set the operating point
      conservatively — a wrong name is worse than no name, so bias toward
      `Unknown`.
- [ ] Record measured EER at 1 s / 2 s / 3 s next to the VoxCeleb published
      numbers. The gap is the honest headline of the whole feature and belongs in
      the research doc.

**Gate:** documented threshold, documented household EER, and an explicit
statement of the false-accept rate at the chosen operating point.

---

### Task 8: Enrolment and persistence (SPKR-2)

**Files:** `crates/server/src/skills/speaker.rs`; modify `crates/server/src/skills/mod.rs`.

- [ ] Enrolment skill following the `skills/persona.rs` pattern: "Babel, I'm
      Roland" → collect over several turns → store TD + TI centroids. The
      TI-duration curve keeps improving past 10 s, so **collect generously**;
      one-shot enrolment is the main way this feature disappoints.
- [ ] Store as JSON next to `.env` (**[assumption]** — ~768 bytes per centroid;
      SQLite is not warranted). Gitignored. Path via `POC_SPEAKER_STORE`.
- [ ] Load at startup; a missing or malformed store logs and degrades to Phase 1
      session labels rather than failing the call.
- [ ] Adaptation: update stored centroids only above a high-confidence threshold,
      with a per-update drift cap. One bad far-field match must not be able to
      poison a voiceprint permanently.
- [ ] Suppress ID (return `Unknown`, record nothing) when the client reports media
      active — the radio/Spotify skills mean background music is common and a
      contaminated match is worse than no match.
- [ ] A "forget me" path that deletes a stored voiceprint. Biometric data with no
      delete story is not something to ship into a household.

**Gate:** SPKR-2 done. Named identification across a server restart, and a stored
voiceprint that survives 50 adaptive updates without drifting past its own
enrolment centroid by more than a documented bound.

---

## Sequencing notes

Tasks 1–4 are the vertical slice and deliver SPKR-1 alone; stop there and the
feature is already useful (the LLM stops confusing two people in one session).
Task 5 is independently valuable and could ship first if the `todo.md` bug is
more annoying than speaker ID is interesting. Task 7 gates Task 8 — do not build
enrolment before knowing the household's real score distribution, or the
threshold will be guessed twice.
