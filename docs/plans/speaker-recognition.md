# speaker-recognition — per-turn speaker ID on the Babel pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-08-28 · **Revised:** 2026-08-29 — review pass; resolves R1 (join
latency is backend-specific), R2 (the pre-roll strip offset was wrong), R3
(nothing is measured). See the ledger below.
**Status:** Not started. Decisions marked **[assumption]** are the author's calls and should be confirmed by the requester.
**Machine:** Mac Studio, Apple M4 Max, 14 cores, 36 GB unified memory.
**Research:** `docs/research/speaker-recognition.md` — read it first; this plan does not repeat it.
**Requirements:** PRD SPKR-1 (within-session), SPKR-2 (voiceprints), SPKR-3 (< ~100 ms). `todo.md` line 1.

**Goal:** Identify who is speaking on each turn, first as session-scoped labels
(SPKR-1) and then as named household members (SPKR-2), with **zero added turn
latency by construction** — the embed is spawned at finalization and the turn
never waits on it. The earlier framing ("it races STT, which is always the longer
leg") is only true of whisper: `BabelStt` decodes the whole utterance inside
`flush()` (`stt.rs:103`), while Moonshine and Nemotron decode incrementally and
their `flush()` merely collects an already-computed final (`moonshine.rs:419`,
`nemotron.rs:214`). ADR-0005/0006 make Nemotron the intended operating point, so
the plan must not buy its latency budget with a race it will lose there.

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

**Nothing in this plan is measured.** Every latency figure is an estimate, and
every EER quoted anywhere in it or in the research note is VoxCeleb-on-strangers
— clean, ~8 s, and not this household. The research note budgets 2–5×
degradation far-field through the Jabra, and family members are the hard case for
verification. Two rules follow:

1. **No threshold, margin, or wait constant ships as a default until Task 7
   measures it.** Tasks 1–6 read them from config with deliberately conservative
   placeholders and log raw cosines regardless of what the placeholders decide.
2. **Every gate below names a number to record, not a box to tick.** A task is
   not done until its ledger row has a value.

| # | Measurement | Source | Value |
| --- | --- | --- | --- |
| M1 | Fbank p50/p95 at 1 / 2 / 3 s | Task 2 bench | — |
| M2 | ERes2NetV2 forward p50/p95 at 1 / 2 / 3 s | Task 2 bench | — |
| M3 | CAM++ forward p50/p95 at 3 s (the fallback's real cost) | Task 2 bench | — |
| M4 | ERes2NetV2 forward p95 at 3 s **with a whisper decode running concurrently** | Task 2 bench | — |
| M5 | Turn-latency delta, feature on vs off, `STT_BACKEND=whisper` | Task 4 | — |
| M6 | Turn-latency delta, feature on vs off, `STT_BACKEND=nemotron` | Task 4 | — |
| M7 | `speaker_ready_rate` per backend — embeds finished before the prompt is built | Task 4 | — |
| M8 | Household EER at 1 / 2 / 3 s across 3 speakers, vs the published numbers | Task 7 | — |
| M9 | False-accept rate at the chosen operating point | Task 7 | — |

Each value is recorded as `number @ git sha, date` — a latency figure without the
tree it came from is not reproducible. **Measurement method** below gives the
harness, the exact command, and what counts as noise for every row.

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
| Join policy | Non-blocking check; `SPEAKER_WAIT_MS` defaults to `0` | A deadline-capped join (the first draft said 150 ms) is free on whisper and a real cost on a streaming backend, where the STT final is already computed when `flush()` is called. `JoinHandle::is_finished()` plus a fall-through to `Unknown` costs a boolean test on every backend, and a late embed still updates the roster, publishes, and logs — so it counts for the next turn. |
| Pre-roll accounting | One documented `SpeechGate` patch: drop the ring across a transport gap | Without it, "strip `preroll_ms` from the front of the buffer" is simply wrong — `SpeechGate` re-injects up to 600 ms of its own ring ahead of the utterance (`cascaded.rs:514-533`) and the tap cannot decompose it, because `SttService` never sees `UserStartedSpeaking` (`adapters.rs:117,132`). Clearing stale ring content makes *buffer start == burst start* an invariant, after which the strip is exactly `preroll_ms` on all three wake paths. |

---

## Global Constraints

- New crate `crates/speaker/`, added to the workspace `members` in the root
  `Cargo.toml`. Do not touch `archive/`. `third_party/flowcat-core` gets
  **exactly one** change, in Task 5 — the `SpeechGate` staleness fix — recorded
  as a fifth bullet in `third_party/flowcat-core/VENDORED.md` beside the four
  local modifications already documented there. Nothing else in the vendored tree
  moves.
- **Feature is off unless configured.** No `SPEAKER_MODEL` → `SpeakerTap` is
  never wrapped and not one byte is copied for speaker ID, exactly as
  `cfg.wake_heads.is_empty()` gates the wake path today. Task 5's
  `WakePrerollTrim` is the deliberate exception: it is always installed, because
  the `todo.md` leakage fix is not part of this feature and must not be switched
  off with it. It touches no model and allocates nothing when no wake is armed.
- **Never block a turn — by construction, not by timeout.** The embed runs on
  `spawn_blocking` at finalization. Where the prompt is built the join is a
  **non-blocking check**: handle finished → take the result; not finished →
  `Unknown` and move on. Default wait is `0 ms`, so the added critical-path cost
  is one boolean test on every STT backend. A late embed is **not** discarded —
  it still updates the roster, publishes its `SPEAKER_EVENT`, and writes its log
  line, so it counts toward the next turn and toward Task 7's calibration. Only
  the prompt line is best-effort.
- Model file in `models/` (gitignored), fetched by a `make models-speaker`
  target next to the existing wake/whisper model targets.
- Config via `.env` following the `WAKE_*` convention: `SPEAKER_MODEL`,
  `SPEAKER_THRESHOLD`, `SPEAKER_MARGIN`, `SPEAKER_STORE`,
  `SPEAKER_WAIT_MS` (default `0`), plus `SPEAKER_LOG` and
  `SPEAKER_LABEL` for the measurement chain (**Measurement method** below).
  A non-zero wait is an explicit trade of
  turn latency for hit rate; it must be justified by an M7 number and recorded in
  the ledger, never adopted quietly to make the feature look better.
- **Nothing in this feature gates a skill, a tool call, or a permission.** The
  speaker is context for the prompt and an event for the UI. Enforcing anything
  on it is out of scope and explicitly discouraged (research §8).
- The repo is public: voiceprint stores, enrolment audio, and `logs/*.jsonl`
  score dumps must be gitignored before the first one is written. `/models/*`
  (`.gitignore:24`) and `/logs/` (`.gitignore:27`) already cover this — verify,
  don't duplicate, and add a rule only for `SPEAKER_STORE` if it lands
  outside those two.
- Commit after every task on branch `speaker-recognition`.

---

## Measurement method

The ledger's nine rows are produced by exactly two binaries and one log file.
Nothing here needs a replay rig or a second clock.

### Environment, recorded with every run

Copy the header style of `archive/poc-tts/bench-m4-max.md`: host and chip, core
count, RAM, OS version, **git sha of the tree**, `STT_BACKEND` and its model
(whisper model file, or the Nemotron right-context operating point from
ADR-0005), `whisper_threads`, the speaker model file with its sha256, and every
`SPEAKER_*` value in force. Machine plugged in, no other load. A number
without this block is not a measurement.

### M1–M4 — `crates/speaker/src/bin/bench.rs`

- [ ] Input is `fixtures/t1_time.wav` and `fixtures/t13_wake.wav`, cut or
      zero-padded to exactly 1 s / 2 s / 3 s at 16 kHz mono, so the duration axis
      is the only thing that varies.
- [ ] **5 warm-up iterations, discarded**, then 20 timed. The first ONNX call
      allocates arenas and pages in the weights; including it reports that setup
      cost as inference cost, which is the difference between passing and failing
      the Task 2 gate for reasons that have nothing to do with the model.
- [ ] Time Fbank, forward, and total separately with `Instant::now()` around each.
      Report p50 and p95 per (model, duration) — never the mean, which hides the
      tail that actually decides whether the non-blocking join hits.
- [ ] `--contend N` (M4) runs N concurrent `BabelStt`-style whisper decodes of a
      3 s buffer alongside the sweep, N = `whisper_threads`. Report p95 with and
      without, and the delta. This is the steady state: `spawn_blocking` and
      whisper share cores.
- [ ] M3 is the same sweep with `--model models/campplus.onnx`; measuring the
      fallback now is what lets the Task 2 gate fail into a decision.
- [ ] One JSON object per timed iteration appended to `logs/speaker-bench.jsonl`;
      the doc quotes percentiles computed from it and links the raw file, exactly
      as `bench-m4-max.md` links `reports/runs.jsonl`.

```bash
cargo run --release -p voice-chatbot-speaker --bin bench -- \
  --model models/eres2netv2.onnx --durations 1,2,3 --iters 20 --warmup 5 \
  --contend 4 --out logs/speaker-bench.jsonl
```

### M5–M7 — one structured line per turn, two ways of reading it

The span the feature can affect is **STT final → LLM request issued**. Measure
that, not time-to-first-token: the LLM leg is cloud-variable and dominates
end-to-end (ADR-0005 measured reply-start swinging by seconds on the cloud LLM
while the STT-relevant numbers stayed within tens of ms). Record `llm_ttft_ms`
too, for context, but never gate on it.

- [ ] **Primary, deterministic:** `crates/server/tests/turn_latency.rs` drives
      `SpeakerTap<WakePrerollTrim<Stub>>` directly with WAV frames plus a `flush()`,
      where `Stub` simulates each backend's finalize cost — whisper: sleep the
      `decode_ms` observed in the real log line at `stt.rs:147-152`; nemotron:
      return immediately. Times `flush()` → speaker resolved → prompt ready. Fully
      repeatable, and it isolates precisely the code this feature adds.
- [ ] **Confirmatory, real:** the same delta computed from `logs/speaker.jsonl`
      over ≥ 20 real turns per condition on the box. No rig needed — every field
      is already in the file.
- [ ] Four conditions: `speaker_enabled` false/true × `STT_BACKEND` whisper
      (**M5**) / nemotron (**M6**), ≥ 20 turns each.
- [ ] **"Within noise" is a number, not a judgement:** p50 delta ≤ **1 ms** and
      p95 delta ≤ **5 ms**. If it fails, read M4 before touching the join — the
      likely cause is CPU contention with the decode, and the fix is a cheaper
      model, not a different join.
- [ ] **M7** is `mean(speaker_ready)` per backend over the same rows, reported
      with its n.

### The log line (written by Task 4, read by everything else)

One object per turn, appended to `logs/speaker.jsonl`. Every `*_ms` is an offset
from `ts`, the VAD falling edge — one clock, one file, no cross-log join:

```json
{
  "ts": "2026-09-05T19:04:11.238Z",
  "turn_id": "c41f…", "backend": "nemotron", "speaker_enabled": true,
  "resolved": "Session(2)", "best": 0.71, "second": 0.44, "margin": 0.27,
  "embedding": ["… 192 floats, 4 dp …"],
  "td_best": null, "ti_best": 0.71, "fused": null,
  "buffer_ms": 1840, "preroll_ms": 500, "trimmed_ms": 500,
  "media_playing": false, "roster_evicted": false,
  "ground_truth": "person_a",
  "stt_final_ms": 412, "embed_spawn_ms": 3, "embed_done_ms": 27,
  "speaker_ready": true, "llm_request_ms": 415, "llm_ttft_ms": 690
}
```

- [ ] **`embedding` is not optional.** Cosines alone cannot be re-scored pairwise,
      so a log without the vector cannot produce Task 7's ROC — the data would have
      to be collected twice. 192 floats at 4 dp is ~1.2 KB/turn: a week of
      household use is a couple of MB.
- [ ] **`ground_truth`** comes from `SPEAKER_LABEL`, set while one person is
      doing their collection sessions. It is the only manual step in the whole
      measurement chain, and it is what makes the six distributions computable.
- [ ] Write the line whenever `SPEAKER_LOG=1`, **including with the feature
      off** (`speaker_enabled: false`, speaker fields null). Without that there is
      no off-condition to subtract for M5/M6.
- [ ] The vector is biometric data. Gitignored (`/logs/`, `.gitignore:27`), and
      Task 8's "forget me" deletes that person's rows here as well as their entry
      in the store.
- [ ] Reuse the timings the server already emits rather than duplicating them:
      whisper's `audio_ms` / `decode_ms` (`stt.rs:147-152`), Moonshine's
      `latency_ms` on "Moonshine utterance finalized" (`moonshine.rs:584`),
      Nemotron's equivalent (`nemotron_native.rs:514`), and the LLM's `ttft_ms`
      (`llm_ollama.rs:435`, `llm_claude.rs:383`).

### M8–M9 — `crates/speaker/src/bin/roc.rs`

- [ ] Reads `logs/speaker.jsonl`, groups by `ground_truth`, and scores **every
      pair** of embeddings: three same-speaker distributions and three cross-pairs
      for three people.
- [ ] Buckets trials by `buffer_ms` into 1 s / 2 s / 3 s bands so the measured
      short-utterance curve can be laid beside ERes2NetV2's published
      0.98 % @ 3 s / 1.48 % @ 2 s. The gap between those two columns is the
      headline (**M8**).
- [ ] Sweeps threshold × margin and prints the operating-point table: threshold,
      margin, false-accept rate (**M9**), false-reject rate, and the share of turns
      resolving to `Unknown` — that last one is the usability cost of a
      conservative point and has to be visible when it is chosen.
- [ ] Writes `docs/research/speaker-bench-m4-max.md` (latency, M1–M7) and the EER
      tables (M8–M9) into the same doc, linking the raw JSONL rather than pasting
      it.

```bash
cargo run --release -p voice-chatbot-speaker --bin roc -- \
  --log logs/speaker.jsonl --buckets 1,2,3 \
  --out docs/research/speaker-bench-m4-max.md
```

---

## Tasks

### Task 1: Crate skeleton and the pure core

**Files:** `crates/speaker/{Cargo.toml,src/lib.rs}`; modify root `Cargo.toml`.

- [ ] `crates/speaker/` with no dependency on `ort`, `flowcat-*`, or `tokio`.
- [ ] `Embedding([f32; 192])` newtype with L2-normalise on construction and
      `cosine(&self, &Self) -> f32`. Normalising at the boundary means scoring is
      a dot product and every downstream comparison is automatically in `[-1, 1]`.
      192 is correct for both ERes2NetV2 and CAM++, but it is a property of the
      model, not a law: `SpeakerEmbedder::load` (Task 2) must check the ONNX
      output dimension and fail loudly on a mismatch rather than silently
      truncating a different embedding space into this one.
- [ ] `SpeakerId` enum: `Session(u32)` (Phase 1) | `Known(String)` (Phase 2) |
      `Unknown`. Call it `SpeakerId` **everywhere**, including in prose: `Speaker`
      is already taken in server scope by `voice_chatbot_wake::Speaker`
      (`Speaker::User` / `Speaker::Bot`, imported at
      `crates/server/src/wake.rs:8`), and a shadowing import would be a genuinely
      confusing bug to read.
- [ ] `Roster` — the online clusterer. `assign(&mut self, Embedding) -> SpeakerId`:
      cosine against every centroid, take the best; assign if `best >= threshold`
      **and** `best - second_best >= margin`, else open a new session speaker.
      Update the winning centroid by running mean, re-normalised.
- [ ] The two-threshold rule is deliberate: a single threshold produces confident
      wrong answers when two household members sit near each other in the space.
      The margin turns those into `Unknown`, which is the correct output.
- [ ] Cap the roster (**[assumption]** 8 session speakers). At the cap, **evict
      the least-recently-assigned centroid** rather than returning `Unknown` for
      the rest of the call: a room with the TV on fills eight slots in a couple of
      minutes, and a hard cap converts that into a permanently dead feature until
      the call ends. Log every eviction so Task 7 can see how often it fires.
- [ ] Unit tests on synthetic vectors only — no audio, no model. Cover: identical
      vectors assign together; orthogonal vectors split; a vector inside the
      margin band returns `Unknown`; centroid drift stays bounded over 100
      updates; the cap evicts the least-recently-assigned centroid and the
      surviving speakers still match themselves afterwards.

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
- [ ] Bench binary `crates/speaker/src/bin/bench.rs`: 1 s / 2 s / 3 s buffers ×
      20 runs, p50/p95 for Fbank, ONNX, and total — **for ERes2NetV2 and CAM++
      both** (M1, M2, M3). Measuring the fallback now costs ten minutes and is the
      only way the Task 2 gate can fail into a decision instead of into a rethink.
      The warm-up rule, the fixture preparation, and the exact command are in
      **Measurement method** above; follow them rather than improvising, because
      the discarded warm-up is the difference between a 20 ms number and a 200 ms
      one.
- [ ] Bench the contended case too (**M4**): the same forward pass while a whisper
      decode of a 3 s utterance runs concurrently, since that is the real
      steady state — `spawn_blocking` and `whisper_threads` share the same cores.
      An embed that is fast alone and slow under load fails SPKR-3 in the room
      while passing it on the bench.
- [ ] Write `docs/research/speaker-bench-m4-max.md` in the format of
      `poc-tts/bench-m4-max.md`, and copy M1–M4 into this plan's ledger.

**Gate:** cosine ≥ 0.99 vs reference, **and** p95 Fbank + forward < **30 ms** at
3 s under contention (M4) — deliberately not SPKR-3's 100 ms. 100 ms is the
ceiling for the whole feature; the number that decides whether this works is
whether the embed fits in the slack between the STT final and the prompt build on
a *streaming* backend, and 100 ms does not fit there. M1–M4 must all have values
before Task 3 starts. If the cosine check fails and an hour of frontend debugging
doesn't fix it, switch to sherpa-onnx (research §4 option B) rather than grinding
— the model is the point, not the frontend. If the 30 ms gate fails, take CAM++
on the strength of M3; if CAM++ also misses, ReDimNet2-B0 (research §4) is the
next stop, not a larger wait.

---

### Task 3: `SpeakerTap` — the STT decorator

**Files:** `crates/server/src/speaker.rs`; modify `crates/server/src/call.rs`.

- [ ] `SpeakerTap<S: SttService>` implementing `SttService`, forwarding every
      method to the inner service and returning its frames unchanged.
- [ ] Accumulate `run_stt`'s `Arc<AudioFrame>` into a local `Vec<f32>` with the
      same resample-to-16 kHz logic as `BabelStt::append` (`stt.rs:85`).
- [ ] Own the finalization rules; do **not** claim to mirror the inner service,
      because only `BabelStt` has all of them — Moonshine and Nemotron finalize
      inside their own workers and never expose a buffer to mirror:
      - `flush()` → finalize
      - the tap's own buffer past `MAX_UTTERANCE_SAMPLES` → finalize (matches
        `stt.rs:215` for whisper, and is a sane bound on the tap's own memory —
        ~1.9 MB at the 29 s cap — for the other three)
      - `set_muted(true)` → **clear** the buffer, don't finalize (`stt.rs:231`)
      - while muted, drop frames instead of accumulating (`stt.rs:211`)
      - buffer under `MIN_UTTERANCE_SAMPLES` (300 ms) → drop, no embed
- [ ] Run Fbank **online**, as frames arrive, via `kaldi-native-fbank`'s
      `OnlineFeature` wrapper, so only the ONNX forward pass is left to do at
      finalization. This is what makes the non-blocking join land in time on a
      streaming backend; start the whole pipeline from scratch at `flush()` and
      M7 will be poor for a reason no threshold can fix.
- [ ] Record, per utterance, the sample length of the **first** frame the tap
      accumulates. Task 5 needs the buffer's geometry and the tap is the only
      component that sees it; capturing it here costs nothing.
- [ ] On finalize, `tokio::task::spawn_blocking` the embed and store the
      `JoinHandle` on the call. **Do not await it here** — `flush()` is on the
      turn's critical path.
- [ ] Wrap the `stt` box built at `call.rs:428` only when a model is configured,
      after the backend match, so all four STT backends are covered by one line.
      Task 5 adds a second, independent decorator *inside* this one; the finished
      composition is `SpeakerTap<WakePrerollTrim<S>>`, because the tap must see
      the wake phrase (it is the TD sample) and the STT must not.
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
- [ ] Resolve the embed handle where the transcript is committed and before the
      prompt is built, with **`JoinHandle::is_finished()` and no wait**: finished
      → take it; not finished → `Unknown`, and leave the task running so its
      result still reaches the roster, the event, and the log. Honour
      `SPEAKER_WAIT_MS` as an operator knob, default `0`. The first draft's
      150 ms hard timeout is removed: it is free on whisper and up to 150 ms of
      real turn latency on Nemotron, which is the backend ADR-0005 selects.
- [ ] Count `speaker_ready_rate` — the share of turns whose embed finished before
      the prompt was built — and log it per backend (**M7**). If it is poor on
      `nemotron`, the fix is to start the work earlier (Task 3's online Fbank) or
      to take a cheaper model (Task 2's M3), **never** to raise the wait.
- [ ] `SPEAKER_EVENT` in `crates/protocol/src/lib.rs` and publish on `CallEvents`,
      mirroring `WAKE_EVENT` (`wake.rs:24`). Payload: label, cosine, margin.
- [ ] Prompt integration: when a turn's speaker differs from the previous turn's,
      prepend a short system line ("this turn is from a different speaker than
      the last"). **[assumption]** Prefix, not a schema change — no skill,
      backend, or tool contract changes.
- [ ] Structured log per turn to `logs/speaker.jsonl`, in exactly the shape given
      in **Measurement method** above — including the raw `embedding` and the
      `ground_truth` label. Both are load-bearing: without the vector Task 7's ROC
      cannot be computed at all and the week of household collection has to be
      done twice. Gate on `SPEAKER_LOG=1`, and write the line with the feature
      **off** as well, or M5/M6 have no baseline to subtract.
- [ ] **This file is the input to Task 7's threshold and to M5–M9**, which is why
      Phase 1 ships before enrolment exists: the calibration data is a by-product
      of using the thing.
- [ ] Confirm `/logs/` (`.gitignore:27`) and `/models/*` (`.gitignore:24`) already
      cover the new files rather than adding duplicate rules; add one only if
      `SPEAKER_STORE` lands outside them.

**Gate:** SPKR-1 done. Two people alternating on one call get stable distinct
labels across ≥ 10 turns.

Latency A/B, and it is not optional: 20 turns per condition, the *same* recorded
utterances replayed each time, measuring VAD falling edge → first LLM token.
Four conditions — feature off and on, under `STT_BACKEND=whisper` (**M5**)
and `STT_BACKEND=nemotron` (**M6**). The delta must be within noise on
**both**; whisper alone proves nothing, because whisper is the backend where the
old racing argument happened to be true. If the delta is real on nemotron with
`SPEAKER_WAIT_MS=0`, the cost is CPU contention with the decode (M4), not the
join — take a cheaper model rather than reaching for the knob.

---

### Task 5: correct pre-roll accounting — the invariant, the wire field, and the `todo.md` leakage fix

**Files:** modify `third_party/flowcat-core/src/pipeline/cascaded.rs`, `third_party/flowcat-core/VENDORED.md`, `crates/protocol/src/lib.rs`, `crates/client/src/wake.rs`, `crates/server/src/wake.rs`, `crates/server/src/main.rs`, `crates/server/src/call.rs`.

Standalone value independent of speaker ID: this is the real fix for the
"Hey, one what?" bug in `todo.md`. It is also a hard prerequisite for Task 6 —
without Step 1's invariant there is no correct place to split TD from TI.

**What the first draft of this task got wrong.** "Strip exactly `preroll_ms` of
leading audio from the STT buffer" assumes the buffer starts where the client's
burst starts. It does not. `SpeechGate` accumulates every `InputAudio` it swallows
into a 600 ms ring (`cascaded.rs:514–522`) and drains that ring into a single
`InputAudio` push at the rising edge (`cascaded.rs:525–533`). The ring is never
cleared across a gap, and a native-wake client sends **nothing at all** while
asleep (`crates/client/src/wake.rs:162` returns an empty `Vec`). So what reaches
the STT is:

```
[stale audio from before the last sleep][the 30–100 ms of burst the VAD needed
 to fire][the rest of the burst: wake phrase, then command]
```

— the first two arriving as one oversized frame, the rest as ordinary ones. The
tap cannot decompose it: the ring's length mixes stale and live audio, and
`SttService` is never shown `UserStartedSpeaking` at all (`adapters.rs:117,132`),
so the rising edge is invisible to it. Stripping `preroll_ms` from the front of
*that* removes stale audio plus the head of the wake phrase and leaves the tail
in — precisely the bug it claims to fix.

**Step 1 — one vendored `SpeechGate` patch, removing the ambiguity at source.**

- [ ] Track the arrival `Instant` of the last swallowed `InputAudio`. Before
      appending to the ring, if more than `PREROLL_STALE_GAP`
      (**[assumption]** 200 ms — far above transport jitter, far below the 500 ms
      wake pre-roll) has elapsed since it, `clear()` the ring first.
- [ ] This earns its place independently of speaker ID: today a fresh utterance
      can be prefixed with up to 600 ms of audio captured before the assistant
      went to sleep and handed to whisper as part of the turn. That it is usually
      silence is luck — a wake session happens to expire *on* silence — not design.
- [ ] Record the change as a fifth bullet in
      `third_party/flowcat-core/VENDORED.md`, in the style of the four already
      there.
- [ ] Test in `cascaded.rs`: audio → 1 s gap → audio → `UserStartedSpeaking`
      re-injects only the post-gap audio; and with no gap the re-injected ring is
      byte-identical to today's, so whisper keeps its 600 ms first-phoneme guard.

**Step 2 — the invariant this buys.** The first sample of the utterance the STT
sees is now the first sample of the burst, on every path:

- [ ] native client: nothing flows while asleep, so the gap clears the ring and it
      holds only the burst head the VAD needed;
- [ ] server `WakeGate`: it swallows audio while idle (`wake.rs:174`), so the gate
      sees the same gap, and it pushes `UserStartedSpeaking` **before** its own
      pre-roll (`wake.rs:161–167`) — the gate opens on an empty ring and the
      pre-roll then arrives as ordinary open-gate audio;
- [ ] push mode: no wake fires, `preroll_ms` is 0, nothing is trimmed, and the
      600 ms pre-roll still protects the first phoneme.
- [ ] Assert it: drive a real `SpeechGate` into the tap and check the tap's buffer
      begins exactly at the burst, for all three paths.

**Step 3 — `preroll_ms` on the wire.**

- [ ] Add `preroll_ms: u32` to `WakeState::Awake` with `#[serde(default)]`
      (`protocol/src/lib.rs:71–76`), matching how `persona` is already optional.
      Milliseconds, not samples — client input rate and the 16 kHz carrier differ.
- [ ] Client populates it from the **actually drained** ring length
      (`crates/client/src/wake.rs:154`), not from `preroll_cap`
      (`crates/client/src/wake.rs:103`) — the ring is short for the first half
      second after start-up. `drained.len() * 1000 / input_rate`.
- [ ] `apply_client_wake` (`main.rs:841`) records `(preroll_ms, Instant::now())`
      on `CallState` beside `wake_armed_at`.
- [ ] Server `WakeGate` populates the same field from its own drain
      (`wake.rs:162`) so both wake paths converge on one representation.
- [ ] Round-trip test: an older client omitting the field → `0` → today's
      behaviour exactly.

**Step 4 — `WakePrerollTrim<S: SttService>`, the decorator that actually strips.**

The strip cannot happen "in the STT buffer": whisper's buffer is private to
`BabelStt`, and Moonshine and Nemotron have already streamed the audio into their
workers by the time the utterance ends — there is nothing left to trim. It has to
happen before the audio reaches the service.

- [ ] `WakePrerollTrim<S>` withholds the first `preroll_ms` of each utterance from
      the inner service and forwards everything after it, splitting a frame when
      the boundary falls mid-frame.
- [ ] Install it at `call.rs:428` **unconditionally**: it is a passthrough when no
      wake is armed, and the `todo.md` fix must not depend on `SPEAKER_MODEL`
      being set. Composition is `SpeakerTap<WakePrerollTrim<S>>` with the speaker
      feature on, `WakePrerollTrim<S>` alone with it off.
- [ ] Decide **at the first frame of an utterance**, lazily — "was a wake armed
      within the grace window?" — never eagerly on the wake frame: the WS frame and
      the WebRTC audio ride different transports. This is the assumption
      `WakeGrace` already runs on (`WAKE_ARM_MAX_AGE`, `crates/server/src/wake.rs:212`)
      and it holds for the same reason — the WS frame goes out the instant the head
      fires, while the audio still has an encoder and a jitter buffer to cross.
      Fail open: a late arm means no trim, i.e. today's behaviour, never an eaten
      command. Count those turns.
- [ ] Clamp: never trim so far that fewer than `MIN_UTTERANCE_SAMPLES` remain, and
      never trim a turn with no armed wake.
- [ ] The head fires at or after the end of the phrase, so trimming to the wake
      instant removes the phrase plus at most a few tens of ms after it. If
      command onsets start clipping ("…lay Bowie"), the fix is a small guard that
      keeps the last N ms (**[assumption]** `WAKE_TRIM_GUARD_MS`, default 0),
      **not** a shorter client pre-roll — that pre-roll is also the TD sample
      Task 6 needs.
- [ ] Tests with a mock inner service: exact-boundary trim; boundary mid-frame; no
      wake armed → byte-identical passthrough; `preroll_ms` longer than the
      utterance → clamped and the inner service still sees a valid short one;
      `set_muted(true)` clears both decorators.

**Gate:** the wake phrase no longer appears in transcripts on the native-client
path, and tool selection on "hey babel play X" improves (the failure `todo.md`
traces to this). Verify on a real Pi session against **both**
`STT_BACKEND=whisper` and `nemotron` — the trim's timing behaviour differs
between a batch and a streaming service and the mock covers only one. Record the
skipped-trim count; if it is not near zero, the lazy resolution is losing the race
and the answer is a bounded hold on the first frames, not a longer grace.

---

### Task 6: TD + TI fusion

**Files:** modify `crates/speaker/src/lib.rs`, `crates/server/src/speaker.rs`.

- [ ] Split the tap's buffer at `preroll_ms` into a TD window (`[0, preroll_ms)`,
      the wake phrase) and a TI window (the remainder — byte-identical to what the
      STT saw after Task 5's trim). The split is correct **only** because of
      Task 5 Step 1's invariant; do not start Task 6 until that test is green.
- [ ] Embed both — two forward passes. Check the cost against M2 and M4 first:
      two passes must still clear the Task 2 gate. If they don't, score TD only on
      turns where there is slack (M7), or drop it — a second pass must never be
      paid for out of the turn.
- [ ] `Roster` carries two centroids per speaker. Fuse by mean of the two cosines
      (**[assumption]** — simplest thing that could work; the alternative is TD as
      primary with TI as tiebreak, and Task 7's data will say which).
- [ ] Fall back to TI-only when `preroll_ms` is 0 or absent (push-mode clients,
      older clients) — TD must never be a hard requirement.
- [ ] Fill in the `td_best`, `ti_best`, and `fused` fields the log schema already
      reserves, side by side on the same turn, so the fusion can be evaluated
      against TI-alone over identical audio rather than a second collection. Log
      the TD embedding as well as the TI one — the ROC tool re-scores pairs
      offline and cannot reconstruct a vector it was never given.

**Gate:** on the household set from Task 7, fused EER beats TI-only. **If it
doesn't, delete this task's fusion and keep TI** — Task 5 stands on its own merits
and the research note's 3.62 % vs 8.86 % figure is from a different corpus and
phrase length. This is a hypothesis, not a certainty.

---

### Task 7: Threshold calibration on real household audio

**Files:** `crates/speaker/src/bin/roc.rs`, `docs/research/speaker-bench-m4-max.md`.

- [ ] Collect the calibration set described in **Enrolling the household**
      below: ≥ 50 utterances each from **three** people, from ordinary use through
      the Jabra, at realistic distance, some with music playing, some with the TV
      on. **Not** clean close-mic recordings — those produce an optimistic
      threshold that fails in the room.
- [ ] Offline ROC tool `crates/speaker/src/bin/roc.rs` over `logs/speaker.jsonl`,
      reporting all six distributions: three same-speaker (A–A, B–B, C–C) and three
      cross-pairs (A–B, A–C, B–C). An average over the three people hides the pair
      that will actually misfire. Command, bucketing, and the operating-point table
      it must print are in **Measurement method** above.
- [ ] Pick threshold and margin from **this** curve, with the margin set by the
      **worst** cross-pair rather than the mean. Bias toward `Unknown` — a wrong
      name is worse than no name.
- [ ] Record measured EER at 1 s / 2 s / 3 s beside the published VoxCeleb numbers
      (**M8**) and the false-accept rate at the operating point (**M9**). The gap
      between the two columns is the honest headline of this whole feature and
      belongs in the research doc, not only here.
- [ ] Only now may a threshold and margin be committed as defaults, per the
      measurement rules at the top of this plan. Until this task lands, every
      constant in `.env.example` stays conservative and provisional.

**Gate:** M8 and M9 filled in, all six distributions plotted, the operating point
stated with its false-accept rate, and a written answer to "which two of the three
are hardest to tell apart, and how often are they confused?". If a cross-pair
cannot be separated at an acceptable false-accept rate, say so and ship those two
as mutually `Unknown` — do not widen the threshold to make the matrix look
complete.

---

### Task 8: Enrolment and persistence (SPKR-2)

**Files:** `crates/server/src/skills/speaker.rs`; modify `crates/server/src/skills/mod.rs`.

- [ ] Enrolment skill following the `skills/persona.rs` pattern: "Babel, I'm
      Roland" → collect over several turns → store TD + TI centroids. The amounts,
      conditions, and stored fields are specified in **Enrolling the household**
      below; the skill's job is to collect exactly that and refuse to finish
      early. One-shot enrolment is the main way this feature disappoints.
- [ ] Store as JSON next to `.env` (**[assumption]** — ~768 bytes per centroid, so
      three people is under 5 KB; SQLite is not warranted), in the shape given
      below, including `model_id` and `frontend_hash`. Gitignored. Path via
      `SPEAKER_STORE`.
- [ ] Load at startup; a missing or malformed store logs and degrades to Phase 1
      session labels rather than failing the call. A store whose `model_id` or
      `frontend_hash` doesn't match the loaded embedder is **not** malformed — it
      is a voiceprint from a different embedding space, and it must be refused
      the same way rather than scored against.
- [ ] Adaptation: update stored centroids only above a high-confidence threshold,
      with a per-update drift cap. One bad far-field match must not be able to
      poison a voiceprint permanently.
- [ ] Suppress ID (return `Unknown`, record nothing) while media is playing — the
      radio/Spotify skills mean background music is common and a contaminated match
      is worse than no match. No protocol work is needed: the server already knows,
      because `MediaController` keeps `playing: Mutex<Option<NowPlaying>>`
      (`crates/server/src/media.rs:19`) from the commands it sent. Phase 1's log
      line records the same flag, so Task 7 can measure what suppression costs in
      coverage before it becomes a rule.
- [ ] A "forget me" path that deletes a stored voiceprint. Biometric data with no
      delete story is not something to ship into a household.

**Gate:** SPKR-2 done. Named identification across a server restart, and a stored
voiceprint that survives 50 adaptive updates without drifting past its own
enrolment centroid by more than a documented bound.

---

## Enrolling the household — what each speaker has to provide

Scoped to the first three people. This is the operational half of Tasks 7 and 8:
Task 7 cannot pick a threshold without the calibration set, and Task 8 cannot
store a voiceprint without the enrolment set. They are collected in that order,
and the calibration set collects itself.

### Per person, actively (≈ 5 minutes of their time)

| What | How much | Why this much |
| --- | --- | --- |
| **Name** | one | the label the LLM sees and the key in the store |
| **Wake phrase(s) they actually use** | all of them | TD centroids are *per phrase*. Someone who uses both "hey babel" and "hey marvin" needs two. |
| **TD sample** — the wake phrase, spoken naturally | **≥ 10 fires per phrase** (~1 s each) | TD is the biggest accuracy lever in this design (research §3: 3.62 % vs 8.86 % EER at 3 s). Ten takes give a centroid that isn't one bad morning. |
| **TI sample** — ordinary commands | **≥ 90 s total, ≥ 30 utterances, across ≥ 3 sessions on different days** | the TI-duration curve keeps improving well past 10 s, and different days capture day-to-day voice variation. One-shot enrolment is the single most likely way this feature disappoints. |
| **Condition coverage**, inside that TI sample | ≥ 5 utterances each: at ~1 m, at ~3 m, with music playing, with the TV on | the published EERs are clean and close-mic; the research note budgets 2–5× degradation in this room. A voiceprint enrolled only at 1 m in silence will not match the same person from the sofa. |

All of it through the **real** Jabra on the **real** box — not a phone, not a
headset, not a close-mic recording. Those produce an optimistic threshold that
fails in the room.

### Per person, passively (no extra effort)

| What | How much | Where it comes from |
| --- | --- | --- |
| **Calibration utterances** | **≥ 50** | `logs/speaker.jsonl`, written by Task 4 on every turn. Ship Tasks 1–4, use the assistant normally for about a week with all three people, and this collects itself. |

### Across the three — which is the point of three rather than one

- [ ] Three speakers means **three** same-speaker distributions and **three**
      cross-pairs (A–B, A–C, B–C). Task 7 reports all six.
- [ ] The margin is set by the **worst** cross-pair, never the average.
      Genetically similar voices — parent/child, siblings — are the hard case, and
      every published EER is computed over strangers.
- [ ] If a cross-pair cannot be separated at an acceptable false-accept rate, the
      honest outcome is that those two stay `Unknown` to each other while the
      third is identified. Do not widen the threshold to make the matrix look
      complete.

### What is stored per person (Task 8)

```json
{
  "name": "…",
  "td": {"hey_babel": {"centroid": ["… 192 floats …"], "count": 12}},
  "ti": {"centroid": ["… 192 floats …"], "count": 34},
  "enrolled_at": "2026-09-05T19:04:11Z",
  "last_adapted_at": "2026-09-19T08:22:40Z",
  "model_id": "eres2netv2-3dspeaker-200k",
  "frontend_hash": "… fbank config digest …"
}
```

- [ ] `model_id` and `frontend_hash` (frame length, mel count, dither, CMVN) are
      not decoration. Change either and every stored centroid is a point in a
      different space. The loader refuses a store whose pair doesn't match and
      falls back to Phase 1 session labels — it must never score against it
      anyway, because that failure is silent and looks like drift.
- [ ] ~768 bytes per centroid: three people with one wake phrase each is under
      5 KB of JSON.

### Ongoing

- [ ] **Re-enrolment cadence.** Children's voices go stale in months; adults' shift
      with illness and season. Re-run the TD + TI collection **every ~6 months**,
      and immediately if someone starts reading `Unknown` more than they used to.
- [ ] **"Forget me"** removes that person's entry entirely — TD, TI, and their
      rows in the logs.
- [ ] Whispering, shouting, and a head cold move an embedding a long way (research
      §8: 13.72 % EER on normal-vs-whispered trials). `Unknown` on those turns is
      correct behaviour, not something to tune away.

### Order of operations

1. Ship Tasks 1–4. Session labels work; nobody enrols anything yet.
2. Use the assistant normally for ~a week with all three people. Check
   `logs/speaker.jsonl` has **≥ 50 utterances each** before going further.
3. Run Task 7's ROC. Fill in M8 and M9. *Now* there is a threshold.
4. Only then run Task 8's enrolment: 10 wake fires + 90 s of commands per person,
   in the conditions listed above.

---

## Sequencing notes

Tasks 1–4 are the vertical slice and deliver SPKR-1 alone; stop there and the
feature is already useful (the LLM stops confusing two people in one session).
Task 5 is independently valuable and could ship first if the `todo.md` bug is
more annoying than speaker ID is interesting — it now carries the vendored
`SpeechGate` fix, which is worth having on its own.

Two hard orderings: **Task 5 Step 1 gates Task 6**, because the TD/TI split is
only meaningful once the buffer starts at the burst; and **Task 7 gates Task 8**,
because enrolment built before the household's real score distribution is known
means guessing the threshold twice. The calibration data for Task 7 starts
accumulating the day Task 4 ships, so the week of ordinary use costs nothing if
it is started early.
