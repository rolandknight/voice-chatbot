# Phase 1 matrix results — FlowCat vs cloud Gemma 4 (T6–T12)

| | |
|---|---|
| **Date** | 2026-08-06 |
| **SUT** | `flowcat-poc` (full-duplex build, branch `poc-python-harness` @ `ef4b928`) on 127.0.0.1:6210 |
| **LLM** | OpenRouter paid `google/gemma-4-26b-a4b-it` (never `:free`) |
| **STT** | whisper.cpp `ggml-base.en.bin`, CPU (no GPU found per log) |
| **TTS** | Kokoro shim (`af_heart`), :8880 |
| **Harness** | `poc/harness/test_matrix.py`, markers per test; run order T6, T7, (stack restart), T9, T11, T10, T12, T8 |

Prior context: T1–T5 green on this build (T5 barge-in verified full-duplex).
All timings use the shared Linux monotonic clock (harness, stub log, and
probes are directly comparable). "speech-end" = last non-silent sample of the
fixture on the wire (send-time + outbound backlog + in-fixture speech end).

## Verdicts

| ID | Test | Verdict | Headline numbers |
|---|---|---|---|
| T6 | Concurrent sessions | **PASS** | A e2e 5.84/5.49 s, B e2e 6.61/5.65 s; correct per-client tools+args+replies; no cross-talk |
| T7 | Abrupt teardown | **PASS** | reconnect 0.16 s (budget 2 s); log 140.2 lines/s (budget 200); post-kill T1 e2e 6.04 s |
| T8 | Idle context wipe | **XFAIL** (expected) | bot recalled after 20 s idle: "you asked what time it is." |
| T9 | In-flight vs idle | **PASS** | tool held 14.4 s (12 s injected); reply delivered; follow-up T1 e2e 5.70 s |
| T10 | Latency bench | **PASS** (19/20 turns) | e2e p50 5.55 s / p95 7.04 s (no gate, cloud) |
| T11 | Tool-failure surfacing | **PASS** | spoken: "i'm sorry, i'm having trouble accessing the weather information right now." |
| T12 | Soak, 30 turns | **PASS** | 0 failed turns; p95 first10 6.69 s vs last10 6.86 s; RSS 615→719 MB (+104 MB) |

## Per-test detail

### T6 — concurrent sessions (@concurrency)

Two simultaneous WebRTC sessions, interleaved turns with different tools:
step 1 A:`t1_time`→`get_current_time` ∥ B:`t2_timer`→`set_timer(minutes=5)`;
step 2 A:`t4_bbc`→`play_bbc_radio` ∥ B:`t3_music`→`play_spotify`.

- Each client's reply audio matched its own question (A time-ish, B
  timer-ish, STT-verified) — the effective cross-talk check.
- Tool calls disambiguated by tool identity + args in the shared stub log;
  all four present and correct.
- **Observation:** zero `transcript-user` events arrived on either events
  WebSocket during the test — either the server emits none for user turns or
  the shapes don't map to the adapter's permissive normalizer. The
  transcript-ownership assertion is therefore vacuous; cross-talk is covered
  by audio + tool-args only. Worth a Rust-side look at what
  `/webrtc/events/{pc_id}` actually emits mid-session.
- Both sessions closed cleanly (graceful close, no server-side errors).

### T7 — abrupt teardown (@teardown)

Long counting reply (`t5_long`), peer killed mid-reply by dropping the
ICE/UDP transport (no SDP bye, no DTLS close_notify, no WS close frame —
adapter `close(graceful=False)`).

- New connection succeeded **0.16 s** after the kill (budget 2 s) and
  completed a full T1 turn (e2e 6.04 s).
- flowcat.log grew at **140.2 lines/s** over the 5 s window — under the
  200 lines/s spin guard, but high. Content inspection: virtually all of it
  is steady-state `ort::logging` INFO chatter (BFCArena allocs per ONNX
  VAD/Smart-Turn inference) plus whisper.cpp stderr — **inference logging,
  not error spin**. No `MediaStreamError`-style loops observed. Recommend
  silencing ort INFO logs in the server build; it would also make the spin
  guard far more sensitive.

### T8 — idle context wipe (@lifecycle, xfail strict=False)

T1 turn, 20 s of silence, then `t8_recall` ("What did I just ask you
about?").

- **Actual behavior: full recall.** Reply verbatim (harness STT): "you asked
  what time it is." Session and transport remained fully usable.
- Recorded finding, as expected: the cascaded builder hard-disables the idle
  timeout, and Babel's CONV-5 idle context wipe is not implemented. The
  xfail marker keeps this visible in every run without failing the suite.

### T9 — in-flight tool latency vs idle (@inflight)

`/admin/latency get_weather=12s`, then `t9_weather`.

- Tool call logged immediately; first reply audio **14.4 s** after the tool
  call (12 s hold + ~2.4 s LLM continuation + TTS). Reply content intact:
  "it is currently 18 degrees and cloudy here."
- No idle-timer kill of the in-flight call (consistent with T8's finding
  that the idle timeout is disabled outright — this test can't distinguish
  "in-flight tracking" from "no timer at all"; re-test if idle timeout is
  ever enabled).
- Same session then completed a clean T1 turn (e2e 5.70 s). Latency cleared
  after.

### T10 — latency bench, 20 warm turns (@latency)

Alternating `t1_time`/`t10_date`, single session, per-turn stub log clear.
19/20 turns completed; turn 8 flaked (see below).

| segment | p50 | p95 |
|---|---|---|
| speech-end → first bot audio (E2E) | **5.55 s** | **7.04 s** |
| speech-end → tool call (≈STT + LLM TTFB) | 3.87 s | 4.55 s |
| tool call → first audio (≈LLM cont. + TTS) | 1.87 s | 2.56 s |

- No hard gate by design: cloud LLM TTFT (0.7–1.8 s), CPU whisper.cpp batch
  windows (~4 s segments dominate the 3.87 s STT+LLM segment), and the VAD
  stop_secs 0.5 s floor together put the local 1.5 s E2E target out of scope
  for Phase 1. These numbers are the framework-overhead baseline for the
  Phase 2 (local) comparison.
- First turn of a fresh session is cold: 10.4 s e2e here (17.2 s in an
  earlier aborted run) — whisper/provider warm-up; excluded from nothing,
  included in the percentiles above (turn 1 is in the sample).
- **Flake pattern:** both bench attempts flaked exactly once on
  `t10_date` — whisper base.en transcribed "What's the date today?" as
  "That the date today" (earlier run) and the LLM then answered nothing /
  no tool call. ~1-in-10 STT miss rate on this particular fixture;
  `t1_time` never flaked. Bench requires ≥15/20 and records flakes.

### T11 — tool-failure surfacing (@failures)

`/admin/fail get_weather=500`, then `t9_weather`.

- The turn did **not** die silently: `get_weather` call logged, and the bot
  spoke "i'm sorry, i'm having trouble accessing the weather information
  right now." (The session layer's "The get_weather service returned an
  error." synthetic tool result reached the LLM and was verbalized.)
- Fail cleared; same session completed T1 (e2e 4.99 s).
- Note: this is the re-scoped T11 — the plan's max_tokens truncation probe
  is not implementable (FlowCat's LLM client exposes no max_tokens knob;
  already recorded as a finding).

### T12 — soak, 30 turns (@soak)

Alternating `t1_time`/`t2_timer`, single session, ~6 min of continuous turns.

- **30/30 turns succeeded** (0 flakes — notably the flake-prone date fixture
  is not in this rotation).
- e2e: p50 5.65 s, min 4.94 s, max 7.26 s. p95 first-10 6.69 s vs last-10
  6.86 s (+2.5%) — no drift.
- flowcat RSS 615 MB → 719 MB (**+104 MB**, budget 200 MB). Passed, but
  +3.5 MB/turn is worth watching: plausibly conversation-context growth in a
  single 30-turn session (all 8 tool schemas + growing history per LLM call)
  rather than a leak. A cross-session soak (fresh session per N turns) would
  separate leak from context growth — suggested for Phase 2.

## Oddities / follow-ups for the Rust side

1. `/webrtc/events/{pc_id}` produced no mappable transcript events during
   T6 (or any matrix test) — confirm what event shapes the server emits.
2. `ort::logging` INFO spam is ~140 lines/s during any active session;
   silence it (it masks real log-spin and bloats logs ~0.5 MB/min).
3. whisper.cpp runs CPU-only ("no GPU found" at every model init — model is
   re-initialized per segment, visible as repeated `whisper_init_state`
   blocks); the 6 GB dev-box GPU is idle. CUDA feature unification or the
   speaches sidecar would cut the dominant latency segment.
4. RSS +104 MB over a 30-turn session (see T12).
5. base.en mishears "What's the date today?" ~1-in-10 ("That the date
   today"); consider small.en for Phase 2 content-accuracy runs.

## Phase 1a/1b (run by coordinator after the harness agent hit its session limit)

### T13 — server-side wake, Listen mode: **PASS** (66 s)

Stack: `POC_WAKE_MODEL=models/wakeword/hey_babel.onnx` (WakeGate between VAD
and SpeechGate; vendored oww_rs detector, threshold 0.5).
- (a) `t1_time.wav` without wake word: swallowed — no bot speech, no tool
  calls for 15 s. ✓
- (b) `t13_wake.wav` ("Hey babel, what time is it?"): turn fired,
  `get_current_time` called, reply "it's 4.16pm." spoken. Probes:
  wake-onset→first-audio **7.59 s**, speech-end→first-audio **6.09 s**
  (CPU whisper + cloud LLM; same latency profile as T10).
- (c) after the 15 s session window expired, wake-less speech was again
  swallowed (gate re-armed). ✓

### T14 — cloned-voice TTS (Chatterbox, CUDA): **PASS** (30 s)

Stack: `POC_TTS_BACKEND=chatterbox`; Chatterbox-TTS-Server on :8004 cloning
from the real production Marvin reference clip (voices/ "look at this
door…", converted to 24 kHz WAV). T1 turn completed through the cloned
voice; pitch discrimination decisive: reply median F0 **82 Hz** vs Kokoro
af_heart **200 Hz** re-synthesis of the same text (ratio 0.41, budget
< 0.8). Segments: tool→first-audio 2.21 s, e2e 5.93 s — Chatterbox adds
~0.3–1 s vs Kokoro, acceptable. One flaky first run failed the F0 assert
(short garbled-STT reply); two consecutive reruns green.

Notes: the agent's earlier Chatterbox 422 was a probe payload missing the
required `model` field — not a server issue. STT oddity logged: "4:16 PM"
transcribed back as "4000 atm" by harness tiny.en in one run (assertion
regex tolerant by design).
