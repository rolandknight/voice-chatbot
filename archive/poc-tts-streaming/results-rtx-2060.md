# poc-tts-streaming on an RTX 2060 — time-to-first-audio

Measured 2026-08-24 on `pop-os`, NVIDIA GeForce RTX 2060 (6 GB, compute
capability sm_75), driver 580.159.03, torch 2.6.0+cu124. Voice: `one-one.mp3`.

**Current defaults** (`config.yaml`, as of the "Follow-up batch" section
below): `num_steps: 4, n_cfm_timesteps: 1, chunk_size: 300, temperature: 0.5,
split_text: true, split_on_clauses: true`, resolved dtype `float16`, backend
`torch` (SDPA — flashinfer is unavailable on this card; see
`poc-tts/bench-rtx-2060.md`), and `engine.block_streaming: true` —
**effective** here (this card resolves to CUDA + torch backend), so the
headline numbers below are the block-streaming engine's. See "Follow-up batch
(2026-08-24): new defaults" for what changed, why, and the full re-bench.

Raw data: [`reports/stream_runs.jsonl`](reports/stream_runs.jsonl).
Reproduce the engine column with `python -m poc_tts_streaming.bench_stream
--block-stream --runs 3` (the effective default engine here) or plain `make
bench-stream` (sentence engine, what non-CUDA/torch backends fall back to).

| sentence | chars | chunks | TTFA engine (s) | TTFA browser (s) | TTFA server (s) | total gen (s) | audio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (30 ch) | 30 | 1 | 0.442 | 0.681 | 0.473 | 0.889 | 2.08 |
| medium (104 ch) | 104 | 1 | 0.440 | 0.739 | 0.501 | 1.845 | 5.60 |
| long (317 ch) | 317 | 2 | 0.441 | 0.706 | 0.500 | 5.593 | 17.70 |
| Dickens excerpt (945 ch) | 945 | 4 | — | 0.820 | 0.512 | 18.557 | 59.98 |

- **TTFA engine** — best-of-3, `t0` (call into `synthesize_stream`) to the
  first `(chunk_text, pcm)` yielded, block-streaming engine
  (`python -m poc_tts_streaming.bench_stream --block-stream --runs 3`).
  Excludes WebRTC/HTTP transport, encode, and the browser's jitter buffer —
  it is the engine's own floor. No row for the Dickens excerpt: outside
  `bench.py`'s three baseline sentences.
- **TTFA browser** / **TTFA server** / **total gen** / **audio** — headless
  Chrome driving the real UI at config-default knobs and the same voice, one
  run per row (not best-of-N), from the controller's live measurement (see
  the follow-up batch section). Browser TTFA is `response.created` sent →
  first non-silent sample at the AnalyserNode (includes Opus encode, jitter
  buffer, decode); server TTFA is `response.created` → `output_audio_buffer.
  started` at the server side of the session (no Opus decode / jitter
  buffer, but still inside the full WebRTC session, unlike the engine
  column).
- **chunks** — transcript chunks (real `chunk_text` splits / `response.
  audio_transcript.delta` events), not the block engine's internal windows
  (see "Which configuration each table below comes from" under Task 16
  spike for the block-join/chunk-join distinction).

Engine TTFA is now flat at ~0.44 s regardless of sentence length (previously
it scaled with sentence length — see the "previous default" table below) and
beats the poc-tts whole-utterance baseline (0.59 s / 1.03 s / 3.38 s for
short/medium/long) by a wider margin than before on every sentence. Browser
TTFA sits roughly 0.24–0.30 s above engine TTFA — WebRTC/session overhead
(client-secret round trip, SDP offer/answer, ICE, Opus encode, jitter buffer)
that generation improvements don't touch, and server TTFA sits in between, as
expected for a narrower measurement window than browser but still inside the
full session.

### Previous default (chunk_size 120, temperature 0.6, sentence streaming only)

The table this doc originally shipped with, kept here for comparison rather
than deleted. Generation config was `chunk_size: 120`, `temperature: 0.6`;
`engine.block_streaming` did not exist yet, so sentence streaming was the
only path. Sections below this point that predate the follow-up batch
(`Server TTFA`, `HTTP chunked-PCM reference points`, `Cold start`, `Gaps
between chunks`) were measured at this same previous-default config unless
individually dated otherwise.

| sentence | chars | chunks | TTFA engine (s) | TTFA browser (s) | total gen (s) | audio (s) | poc-tts whole-utterance (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 30 | 1 | 0.524 | 1.136 | 0.52 | 2.32 | 0.59 |
| medium | 104 | 1 | 0.847 | 1.455 | 0.85 | 5.16 | 1.03 |
| long | 317 | 4 | 0.843 | 1.424 | 3.51 | 18.56 | 3.38 |

- **TTFA engine** — `t0` (call into `synthesize_stream`) to the first
  `(chunk_text, pcm)` yielded, from `reports/stream_runs.jsonl` (best-of-2,
  measured by this task's `make bench-stream` run). It excludes WebRTC/HTTP
  transport, encode, and the browser's jitter buffer — it is the engine's own
  floor.
- **TTFA browser** = `response.created` sent → first non-silent sample at the
  AnalyserNode, via headless Chrome driving the real UI at config-default
  knobs and the same voice. Includes Opus encode, jitter buffer, and decode.
  Measured by the controller, not by this bench script.
- **total gen (s)** / **audio (s)** are the engine bench's totals across all
  chunks of that sentence (`gen_s`, `audio_s` in the JSONL row).
- **poc-tts whole-utterance (s)** — `poc-tts/bench-rtx-2060.md`'s tuned row
  (`drf_block_size=32, num_steps=4, n_cfm_timesteps=1`); poc-tts has no
  streaming path, so this is the time until *any* audio exists at all —
  the number poc-tts-streaming's TTFA is meant to beat.

Engine TTFA already beats the poc-tts whole-utterance baseline on short and
medium (0.52 s vs 0.59 s; 0.85 s vs 1.03 s) and by a wide margin on long
(0.84 s vs 3.38 s), because only the first chunk has to finish before audio
starts, not the whole paragraph. Browser TTFA is roughly 0.6 s higher across
the board — that gap is WebRTC/session overhead (client-secret round trip,
SDP offer/answer, ICE, Opus encode, jitter buffer), not generation.

Long sentence: 4 chunks, matching the engine bench's own 4-chunk split.

## Server TTFA (response.created → output_audio_buffer.started)

A second controller measurement, taken at the server side of the WebRTC
session rather than the browser's AnalyserNode — narrower than browser TTFA
(no Opus decode / jitter buffer) but still inside the full session, unlike
the engine column above:

| sentence | server TTFA (s) |
|---|---:|
| short | 0.772 |
| medium | 1.228 |
| long | 1.187 |

Long, at the server: 4 chunks, total gen 5.153 s, audio 28.711 s (a
different run from the engine-bench row above — Flash's output length is
not deterministic between runs; see `poc-tts/bench-rtx-2060.md`'s
"Read gen_s, not RTF" section). Gaps between chunks: none observed — the
level meter stayed continuous, and total generation time is far below
total audio duration on every sentence, so the synthesis worker always
stays ahead of playback.

## HTTP chunked-PCM reference points (`/v1/audio/speech`, warm)

For a 319-char Dickens paragraph, at `chunk_size: 120` (config default):
TTFA 1.24 s. Reducing `chunk_size` to 50 (a smaller first chunk) drops that
to 0.58 s. Whole generation ranged 3.3–6.0 s for 21–24 s of audio across
these runs — consistent with the non-deterministic output length noted
above.

## Cold start

The very first generation after model load took about 14.6 s (CUDA
warm-up / kernel compilation). Every number above is warm — `make
bench-stream` runs an explicit warm-up request before measuring, and the
browser/server numbers were taken against an already-warm server.

## Gaps between chunks

Not observed, by ear or by the level meter, on any sentence: the worker's
per-chunk generation time is consistently far shorter than that chunk's
audio duration (e.g. long: 0.80–1.02 s to generate each ~4–5 s chunk), so
the next chunk is always ready well before the previous one finishes
playing.

## Task 16 spike: intra-sentence block streaming — **GO**

Sentence pipelining still leaves the *first* sentence's whole generation on
the critical path. T3 fills its speech-token buffer a `drf_block_size` (32)
block at a time, so this spike asked whether each finished block can be
vocoded and played while T3 is still working on the next one — and whether
the result still sounds like one sentence.

Code: [`poc_tts_streaming/engine_blockstream.py`](poc_tts_streaming/engine_blockstream.py),
behind `engine.block_streaming` in `config.yaml`, **default `false`**.

Evidence is reproducible — the analysis scripts are committed:

```
python -m poc_tts_streaming.bench_stream --block-stream --runs 3   # TTFA -> reports/stream_runs.jsonl
python -m poc_tts_streaming.spike_analysis seams --runs 3          # TTFA + seams -> reports/spike_seams.json
python -m poc_tts_streaming.spike_analysis paired                  # windowed vs one-shot vocoding
python -m poc_tts_streaming.spike_analysis lookahead               # what the finalize workaround changes
```

### Which configuration each table below comes from

Two configurations appear, and they are not interchangeable:

- **Benched (chunked).** `config.yaml`'s knobs — `chunk_size: 120`,
  `split_text: true` — so text is chunked exactly as the shipping path chunks
  it. The long paragraph becomes 4 chunks. This is what `bench_stream` and
  `spike_analysis seams` run, and it is the configuration the TTFA and seam
  numbers below come from. Block streaming adds **block joins** inside each
  chunk; the **chunk joins** between chunks are the same ones the
  sentence-level path already has.
- **Single-chunk (paired).** `spike_analysis paired` only, and only because
  its point is that the windowed and one-shot vocoders see an *identical*
  token sequence — which forces one chunk. It is the exact-tiling and
  log-mel evidence, not the seam-at-a-join evidence.

### Prosody is preserved by construction

Block streaming changes *nothing* about token generation: T3 still sees the
whole sentence's text and still runs its full block-diffusion loop.
`generate_blocks()` is a copy of `ChatterboxFlashT3.generate`'s single-sample
torch-SDPA path with an `on_block` callback added after each block is
committed to the KV cache; every RNG-consuming line is untouched.
`tests/test_blockstream.py::test_generate_blocks_matches_generate` pins that
down — same seed in, byte-identical speech tokens out — so the prosody
question reduces to "does windowed vocoding of the same tokens sound the
same", not "does chopping the sentence change the delivery". It does not:
the sentence is never chopped.

**The invariant was violated on the shipping path, and is now fixed.** Found
2026-08-24: the identity test drove `on_block` with a callback that does
nothing, and *there* the loop was byte-identical — but the **production**
callback vocodes, and vocoding consumes torch RNG.
`CausalConditionalCFM.forward` does `z = torch.randn_like(mu)` unconditionally
even when `noised_mels` is supplied (`s3gen/flow_matching.py:216`, the draw is
made and *then* overwritten), and HiFTGAN's NSF source draws `torch.randn_like`
per call (`s3gen/hifigan.py:226,282`) plus a `Uniform.sample` for the sine
phase (`:212`). Every window therefore advanced the same global stream the
decode loop samples its *later* blocks from, and `_MelWindow.__init__`'s noise
pre-draw shifted it once more before the loop even started. Over 40 seeds the
no-op callback reproduced `t3.generate`'s token counts exactly (40/40); the
vocoding callback did not. Same distribution, different utterance — so
"byte-identical tokens / same prosody as the sentence path" was false of the
path that ships.

The fix (2026-08-24) is `engine_blockstream._fork_rng`: `torch.random.fork_rng`
around the noise pre-draw and around each `_MelWindow.push`, snapshotting the
CPU generator and the CUDA device's and restoring both on exit. `randn_like`
takes no `generator` argument, so routing the vocoder at a private
`torch.Generator` would mean forking the installed package; fencing the global
one is the practical route. Inside the fence the vocoder still gets
deterministic noise for a fixed seed; outside it the decode loop walks exactly
the stream `t3.generate` would. One fence per window and one for the pre-draw
— never one around the whole utterance, which would put the T3 draws inside it.

- **Overhead: 31.8 µs per fence** (5056-byte CPU state copy + 16-byte CUDA
  one, each way), against a 130 ms mean window on this box — **0.024 %**. A
  5.45 s utterance pays 0.22 ms across 6 windows and the pre-draw, against
  782 ms of vocoding. Not measurable in TTFA.
- **Cover.** `test_generate_blocks_matches_generate` now runs twice: with the
  no-op callback (pins the loop copy) *and* with the real vocoding one (pins
  the fence). `test_block_and_sentence_engines_draw_the_same_tokens` compares
  the two engines end to end — one sentence, one seed, identical trimmed token
  tensors. Both were confirmed to fail with the fence disabled: the vocoding
  half drew 109 tokens against `t3.generate`'s 120, and the two engines drew
  82 against 99.
- **Scope of the guarantee: per chunk, from the same RNG state.** Across a
  multi-chunk utterance from a single seed the two engines still diverge from
  chunk two on, for the mirror-image reason: the *sentence* path's vocoder
  consumes global RNG between chunks (it is one `model.generate` call, T3 and
  S3Gen together) and the fenced block path's no longer does. Fixing that
  would mean fencing the sentence path too, which changes its output for a
  given seed on the flag-off path. Not done here.
- **Audio, same tokens.** The two renders still differ, because their vocoder
  noise differs: the block path slices one fixed pre-draw across its windows,
  the sentence path takes a fresh CFM `z` for the utterance. Reported by the
  end-to-end test, not asserted: rel RMS 125.5 %, **log-mel 4.55 %** — against
  a control of the same tokens through `s3gen.inference` *twice*, which is
  rel RMS 112.6 % / **log-mel 4.12 %**. Two renders of one token sequence are
  near-uncorrelated sample-wise whatever you do (the CFM redraws a full-length
  `z` per call, so each is a fresh sample of the same distribution), which is
  why only the phase-blind number is worth reading — and on it the block path
  sits at the redraw floor. Note this control is *not* `spike_analysis
  paired`'s 0.06–0.16 % one: that holds the mel fixed and re-runs only
  HiFTGAN, isolating the windowed vocoder; this one holds only the tokens
  fixed, which is what "same seed, two engines" actually means.

### TTFA (benched configuration)

`spike_analysis seams --runs 3`, both paths off one set of loaded weights,
warm, best of three:

| sentence | windows (sentence) | TTFA sentence-level (s) | windows (block) | TTFA block-stream (s) | change |
|---|---:|---:|---:|---:|---:|
| short  | 1 | 0.534 | 3–4   | 0.428 | **−19.8 %** |
| medium | 1 | 0.976 | 6     | 0.464 | **−52.5 %** |
| long   | 4 | 0.958 | 20–22 | 0.422 | **−55.9 %** |

Per-run TTFA, so the spread is visible:

```
[blockstream]   short: [0.4282, 0.4351, 0.4638]   [sentence]   short: [0.5671, 0.5336, 0.5637]
[blockstream]  medium: [0.4684, 0.4639, 0.4973]   [sentence]  medium: [0.9757, 1.1486, 1.0835]
[blockstream]    long: [0.4620, 0.4225, 0.4458]   [sentence]    long: [1.1333, 0.9653, 0.9576]
```

`bench_stream --runs 3` agrees independently (0.529 → 0.430, 1.049 → 0.433,
1.124 → 0.447 = −18.7 % / −58.7 % / −60.2 %). **Medium clears the ≥ 40 % gate
in every measurement set** (−52 % to −59 %). Short is the noisy one, and for a
structural reason: its floor is the fixed per-chunk cost (reference-clip load,
`prepare_conditionals`, tokenisation, T3 prefix forward), which block
streaming does not touch. Block-streamed TTFA is essentially *flat* at
0.42–0.50 s across all three sentences — it no longer depends on sentence
length, which is the real result.

### Cost

Re-running the flow over the whole token prefix each block is quadratic
within a chunk. Measured RTF (`gen_s / audio_s`) rose from 0.18–0.30 to
0.29–0.38 — about 1.6× the GPU work, still far below 1.0, so the synthesis
worker stays comfortably ahead of playback on every sentence. `chunk_size:
120` caps the quadratic term at one chunk, which is why the long paragraph is
no worse than the medium sentence.

### Seams (benched configuration, 3 renders per sentence)

`spike_analysis seams --runs 3` → `reports/spike_seams.json`. Listening
samples are in `reports/spike-wavs/` (gitignored).

**Step ratio at joins** — `|x[j] − x[j−1]|` over the median `|diff(x)|` in the
surrounding ±50 ms, with a control drawn from 40 evenly-spaced offsets in the
same render. Pooled over all three runs:

| render | join type | n | median | max | control: median / p95 / max |
|---|---|---:|---:|---:|---|
| blockstream / short  | block | 7  | 0.56 | 4.69 | 1.05 / 4.07 / 7.81 |
| blockstream / medium | block | 15 | 0.56 | 1.82 | 0.96 / 4.91 / 93.79 |
| blockstream / long   | block | 51 | 0.72 | 5.31 | 1.11 / 6.31 / 13.20 |
| blockstream / long   | chunk | 9  | 0.29 | 0.82 | 1.11 / 6.31 / 13.20 |
| sentence / long      | chunk | 9  | 0.44 | 1.41 | 0.88 / 4.80 / 13.84 |

Pooled across all nine block-streamed renders: **73 block joins, median step
ratio 0.60, p95 2.98, max 5.31, against a control median of 1.05 — and 1 of 73
exceeds its own render's control p95.** The joins block streaming *adds* are
quieter than an arbitrary offset in the same speech.

**RMS jump**, as a percentile of the same statistic at every other offset:

| render | join type | n | jump % (median) | percentile (median / max) | >20 % at an arbitrary offset |
|---|---|---:|---:|---:|---:|
| blockstream / short  | block | 7  | 10.6 %  | 48.4 / 95.9 | 37.4 % |
| blockstream / medium | block | 15 | 26.0 %  | 65.1 / 95.2 | 41.1 % |
| blockstream / long   | block | 51 | 12.3 %  | 46.1 / 99.3 | 42.1 % |
| blockstream / long   | chunk | 9  | 200.0 % | 99.6 / 99.7 | 42.1 % |
| sentence / long      | chunk | 9  | 200.0 % | 99.6 / 99.6 | 42.1 % |

The raw ">20 %" threshold does not discriminate for speech: 37–42 % of *all*
offsets in these very utterances already clear it. Calibrated as a percentile,
block joins sit at the 46th–65th — typical.

**Chunk joins are inherited, not improved.** The block-streamed long paragraph
has the *same* chunk joins as the sentence-level path — both read 200 % jump
at the 99.6th percentile there, because a chunk boundary is a
silence-to-silence transition between two independent `generate()` draws. That
is a property of sentence chunking, present in the shipping path today, and
block streaming neither fixes nor worsens it. The honest statement is: **the
joins block streaming adds are quieter than the joins the pipeline already
has** — not that overall join quality improves.

**Glitch scan** — the same step ratio evaluated *everywhere*, to catch an
artefact that is not at a join at all. Worst offsets in each run-0 render, with
the nearest join and its type:

```
blockstream/short    1.332s r=16.5   nearest join 1.000s (block, +0.332s)
blockstream/medium   4.431s r=22.2   nearest join 4.840s (block, -0.409s)
blockstream/medium   3.507s r=20.8   nearest join 3.560s (block, -0.053s)
blockstream/long     6.248s r=49.7   nearest join 6.280s (chunk, -0.032s)
blockstream/long    18.564s r=33.7   nearest join 18.320s (block, +0.244s)
sentence/short       0.751s r=16.1   (no joins in this render)
sentence/medium      0.460s r=19.1   (no joins in this render)
sentence/long        9.688s r=43.6   nearest join 9.720s (chunk, -0.032s)
```

Two things to read here. First, the per-render maxima are comparable between
the two paths on the same text (short 16.5 vs 16.1; medium 22.2 vs 19.1; long
49.7 vs 43.6) — block streaming is not introducing a class of transient the
sentence-level path lacks. Second, the largest values in *both* paths sit
32 ms before a **chunk** join, i.e. they are the tail of a chunk, again
inherited rather than new.

### Windowed vs one-shot vocoding (single-chunk, `spike_analysis paired`)

Same tokens, same CFM noise draw, windowed vocoder vs a single-shot vocode.
This isolates the vocoder; it is not a chunk-join measurement.

| sentence | tokens | expected (960 × tokens) | streamed | one-shot | tiling |
|---|---:|---:|---:|---:|---|
| short  | 50  | 48 000  | 48 000  | 48 000  | OK |
| medium | 157 | 150 720 | 150 720 | 150 720 | OK |
| long   | 451 | 432 960 | 432 960 | 432 960 | OK |

Exact in all three: nothing repeated, nothing truncated.

| sentence | log-mel: streamed vs one-shot | control (one-shot × 2) | waveform rel RMS | control |
|---|---:|---:|---:|---:|
| short  | 3.29 % | 0.06 % | 71.45 % (corr 0.74) | 0.76 % (corr 0.99997) |
| medium | 3.72 % | 0.16 % | 109.11 % (corr 0.39) | 1.93 % (corr 0.99981) |
| long   | 4.75 % | 0.23 % | 99.59 % (corr 0.46) | 3.20 % (corr 0.99949) |

The waveform figure is pure phase — HiFTGAN restarts its sine phase
accumulator per call, so the streamed excitation is phase-shifted relative to
a one-shot render; absolute excitation phase is inaudible. The meaningful
number is **log-mel: 3.3–4.8 %**, against a 0.06–0.23 % floor, growing with
window count. That is a content difference, not a discontinuity, and it has
two plausible contributors, both inherent to windowing:

1. **Right context.** Each non-final window's mel is computed from the tokens
   generated so far, so frames near the window's end see less future than a
   one-shot pass gives them. `spike_analysis lookahead` measures the tail
   effect directly (below).
2. **Cross-fade cancellation.** 8 mel frames per join are rendered twice, with
   different excitation phase, and Hamming-cross-faded; partial cancellation
   in the overlap is expected.

Which dominates has not been separated. Both are listed rather than one being
asserted.

### What broke on the way (and how)

1. **`flow_inference(..., finalize=False)` raises.** In chatterbox-tts 0.1.7,
   `CausalMaskedDiffWithXvec.inference` (`flow.py:170-182`) truncates `h` by
   `pre_lookahead_len * token_mel_ratio` and then multiplies it by a `mask`
   built from the *untruncated* `h_lengths`:
   `RuntimeError: The size of tensor a (558) must match the size of tensor b
   (564) at non-singleton dimension 2`. The `S3GenStreamer` the `s3gen.py:278`
   docstring points at does not exist in this package either.

   The spike passes `finalize=True` and drops the trailing frames from the
   *output* instead. **That is not the same computation.** Upstream truncates
   before `encoder_proj` and the CFM, so the lookahead positions never enter
   `mu` and cannot condition the retained frames; we leave them in and discard
   output frames, so our retained frames get *more* right context. Measured
   (`spike_analysis lookahead`, medium sentence, 131 tokens):

   ```
   ours vs upstream-truncated : rel RMS 0.7670%  corr 0.999814
   CONTROL truncated, rerun   : rel RMS 0.6385%  corr 0.999869
   per-frame rel diff, last 8 frames:
     ours    : 6.72e-03 6.69e-03 6.63e-03 8.04e-03 1.28e-02 1.91e-02 8.61e-03 3.77e-02
     control : 6.33e-03 5.98e-03 5.89e-03 5.96e-03 5.18e-03 4.17e-03 3.58e-03 5.00e-03
   ```

   Most of the 0.77 % is the control floor (the CFM redraws prompt-region
   noise every call). The real effect is localised exactly where theory says
   it should be: the last ~5 frames differ by 3–7× the control, earlier frames
   sit at the control level.

2. **The meanflow CFM redraws its noise on every call.** `flow_inference`
   unconditionally does `noise = torch.randn(1, 80, n_tokens*2)`, so each
   window would be a different stochastic draw of the same speech and the
   emitted frames would not be the ones the next window continues. Fixed by
   drawing the noise once per utterance and slicing it, which needs
   `S3Token2Mel.forward` with an explicit `noised_mels` rather than
   `flow_inference`. Before the fix, mel prefix drift between window sizes was
   3.2–4.3 %, against a 3.3–3.6 % redraw floor — i.e. entirely noise.

3. **`cache_source` alone does not make the vocoder continuous.** Seeding
   `hift_inference(cache_source=…)` fixes the NSF phase only for the cached
   samples; the fresh `theta = cumsum(f0)` restarts at 0 each call and meets
   the cached phase with a step. The first implementation did exactly what the
   task brief describes and produced a measurable click at every join — step
   ratio 9–45× the local median against a control of ~1. CosyVoice2's actual
   answer is a third cache: hold the last 8 mel frames' *audio* back,
   re-render it with the next window, and cross-fade the two renderings
   (`fade_in_out`, Hamming over 2 × 3840 samples). With that, the joins
   measure as tabulated above.

### Post-EOS hallucination report (2026-08-24) — **not reproduced**

A user running the flag on in the browser reported hallucinated trailing
speech — babble after the intended text, inconsistent across runs — and the
working hypothesis was that the streamed path never trims at
`stop_speech_token` and so speaks out the rest of the token budget. **That is
not what happens.** The evidence, all on the RTX 2060 at `config.yaml`'s knobs
(`num_steps: 4`, `n_cfm_timesteps: 1`, `drf_block_size: 32`):

1. **`generate_blocks` already stops at EOS.** It carries the same early
   return as upstream `ChatterboxFlashT3.generate`: the EOS block is truncated
   at the stop token (exclusive) and the loop returns. Over 40 seeds on the
   reported sentence, `generate_blocks` with a no-op callback produced
   *identical* token counts to `t3.generate` — including which runs
   terminated where. Typical run: budget 300 tokens / 10 blocks, EOS in
   block 3, 4 callbacks, 103 tokens returned, 0 stop tokens in the result.

2. **There is a second, independent guard.** `_stream_chunk`'s `on_block`
   applies `model._trim_to_eos` to *every* block prefix before it reaches the
   vocoder, so even a loop that over-ran could not push post-EOS tokens
   through S3Gen — `_MelWindow` would see a prefix that stopped growing and
   emit nothing.

3. **Sample counts are exact.** Streamed samples minus `trimmed_tokens × 960`
   was **0 on every one of 160 runs** (2 voices × 2 knob sets × 2 engines ×
   20 runs). Same-token A/B — one runaway token stream vocoded (a) one-shot
   through `s3gen.inference` and (b) through `_MelWindow` as it ships — gave
   288000 samples both ways, delta 0.

4. **No extra speech, either.** Measuring *active* (non-silent) audio rather
   than total, block streaming is level with or slightly below the
   sentence-level path:

   | text | engine | total_s med / max | active_s med / max |
   |---|---|---:|---:|
   | "The time is 5 oclock. it also is sunny outside." | blockstream | 3.88 / 7.56 | 2.71 / 3.06 |
   | | sentence | 3.88 / 6.36 | 2.83 / 3.14 |
   | 3-sentence reply (multi-chunk) | blockstream | 8.36 / 14.96 | 6.13 / 6.94 |
   | | sentence | 8.30 / 15.12 | 6.25 / 7.38 |

   30 runs each. Repeated with `marvin.mp3` and with the hot preset knobs
   (temperature 0.85, exaggeration 1.2, cfg 0.55): same picture.

**What is real** is a different, engine-agnostic failure. Roughly **1–2 % of
chunks never emit EOS at all** and run the full
`_speech_len_for_text_tokens` budget — `max(6 × n_text, 300)`, so a 46-char
sentence gets a 300-token / **12 s** floor. Measured over 200 trials on the
reported sentence: `t3.generate` 2/200, `generate_blocks` with an
RNG-consuming callback 4/200 — statistically indistinguishable, and the
sentence-level path hit it too (2/30 on the multi-chunk text). **The excess is
silence, not babble**: across 8 captured runaways the RMS after the sentence
ends is 0.000 for the remaining 8–9 s. Listening pairs in
`reports/spike-wavs/eos-*.wav` (gitignored), including
`eos-runaway-{oneshot,streamed}.wav` — the same runaway tokens through both
vocoders.

So the trailing-audio complaint is a **T3 budget-exhaustion bug affecting both
engines**, and what it produces is a long silent tail. Whether that is what
the user heard is unresolved; nothing in 160 paired runs produced trailing
*speech* on either path. Two follow-ups were opened and neither belonged to the
streaming spike: cap or trim the unterminated budget (it changes the
flag-off path too), and decide whether the RNG re-roll noted under "Prosody"
should be fenced off with `torch.random.fork_rng`. Both are now done — the
first in the next subsection, the second under "Prosody is preserved by
construction" above (fenced, 31.8 µs per window).

Regression cover added:
`tests/test_blockstream.py::test_generate_blocks_stops_at_the_eos_block`
(no block is emitted after the EOS block; the result carries no stop token)
and `::test_streamed_samples_match_trimmed_tokens` (three bench sentences,
chunked as they ship; since the silence trim landed it pins the *vocoded*
length at `trimmed_tokens × 960` and the *emitted* length at exactly a
whole-chunk `trim_edge_silence` of it). The first was verified to fail when the
early return is removed from the copied loop.

### Chunk-edge silence trim (2026-08-24) — the fix for the long pauses

Both defects above are silence at a chunk's edges, so both are fixed in one
place: `audio.trim_edge_silence` / `audio.TrailingSilenceGate`, cutting each
chunk back to **120 ms of natural pause per edge** at a **−45 dBFS** floor
(`engine.trim_silence` in config.yaml; set it false to stream the raw draw).

The sentence engine trims each chunk before it yields. The block engine
cannot — a runaway silent tail spans many windows that have already gone out
by the time the chunk ends — so it runs one `TrailingSilenceGate` per chunk on
the emission path: a window's speech goes out immediately, silence after it is
buffered and released in full the moment more speech arrives (a real mid-chunk
pause survives), and whatever is still buffered at chunk end is cut to 120 ms.
The gate never alters a sample, so cross-fade continuity between emitted
windows is untouched, and what it emits is byte-for-byte
`trim_edge_silence` of that chunk's concatenation.

Measured, `Sure, the kitchen light is on.` at config.yaml's knobs, on the
captured budget-exhaustion seeds (200-seed scan, 2/400 chunk-renders hit the
300-token ceiling — the 1–2 % rate above):

| engine | seed | before | after | speech span |
|---|---:|---:|---:|---:|
| sentence | 8 | 12.00 s | **1.75 s** | 1.60 s |
| blockstream | 15 | 12.00 s | **2.85 s** | 2.64 s |

Ordinary (non-runaway) renders of the same sentence lose 0.15–0.64 s each,
which is the per-join stacking: seeds 0–4, sentence 2.56→2.24, 2.28→1.95,
2.44→2.00, 2.52→2.08, 1.92→1.77 s; blockstream 2.28→1.85, 2.44→2.08,
2.44→2.10, 2.48→1.84, 2.64→2.39 s.

On the two-chunk `long` bench sentence, four seeds each: sentence
16.92→16.65, 17.96→16.69, **25.52→17.54** (a runaway in the *middle* of the
utterance — the user's mid-passage pause), 18.00→17.75 s; blockstream
18.08→17.24, 18.48→18.30, 18.08→17.54, 18.68→17.11 s.

`reports/spike-wavs/trim-runaway-{sentence,blockstream}-{before,after}.wav`
are those two seeds rendered both ways.

### Punctuation normalisation for clause fragments (2026-08-24)

`chunk_text`'s clause split keeps the clause mark on the fragment it splits
off (regex `(?<=[,;:])\s+`), so a fragment like `"it was the age of
wisdom,"` used to reach the model with a comma where a period belongs — a
weak EOS signal for a model trained to stop at sentence-final punctuation.
`engine_flash.speakable(text)` normalises that: a trailing clause mark (`,
; :`) becomes `.`, anything with no terminal punctuation gets `.` appended,
and an already-terminal chunk (`"Hello."`, `"Really?!"`, `'He said "go."'`,
`"..."`) passes through unchanged. Both engines apply it only at the model
boundary — `FlashEngine.synthesize_stream` calls `generate(speakable(chunk),
...)`, `BlockStreamEngine._stream_chunk` encodes `speakable(chunk)` — while
the yielded `(chunk_text, pcm)` label stays the original `chunk`, so
transcripts and `bench_stream`'s char counts are unaffected.

**Method.** The hypothesis under test: does the missing terminal punctuation
raise the no-EOS runaway rate this doc's "Chunk-edge silence trim" section
above measured at 1–2 %? The Dickens opening sentence (the same text as the
`ui/presets.yaml` "Long Story Excerpt" preset) split by `chunk_text(text,
120)` gives 6 clause fragments — one already ends in a period (`"...in the
superlative degree of comparison only."`), a built-in control since
`speakable()` is a no-op on it. Each fragment was rendered N=60 times through
the sentence engine's model boundary (`ChatterboxFlashTTS.generate`, what
`FlashEngine.synthesize_stream` calls) at `config.yaml`'s tuned knobs
(`num_steps: 4, n_cfm_timesteps: 1, drf_block_size: 32, temperature: 0.5`),
seeds 0–59 fixed and shared between arm A and arm B for a given fragment so
any rate difference isn't RNG-draw noise. Arm A is the raw fragment (what the
model saw before this task); arm B is `speakable(fragment)`. Runaway = T3
never emitted `stop_speech_token` and ran the whole
`_speech_len_for_text_tokens` budget, detected by comparing the token count
`t3.generate` returns to the budget it was called with — **not** by checking
whether the returned tensor contains the stop token, which is always absent
either way (T3's early return truncates *exclusive* of it, same as the
`generate_blocks` copy this doc's post-EOS section already documents). A
first pass at this harness used the stop-token-presence check and got 60/60
"runaways" on the first fragment — a methodology bug caught before any real
numbers were recorded, not a finding.

| # | fragment (arm A tail → arm B tail) | A runaways | A audio_s | A gen_s | B runaways | B audio_s | B gen_s |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | "...age of foolishness," → "...foolishness." | 0/60 | 6.944 | 1.224 | 0/60 | 6.951 | 1.252 |
| 1 | "...season of Darkness," → "...Darkness." | 1/60 | 8.950 | 1.554 | 0/60 | 8.700 | 1.526 |
| 2 | "...had nothing before us," → "...before us." | 0/60 | 7.541 | 1.311 | 0/60 | 7.636 | 1.323 |
| 3 | "...other way – in short," → "...in short." | 0/60 | 4.971 | 0.954 | 1/60 | 5.061 | 0.967 |
| 4 | "...insisted on its being received," → "...received." | 3/60 | 7.650 | 1.340 | 2/60 | 7.667 | 1.299 |
| 5 | "...comparison only." (already terminal — control, A = B) | 2/60 | 5.469 | 1.052 | 2/60 | 5.469 | 1.058 |
| **all 6** | | **6/360 (1.67 %)** | **6.921** | **1.239** | **5/360 (1.39 %)** | **6.914** | **1.237** |

Fragment 5 is the control: identical text, identical seeds → identical
runaway count and audio length in both arms (the sub-millisecond `gen_s`
difference is wall-clock noise, not a computation difference), which is the
harness's own sanity check on the seed-pairing.

**Conclusion: within noise.** 6/360 vs 5/360 is a 1.67 % vs 1.39 % runaway
rate — a gap of one trial in 360. At this sample size the per-arm standard
error on ~1.5 % is about 0.65 percentage points, several times the 0.28-point
gap, so this is not a detectable effect; Task A's own post-EOS section notes
the same power limit at a similar sample size. Mean audio duration (6.921 s
vs 6.914 s) and mean generation time (1.239 s vs 1.237 s) are also
indistinguishable — normalisation costs nothing. **The normalisation still
ships**: it does not make anything worse, it is what the fix was already
argued for on prosody grounds (a correct EOS signal for a model trained on
sentence-final punctuation, not a runaway-rate claim), and it fixes a real
mismatch between what a clause fragment says and what mark it ends on. It
just isn't the runaway-rate fix a 1–2 % baseline and a 720-trial harness
could show one way or the other — the fix for that remains the edge-silence
trim above.

### Verdict

**GO**, with caveats.

- TTFA, medium sentence: −52 % … −59 % across two independent measurement
  sets. Gate: ≥ 40 %. ✅
- Seams, benched configuration, 3 sentences × 3 renders: 73 block joins with a
  median step ratio of 0.60 against a control median of 1.05, 1 of 73 above
  its own render's control p95; exact tiling in the paired test; and a
  whole-waveform glitch scan whose maxima are comparable between the two paths
  on the same text. ✅
- Prosody: T3 sees the whole sentence, and the block loop is byte-identical to
  `generate` for a given RNG stream (identity test). The shipping callback used
  to break that by consuming RNG while vocoding; since 2026-08-24 those draws
  are fenced with `torch.random.fork_rng` (31.8 µs per window, 0.024 %) and
  both engines draw the same tokens from the same seed on a chunk. ✅ — read
  the scope note under "Prosody is preserved by construction": the guarantee is
  per chunk, and multi-chunk utterances still diverge from chunk two.
- Chunk-edge silence: trimmed to 120 ms per edge on both engines
  (2026-08-24). Runaway chunks go from 12.00 s to 1.75 s / 2.85 s, and every
  ordinary chunk join sheds 0.15–0.6 s. ✅
- Post-EOS hallucination: investigated 2026-08-24 and **not reproduced** —
  streamed samples equal `trimmed_tokens × 960` on 160/160 runs. The real
  defect found is a 1–2 % T3 budget-exhaustion runaway that affects both
  engines and produces trailing silence. See the subsection above. ✅

Caveats a follow-up must clear before this ships:

- **Listened, and it holds up.** Every check above is numeric, but the user
  listened to the A/B pairs in `reports/spike-wavs/` (holds
  `{short,medium,long}-{sentence,blockstream}.wav` and the paired pair) on
  2026-08-24 and confirmed the audio is good — no audible seams.
- **The 3.3–4.8 % log-mel divergence** from a one-shot render is real, grows
  with window count, and has two unseparated causes (above).
- **Chunk joins are untouched.** The 200 %-jump / 99.6th-percentile chunk
  boundaries the pipeline already has are still there. If those turn out to be
  audible, this spike does not help.
- **`generate_blocks` is a copy of installed-package code.** It will drift when
  `chatterbox-flash` is upgraded. The identity test catches that, but only on a
  machine with a GPU.
- **1.6× the GPU work** for the same audio. Fine on an idle RTX 2060 at
  RTF 0.38; not obviously fine with a concurrent ASR/LLM load on the same card.
- **Not run through the WebRTC session.** Browser TTFA gain is projected from
  Task 14's ~0.6 s transport overhead (≈1.45 s → ≈1.05 s on medium), not
  measured.
- **Six times as many sink pushes** per utterance (20–22 windows vs 4 chunks on
  the long paragraph). `PcmQueueTrack` should not care, but that has not been
  exercised end to end.

## Follow-up batch (2026-08-24): new defaults

The GO above shipped block streaming behind a flag, `false` by default. This
batch flips the defaults this doc now opens with, fixes the chunk-edge
silence the flag exposed, and closes out the post-EOS hallucination report
that was still open. Raw data for the re-bench:
[`reports/stream_runs.jsonl`](reports/stream_runs.jsonl) (appended, previous
rows kept) and [`reports/spike_seams.json`](reports/spike_seams.json)
(overwritten by `spike_analysis seams`, so it reflects only this run).

### What changed

- Defaults: `chunk_size` 120 → 300, `temperature` 0.6 → 0.5,
  `engine.block_streaming: true` — effective on CUDA + torch (this card),
  with a logged fallback to sentence streaming on every other resolved
  device/backend.
- Chunk-edge silence trim (sentence path) + trailing-silence gate (block
  path): 12 s no-EOS runaway tails now cut to `trim_keep_ms`; ordinary
  chunks shed 0.15–0.64 s of stacked silence.
- `speakable()` punctuation normalisation at both model boundaries (A/B: the
  runaway-rate change is within noise; it ships on prosody grounds).
- RNG fork (`torch.random.fork_rng`) around block-path vocoding: tokens now
  identical to the sentence path per chunk (the guarantee is per-chunk —
  multi-chunk utterances still diverge from chunk two, see "Prosody is
  preserved by construction" above).
- UI: `split_on_clauses` toggle; every generation knob now initialises from
  `generation_defaults` (`config.yaml`'s `generation:` block) instead of
  hardcoded slider defaults; curated voice list (`voices.paths: [../voices]`
  → `babel.mp3`, `marvin.mp3`, `one-one.mp3`).
- Task A diagnosis: no post-EOS vocoding bug existed; the reported
  "hallucination" was silent runaways (the 1–2 % T3 budget-exhaustion bug,
  now trimmed) plus RNG-order divergence between the two engines past the
  first chunk. See "Task A diagnosis" below.

### Browser measurements (controller, headless Chrome, defaults, warm)

One run per row, against the live UI/WebRTC session, block streaming active
(the effective default on this card):

| sentence | browser TTFA | server TTFA | total gen | audio | chunks |
|---|---:|---:|---:|---:|---:|
| short (30 ch) | 0.681 s | 0.473 s | 0.889 s | 2.08 s | 1 |
| medium (104 ch) | 0.739 s | 0.501 s | 1.845 s | 5.60 s | 1 |
| long (317 ch) | 0.706 s | 0.500 s | 5.593 s | 17.70 s | 2 |
| Dickens excerpt (945 ch) | 0.820 s | 0.512 s | 18.557 s | 59.98 s | 4 (4 transcript deltas, 0 errors) |

Against the previous default (sentence streaming, chunk 120, temp 0.6):
browser 1.136 / 1.455 / 1.424 s; server 0.772 / 1.228 / 1.187 s (short /
medium / long — no Dickens row was taken under the previous default). Browser
TTFA drops 40–50 % on short/medium and 50 % on long; the gain widens with
length because block streaming's TTFA no longer depends on how much of the
sentence remains once the first block is vocoded.

### Engine bench — both engines at the new defaults

`bench_stream.py` selects its engine by CLI flag
(`--block-stream` → `BlockStreamEngine`, absent → `FlashEngine`) and tags
each row `engine: "sentence"` or `"blockstream"` in
`reports/stream_runs.jsonl`; no code change was needed to bench both.

Sentence engine (`make bench-stream`, default 2 runs, best-of-N; what every
non-CUDA/torch backend falls back to):

```
  short: ttfa 0.545s  gen 0.54s  audio 1.97s  chunks 1
 medium: ttfa 1.061s  gen 1.06s  audio 5.84s  chunks 1
   long: ttfa 2.183s  gen 3.01s  audio 17.43s  chunks 2
```

Block-streaming engine (`python -m poc_tts_streaming.bench_stream
--block-stream --runs 3`; the effective default here — "chunks" in this
script's output is windows, not transcript chunks):

```
  short: ttfa 0.442s  gen 0.84s  audio 1.93s  chunks 3
 medium: ttfa 0.440s  gen 1.65s  audio 5.39s  chunks 7
   long: ttfa 0.441s  gen 5.28s  audio 17.58s  chunks 17
```

Block-streamed TTFA is flat at 0.44 s regardless of sentence length, same
shape as the Task 16 spike found at the old defaults. One side effect of
raising `chunk_size` to 300 while relying on block streaming to hide it:
**the sentence-only fallback path (cpu / mlx / flashinfer) got slower on
long text**, not faster. `long`'s sentence-engine TTFA rose from 0.843 s at
the old defaults (chunk_size 120, 4 chunks) to 2.183 s here (chunk_size 300,
2 chunks) — the first chunk is now roughly twice the text, so it takes
roughly twice as long to generate before any audio plays. `chunk_size: 300`
is a deliberate trade for prosody that block streaming is expected to pay
for; a machine that falls back to sentence streaming pays the chunk_size
cost with none of the offsetting TTFA win. Worth a follow-up if the
sentence-only fallback path needs to stay fast on its own (e.g. a
`block_streaming_effective`-conditioned `chunk_size`), not something this
batch changes.

### Seam re-measure (post-gate)

`spike_analysis seams --runs 3`, re-run at the new defaults (`chunk_size:
300`) with the chunk-edge silence trim now live on both paths — the Task B
review asked for this because trimming changes what sits on either side of
every join, which the original seam numbers (above, pre-trim) don't reflect.

**TTFA (best of 3 runs)** — same shape as before, wider gap on `long` because
of the `chunk_size` effect noted above:

```
sentence  sentence-level  block-stream    change
   short           0.478         0.431     -9.8%
  medium           0.990         0.435    -56.1%
    long           1.965         0.447    -77.2%
```

(Previously: short −19.8 %, medium −52.5 %, long −55.9 %. Short's gate isn't
comparable release-to-release — it's dominated by fixed per-chunk cost, and
is the noisiest of the three in both runs.)

**Step ratio at joins**, pooled across the nine block-streamed renders (3
sentences × 3 runs):

| join type | n | median | max | control median (per render) | exceeds own render's control p95 |
|---|---:|---:|---:|---:|---:|
| block (all 3 sentences) | 68 | 1.40 | 7.49 | 0.67–1.71 | 1/68 |
| chunk (blockstream/long only) | 3 | 49.30 | 51.87 | 1.25–1.71 | 3/3 |
| chunk (sentence/long only) | 3 | 1.05 | 30.40 | 1.12–1.27 | 1/3 |

Previously (pre-trim, pooled over all nine blockstream renders): 73 block
joins, median 0.60, max 5.31, 1/73 above its own render's control p95.
**By the calibrated metric — how often a join exceeds that same render's own
control p95 — block joins are unchanged: roughly 1 in 70 either way.** The
raw median step ratio at a block join more than doubled (0.60 → 1.40), which
looks alarming in isolation but tracks the calibrated metric holding steady,
not a new defect.

**Chunk joins moved a lot more**, and that's the trim doing its job, not a
regression: with `trim_silence` now live, every chunk edge is cut back to
120 ms of near-silence at −45 dBFS. Checking the raw samples at each
blockstream/long chunk join: `rms_before` is 0.0006–0.0013 (already near the
noise floor — that's the 120 ms trim residue) and `rms_after` is exactly
0.0. Before the trim, a chunk join's neighbourhood held several times more
near-silent padding, which set a small-but-stable local step-ratio
denominator; after trimming there's almost nothing left to average over, so
the same absolute (near-zero) discontinuity produces a much larger ratio.
The RMS-jump-percentile metric, which isn't sensitive to this scaling, tells
the same "inherited, not improved" story as before: chunk joins still read
200 % jump at the 99.6th–99.8th percentile on both engines (99.6–99.7 %
pre-trim, 99.8 % here) — a silence-to-silence transition between independent
`generate()` draws, present on both paths, that block streaming neither
fixes nor worsens. No listening evidence suggests this reads as an audible
regression; it's a metric artifact of measuring a much quieter join.

**RMS jump, as a percentile of the same statistic everywhere:**

| render | join type | n | jump % (median) | percentile (median / max) | of all offsets |
|---|---|---:|---:|---:|---:|
| blockstream / short  | block | 7  | 41.3 % | 79.1 / 93.9 | 45.5 % |
| blockstream / medium | block | 16 | 12.6 % | 43.5 / 91.3 | 43.5 % |
| blockstream / long   | block | 45 | 19.3 % | 53.6 / 97.4 | 45.3 % |
| blockstream / long   | chunk | 3  | 200.0 % | 99.8 / 99.8 | 45.3 % |
| sentence / long      | chunk | 3  | 200.0 % | 99.8 / 99.8 | 43.0 % |

**Glitch scan** (worst step ratio anywhere in run 0 of each render, and its
distance to the nearest join):

```
blockstream/short    0.648s r=25.8   nearest join 1.000s (block, -0.352s)
blockstream/medium   3.248s r=30.8   nearest join 3.556s (block, -0.308s)
blockstream/long     12.365s r=36.4  nearest join 12.365s (chunk, +0.000s)
sentence/short       0.671s r=80.7   (no joins in this render)
sentence/medium      0.476s r=24.9   (no joins in this render)
sentence/long        3.014s r=29.0   nearest join 12.492s (chunk, -9.478s)
```

The worst glitch on `blockstream/long` now sits exactly at a chunk join
(+0.000 s) — this is the same trimmed near-silence-to-silence transition the
step-ratio and jump tables above already flag, not an independent finding.
Per-render maxima otherwise remain comparable between the two paths, as
before (block streaming is not introducing a class of transient the
sentence-level path lacks).

**Conclusion: seam statistics moved, mostly as an artifact of what the
edge-silence trim measures, not as a quality regression.** Block joins are
unchanged by the calibrated (exceeds-own-control-p95) metric. Chunk joins
show much larger raw numbers post-trim because trimming shrank the silence
padding the old numbers were partly measuring; the RMS-jump-percentile view,
which isn't sensitive to that scaling, is essentially unchanged (99.6–99.7 %
→ 99.8 %). Nothing here revises the Verdict above.

### Task A diagnosis: post-EOS hallucination — hypothesis vs. finding

The original hypothesis (see "Post-EOS hallucination report" above) was that
the streamed path never trims at `stop_speech_token` and speaks out the rest
of the token budget as babble. **That hypothesis was not confirmed.**
`generate_blocks` already stops at EOS, a second independent guard
(`_trim_to_eos`) sits in front of the vocoder, and streamed sample counts
matched `trimmed_tokens × 960` on 160/160 paired runs — there is no
post-EOS vocoding bug on either engine.

What *is* real, and engine-agnostic: roughly 1–2 % of chunks never emit EOS
and run the full token budget (a 300-token / 12 s floor on short text), and
the excess is **silence, not babble** — RMS after the sentence ends measured
0.000 across every captured runaway. That bug is exactly what this batch's
chunk-edge silence trim (section above) fixes: runaway tails now cut from
12.00 s to 1.75 s (sentence) / 2.85 s (blockstream). Whether that silent
runaway, or the RNG-order divergence between the two engines past the first
chunk (see "Prosody is preserved by construction"), is what the original
user heard as "hallucination" is unresolved — nothing in 160 paired runs
produced trailing *speech* on either path — but both known defects in that
area are now closed: the runaway by the silence trim, and the RNG mismatch
by the `torch.random.fork_rng` fence, both in this batch.
