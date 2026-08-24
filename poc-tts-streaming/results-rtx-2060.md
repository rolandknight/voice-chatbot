# poc-tts-streaming on an RTX 2060 — time-to-first-audio

Measured 2026-08-24 on `pop-os`, NVIDIA GeForce RTX 2060 (6 GB, compute
capability sm_75), driver 580.159.03, torch 2.6.0+cu124. Generation config
is `config.yaml`'s defaults (`num_steps: 4, n_cfm_timesteps: 1,
chunk_size: 120, split_text: true, split_on_clauses: true`), resolved
dtype `float16`, backend `torch` (SDPA — flashinfer is unavailable on this
card; see `poc-tts/bench-rtx-2060.md`). Voice: `one-one.mp3`.

Raw data: [`reports/stream_runs.jsonl`](reports/stream_runs.jsonl).
Reproduce the engine column with `make bench-stream`.

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

### Verdict

**GO**, with caveats.

- TTFA, medium sentence: −52 % … −59 % across two independent measurement
  sets. Gate: ≥ 40 %. ✅
- Seams, benched configuration, 3 sentences × 3 renders: 73 block joins with a
  median step ratio of 0.60 against a control median of 1.05, 1 of 73 above
  its own render's control p95; exact tiling in the paired test; and a
  whole-waveform glitch scan whose maxima are comparable between the two paths
  on the same text. ✅
- Prosody: T3 sees the whole sentence and the token stream is byte-identical
  (identity test). ✅

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
