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
Reproduce with `.venv/bin/python -m poc_tts_streaming.bench_stream --block-stream --runs 3`.

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

### TTFA

`make bench-stream` with and without `--block-stream`, three runs each,
best `ttfa_s` per sentence, warm engine, config defaults:

| sentence | chunks (sentence) | TTFA sentence-level (s) | windows (block) | TTFA block-stream (s) | change |
|---|---:|---:|---:|---:|---:|
| short  | 1 | 0.529 | 3  | 0.430 | **−18.7 %** |
| medium | 1 | 1.049 | 6  | 0.433 | **−58.7 %** |
| long   | 4 | 1.124 | 24 | 0.447 | **−60.2 %** |

A separate paired harness (both engines in one process, three runs each,
same warm-up) gave −13.8 % / −55.4 % / −62.5 %, and an earlier run
−27.5 % / −54.1 % / −60.3 %. **Medium clears the ≥ 40 % gate in every run**
(−54 % to −59 %). Short is the noisy one, and for a structural reason: its
TTFA floor is dominated by the fixed per-chunk cost (reference-clip load,
`prepare_conditionals`, text tokenisation, T3 prefix forward), which block
streaming does not touch. Block-streamed TTFA is essentially *flat* at
0.41–0.45 s across all three sentences — it no longer depends on sentence
length at all, which is the real result here.

### Cost

Re-running the flow over the whole token prefix each block is quadratic
within a chunk. Measured RTF (`gen_s / audio_s`) rose from 0.18–0.30 to
0.29–0.38 — about 1.6× the GPU work, still far below 1.0, so the synthesis
worker stays comfortably ahead of playback on every sentence. `chunk_size:
120` caps the quadratic term at one chunk, which is why the long paragraph
is no worse than the medium sentence.

### Seams: none detectable

Three independent checks. Listening samples for the user are in
`reports/spike-wavs/` (`{short,medium,long}-{sentence,blockstream}.wav`, plus
`*-paired-{streamed,singleshot}.wav` from the paired harness). That directory
is gitignored.

**1. Structural — exact tiling.** Driving the same token sequence and the
same CFM noise through both the windowed vocoder and a single-shot vocode:

| sentence | tokens | expected samples (960 × tokens) | streamed | single-shot |
|---|---:|---:|---:|---:|
| short  | 50  | 48 000  | 48 000  | 48 000  |
| medium | 157 | 150 720 | 150 720 | 150 720 |
| long   | 451 | 432 960 | 432 960 | 432 960 |

Exact in all three: nothing is repeated, nothing is truncated.

**2. Step discontinuity at the joins.** `|x[j] − x[j−1]|` divided by the
median `|diff(x)|` in the surrounding ±50 ms (a local control — speech is
loud in places, so a global percentile over-reports):

| sentence | joins | joins: median / max | control offsets: median / p95 / max |
|---|---:|---:|---:|
| short  | 2  | 1.44 / 2.21 | 1.22 / 3.37 / 6.16 |
| medium | 5  | 0.52 / 2.67 | 0.93 / 4.73 / 13.10 |
| long   | 15 | 0.63 / 5.08 | 1.04 / 3.51 / 12.12 |

The joins are indistinguishable from arbitrary offsets — if anything
smoother. **This only became true after the cross-fade was added** (see
"What broke" below): before it, the same measurement read 45 / 9 / 9 at the
joins against a control median of ~1.

**3. Glitch scan over the whole waveform.** `|diff|` divided by the local
mean `|diff|` (±50 ms), everywhere, then the top offsets:

```
short : streamed p99.99 12.7 max 15.8 | single-shot p99.99 12.9 max 15.2
medium: streamed p99.99 15.0 max 47.3 | single-shot p99.99 16.3 max 48.5
long  : streamed p99.99 16.8 max 33.9 | single-shot p99.99 16.4 max 33.2
```

The extreme offsets are the *same timestamps with the same magnitudes* in
both — they are plosives in the speech, not artefacts — and none of them is
near a join (nearest join is 130–490 ms away in every case). The point
`join + 3840` samples, where HiFTGAN's NSF source switches from the cached
phase to its fresh accumulator, scores 0.37–0.99 median (below average).

**On the "> 20 % RMS jump" threshold.** Taken literally it is not a usable
seam test for speech: 40–45 % of *all* offsets in these very utterances have
a >20 % jump between adjacent 10 ms windows. Calibrated as a percentile of
that same distribution, block-stream joins sit at the 51st–59th percentile
(median), i.e. typical. For contrast, the sentence-level output's own
chunk joins sit at the **99.6th** percentile — the existing, shipped,
un-complained-about path has bigger level discontinuities at its joins than
the block-streamed one does.

**Residual difference.** Streamed and single-shot audio differ by 3.3–4.8 %
in log-mel (control: two single-shot vocodes of the identical mel differ by
0.06–0.23 %). That is content, not a seam — each window's mel is computed
with less right context than a single-shot pass has. In the raw waveform the
difference looks enormous (70–109 % rel RMS, corr 0.39–0.74) but that is
pure phase: HiFTGAN's sine source restarts its phase accumulator per call, so
the streamed excitation is phase-shifted relative to a one-shot render.
Absolute excitation phase is inaudible; the log-mel figure is the meaningful
one. **A human still needs to confirm by ear** — the WAVs above are saved for
exactly that.

### What broke on the way (and how)

Three things in the base package had to be worked around. All three are
recorded here because they are the parts most likely to bite a follow-up.

1. **`flow_inference(..., finalize=False)` raises.** In chatterbox-tts 0.1.7,
   `CausalMaskedDiffWithXvec.inference` (`flow.py:170-182`) truncates `h` by
   `pre_lookahead_len * token_mel_ratio` and then multiplies it by a `mask`
   built from the *untruncated* `h_lengths`:
   `RuntimeError: The size of tensor a (558) must match the size of tensor b
   (564) at non-singleton dimension 2`. `finalize` gates nothing else in that
   method, so the spike passes `finalize=True` and drops the 6 lookahead mel
   frames itself. The `S3GenStreamer` the docstring points at does not exist
   in this package.
2. **The meanflow CFM redraws its noise every call.** `flow_inference`
   unconditionally does `noise = torch.randn(1, 80, n_tokens*2)`, so each
   window would be a different stochastic draw of the same speech and the
   emitted frames would not be the ones the next window continues. Fixed by
   drawing the noise once per utterance and slicing it, which means calling
   `S3Token2Mel.forward` with an explicit `noised_mels` instead of
   `flow_inference`. With the noise fixed, mel prefix drift between window
   sizes drops to the level of the CFM's own redraw floor (measured before
   the fix: 3.2–4.3 % drift vs a 3.3–3.6 % redraw floor).
3. **`cache_source` alone does not make the vocoder continuous.** Seeding
   `hift_inference(cache_source=…)` fixes the NSF phase only for the cached
   samples; the fresh `theta = cumsum(f0)` restarts at 0 each call and meets
   the cached phase with a step. The first implementation did exactly what
   the task brief describes (emit only the new samples, seed `cache_source`)
   and produced a measurable click at every join — step ratio 9–45× the local
   median against a control of ~1. CosyVoice2's actual answer is a third
   cache: hold the last 8 mel frames' *audio* back, re-render it with the
   next window, and cross-fade the two renderings (`fade_in_out`, Hamming
   over 2 × 3840 samples). With that, the joins measure as above.

### Verdict

**GO**, with caveats. TTFA drops 54–59 % on the medium sentence across every
run (gate: ≥ 40 %), no seam is detectable by any of the three measures on any
of the three sentences over three runs, and prosody is provably unchanged
because the token stream is identical.

Caveats a follow-up must clear before this ships:

- **Listen to the WAVs.** Every seam check here is numeric. The numbers say
  the joins are quieter than the speech around them, but no one has heard it.
- **The 3.3–4.8 % log-mel divergence** from a single-shot render is real. It
  is not a seam, but it is a quality difference, and it grows with the number
  of windows (3.29 % short → 4.75 % long).
- **`generate_blocks` is a copy of installed-package code.** It will drift
  when `chatterbox-flash` is upgraded. The identity test catches that, but
  only on a machine with a GPU.
- **1.6× the GPU work** for the same audio. Fine on an idle RTX 2060 at
  RTF 0.38; not obviously fine with a concurrent ASR/LLM load on the same
  card.
- **Cancellation** unwinds through an exception raised inside `on_block`,
  which is workable but means a cancelled utterance still pays for the block
  that was in flight — same as the sentence-level path, but the blocks are
  smaller, so it is strictly better.
