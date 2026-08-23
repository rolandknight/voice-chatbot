# Chatterbox Flash on an RTX 2060 — tuning sweep

Measured 2026-08-23 on `pop-os`, NVIDIA GeForce RTX 2060 (6 GB, compute capability
sm_75), driver 580.159.03, torch 2.6.0+cu124, `chatterbox-flash==0.1.0`.

Resolved runtime: **device `cuda`, dtype `float16`, backend `torch` (SDPA)**.

Raw data: [`reports/runs.jsonl`](reports/runs.jsonl) — 72 rows from two independent
sweeps. Reproduce with `make bench`.

## Headline

**`drf_block_size=32, num_steps=4` is roughly 3× faster than the library defaults**,
at the same output quality and about 1.2 GB more VRAM.

| | library default<br>`blk16 / steps10 / cfm2` | tuned<br>`blk32 / steps4 / cfm1` | speedup |
|---|---:|---:|---:|
| short sentence (30 chars) | 1.45 s | **0.59 s** | 2.5× |
| medium (104 chars) | 3.42 s | **1.03 s** | 3.3× |
| long paragraph (317 chars) | 10.60 s | **3.38 s** | 3.1× |
| VRAM peak | 2506 MB | 3678 MB | +1.2 GB |

`gen_s` above is median generation time, best-of-2 per measurement, across both runs.

## Method

Twelve configurations — `drf_block_size` × `num_steps` × `n_cfm_timesteps` — each
against three fixed sentences, best-of-2, run twice independently. `drf_block_size` is
a constructor argument so it forms the outer loop (two model loads total); the other
two are per-request parameters of `generate()`.

The three sentences are the same ones used for every other baseline in this project
(Chatterbox Turbo on CUDA, Turbo on CPU), so the numbers are directly comparable.

## Read `gen_s`, not RTF

**RTF is contaminated on this model and should not drive the decision.** It is
`gen_s / audio_s`, and Flash's output length varies substantially for identical input,
so a configuration that over-generates earns a flatteringly low RTF.

The clearest case is `blk16 / steps4 / cfm1`. Its RTF of 0.158 on the medium sentence
is the second-best number in the whole grid — but it produced a mean of **10.00 s of
audio for a 104-character sentence** that should run about 5.5 s. It was not fast; it
was verbose. The tuned config produced 5.58 s for the same input in 1.03 s.

| sentence | expected | observed range across 24 measurements |
|---|---|---|
| short (30 chars) | ~2 s | 1.88 – 2.96 s, plus one 12.0 s outlier |
| medium (104 chars) | ~5.5 s | 4.48 – 14.88 s |
| long (317 chars) | ~17 s | 14.60 – 27.96 s |

## Full grid

Mean RTF over both runs. Included for completeness — see the caveat above before
reading the low numbers as fast.

| blk | steps | cfm | short | medium | long | mean | VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 1 | 0.334 | 0.205 | 0.294 | 0.277 | 2577 |
| 16 | 4 | 2 | 0.416 | 0.335 | 0.320 | 0.357 | 2577 |
| 16 | 6 | 1 | 0.407 | 0.453 | 0.390 | 0.417 | 3210 |
| 16 | 6 | 2 | 0.433 | 0.432 | 0.404 | 0.423 | 2547 |
| 16 | 10 | 1 | 0.706 | 0.581 | 0.549 | 0.612 | 2578 |
| 16 | 10 | 2 | 0.740 | 0.627 | 0.631 | 0.666 | 2506 |
| **32** | **4** | **1** | **0.140** | **0.185** | **0.196** | **0.174** | **3678** |
| 32 | 4 | 2 | 0.293 | 0.186 | 0.207 | 0.229 | 3636 |
| 32 | 6 | 1 | 0.289 | 0.241 | 0.243 | 0.258 | 3640 |
| 32 | 6 | 2 | 0.383 | 0.267 | 0.262 | 0.304 | 3642 |
| 32 | 10 | 1 | 0.435 | 0.339 | 0.353 | 0.376 | 3638 |
| 32 | 10 | 2 | 0.365 | 0.347 | 0.363 | 0.359 | 3664 |

`drf_block_size=32` beats 16 at every step count. That matches the paper's own best
row (D=32), so the sweep independently reproduces upstream's guidance.

Reproducibility on the tuned config is good where output length is stable: RTF differed
by 0.003 (medium) and 0.016 (long) between the two runs. The short sentence differed by
0.177 — entirely because of the 12 s over-generation outlier, not timing noise.

## Time to first speech

**Today it equals full generation time.** `/tts` returns a complete WAV
(`poc_tts/server.py`, `Response(content=...)`), so nothing is emitted until the whole
utterance is synthesised: **0.59 s** for a short sentence, **3.38 s** for a long
paragraph. Model load (~10 s cold) is paid once at server start, not per request.

Chunked streaming would put first audio at roughly the first chunk's generation time.
At the default `chunk_size: 120` that is about **1.0 s**; deliberately making the first
chunk a single short sentence brings it to about **0.6 s**.

## Is it fast enough to stream?

**Yes, with margin — but the PoC is not plumbed for it.**

Sustainability is not in question. At RTF 0.19–0.23 on realistic input, generation
outruns playback by 4–5×, so once first audio is out the buffer cannot underrun. The
only real constraint is initial latency, and ~0.6 s is far below this project's current
end-to-end `reply_start_latency` of 11–14 s.

It will not reach Flash's advertised sub-200 ms time-to-first-packet. That figure comes
from the FlashInfer streaming path, which **cannot run on this GPU** — see below.

Three things stand in the way of actually streaming:

1. `chatterbox_flash` exposes only `generate` and `generate_batch`. There is no
   streaming API, so chunk-level streaming is the only route.
2. `synthesize` would need to become a generator and `/tts` a `StreamingResponse`. The
   chunking already exists internally.
3. **`poc-tts` does not implement `/v1/audio/speech`.** The voice pipeline points at
   `127.0.0.1:8004/v1` with `model: chatterbox-turbo` (`config.yaml`), and the vendored
   Turbo server implements that endpoint with `stream: true` support. Until poc-tts
   exposes it, this is a GUI test bench rather than a pipeline component.

## FlashInfer does not work on this GPU

Installing a CUDA 12.4 toolkit is necessary but not sufficient. With `nvcc` present,
FlashInfer's JIT compiles for `sm_75` and then fails a static assertion, twelve times:

```
prefill.cuh(342): error: static assertion failed with
  "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math then recompile
   to support fp16 reduction"
```

On Turing, FlashInfer selects fp16 QK accumulation because sm_75 lacks Ampere's
fp32-accumulate path, and the stock wheel is built without `FP16_QK_REDUCTION_SUPPORTED`.
Making it work would mean rebuilding FlashInfer from source with that define plus
boost_math.

`_flashinfer_available()` therefore requires compute capability ≥ 8.0 in addition to a
findable `nvcc`, and fails closed to torch SDPA. On an Ampere-or-newer card the same
code path would engage FlashInfer normally.

## Caveats

- Every number is single-stream on an otherwise idle GPU. No concurrency was measured.
- `vram_peak_mb` is per-configuration, reset between configurations but not between the
  three sentences within one, so it is an upper bound for that configuration.
- The over-generation behaviour above is a model-quality question, not a code defect. It
  is worth understanding before Flash goes into a latency-sensitive pipeline: one run in
  twenty-four produced 12 s of speech from a 30-character input — in 0.6 s, so latency
  stayed normal while the content ran long.
