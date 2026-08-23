# Chatterbox Flash on Apple M4 Max (Metal / MLX) — tuning sweep

Companion to [`bench-rtx-2060.md`](bench-rtx-2060.md) (NVIDIA) and
[`bench-mac-cpu.md`](bench-mac-cpu.md) (the CPU-only baseline this replaces), and the
result of executing [`mac-gpu-build-plan.md`](mac-gpu-build-plan.md).

Measured 2026-08-23 on `Rolands-Mac-Studio.local`, Apple M4 Max (14 cores: 10P + 4E,
32-core GPU), 36 GB unified memory, macOS 26.4, torch 2.6.0, `chatterbox-flash==0.1.0`,
mlx 0.32.1 / mlx-lm 0.31.3.

Resolved runtime: **device `cpu`, dtype `float16`, backend `mlx`**. The LLaMA backbone
runs on Metal; S3Gen and the voice encoder stay on PyTorch CPU. That split is upstream's
design, not a limitation of this integration.

Raw data: [`reports/runs.jsonl`](reports/runs.jsonl) — 72 `backend: mlx` rows (36 fp16,
36 4-bit) plus the 36 `backend: torch` CPU rows.
[`reports/tuned-repeats-mac.jsonl`](reports/tuned-repeats-mac.jsonl) — 45 rows, five
repeats of the tuned config per variant, which is where the headline numbers come from.

## Headline

**Route 1 (MLX) works, and needed no changes to `engine_flash.py` at all.** It roughly
doubles CPU throughput and lands within 1.2–1.6× of the RTX 2060.

Tuned config `drf_block_size=32, num_steps=4, n_cfm_timesteps=1`, median `gen_s` of five
repeats:

| sentence | CPU (fp32, torch) | **MLX fp16** | MLX 4-bit | RTX 2060 (fp16, CUDA) |
|---|---:|---:|---:|---:|
| short (30 chars) | 1.48 s | **0.92 s** | 0.91 s | 0.59 s |
| medium (104 chars) | 2.88 s | **1.37 s** | 1.25 s | 1.03 s |
| long (317 chars) | 8.62 s | **4.20 s** | 4.19 s | 3.38 s |
| speed-up vs CPU | 1.0× | **1.6–2.1×** | 1.6–2.3× | — |
| vs the 2060 | 2.5–2.8× slower | **1.2–1.6× slower** | 1.2–1.5× slower | 1.0× |

The honest question the plan posed was *"is Apple Silicon fast enough to deploy on"*.
**Yes.** A 4-second long-paragraph turn on a machine with no discrete GPU is inside the
same envelope as the CUDA box.

## How MLX is selected

`config.yaml` is shared with the CUDA box, so `backend: mlx` cannot be committed there —
the next pull would break it. Selection therefore comes from the environment, via the
gitignored `.env` the Makefile already sources into every recipe:

    POC_TTS_ENGINE_BACKEND=mlx
    POC_TTS_ENGINE_DTYPE=float16

`dtype` matters: `resolve_dtype("auto", "cpu", ...)` returns `float32`, and the MLX path
expects fp16. Leaving it on auto silently runs a slower, heavier configuration rather
than failing.

Code changes, both additive and both no-ops when the environment is unset:

- **`poc_tts/config.py`** — `apply_engine_overrides()` overlays `POC_TTS_ENGINE_DEVICE`
  / `_DTYPE` / `_BACKEND` onto the engine section at load time. Values pass through
  untouched; `resolve_device` / `resolve_dtype` / `resolve_backend` remain the single
  validation site.
- **`poc_tts/bench.py`** — `load_time_configs()` adds backend, dtype and quantization as
  outer-loop sweep axes (`POC_TTS_BENCH_BACKENDS`, `_DTYPE`, `_QUANT_BITS`).
  Quantization has to be an outer-loop axis: it is applied when `chatterbox_flash`
  builds its MLX engine, so each bit width needs its own model load.

**`engine_flash.py` was not touched.** `resolve_backend` already accepted an explicit
`"mlx"` and `FlashEngine` already forwarded `self.backend` into `generate()`, exactly as
the plan predicted. No `mps` branch was needed, so Route 2 was never opened.

### The plan's quantization interface is wrong

The plan describes `quantize_bits={4,8}` as an argument. There is no such argument
anywhere in the `chatterbox_flash` Python API. The only interface is the environment
variable `CHATTERBOX_FLASH_MLX_QUANT_BITS`, read in `engines/mlx.py:224` when the MLX
engine is constructed, and it accepts `{2, 3, 4, 6, 8}` rather than `{4, 8}`.

## Versus Resemble's published Mac figures

The plan lists RTF **0.778** (fp16) and **0.665** (4-bit) on an M4 as the figures to
reproduce or refute. **Both are refuted, in the favourable direction, at every
configuration measured** — and the relative ordering does not reproduce either.

| | published (M4) | measured, library defaults | measured, tuned |
|---|---:|---:|---:|
| fp16 | 0.778 | 0.485 | 0.282 |
| 4-bit | 0.665 | 0.417 | 0.283 |

Measured columns are mean RTF over the three sentences; the tuned column comes from the
five-repeat runs, so it is not contaminated by the over-generation described below.

Two caveats before reading this as a win. An M4 Max is a much larger chip than the base
M4 those figures presumably used, so the gap is not all software. And **the 15%
fp16 → 4-bit gain does not reproduce at all**: at the tuned config the two are
indistinguishable (0.282 vs 0.283), and the reason is visible in the memory numbers
below.

## Read `gen_s`, not RTF

The contamination the 2060 document warns about is present here, and it bit this
sweep's headline cell directly. The MLX fp16 tuned config's medium sentence produced
**14.88 s of audio for a 104-character sentence** that should run about 5.5 s, posting
an RTF of 0.117 — the best number in the entire grid, from a run that was not fast but
verbose.

That is why the headline table above comes from five repeats rather than the sweep. Over
five runs the same cell produced a median of 6.32 s of audio in 1.37 s; the 14.88 s
outlier recurred once.

Rate of over-generation (any sentence exceeding 1.5× its expected length), 36 cells per
variant:

| variant | over-generating cells | worst observed |
|---|---:|---|
| CPU torch fp32 | 2 / 36 | 26.72 s for the long sentence |
| MLX fp16 | 3 / 36 | 29.84 s for the long sentence |
| MLX 4-bit | 3 / 36 | 26.80 s for the long sentence |

**MLX does not introduce a new output-length pathology.** The rate matches the CPU path
on the same machine, and nothing resembles the FlashInfer-on-sm_75 failure the plan
describes, where 9 of 12 runs hallucinated ~9 extra seconds. This is the pre-existing
model behaviour, on a different backend.

That said, three-in-thirty-six is a rate, not a verdict on quality: no transcription or
listening test was run, so this rules out the gross failure mode, not subtle degradation.

## Full grid

Twelve configurations × three sentences, best-of-2 per cell, one sweep per variant.
`gen_s` in seconds; mean RTF over the three sentences.

### MLX fp16

| blk | steps | cfm | short | medium | long | mean rtf |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 1 | 1.02 | 1.68 | 6.04 | 0.351 |
| 16 | 4 | 2 | 1.17 | 1.73 | 6.60 | 0.331 |
| 16 | 6 | 1 | 1.08 | 1.88 | 6.66 | 0.414 |
| 16 | 6 | 2 | 1.24 | 2.05 | 7.59 | 0.453 |
| 16 | 10 | 1 | 1.60 | 2.45 | 7.36 | 0.453 |
| 16 | 10 | 2 | 1.41 | 2.65 | 8.69 | 0.485 |
| **32** | **4** | **1** | **0.93** | **1.75**\* | **4.49** | **0.250**\* |
| 32 | 4 | 2 | 1.07 | 1.44 | 4.68 | 0.312 |
| 32 | 6 | 1 | 0.96 | 1.43 | 4.75 | 0.306 |
| 32 | 6 | 2 | 1.11 | 1.73 | 5.12 | 0.339 |
| 32 | 10 | 1 | 1.11 | 1.76 | 6.33 | 0.353 |
| 32 | 10 | 2 | 1.17 | 1.89 | 6.68 | 0.462 |

\* the over-generating cell — its medium `gen_s` covers 14.88 s of audio, and its mean
RTF is correspondingly flattered. The five-repeat median is 1.37 s.

### MLX 4-bit

| blk | steps | cfm | short | medium | long | mean rtf |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 1 | 0.96 | 1.57 | 5.12 | 0.329 |
| 16 | 4 | 2 | 1.16 | 1.57 | 5.62 | 0.341 |
| 16 | 6 | 1 | 1.06 | 1.69 | 6.26 | 0.337 |
| 16 | 6 | 2 | 1.23 | 2.18 | 6.16 | 0.363 |
| 16 | 10 | 1 | 1.21 | 1.99 | 8.26 | 0.405 |
| 16 | 10 | 2 | 1.47 | 2.67 | 8.01 | 0.417 |
| **32** | **4** | **1** | **0.94** | **1.40** | **4.12** | **0.271** |
| 32 | 4 | 2 | 1.04 | 1.45 | 4.71 | 0.302 |
| 32 | 6 | 1 | 0.90 | 1.45 | 4.75 | 0.304 |
| 32 | 6 | 2 | 1.31 | 1.79 | 5.32 | 0.349 |
| 32 | 10 | 1 | 1.11 | 2.06 | 6.32 | 0.379 |
| 32 | 10 | 2 | 1.26 | 2.17 | 6.46 | 0.411 |

`drf_block_size=32` beats 16 on total `gen_s` in **all twelve** matched pairs across both
variants — a cleaner reproduction of the paper's D=32 guidance than either the 2060 sweep
or the CPU sweep managed. `num_steps` remains the dominant cost axis; `n_cfm_timesteps`
is close to free.

## Memory

Peak of load plus one short generation, tuned config:

| variant | process RSS | MLX (Metal) peak |
|---|---:|---:|
| CPU torch fp32 | 6658 MB | — |
| MLX fp16 | 6041 MB | 2598 MB |
| MLX 4-bit | 6039 MB | 2220 MB |

**This is why 4-bit buys nothing.** Quantization touches only the LLaMA backbone —
S3Gen and the voice encoder stay in fp16 on the PyTorch side — so it saves 378 MB of
Metal buffers and *zero* process RSS, on a machine with 36 GB. The backbone is not the
bottleneck on this chip, which is equally why the speed gain fails to materialise.

4-bit is worth revisiting only on an 8 GB Mac, where 378 MB is a meaningful fraction of
what is available. On this hardware, run fp16.

## Time to first speech

**Today it equals full generation time.** `/tts` returns a complete WAV
(`poc_tts/server.py`, `Response(content=...)`), so nothing is emitted until the whole
utterance is synthesised: **0.92 s** for a short sentence, **4.20 s** for a long
paragraph.

Model load is **~5.6 s** from a warm Hugging Face cache, paid once at server start. The
MLX backbone is built lazily on the first `generate()`, not at load: the first generation
after start costs an extra ~2.7 s for PyTorch → MLX weight conversion. `bench.py`'s
warm-up call absorbs it, and so does any real server's first request — but it is not
free, and it recurs whenever the engine is rebuilt for a longer sequence.

Chunked streaming would put first audio at roughly the first chunk's generation time. At
the default `chunk_size: 120` that is about **1.4 s**; making the first chunk a single
short sentence brings it to about **0.9 s** — within 1.5× of the 2060's 0.6 s.

## Is it fast enough to stream?

**Yes, comfortably.** Mean RTF at the tuned config is 0.28, so generation outruns
playback by 3.6×. More importantly the *whole grid* is safe: the worst single measurement
across all 72 MLX rows is RTF 0.647, and no cell reaches 0.9. Unlike the CPU path — where
two of twelve configurations cannot stream at all — every MLX configuration sustains
real-time output. Tuning improves latency here; it is no longer the difference between
working and not.

The three blockers from the 2060 write-up are unchanged and remain the actual obstacle:
`chatterbox_flash` exposes no streaming API, `synthesize`/`/tts` are not a
generator/`StreamingResponse` pair, and poc-tts does not implement `/v1/audio/speech`,
which is the endpoint the voice pipeline actually calls.

## Browser check

Verified in headless Chromium driving the real GUI — not by inspecting served files,
which the plan flags as having shipped two Critical defects on the NVIDIA side.

`/api/model-info` reported `backend: mlx, dtype: float16, device: cpu`, confirming the
env override reaches the server and not just the benchmark. Filling the textarea,
selecting a predefined voice and clicking Generate produced a 73004-byte WAV that the
browser decoded to 1.52 s at 24 kHz, with the waveform, Play button and Download WAV link
all rendered. Zero console errors, zero failed requests, zero HTTP ≥ 400.

One incidental finding: **`vendor/woosh` already occupies 127.0.0.1:8005**, the port
`config.yaml` assigns to poc-tts, so `make run` dies with `[Errno 48] address already in
use`. The browser check ran on port 8015 instead. Everything else — engine construction,
`create_app`, the served assets — was the real code path.

## Reproducing this run

`setup.sh` requires `mise`, which is not installed on this machine. The venv here was
built with the repo's own hermit Python 3.10, which satisfies the same 3.10 pin:

    ../bin/python3.10 -m venv .venv
    ./.venv/bin/python -m pip install -r requirements.txt
    ./.venv/bin/python -m pip install 'chatterbox-flash[mlx]==0.1.0'
    touch .venv/.setup-stamp    # satisfies the Makefile's setup prerequisite

    POC_TTS_BENCH_BACKENDS=mlx POC_TTS_BENCH_DTYPE=float16 \
        POC_TTS_BENCH_QUANT_BITS=,4 make bench

The `[mlx]` extra is deliberately not in `requirements.txt`: mlx is macOS-only and would
break the CUDA box's install. A `sys_platform == "darwin"` marker would fix that if the
Mac path is ever meant to be first-class.

## Caveats

- **One sweep per variant**, not two. The 2060 document averages two independent runs.
  Grid ordering below ~10% differences is unresolved here; the headline numbers are the
  exception, being medians of five.
- Single-stream on an otherwise idle machine. No concurrency was measured. This is the
  number most likely to mislead — poc-tts serving two requests at once was never tested.
- Thermals were not controlled. A Mac Studio sustains load; a fanless Mac would not
  reproduce these figures over a full sweep.
- Memory figures come from a separate one-off measurement, not from the sweep —
  `bench.py` records `vram_peak_mb` only under `torch.cuda.is_available()`.
- **No audio quality test was run.** Output length distribution rules out the gross
  hallucination failure mode; it says nothing about whether MLX output is subtly worse
  than the torch path. A WER check against the reference sentences is the obvious next
  step, and `chatterbox_flash.eval.wer_seedtts` exists for it.
- Route 2 (PyTorch MPS) was never attempted, because Route 1 succeeded. `device: mps`
  still raises `ValueError` in `resolve_device`.
