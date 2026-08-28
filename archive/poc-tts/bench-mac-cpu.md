# Chatterbox Flash on Apple M4 Max (CPU) — tuning sweep

> **CPU-only baseline.** The Metal/MLX path is roughly 2× faster and is measured in
> [`bench-m4-max.md`](bench-m4-max.md). Read this one for the CPU floor.

Measured 2026-08-23 on `Rolands-Mac-Studio.local`, Apple M4 Max (14 cores: 10P + 4E),
36 GB unified memory, macOS 26.4, torch 2.6.0, `chatterbox-flash==0.1.0`.

Resolved runtime: **device `cpu`, dtype `float32`, backend `torch` (SDPA)**.

Raw data: [`reports/runs.jsonl`](reports/runs.jsonl) — the 36 `machine: arm64` rows
(one sweep; the 72 `x86_64` rows are the RTX 2060 baseline). Reproduce with `make bench`,
but see [Reproducing this run](#reproducing-this-run) — `make setup` does not work on
macOS as written.

## Headline

**Flash runs faster than real time on Apple-silicon CPU once tuned** — but not at every
config: two of the twelve are slower than real time. `drf_block_size=32, num_steps=4` is
roughly 2.3× faster than the library defaults, the same direction as the 2060 at about a
third of the magnitude.

| | library default<br>`blk16 / steps10 / cfm2` | tuned<br>`blk32 / steps4 / cfm1` | speedup |
|---|---:|---:|---:|
| short sentence (30 chars) | 2.60 s | **1.36 s** | 1.9× |
| medium (104 chars) | 6.66 s | **2.66 s** | 2.5× |
| long paragraph (317 chars) | 20.26 s | **8.69 s** | 2.3× |
| mean RTF | 0.857 | **0.539** | 1.6× |

`gen_s` is best-of-2 per measurement, one sweep. The default's mean RTF of 0.857 flatters
it — one of its three sentences over-generated; see [Read `gen_s`, not RTF](#read-gen_s-not-rtf).

## Versus the RTX 2060

Same three sentences, same tuned config (`blk32 / steps4 / cfm1`), so these are directly
comparable.

| | RTX 2060 (fp16, CUDA) | M4 Max (fp32, CPU) | CPU penalty |
|---|---:|---:|---:|
| short | 0.59 s | 1.36 s | 2.3× |
| medium | 1.03 s | 2.66 s | 2.6× |
| long | 3.38 s | 8.69 s | 2.6× |
| mean RTF | 0.174 | 0.539 | 3.1× |

A six-year-old 6 GB GPU is about 2.5× faster than this machine's CPU on this model. That
is a much smaller gap than the usual CPU-versus-CUDA story, and it is the main result
here: **CPU-only Flash is a viable fallback on Apple silicon**, not a degraded mode you
would only use for smoke tests. Note that the 2060 number is itself the *slow* CUDA path
— torch SDPA, not FlashInfer, which will not compile on sm_75.

## Read `gen_s`, not RTF

The same contamination the 2060 sweep documented is present here: RTF is `gen_s /
audio_s`, and Flash's output length varies for identical input, so a config that
over-generates earns a flatteringly low RTF.

`blk16 / steps10 / cfm1` and `blk16 / steps10 / cfm2` differ by 0.28 in mean RTF (1.140
vs 0.857), which reads as a large win for `cfm2`. It is not. `cfm2` produced **14.88 s of
audio for the 104-character medium sentence** — nearly three times the expected length —
which pushed its medium RTF to 0.448 while its `gen_s` (6.66 s) was actually *worse* than
`cfm1`'s (6.13 s). It was not fast; it was verbose.

| sentence | expected | observed range across 12 measurements |
|---|---|---|
| short (30 chars) | ~2 s | 1.92 – 2.60 s |
| medium (104 chars) | ~5.5 s | 5.12 – 7.16 s, plus one 14.88 s outlier |
| long (317 chars) | ~17 s | 16.76 – 26.72 s |

Output length was noticeably better behaved here than on the 2060 run, which saw a 12 s
short sentence and a 4.48–14.88 s medium spread. One sweep is too little to say whether
that is a real device/dtype effect (fp32 versus fp16) or luck.

## Full grid

Twelve configurations, three sentences each, best-of-2 per cell, single sweep.

| blk | steps | cfm | short<br>gen_s | medium<br>gen_s | long<br>gen_s | short<br>rtf | medium<br>rtf | long<br>rtf | mean<br>rtf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 1 | 1.66 | 3.21 | 10.36 | 0.660 | 0.586 | 0.590 | 0.612 |
| 16 | 4 | 2 | 1.72 | 3.20 | 10.66 | 0.879 | 0.447 | 0.630 | 0.652 |
| 16 | 6 | 1 | 2.52 | 3.87 | 13.71 | 0.983 | 0.755 | 0.513 | 0.750 |
| 16 | 6 | 2 | 2.54 | 4.57 | 13.87 | 0.990 | 0.793 | 0.801 | 0.861 |
| 16 | 10 | 1 | 2.51 | 6.13 | 18.85 | 1.278 | 1.056 | 1.086 | 1.140 |
| 16 | 10 | 2 | 2.60 | 6.66 | 20.26 | 1.102 | 0.448 | 1.021 | 0.857 |
| **32** | **4** | **1** | **1.36** | **2.66** | **8.69** | 0.709 | 0.458 | 0.450 | **0.539** |
| 32 | 4 | 2 | 1.48 | 2.71 | 8.19 | 0.629 | 0.521 | 0.489 | 0.546 |
| 32 | 6 | 1 | 1.44 | 3.34 | 10.23 | 0.552 | 0.556 | 0.606 | 0.572 |
| 32 | 6 | 2 | 1.58 | 3.39 | 11.01 | 0.731 | 0.540 | 0.610 | 0.627 |
| 32 | 10 | 1 | 2.29 | 4.65 | 14.28 | 0.880 | 0.825 | 0.783 | 0.829 |
| 32 | 10 | 2 | 2.40 | 4.23 | 16.68 | 0.925 | 0.660 | 0.793 | 0.792 |

`drf_block_size=32` beats 16 at every step count on total `gen_s`, reproducing both the
2060 sweep and the paper's own best row (D=32). `num_steps` is the dominant cost axis: going
4 → 10 adds 70–80% to total generation time. `n_cfm_timesteps` costs little on CPU — under
10% in every matched pair, and not consistently in one direction.

**`blk32/4/1` and `blk32/4/2` are a tie, not a ranking.** They differ by 0.007 mean RTF —
well inside single-sweep noise, and `blk32/4/2` is actually the faster of the two on the
long sentence (8.19 s vs 8.69 s). The bolded row is the tuned config only because it
matches the 2060's pick. Unlike the 2060 document, this grid rests on **one** sweep, so
treat cell-to-cell gaps under ~10% as unresolved.

## Memory

`bench.py` records `vram_peak_mb` only under `torch.cuda.is_available()`, so the sweep
captured no memory figure on this machine. Measured separately on the tuned config: peak
process RSS **~6.5 GB** for load plus one short generation. That is whole-process RSS —
torch runtime, librosa, and fp32 weights together — not a model footprint, and it is
roughly what you would expect from float32 against the 2060's 3.7 GB of fp16 VRAM. On a
36 GB machine it is not a constraint; on an 8 GB Mac it would be.

## Time to first speech

**Today it equals full generation time**, exactly as on the 2060: `/tts` returns a
complete WAV (`poc_tts/server.py`, `Response(content=...)`), so nothing is emitted until
the whole utterance is synthesised — **1.36 s** for a short sentence, **8.69 s** for a
long paragraph.

Model load is **~5.7 s** from a warm Hugging Face cache, paid once at server start. The
first run also downloads ~3.2 GB of weights.

Chunked streaming would put first audio at roughly the first chunk's generation time. At
the default `chunk_size: 120` that is about **2.7 s**; making the first chunk a single
short sentence brings it to about **1.4 s**.

## Is it fast enough to stream?

**Sustainably yes; responsively, marginally.**

At mean RTF 0.54 on the tuned config, generation outruns playback by about 1.9×. The
buffer will not underrun once first audio is out, but the margin is 1.9× rather than the
2060's 4–5×, and it is thinner than that on the worst cells: both
`blk16 / steps10` configs sit at RTF ≥ 1.0 on at least one sentence and **cannot** stream
at all, and five of the twelve reach RTF ≥ 0.9 somewhere. Tuning is not
optional on CPU — it is the difference between streaming and not.

Initial latency is the real cost. ~1.4–2.7 s to first audio is 2–4× the 2060's ~0.6 s.
That is still well inside this project's current end-to-end `reply_start_latency` of
11–14 s, so it would not be the bottleneck today, but it would become one as the rest of
the pipeline tightens.

The same three blockers from the 2060 write-up apply unchanged — `chatterbox_flash`
exposes no streaming API, `synthesize`/`/tts` are not a generator/`StreamingResponse`
pair, and poc-tts does not implement `/v1/audio/speech`, which is the endpoint the voice
pipeline actually calls.

## Superseded: there *is* an accelerated path

> **Update.** This section originally concluded that no GPU path was reachable. That was
> wrong about MLX, and the correction is measured in
> [`bench-m4-max.md`](bench-m4-max.md): installing `chatterbox-flash[mlx]` and setting
> `backend: mlx` + `dtype: float16` runs the LLaMA backbone on Metal and roughly doubles
> the numbers below, with no change to `engine_flash.py`. **Treat this document as the
> CPU-only baseline, not as the state of the art on this hardware.**

What the original reasoning got right and wrong:

- **mlx.** `resolve_backend` accepts `backend: mlx` when named explicitly (upstream marks
  it experimental, so `auto` never picks it). The observation that the engine still
  builds the model on CPU tensors was correct — but that is upstream's *design*, not a
  failure: MLX runs the backbone on Metal while the PyTorch side stays on CPU. Concluding
  it was therefore unusable was the error.
- **MPS.** Still accurate. `resolve_device` accepts only `auto | cuda | cpu`, so
  `device: mps` raises `ValueError` even though `torch.backends.mps.is_available()` is
  `True` here. It was never needed — MLX made Route 2 unnecessary.
- **FlashInfer** is CUDA-only and correctly not selected: `_flashinfer_available()`
  requires `torch.cuda.is_available()`. `setup.sh`'s `pip install flashinfer-python` step
  is a no-op-or-failure on macOS and is already written to never fail the setup.

Every number in this document is the portable SDPA path on CPU cores — the floor for this
hardware, and now a known one.

## Reproducing this run

`make bench` works, but `make setup` does not: `setup.sh` requires `mise`, which is not
installed on this machine, and `mise.toml` pins Python 3.10. The venv here was built with
the repo's own hermit Python 3.10 instead, which satisfies the same pin:

    ../bin/python3.10 -m venv .venv
    ./.venv/bin/python -m pip install -r requirements.txt
    touch .venv/.setup-stamp    # satisfies the Makefile's setup prerequisite
    make bench

The stamp is what lets `make bench` skip `setup.sh`. Installing `mise` (`brew install
mise`) would make the documented path work unchanged.

## Caveats

- **One sweep, not two.** The 2060 document averages two independent runs; this is a
  single run, so per-cell numbers carry more noise and the grid ordering is weakly
  determined below ~10% differences.
- Single-stream on an otherwise idle machine. No concurrency was measured, and CPU
  inference degrades far more sharply than GPU under concurrent load — this is the
  number most likely to mislead if poc-tts ever serves more than one request at a time.
- Thermals were not controlled. A Mac Studio sustains load well, but a fanless Mac would
  not reproduce these figures over a long sweep.
- The ~6.5 GB RSS figure comes from a separate one-off measurement, not from the sweep.
- The over-generation behaviour is a model-quality question, not a code defect. It is
  worth understanding before Flash goes into a latency-sensitive pipeline.
