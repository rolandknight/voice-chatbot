# Build plan — GPU-accelerated Chatterbox Flash on Apple Silicon

Companion to [`bench-rtx-2060.md`](bench-rtx-2060.md), which covers the NVIDIA side.
This plan is written to be handed to a fresh session on the Mac verbatim; it assumes no
prior context beyond this repository.

**Goal:** get Chatterbox Flash onto Apple Silicon's GPU, benchmark it against the
recorded RTX 2060 baselines, and write the results to `poc-tts/bench-<chip>.md`
mirroring the structure of `bench-rtx-2060.md`.

## Current state on Mac

`poc-tts/` runs today on Apple Silicon, but **CPU-only**. `resolve_device`
(`poc_tts/engine_flash.py:36`) accepts `auto | cuda | cpu`; `auto` resolves to `cpu`
when CUDA is absent, and an explicit `device: mps` raises `ValueError`.

Nothing crashes — every CUDA-specific call is guarded (`_vram_report`,
`_bf16_supported`, all five `torch.cuda.*` calls in `bench.py`), `setup.sh` has no
Linux assumptions, and `torch.cuda.OutOfMemoryError` resolves even without CUDA. The
PoC simply never reaches the GPU.

Start by confirming that behaviour on the actual machine before changing anything.

## Route 1 — MLX (try this first)

`chatterbox-flash` ships a first-party Apple Silicon backend. From its own backend
table:

| `--backend` | Engine | Device | dtype | CUDA graph | Notes |
| --- | --- | --- | --- | --- | --- |
| `gpu` (default) | torch SDPA | cuda | bf16 | on | No JIT cold start |
| `flashinfer` | FlashInfer | cuda | bf16 | on | Paged KV; CUDA only |
| `cpu` | torch SDPA | cpu | fp16 | off | CPU-only validation |
| `mlx` | mlx (Metal) | **cpu\*** | fp16 | off | Apple Silicon native (`[mlx]` extra) |

\* MLX runs the LLaMA backbone on Metal; the PyTorch side stays on CPU.

**This likely needs no `resolve_device` change.** The torch device stays `cpu` and the
backend does the Metal work. `resolve_backend` already accepts an explicit `"mlx"`, and
`FlashEngine` forwards `self.backend` into `generate()` — so `backend: mlx` in
`config.yaml` may work as-is. **Verify that before writing any code.**

Install: `chatterbox-flash[mlx]` (pulls `mlx>=0.13`, `mlx-lm>=0.10`; macOS only).

Two gotchas:

- **MLX is never selected automatically.** `build_engine` resolves `auto` to
  flashinfer-or-torch only; the module docstring is explicit that MLX "must be selected
  explicitly via `backend="mlx"` (or the `CHATTERBOX_FLASH_ENGINE=mlx` env var)".
- **Set `dtype: float16` in `config.yaml`.** `resolve_dtype("auto", "cpu", ...)` returns
  `float32`, but the MLX path expects fp16.

### Quantization is MLX-only

`quantize_bits={4,8}` quantizes the T3 LLaMA backbone via `mlx.nn.quantize`; the S3Gen
vocoder and voice encoder stay at `dtype`. Group size is tunable via
`CHATTERBOX_FLASH_MLX_QUANT_GROUP` (default 64). There is no CPU quantization path.

Benchmark fp16 and 4-bit. Resemble publishes RTF **0.778** and **0.665** respectively on
a Mac M4 — those are the figures to reproduce or refute.

## Route 2 — PyTorch MPS (only if MLX disappoints)

Flash's `TorchSDPAEngine` docstring claims it "runs on any CUDA / CPU / MPS device the
underlying T3 supports." Pursuing this needs:

- an `mps` branch in `resolve_device`, plus tests (patch the capability probes; do not
  depend on the local machine);
- a dtype rule for MPS — `resolve_dtype` currently returns `float32` for any non-CUDA
  device.

Expect friction. The vendored Turbo server carries a patch for exactly this class of
bug — `_patch_chatterbox_mps_float32` in `vendor/chatterbox-tts-server/start.py` forces
float32 in `s3tokenizer` and `voice_encoder` to work around *"Cannot convert a MPS
Tensor to float64 dtype"*. Read that patch before debugging from scratch.

## Benchmarking

`make bench` sweeps configurations and appends to `reports/runs.jsonl`.

**Do not change the three `SENTENCES` in `poc_tts/bench.py`.** They are byte-identical
across every baseline in this project — the RTX 2060 sweeps, the Chatterbox Turbo CUDA
run, and the Turbo CPU run. Changing them invalidates all comparisons, which is the
entire point of the sweep.

Add Mac-specific axes (backend, quantization) as needed, but keep the recorded row
schema: `ts`, `host`, `machine`, `model`, `device`, `dtype`, `backend`, `config`,
`sentence`, `results{audio_s, gen_s, rtf}`.

### Report `gen_s`, not RTF

**RTF is contaminated on this model.** It is `gen_s / audio_s`, and Flash's output
length varies substantially for identical input, so a configuration that over-generates
earns a flatteringly low RTF.

The clearest case from the 2060 sweep: `blk16 / steps4 / cfm1` posted the grid's
second-best RTF (0.158) while producing a mean of **10.00 s of audio for a
104-character sentence** that should run about 5.5 s. It was not fast, it was verbose.

Watch for the same effect here, and record output lengths. On the 2060, one measurement
in twenty-four produced **12 s of speech from a 30-character input** — in 0.6 s, so
latency stayed normal while the content ran long.

### Baselines to compare against

RTX 2060 (6 GB, sm_75), fp16, torch SDPA, tuned `drf_block_size=32 num_steps=4`, median
`gen_s`:

| sentence | gen_s |
| --- | ---: |
| short (30 chars) | 0.59 s |
| medium (104 chars) | 1.03 s |
| long (317 chars) | 3.38 s |

Library defaults (`blk16 / steps10 / cfm2`) were 2.5–3.3× slower. `drf_block_size=32`
beat 16 at every step count, independently reproducing the paper's D=32 guidance.

**Calibrate expectations:** the published Mac figures are worse than these. Resemble's
RTF 0.778 on an M4 compares against roughly 0.19 measured on the 2060. Even allowing for
a Max or Ultra chip and the RTF caveat above, the Mac may well lose. The honest question
this benchmark answers is *"is Apple Silicon fast enough to deploy on"*, not *"does it
beat the laptop"*.

## Settled findings — do not re-derive

- **FlashInfer is CUDA-only and irrelevant on Mac.** It also cannot run on sm_75 at all:
  with a CUDA 12.4 toolkit installed its JIT compiles for sm_75 and then fails a static
  assertion on fp16 QK reduction, twelve times. That is why the 2060 numbers are the
  SDPA path rather than the paper's fast path.
- **The bf16 trap.** `torch.cuda.is_bf16_supported()` defaults to
  `including_emulation=True` and returns `True` on hardware where bf16 is emulated and
  slow — which silently defeated the guard the whole design was built around.
  `_bf16_supported()` handles it. CUDA-specific, but do not "simplify" it away.
- **UI asset caching.** The hand-served UI files carry `no-store` and versioned
  references. A stale `script.js` once made browser verification report old behaviour
  while the server served new code, costing hours. If you edit the UI and the browser
  disagrees with the file, bust the page URL with a query string.

## Constraints

- **Never touch `vendor/chatterbox-tts-server/` or its venv.** It is pinned at
  `torch 2.5.1+cu121` and Flash requires `>=2.6`; that conflict is precisely why
  `poc-tts/` has its own environment.
- Python is mise-pinned to 3.10 for `poc-tts/` only. The rest of the repo is
  hermit-managed — do not add a root `mise.toml` or touch any `bin/` shim.
- Keep the suite green (78 tests) and GPU-free. Endpoint tests inject a mock engine into
  `create_app`; a test that loads real weights or needs a GPU is a defect.
- **Verify in a real browser, not by grepping served files.** That shortcut shipped two
  Critical defects on the NVIDIA side — a 422 on every Generate click, and a missing
  `wavesurfer.min.js` that broke playback. Both were invisible to curl and to substring
  checks of the served assets.

## Definition of done

1. Flash runs on the Mac GPU, with the route and any code changes documented.
2. `make bench` produces a full sweep including the Mac backend, appended to
   `reports/runs.jsonl`.
3. `poc-tts/bench-<chip>.md` written in the shape of `bench-rtx-2060.md`: headline
   config, method, the `gen_s`-over-RTF caveat, full grid, time-to-first-speech, and
   honest caveats.
4. A browser check confirming the GUI still works on the Mac path.
5. Suite green.
