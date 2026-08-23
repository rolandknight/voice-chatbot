# poc-tts: Chatterbox Flash on the fastest available path

**Date:** 2026-08-23
**Status:** Approved, pending implementation plan

## Purpose

Stand up a self-contained proof of concept that runs
[Chatterbox Flash](https://huggingface.co/ResembleAI/chatterbox-flash) as fast as
the hardware allows, driven through the same Chatterbox web GUI already used for
Turbo, so cloned-voice quality and latency can be judged by ear and by number
against the incumbent.

Flash earns the PoC on measured evidence, not marketing. On the development
laptop's RTX 2060 it is faster than Turbo at every utterance length while using
40% less VRAM — and that was Flash's *slow* path, with FlashInfer absent and all
tuning at defaults.

### Baseline measurements (RTX 2060 6 GB, i7-10875H, 2026-08-23)

Identical three sentences throughout; RTF, lower is better.

| config | short | medium | long | memory |
|---|---:|---:|---:|---|
| Turbo, CUDA fp32 | 0.835 | 1.099 | 1.121 | 4.7 GB |
| Flash, CUDA fp16, torch SDPA | 0.793 | 0.724 | 0.579 | 2.8 GB peak |
| Turbo, CPU fp32 (8 threads) | 2.588 | 2.606 | 2.644 | — |

Flash is the only configuration faster than real time at all three lengths, and
the only one whose RTF *improves* with length (0.79 → 0.58) rather than
degrading — the block-diffusion decoder amortising over longer sequences instead
of paying per token.

Published reference points for context: RTF 0.076 with FlashInfer and CUDA graphs
on modern hardware; RTF 0.778, or 0.665 at 4-bit, via MLX on a Mac M4.

## Non-goals

- Replacing Turbo anywhere in the production path. `personas.yaml` routing,
  `config.yaml:89-91`, and the 8004 server are untouched.
- Mac / MPS / MLX support. Flash's MLX engine is self-labelled experimental and
  is a separate investigation.
- Multilingual. Flash is English-only by construction (English BPE tokenizer).
- Process supervision, auth, or production hardening.

## Decisions taken

Three decisions were settled with the requester before design:

1. **Standalone server with a copied UI**, not a patch against the vendored
   server and not a fork of it. `/vendor/` is gitignored (`.gitignore:10`) and
   re-cloned by `setup_chatterbox.sh`, so anything patched there is lost. A
   standalone app is git-tracked and disposable.
2. **mise scoped to `poc-tts/` only.** The repo is hermit-managed
   (`bin/activate-hermit`, `python3@3.10`, `rust-1.97.1`). A single
   `poc-tts/mise.toml` pins python for this directory; hermit keeps everything
   else. No repo-wide migration rides along with a TTS PoC.
3. **Empirical fast path.** Attempt FlashInfer and record the result either way,
   then sweep the tuning levers rather than guessing a config.

## Architecture

```
poc-tts/
  mise.toml           python = "3.10"
  setup.sh            mise exec -> .venv, deps, flashinfer probe
  config.yaml         device, dtype, drf_block_size, generation defaults
  engine_flash.py     ChatterboxFlashTTS wrapper
  server.py           FastAPI: serves ui/ and the ten endpoints
  bench.py            sweep -> reports/runs.jsonl
  ui/                 index.html, script.js, styles.css, presets.yaml
  reports/runs.jsonl
  tests/
  .gitignore          .venv/, model_cache/, outputs/
```

Root `Makefile` gains `poc-tts-setup`, `poc-tts`, and `poc-tts-bench`, following
the existing `poc-*` target convention (`Makefile:242-292`).

**Port 8005.** Turbo stays on 8004, so both servers run side by side and the same
reference voice can be A/B'd in two browser tabs. This is the primary way the
PoC gets judged.

Python 3.10 is pinned because that exact combination is already verified working:
3.10 + torch 2.6.0+cu124 + chatterbox-flash 0.1.0.

### Component boundaries

| unit | does | depends on |
|---|---|---|
| `engine_flash.py` | resolve device/dtype/backend, load model once, synthesize one utterance, chunk long text | `chatterbox_flash`, `config.yaml` |
| `server.py` | HTTP surface, static UI, request validation, error mapping | `engine_flash` |
| `bench.py` | sweep configurations, record results | `engine_flash` |

`server.py` never imports `chatterbox_flash` directly, and `engine_flash.py`
never imports FastAPI. That boundary is what lets the endpoint tests run with the
engine mocked and no GPU present.

## Engine

The model loads once at startup and is held for the process lifetime.

### Dtype resolution

The critical correctness detail. `ChatterboxFlashTTS.from_pretrained` defaults to
`dtype=torch.bfloat16` (`tts.py:157`), and the RTX 2060 is sm_75 — verified:
`torch.cuda.is_bf16_supported()` returns `False`. Taking the default on Turing
gives emulated-bf16 speeds or an outright failure.

Resolution order:

1. Explicit `config.yaml` value wins.
2. CUDA and `torch.cuda.is_bf16_supported()` → `bfloat16`.
3. CUDA otherwise → `float16`.
4. CPU → `float32`.

This mirrors the auto-detect shape of `scripts/start_chatterbox.sh:36-48`, so it
reads as familiar to anyone who knows the Turbo launcher.

Note that `dtype` applies **only to T3**. In `from_local` (`tts.py:104-149`) the
next two lines are `s3gen.to(device)` and `ve.to(device)` with no dtype, so the
vocoder and voice encoder stay fp32 regardless. Measured module footprints:

| module | params | fp32 | fp16 |
|---|---:|---:|---:|
| T3 (flash decoder) | 532,406,272 | 2031 MB | 1016 MB |
| S3Gen (vocoder) | 266,030,919 | 1015 MB | forced fp32 |
| VoiceEncoder | 1,423,618 | 5 MB | forced fp32 |
| **total** | **799.9M** | **3052 MB** | **2036 MB** |

Half precision therefore saves roughly 1.0 GB, not 1.5 GB. Measured VRAM at
fp16: 2062 MB allocated at load, 2834 MB reserved at peak.

### Backend resolution

`build_engine` resolves `auto` to `flashinfer` when available and `torch`
otherwise; MLX is never auto-selected and must be named explicitly. The engine
logs which backend actually resolved at startup and reports it via
`/api/model-info`, so a silent fallback to the slow path can never be mistaken
for the fast one.

### Reference voices

The repo-tracked `voices/` directory is the source of truth, since it is
committed and always present. The vendored server's `reference_audio/` is also
scanned when it exists, so the `marvin.wav` already staged there stays usable for
direct A/B against Turbo — but the PoC must run correctly with the vendor clone
absent. No third copy of the reference clips is created.

## GUI

Eleven endpoints, matching what `ui/script.js` actually calls:

`/`, `/styles.css`, `/script.js`, `/api/ui/initial-data`, `/api/model-info`,
`/get_predefined_voices`, `/get_reference_files`, `/save_settings`,
`/reset_settings`, `/tts`, and `/restart_server`.

`/restart_server` returns a clear "not supported in the PoC" message rather than
404 — the UI calls it, and a silent failure would read as a bug.

`/api/ui/initial-data` must return the shape `script.js` consumes:
`{config, reference_files, predefined_voices, presets, initial_gen_result,
model_info}`.

### Knob mapping

The UI already reshapes itself per model — `script.js:373` hides exaggeration and
CFG for Turbo, driven by `model_info`. The design extends that mechanism instead
of working around it.

| UI control | Flash parameter | status |
|---|---|---|
| temperature | `temperature` | exists |
| exaggeration | `exaggeration` | exists |
| CFG weight | `cfg_scale` | exists, relabel |
| chunk size / split text | server-side chunking | exists |
| — | `num_steps` | new slider |
| — | `n_cfm_timesteps` | new slider |
| — | `drf_block_size` | new select (16 / 32) |
| — | `backend` | new select |
| seed, language | — | hidden via `model_info` |

The four new controls are an additive edit to the committed `ui/` copy.

## Bench

`bench.py` sweeps `backend × drf_block_size × num_steps × n_cfm_timesteps` over
the same three sentences used for the baseline table, best-of-2 per cell, and
appends to `poc-tts/reports/runs.jsonl` in the field style of
`poc/reports/runs.jsonl` — plus `vram_peak_mb`, resolved `dtype`, and resolved
`backend`.

Using the same three sentences is what makes the sweep comparable to the Turbo
and CPU rows already recorded.

`setup.sh` attempts `pip install flashinfer-python` and records whether it builds
for sm_75. FlashInfer's documented floor is sm_75, so the 2060 nominally
qualifies, but its kernels lean on newer tensor-core features and practical
support is unproven here. Either outcome is a finding worth recording. Failure is
logged, never fatal.

## Testing

TDD, with the engine mocked so the suite runs GPU-free:

- Dtype resolution across capability combinations. The sm_75 bf16 trap gets an
  explicit regression test — it is the single most likely way this PoC silently
  runs slow.
- Backend resolution and fallback, including that a FlashInfer-absent
  environment resolves to `torch` and says so.
- Text chunking boundaries.
- Endpoint contract: `/api/ui/initial-data` returns the keys `script.js` reads.

GPU generation itself is covered by `bench.py`, not pytest.

## Error handling

- **CUDA OOM** → report VRAM required against VRAM free, and name the processes
  holding the rest. This failure has already occurred twice during
  investigation; a bare `CUDA error: out of memory` wastes the reader's time.
- **bf16 requested on unsupported hardware** → refuse at startup with the
  capability reported, rather than running slowly.
- **Missing reference wav** → name the path searched.
- **FlashInfer absent** → log once at startup, fall back, continue.

## Risks

- **UI drift.** The `ui/` copy will not track upstream. Accepted as the cost of
  the standalone path.
- **`chatterbox-flash` is 0.1.0** — a single PyPI release, with its MLX engine
  self-labelled experimental. Pin the version exactly.
- **First run downloads 3.2 GB** and took ~160 s cold.
- **Torch version divergence.** `chatterbox-flash` requires `torch>=2.6`; the
  Turbo server venv is pinned at 2.5.1+cu121. The separate venv under `poc-tts/`
  is what keeps these from colliding — the Turbo environment must not be
  upgraded to satisfy Flash.
