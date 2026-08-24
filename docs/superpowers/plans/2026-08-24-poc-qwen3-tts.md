# poc-qwen3-tts — Qwen3-TTS voice-cloning demo on the M4 Max (MLX)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-08-24
**Status:** Draft — written autonomously; decisions marked **[assumption]** are the author's calls and can be overridden before Task 1 starts.
**Machine:** Mac Studio, Apple M4 Max, 14 cores, 36 GB unified memory, macOS 25.4 (Darwin).
**Research:** `docs/research/streamable-tts-mac/02-chinese-llm-tts.md` (Qwen3-TTS section) — read it first; this plan does not repeat it.

**Goal:** Stand up `poc-qwen/` (directory name chosen by the requester) — a Gradio app on port **8007** that mirrors the three tabs of https://huggingface.co/spaces/Qwen/Qwen3-TTS (Voice Design / Voice Clone / TTS with preset speakers), runs Qwen3-TTS on the Apple Silicon GPU through **mlx-audio**, and lets us hear zero-shot clones of the `voices/` clips within a day. Whole-utterance generation only. A bench target records latency and RTF next to the Chatterbox numbers so the two models can be compared on the same sentences.

**Non-goal (this iteration):** low time-to-first-audio streaming. The engine is shaped so that iteration 2 can flip `stream=True` and pipe chunks into `poc-tts-streaming`'s Realtime/WebRTC session without rewriting the model code — Task 8 is a 30-minute spike that measures what `stream=True` gives for free, nothing more.

**Architecture:** Same hard boundary as `poc-tts`: `engine.py` owns model lifecycle and every `mlx_audio` import and never imports Gradio; `app.py` owns the UI and never imports `mlx_audio`; `bench.py` drives the engine directly. Every UI test runs with the engine mocked.

**Tech stack:** Python 3.12 pinned via `mise` (installed with Homebrew on 2026-08-24, same convention as `poc-tts/`), `mlx-audio>=0.5.0` (Qwen3-TTS first-class; v0.5.0 released 2026-08-17), `mlx`, `gradio>=5`, `soundfile`, `mlx-whisper` (optional, for auto-transcribing reference clips), `pytest`.

---

## Why these choices

| Decision | Choice | Why |
| --- | --- | --- |
| Runtime | **mlx-audio** on MLX | Only Mac path for Qwen3-TTS that is maintained and streams (needed for iteration 2). The official `qwen-tts` package is torch-only, does not stream, and MPS support is unproven. |
| Models | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (clone, default), `…-0.6B-Base-bf16` (clone, fast), `…-1.7B-CustomVoice-bf16` (presets), `…-1.7B-VoiceDesign-bf16` (design) | Exactly the model set behind the HF Space. ~4.5 GB each in bf16; loading all four is ~15 GB, fine in 36 GB but load lazily (see Task 3). The torch `Qwen/Qwen3-TTS-12Hz-1.7B-Base` already in `~/.cache/huggingface` (4.2 GB) is **not** used by the MLX path — leave it; it's a fallback for an A/B against the official package if MLX quality looks off. |
| GUI | **Gradio**, three tabs cloned from the Space's `app.py` | Fastest route to "looks like the Space". Gradio's `gr.Audio(streaming=True)` + generator handlers cover iteration 2's browser-side streaming too, so the UI isn't throw-away. **[assumption]** The custom `poc-tts-streaming/ui` is not reused here; the Realtime/WebRTC client there is the iteration-2 target, not this one. |
| Reference transcript | Required by Qwen3-TTS `ref_text`; auto-filled with `mlx-whisper` (`mlx-community/whisper-base.en-mlx` is already cached) | The Space makes users type it. Auto-transcription makes drag-and-drop-a-clip demos work; the field stays editable. |
| Python | 3.12 via `mise` | mlx-audio wheels target 3.10–3.13; 3.14 (Homebrew default) is too new for the mlx/torch ecosystem. |

---

## Global Constraints

- New directory `poc-qwen/` at repo root, sibling of `poc-tts/`. Do not touch `poc-tts/`, `poc-tts-streaming/`, or `vendor/`.
- Toolchain is **`mise`** scoped to this directory (`poc-qwen/mise.toml` pins python 3.12), venv at `.venv`, deps via pip — the `poc-tts/` pattern. The repo root stays on hermit (`bin/`). Root `Makefile` gets `poc-qwen-*` targets that delegate with `$(MAKE) -C poc-qwen`, placed next to the existing `poc-tts-*` block (`Makefile:57`).
- Port **8007** (8004 Turbo, 8005 poc-tts, 8006 poc-tts-streaming).
- Bind `127.0.0.1` by default; `HOST=0.0.0.0` env override for LAN demos. Gradio's `share=` is never enabled.
- `voices/` (repo root) is the source of predefined clone references; `voices/README.md` conventions apply (5–15 s, mono, clean). The Voice Clone tab also accepts arbitrary uploads/mic recordings.
- Reports go to `poc-qwen/reports/runs.jsonl` (committed) using the same three bench sentences as `poc-tts/poc_tts/bench.py:39` so rows are comparable with `poc-tts/bench-m4-max.md` (Chatterbox Flash MLX fp16 on this machine: 0.92 s / 1.37 s / 4.20 s short/medium/long, whole-utterance).
- Text longer than ~40 s of speech must be split before generation: the Metal watchdog kills MLX kernels on long single calls (~500-token cap reported by soniqo). Sentence-chunk and concatenate — see Task 4.
- Model ids and every knob live in `config.yaml`; env overrides `POC_QWEN_<SECTION>_<KEY>` mirror `poc-tts`'s `config.py` pattern.
- Commit after every task on branch `poc-qwen`.

---

### Task 1: Skeleton, toolchain, smoke test

**Files:** `poc-qwen/{Makefile,requirements.txt,setup.sh,config.yaml,.gitignore,README.md,poc_qwen/__init__.py,tests/__init__.py}`; modify root `Makefile`.

- [ ] `setup.sh` (idempotent): fail with a clear message if `mise` missing; `mise install`; `mise exec -- python -m venv .venv` if absent; `pip install -r requirements.txt`; write `reports/env_probe.json` with `platform.mac_ver()`, `mlx.__version__`, `mlx_audio.__version__`, `mlx.core.metal.device_info()` (chip name, memory). Never fail setup on the probe.
- [ ] `requirements.txt`: `mlx-audio>=0.5.0,<0.6`, `gradio>=5.0,<6`, `soundfile`, `pyyaml`, `numpy`, `mlx-whisper`, `pytest`, `httpx`. Pin exact versions after the first successful install (record them in the README).
- [ ] `Makefile` mirroring `poc-tts/Makefile`: `run` (default), `setup` (stamp file), `bench`, `test`, `smoke`, `clean`, `help`. `smoke` runs `python -m poc_qwen.smoke` — a 10-line script that loads 0.6B-Base, clones `voices/one-one.mp3` with a hardcoded transcript, writes `reports/smoke.wav`, prints wall time. **This is the go/no-go gate for the whole plan**: if mlx-audio's Qwen3-TTS path does not produce intelligible audio on this box, stop and reassess (fallback: official `qwen-tts` on MPS/CPU with the cached torch weights).
- [ ] `config.yaml`:
  ```yaml
  server: { host: 127.0.0.1, port: 8007 }
  models:
    clone_default: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
    clone_small:   mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
    custom_voice:  mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
    voice_design:  mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
  generation: { temperature: 0.9, top_p: 0.9, max_chunk_chars: 300 }
  voices: { paths: [../voices] }
  transcribe: { model: mlx-community/whisper-base.en-mlx, enabled: true }
  bench: { voice: one-one.mp3, repeats: 3 }
  ```
- [ ] Root `Makefile`: `poc-qwen`, `poc-qwen3-tts-setup`, `poc-qwen3-tts-bench`, `poc-qwen3-tts-test`, each `@$(MAKE) -C poc-qwen <target>`; add to `.PHONY` and `help`.
- [ ] `.gitignore`: `.venv/`, `__pycache__/`, `reports/*.wav`, `.env`.
- [ ] Commit: `feat(poc-qwen3-tts): skeleton, uv toolchain, MLX smoke test`.

**Verify:** `make smoke` produces `reports/smoke.wav` that sounds like the one-one clip. Record wall time and model download size in the README.

### Task 2: Config loading with env overrides

**Files:** `poc_qwen/config.py`, `tests/test_config.py`.

- [ ] Port `poc-tts/poc_tts/config.py` (dataclass sections + `POC_QWEN_*` env overrides + type coercion). Reuse its tests' shape (`poc-tts/tests/test_config_overrides.py`).
- [ ] Resolve `voices.paths` relative to the config file; tolerate missing directories.
- [ ] Commit.

### Task 3: Engine — lazy model registry and synthesis

**Files:** `poc_qwen/engine.py`, `tests/test_engine.py`.

Interface (the only surface `app.py` and `bench.py` may call):

```python
class Qwen3Engine:
    def __init__(self, cfg: Config): ...
    def clone(self, text, ref_audio: str | np.ndarray, ref_text: str | None, language="Auto", size="1.7B", *, xvector_only=False) -> Result
    def custom_voice(self, text, speaker, language="Auto", instruct="", size="1.7B") -> Result
    def voice_design(self, text, instruct, language="Auto") -> Result
    def transcribe(self, audio_path) -> str          # mlx-whisper; "" if disabled
    def speakers(self) -> list[str]                 # from the CustomVoice model's config
    def languages(self) -> list[str]
    def model_info(self) -> dict                    # loaded models, memory in use, versions
@dataclass
class Result: audio: np.ndarray; sample_rate: int; timings: dict  # load_s, prefill_s?, gen_s, audio_s, rtf
```

- [ ] Model registry keyed by HF id; `load_model()` from `mlx_audio.tts.utils` on first use only; `mx.clear_cache()` and drop the reference on `unload()`. **[assumption]** Keep at most two models resident (LRU) so a demo that hops between tabs stays under ~10 GB and leaves room for an LLM.
- [ ] `clone()` calls `model.generate(text=..., ref_audio=..., ref_text=..., lang_code/language=...)` — confirm the exact kwarg names against the installed mlx-audio version's `qwen3_tts` README on day 1 and pin them in one adapter function. `xvector_only=True` passes `ref_text=None` (the Base model supports x-vector-only conditioning; if mlx-audio doesn't expose it, the checkbox is disabled with a tooltip, not faked).
- [ ] Reference-audio cache: hash the reference file → keep the computed ICL prompt/x-vector if mlx-audio exposes it (v0.4.4 added an "ICL cache"); otherwise just cache the loaded, resampled waveform. Measure whether repeat clones of the same voice get faster; record in README.
- [ ] Timings: `time.perf_counter()` around load and generate; `mx.eval` before stopping the clock so lazy evaluation doesn't fake fast numbers. `rtf = gen_s / audio_s`.
- [ ] Warm-up: after each first load, run a 5-word generation so Metal kernel compilation is not charged to the first demo utterance. Log both cold and warm numbers in `model_info()`.
- [ ] Tests with `mlx_audio` mocked via a fake `load_model` returning an object whose `generate` yields a fixed 24 kHz sine: registry LRU, kwarg mapping, timings, xvector path.
- [ ] Commit.

### Task 4: Text chunking and long-input safety

**Files:** `poc_qwen/text.py`, `tests/test_text.py`.

- [ ] Sentence splitter (port `chunk_text` from `poc-tts-streaming/poc_tts_streaming/audio.py` if it's engine-agnostic) with `max_chunk_chars` from config; never split inside a number/abbreviation.
- [ ] Engine generates chunk-by-chunk with the **same** reference, concatenates with a 20 ms crossfade at seams, sums timings. This keeps every Metal call short of the watchdog and is also the shape iteration 2 pipelines.
- [ ] Test: 900-char paragraph → ≥3 chunks, total audio length = sum of parts minus overlaps.
- [ ] Commit.

### Task 5: Gradio app — three tabs matching the Space

**Files:** `poc_qwen/app.py`, `tests/test_app.py`.

Reproduce the Space's layout and labels (from its `app.py`), tab order as in the Space:

- [ ] **Voice Design** tab: `Text to Synthesize`, `Language` dropdown (Auto + 10 languages), `Voice Description`, `Generate` button → `Generated Audio` + status Markdown (model, gen time, RTF). Example row with the Space's example ("It's in the top drawer... wait, it's empty?").
- [ ] **Voice Clone** tab: `Reference Audio` (`gr.Audio(sources=["upload","microphone"], type="filepath")`), a **`Preset voice`** dropdown listing `voices/*.{wav,mp3}` that fills the audio component when chosen (our addition), `Reference Text` textbox with an `Auto-transcribe` button (mlx-whisper; also fired automatically when a preset is chosen and its `<name>.txt` sidecar exists in `voices/`), `Use x-vector only` checkbox, `Target Text`, `Language`, `Model Size` radio (0.6B / 1.7B), `Generate`. Output as above.
- [ ] **TTS (CustomVoice)** tab: `Text to Synthesize`, `Language`, `Speaker` dropdown populated from `engine.speakers()`, `Style Instruction (Optional)`, `Model Size`, `Generate`.
- [ ] Shared: a header line with `model_info()` (chip, mlx-audio version, resident models, memory) and a `Unload models` button. Every handler catches exceptions and returns them in the status box; nothing crashes the server.
- [ ] Each generation appends a row to `reports/ui_runs.jsonl` (tab, model, chars, gen_s, audio_s, rtf) — cheap telemetry for the demo day.
- [ ] `python -m poc_qwen.app` → `demo.queue().launch(server_name=cfg.server.host, server_port=8007)`.
- [ ] Tests: build the Blocks with a fake engine, call each handler function directly (not through HTTP) and assert the returned `(sample_rate, ndarray)` and status text; assert the preset dropdown lists `one-one`, `babel`, `marvin`.
- [ ] Commit.

### Task 6: Reference transcripts for the repo voices

**Files:** `voices/{one-one,babel,marvin}.txt`, update `voices/README.md`.

- [ ] Run `engine.transcribe()` on each clip, correct by ear, save as sidecar `.txt` (Qwen3-TTS clone quality depends on an accurate transcript; whisper-base is not accurate enough to trust blindly).
- [ ] README: document the sidecar convention (Chatterbox ignores it; Qwen3-TTS requires it).
- [ ] Commit.

### Task 7: Bench

**Files:** `poc_qwen/bench.py`, `tests/test_bench.py`, `reports/runs.jsonl`, `bench-m4-max.md`.

- [ ] Same three sentences as `poc-tts/poc_tts/bench.py:39`, `voices/one-one.mp3` reference, `repeats: 3`, discard the first (cold) run. Matrix: `{0.6B, 1.7B} × {bf16}`; add `mlx-community/…-6bit` variants if they exist for Base (they exist for 1.7B-CustomVoice; check HF).
- [ ] Row schema compatible with `poc-tts/reports/runs.jsonl` plus `model`, `ref_cache_hit`, `peak_mem_gb` (`mx.metal.get_peak_memory()`).
- [ ] `bench-m4-max.md`: table of Qwen3-TTS 0.6B / 1.7B vs Chatterbox Flash MLX fp16 (0.92 / 1.37 / 4.20 s) on the same rows; a paragraph on subjective clone quality of the three repo voices vs Chatterbox (accent, prosody, transcript sensitivity); memory; and the go/no-go for iteration 2. Success bar for this iteration: **1.7B whole-utterance medium sentence ≤ 1.5 s warm, RTF ≤ 0.6, clone clearly recognizable.**
- [ ] Commit.

### Task 8: Streaming spike (time-boxed, 30 min, read-only for iteration 2)

**Files:** `poc_qwen/spike_stream.py`, `reports/stream_spike.jsonl`.

- [ ] Call `model.generate(..., stream=True, streaming_interval=0.32)` for the medium sentence, 1.7B and 0.6B, and log time-to-first-chunk, chunk cadence, and whether chunks concatenate seamlessly (save the wav). Do not wire into the UI.
- [ ] Add the numbers and a "what iteration 2 needs" section to `bench-m4-max.md`: expected TTFA, whether mlx-audio's chunk boundaries click, and how `Result` + chunk generator maps onto `poc-tts-streaming/poc_tts_streaming/realtime/session.py`'s audio push.
- [ ] Commit. Open PR `poc-qwen` → `main`.

---

## Risks and fallbacks

| Risk | Signal | Fallback |
| --- | --- | --- |
| mlx-audio Qwen3-TTS API drift (kwarg names, `generate` return shape) — ~72 open issues | Task 1 smoke fails or Task 3 adapter mismatch | Pin the exact mlx-audio version that works; keep all kwarg mapping in one adapter function. |
| Clone quality below Chatterbox for English (Chinese-accent reports are for presets, not clones) | Task 6 listening | Try 1.7B vs 0.6B, transcript accuracy, longer reference (10–15 s); document honestly in bench-m4-max.md. |
| Metal watchdog on long inputs | Generation aborts or the GPU resets on the long sentence | Task 4 chunking; lower `max_chunk_chars`. |
| Memory pressure with two 1.7B models + whisper | `model_info()` memory > ~12 GB, swap | LRU=1, or 6-bit variants. |
| `xvector_only` not exposed by mlx-audio | AttributeError in Task 3 | Disable the checkbox with a tooltip; note as a gap. |
| Gradio 5 mic recording gives 48 kHz stereo | Clone sounds wrong from mic input | Engine resamples/mono-mixes every reference with `soundfile` + `numpy` before use (do this unconditionally). |

## Self-review

Space parity: three tabs, labels, language list, model-size selector, x-vector checkbox → Task 5. Voice cloning of repo voices → Tasks 5–6. Speed on this Mac → Tasks 3 (lazy load, warm-up, ref cache) and 7 (measured). Iteration-2 readiness → engine/UI boundary (Task 3), chunked generation (Task 4), streaming numbers (Task 8). Comparability with prior PoCs → shared bench sentences and jsonl schema (Task 7). Toolchain → mise, matching poc-tts (Task 1).
