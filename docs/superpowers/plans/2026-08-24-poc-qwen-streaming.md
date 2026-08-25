# poc-qwen-streaming — Rust server embedding the mlx-audio engine via PyO3, streaming Qwen3-TTS to the browser with minimum TTFA

**Date:** 2026-08-24 · **Branch:** `poc-qwen3-tts` (continues from poc-qwen) · **Dir:** `poc-qwen-streaming/`

## Goal

Rebuild the poc-qwen GUI (Voice Design / Voice Clone / TTS tabs) on a Rust
server that streams audio to the browser as mlx-audio emits it, so the
browser's time-to-first-audio tracks the measured in-process 0.12–0.18 s
time-to-first-chunk (`poc-qwen/bench-m4-max.md`) instead of Gradio's
whole-utterance round-trip (1–6 s). Answer the question "is embedding the
Python engine in a Rust process viable for the main app?".

## Decision: PyO3 embed, not a sidecar and not a Rust model

- Rust Qwen3-TTS ports exist (second-state on MLX: RTF 1.5–3.4 on M4 —
  slower than real time; TrevorS on candle: full model set, unmeasured on
  Metal, "an experiment"). mlx-audio measured RTF 0.28–0.38. The Python engine
  stays. See conversation notes in the README.
- PyO3 embedding (one process) over a UDS sidecar: no wire protocol, the
  engine trait leaves a slot for a Rust engine later, and the experiment tells
  us whether embedding is viable for the main app. Cost: a Metal crash takes
  the server down — acceptable for a PoC.

## Architecture

```
browser ──WS /ws──▶ axum (tokio) ──mpsc cmd──▶ python thread (GIL) ──▶ Qwen3Engine (poc-qwen)
        ◀─ int16 PCM frames ◀── tokio mpsc ◀── generator chunks         └─ mlx-worker daemon thread (Metal)
```

- **One dedicated OS thread owns Python.** All PyO3 calls happen on it; axum
  handlers send `EngineCmd` over a `std::sync::mpsc` and await a oneshot /
  bounded `tokio::mpsc` for results. Nothing touches Python from tokio
  workers. The engine's own `mlx-worker` daemon thread (poc-qwen `faca18a`)
  is kept: it owns the Metal state, and `Future.result()` / `queue.get()`
  release the GIL while the Rust thread waits, so the two threads never
  deadlock.
- **Python side is a thin bridge** (`poc_qwen_streaming/bridge.py`) that
  imports `poc_qwen.engine` from `../poc-qwen` and adds a single
  `stream(kind, params, stop)` generator covering all three tabs with
  `stream=True`, sentence-chunked (`chunk_text`, 300 chars) with a 20 ms
  crossfade held back across chunk seams. The stop flag is a
  `threading.Event` Rust sets when the client goes away.
- **Transport: WebSocket, raw int16 PCM 24 kHz binary frames.** Lowest
  possible TTFA on a LAN; no Opus/WebRTC dependency to build. WebRTC is a
  follow-up once TTFA is measured.
- **Playback:** Web Audio, each frame scheduled back-to-back from the first;
  the full take is also assembled into a WAV blob for an `<audio>` control.
- **Interpreter wiring:** `PYO3_PYTHON` = poc-qwen's mise Python 3.12 at
  build time; `build.rs` adds the libpython rpath. At start-up
  `config.yaml → python.paths` are pushed onto `sys.path` (the poc-qwen venv's
  site-packages and the two package dirs), so the binary runs without
  activating anything.
- **Telemetry:** every generation appends to `reports/rs_runs.jsonl` with
  server-side `ttfa_s` (request → first chunk), `gen_s`, `audio_s`, `rtf`;
  the browser reports its own TTFA in the status line. `bench` subcommand
  runs the three bench sentences headless.

## Tasks

### Task 1: Skeleton, build wiring, Python embed smoke
**Files:** `Cargo.toml`, `build.rs`, `src/main.rs`, `src/config.rs`, `config.yaml`, `Makefile`
- [x] `make setup` delegates to `poc-qwen` (venv) and checks cargo/PYO3_PYTHON.
- [x] `cargo run -- info` prints `model_info()` from the embedded engine.

### Task 2: Python bridge with streaming for all three tabs
**Files:** `poc_qwen_streaming/bridge.py`, `tests/test_bridge.py`
- [x] `Bridge(cfg)`: `stream`, `transcribe`, `voices`, `speakers`, `languages`, `model_info`, `unload`.
- [x] Tests with poc-qwen's `FakeModel` (fake loader): chunking, crossfade holdback, stop flag, kwarg mapping per tab.

### Task 3: Rust engine thread
**Files:** `src/engine.rs`, `src/pcm.rs`
- [x] `EngineCmd::{Info, Voices, Speakers, Transcribe, Unload, Generate{..}}`; `Generate` returns a `Receiver<Frame>` of `Vec<i16>` with the first-chunk timestamp.
- [x] Sets the stop event when the receiver is gone.

### Task 4: axum server + WebSocket protocol
**Files:** `src/server.rs`, `ui/*`
- [x] `GET /` static UI; `GET /api/info|voices|speakers|languages`; `POST /api/upload` (WAV from the browser); `POST /api/transcribe`; `POST /api/unload`; `GET /ws`.
- [x] WS: client `{"type":"generate", tab, ...}` → server `start` → binary frames → `done{timings}` | `error`.
- [x] Same three tabs/labels as poc-qwen; presets from `../voices` with `.txt` sidecars; mic capture rendered to WAV in JS (no MediaRecorder container issues).

### Task 5: Bench + README
- [x] `cargo run -- bench` → `reports/rs_runs.jsonl`; README with results vs the 0.9 s bar and the Gradio numbers.

## Results (2026-08-24)

`make bench` (1.7B, warm): TTFA 0.178–0.187 s, RTF 0.35. `tests/e2e_ws.py`
over the WebSocket: client TTFA 0.182 s (1.7B) / 0.124 s (0.6B) — equal to
server-side to the ms. First request of a fresh process was 6.1 s (model
load + kernel compilation + reference encoding) → added `preload:` (models +
preset ICL cache at start-up, 8 s) → first request 0.185 s. Embedding verdict: viable. See `poc-qwen-streaming/README.md`.

## Risks

| Risk | Signal | Fallback |
| --- | --- | --- |
| pyo3 can't find libpython / wrong interpreter | link error or `ModuleNotFoundError: mlx` | `PYO3_PYTHON` + rpath in `build.rs`; `python.paths` in config |
| GIL held by the generator starves other requests | `/api/info` hangs during generation | it doesn't: the engine's queue waits release the GIL; requests still serialize on the python thread by design |
| Metal crash kills the process | server exits mid-demo | `make run` loops; move to a sidecar later behind the same `EngineCmd` seam |
| Chunk seams click across sentence chunks | listen to `reports/*.wav` | crossfade holdback in the bridge (20 ms) |
