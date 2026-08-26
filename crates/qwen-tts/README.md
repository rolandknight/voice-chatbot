# qwen-tts — Qwen3-TTS (mlx-audio) embedded in the server via PyO3

The chatbot's TTS backend for `POC_TTS_BACKEND=qwen`: a Rust `Engine` that
owns one Python thread, and a Python package (`python/qwen_tts`) that runs
mlx-audio's Qwen3-TTS models on Metal. Audio streams out of `Engine::generate`
as int16 PCM chunks ~0.18 s after the request (1.7B, M4 Max).

    make -C crates/qwen-tts setup     # mise python 3.12, .venv, mlx-audio, qwen_tts installed editable
    make server-build                 # root Makefile: PYO3_PYTHON = crates/qwen-tts/.venv/bin/python
    make -C crates/qwen-tts test-py   # GPU-free unit tests (fake model behind the engine's loader hook)
    make -C qwen-tts-tester           # browser GUI for trying models/voices through this exact engine

## Layout

```
Cargo.toml build.rs        Rust half: Engine (src/engine.rs), Config (src/config.rs), PCM helpers
config/server.yaml         the server's engine profile: clone-only, one model resident, presets primed
python/qwen_tts/
  engine.py                Qwen3Engine: model load/LRU/warm on the mlx-worker thread, clone/custom/design, transcribe
  bridge.py                Bridge: the object the Rust thread calls; stream(tab, params, stop) for all three tabs
  text.py config.py        sentence chunking; profile loading with POC_QWEN_<SECTION>_<KEY> overrides
  bench.py                 the three bench sentences shared with the other TTS PoCs
python/tests/              pytest, no GPU (tests/config.yaml is the test profile)
requirements.txt setup.sh mise.toml   the venv; .venv/ is gitignored
```

`config/server.yaml` is read twice: `server:`, `python:` (sys.path entries,
relative to the file) and `bench:` by Rust; `models:`, `generation:`,
`voices:`, `preload:`, `transcribe:` by `qwen_tts.config`.

## How the embed works

```
tokio ──Cmd (std mpsc)──▶ "python" thread (holds the GIL) ──▶ Bridge → Qwen3Engine
      ◀── StreamEvent (tokio mpsc) ◀── Bridge.stream() chunks     └─ mlx-worker daemon thread (Metal)
```

- **One thread touches Python.** MLX keeps per-thread Metal state and
  touching it from short-lived pool threads segfaults (learned in the PoC,
  commit `faca18a`). `Engine::start` spawns the `python` thread, which attaches
  to the interpreter, imports `qwen_tts.bridge`, and serves `Cmd`s from a
  channel for the life of the process; tokio never sees a Python object. The
  engine's own `mlx-worker` daemon thread owns the GPU; the queue waits between
  them release the GIL.
- **Streaming and cancel.** `generate()` returns a `tokio::mpsc::Receiver`;
  chunks are `f32 → i16` converted on the Python thread with the GIL released
  for the send. Dropping the receiver sets a `threading.Event` the bridge
  checks per chunk, so the model stops after the current chunk.
- **Preload.** `preload:` loads and warms the listed models and runs one tiny
  clone per preset voice so mlx-audio's per-model ICL cache is primed; the
  server's first request then pays ~0.2 s instead of ~6 s.
- **Interpreter wiring.** `PYO3_PYTHON` must be this crate's venv Python at
  build time (the root Makefile and `poc/platform.sh` set it). `build.rs`
  adds libpython's directory to the rpath and bakes the interpreter path in
  as `POC_PYTHON`, which `init_bridge` installs as `sys.executable` — an
  embedded interpreter otherwise reports the host binary, and libraries that
  spawn `sys.executable -c …` would launch the server. A dependency's
  `rustc-link-arg` is not transitive, so `crates/server/build.rs` and
  `qwen-tts-tester/build.rs` repeat the rpath step.
- **Shutdown.** The interpreter is never finalized (its Metal state belongs to
  another thread), so `Engine::shutdown` runs Python's `atexit` handlers on the
  Python thread — that is what unlinks the `multiprocessing` semaphore a
  library creates at model load and keeps `resource_tracker` quiet. A library
  also installs Python's SIGINT handler, so a ctrl-c sits pending in Python;
  `shutdown` drains it (`PyErr_CheckSignals`) and sets `SIG_IGN` before running
  the handlers.

## Measured (M4 Max, 1.7B-Base, warm, `streaming_interval_s: 0.32`)

TTFA 0.18–0.20 s, RTF ≈ 0.35, one model ≈ 4.3 GiB active. Details, the
bench, and the profiles for trying other models live in `qwen-tts-tester/`.
Origins: `poc-qwen` (engine, Gradio GUI) and `poc-qwen-streaming` (PyO3 embed,
streaming GUI); plan: `docs/plans/qwen-tts-finalize.md`.
