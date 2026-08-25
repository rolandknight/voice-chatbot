# poc-qwen-streaming — Qwen3-TTS streamed from a Rust server (PyO3-embedded mlx-audio)

Same three tabs as `poc-qwen` (**Voice Design**, **Voice Clone**, **TTS
(CustomVoice)**), but the server is Rust (axum) with poc-qwen's `Qwen3Engine`
embedded through PyO3, and audio is streamed to the browser over a WebSocket
as mlx-audio emits it — so the browser hears the first chunk ~0.18 s after
clicking Generate instead of waiting for the whole utterance.

    make              # build (needs poc-qwen's venv; created on demand), serve http://127.0.0.1:8008
    make bench        # headless TTFA bench -> reports/rs_runs.jsonl + reports/bench_*.wav
    make info         # print model_info() from the embedded engine and exit
    make test         # GPU-free: bridge unit tests (fake model) + Rust unit tests
    ../poc-qwen/.venv/bin/python tests/e2e_ws.py   # real server + real model over /ws (GPU)

Run from this directory. `HOST=0.0.0.0 make` exposes it on the LAN. The
repo-root `make poc-qwen-streaming*` targets delegate here.

## Results (2026-08-24, M4 Max, 1.7B-Base, `one-one` clone, warm)

| path | TTFA | gen | audio | RTF |
| --- | --- | --- | --- | --- |
| poc-qwen Gradio (whole utterance, medium sentence) | 2.21 s | 2.21 s | 5.6 s | 0.38 |
| mlx-audio in-process spike (`poc-qwen/bench-m4-max.md`) | 0.18 s | 1.96 s | 5.6 s | — |
| **this PoC, `make bench` (Rust ← PyO3 ← engine)** | **0.178–0.187 s** | 0.79 / 1.98 / 6.7 s | 2.1 / 5.6 / 19.4 s | 0.35–0.38 |
| **this PoC, WebSocket client (`tests/e2e_ws.py`)** | **0.182 s** | 1.94 s | 5.5 s | 0.35 |

- The PyO3 hop and the WebSocket add nothing measurable: client TTFA equals
  the server's request→first-chunk time to the millisecond on localhost. The
  0.9 s research bar is beaten by ~5×; the 0.18 s floor is the model's
  time-to-first-chunk at `streaming_interval_s: 0.32`.
- **First request of a fresh process: 0.185 s** — because `preload:` in
  `config.yaml` loads + warms the three demo models and runs one tiny clone
  per preset voice at start-up (8 s total, right after the port binds;
  `/api/info` shows progress and the UI's info bar says "⏳ preloading").
  Without it the first click paid ~6 s: model load from the HF cache (~4 s)
  + Metal kernel compilation + the first encoding of the reference clip
  (mlx-audio's per-model `_icl_cache`, keyed on `(ref_text, audio)`).
  `max_resident: 3` keeps all three models loaded so tab-hopping never
  reloads. A clip that was *not* preloaded (upload/mic, or a preset with a
  changed transcript) still pays its one-off reference encoding, ~0.5–1 s,
  on its first use.
- **Memory (measured via `/api/info`):** one 1.7B bf16 model ≈ 4.3 GiB
  active; all three resident = **12.8 GiB active, 14.3 GiB peak**, out of
  MLX's 28 GiB recommended working set on the 36 GB M4 Max. The per-voice
  ICL cache (reference codes + token ids) is kilobytes — preloading every
  preset voice is free; the models are the cost. If memory is short, trim
  `preload.models` / `max_resident` (Base + CustomVoice ≈ 9 GiB) or point
  `clone_default` at the 0.6B model.
- **If TTFA suddenly reads seconds and RTF > 1 on a warm model, the box is
  swapping**, not the engine: one run showed TTFA 8.1 s / RTF 2.9 while
  `vm.swapusage` was 29 of 30 GB (an Ollama `llama-server` held 5 GiB and
  the compressor 12 GB). The next run of the same voice was 0.54 s, then
  0.19 s, once the weights were paged back in. Check `sysctl vm.swapusage`
  before blaming the cache.
- 0.6B: TTFA 0.12 s in the spike; run `tests/e2e_ws.py --size 0.6B`.
- Long sentence (317 chars) streams as 57–63 chunks; the sentence-chunk seams
  are crossfaded 20 ms in the bridge (`Seam`). Listen to `reports/bench_long.wav`.

## How it is built

```
browser ──WS /ws──▶ axum (tokio) ──mpsc──▶ "python" thread (GIL) ──▶ Bridge → Qwen3Engine (poc-qwen)
        ◀─ int16 PCM ◀── tokio mpsc ◀── Bridge.stream() chunks         └─ mlx-worker daemon thread (Metal)
```

- `src/engine.rs` — the **only** code that touches Python. One OS thread
  attaches to the interpreter at start-up, imports `poc_qwen_streaming.bridge`,
  and serves `Cmd`s from a channel forever; tokio never sees a Python object.
  MLX keeps per-thread Metal state (poc-qwen `faca18a`), so the engine's own
  `mlx-worker` daemon thread stays in charge of the GPU; the Python queue
  waits release the GIL, so the Rust thread blocking on the generator never
  starves it. Dropping a generation's receiver sets a `threading.Event` the
  bridge checks per chunk (cancel/disconnect).
- `poc_qwen_streaming/bridge.py` — imports `poc_qwen.engine` from
  `../poc-qwen` (nothing copied) and adds `Bridge.stream(tab, params, stop)`:
  one generator for all three tabs using `generate(stream=True)`, sentence
  chunking (`chunk_text`, 300 chars) and the crossfade holdback. The kwarg
  mapping onto mlx-audio's API lives in `_kwargs()` alone.
- `src/server.rs` — `/api/{info,catalog,upload,transcribe,unload}`,
  `/voice/{name}` (preset playback), `/ws`. Protocol at the top of the file.
- `ui/` — static HTML/JS. Plays frames with Web Audio scheduled
  back-to-back from the first one; assembles a WAV for the replay control;
  mic capture is encoded to WAV in JS so the server gets a container
  mlx-audio loads without guesswork. Shows browser TTFA and server TTFA.
- `src/bench.rs` — `bench` subcommand; `config.yaml` — `server:`, `python:`
  (sys.path entries), `bench:` for Rust, the rest for poc-qwen's loader.

### Interpreter wiring (the part that bites a fresh checkout)

`PYO3_PYTHON` must be poc-qwen's venv Python (mise 3.12, shared libpython) at
**build** time — the Makefile exports it. `build.rs` adds libpython's dir to
the rpath and bakes the interpreter path in as `sys.executable` (an embedded
interpreter otherwise reports the Rust binary, and libraries that spawn
`sys.executable -c …` would launch the server). `config.yaml → python.paths`
is prepended to `sys.path` at start-up, so nothing needs activating. Hermit's
cargo (`../bin/cargo`) builds it; the Python is mise's, as in poc-qwen.

## Verdict on embedding

Viable. Zero latency cost, one binary to launch, and the `Cmd` seam is a
clean place to swap in a Rust engine (e.g. the `qwen_tts` candle port) or
move Python to a sidecar if Metal crashes ever become a problem. Things to
carry into the main app: the single-thread rule, `sys.executable`, and
`stop` events for cancellation.

Plan: `docs/superpowers/plans/2026-08-24-poc-qwen-streaming.md`.
