# qwen-tts-tester — model-testing GUI and bench for `crates/qwen-tts`

The three Qwen3-TTS tabs (**Voice Design**, **Voice Clone**, **TTS
(CustomVoice)**) served from a small axum binary that embeds the *same*
engine the chatbot server uses — `crates/qwen-tts` (Rust `Engine`, PyO3
bridge, `qwen_tts` Python package) — and streams audio to the browser over a
WebSocket as mlx-audio emits it. What you hear here is what the chatbot says.

    make              # build, serve http://127.0.0.1:8008 with gui.yaml (all three models resident)
    make explore      # explore.yaml: CustomVoice preloaded only, LRU 2 — compare built-in speakers vs clones next to a resident LLM
    make bench        # headless TTFA bench -> reports/rs_runs.jsonl + reports/bench_*.wav
    make info         # print model_info() from the embedded engine and exit
    make e2e          # real server + real model over /ws (GPU): client-side TTFA
    make -C ../crates/qwen-tts test-py   # GPU-free unit tests of the Python package

Run from this directory. `HOST=0.0.0.0 make` exposes it on the LAN. The
venv is the crate's (`make -C ../crates/qwen-tts setup`, run on demand); the
binary lands in the workspace `../target/release/qwen-tts-tester`.

This is a dev tool, not part of the product: it lives outside `crates/` and
is never linked into the server. It is `poc-qwen-streaming`'s GUI with the
engine code removed in favour of a path dependency on the crate.

## Profiles

| file | resident | preload | transcribe | for |
| --- | --- | --- | --- | --- |
| `gui.yaml` | 3 | Base, CustomVoice, VoiceDesign + every preset voice | on | tab-hopping without reloads (≈12.8 GiB active) |
| `explore.yaml` | 2 | CustomVoice only | off | voice comparisons next to a resident LLM |
| `../crates/qwen-tts/config/server.yaml` | 1 | Base + preset voices | off | the chatbot server (clone-only) |

Engine sections (`models:`, `generation:`, `voices:`, `preload:`,
`transcribe:`) are read by `qwen_tts.config`; `server:`, `python:`, `bench:`
by the Rust side. Scalars can be overridden with
`POC_QWEN_<SECTION>_<KEY>=value`.

## Results (M4 Max, 1.7B-Base, `one-one` clone, warm)

| path | TTFA | gen | audio | RTF |
| --- | --- | --- | --- | --- |
| Gradio PoC, whole utterance (medium sentence) | 2.21 s | 2.21 s | 5.6 s | 0.38 |
| mlx-audio in-process spike | 0.18 s | 1.96 s | 5.6 s | — |
| `make bench` (Rust ← PyO3 ← engine), 2026-08-24 | 0.178–0.187 s | 0.79 / 1.98 / 6.7 s | 2.1 / 5.6 / 19.4 s | 0.35–0.38 |
| `make e2e` (WebSocket client), 2026-08-26 via the crate | 0.187–0.204 s | 1.87–2.09 s | 5.2–5.9 s | 0.35–0.36 |

- The PyO3 hop and the WebSocket add nothing measurable: client TTFA equals
  the server's request→first-chunk time to the millisecond on localhost. The
  0.18 s floor is the model's time-to-first-chunk at `streaming_interval_s: 0.32`.
- **First request of a fresh process ≈ 0.2 s** because `preload:` loads and
  warms the models and runs one tiny clone per preset voice at start-up
  (~14 s, right after the port binds; `/api/info` shows progress and the UI's
  info bar says "⏳ preloading"). Without it the first click paid ~6 s (model
  load, Metal kernel compilation, first encoding of the reference clip —
  mlx-audio's per-model `_icl_cache`, keyed on `(ref_text, audio)`). A clip
  that was not preloaded (upload/mic, or a preset with a changed transcript)
  still pays its one-off reference encoding, ~0.5–1 s.
- **Memory (via `/api/info`):** one 1.7B bf16 model ≈ 4.3 GiB active; all
  three resident = 12.8 GiB active, 14.3 GiB peak. The per-voice ICL cache is
  kilobytes. If memory is short, trim `preload.models` / `max_resident` or
  point `clone_default` at the 0.6B model (TTFA 0.12 s).
- **If TTFA suddenly reads seconds and RTF > 1 on a warm model, the box is
  swapping**, not the engine. Check `sysctl vm.swapusage` before blaming the
  cache.
- Long sentence (317 chars) streams as ~60 chunks; sentence-chunk seams are
  crossfaded 20 ms in the bridge (`Seam`). Listen to `reports/bench_long.wav`.

## How it is built

```
browser ──WS /ws──▶ axum (tokio) ──mpsc──▶ "python" thread (GIL) ──▶ Bridge → Qwen3Engine
        ◀─ int16 PCM ◀── tokio mpsc ◀── Bridge.stream() chunks         └─ mlx-worker daemon thread (Metal)
```

- `../crates/qwen-tts/src/engine.rs` — the only code that touches Python
  (see the crate README for the single-thread rule, `sys.executable`, stop
  events, and shutdown).
- `src/server.rs` — `/api/{info,catalog,upload,transcribe,unload}`,
  `/voice/{name}` (preset playback), `/ws`. Protocol at the top of the file.
- `src/bench.rs` — `bench` subcommand: the three shared bench sentences
  (`qwen_tts.bench.SENTENCES`), cloned from `bench.voice`.
- `ui/` — static HTML/JS. Plays frames with Web Audio scheduled back-to-back
  from the first one; assembles a WAV for the replay control; mic capture is
  encoded to WAV in JS. Shows browser TTFA and server TTFA.
- `build.rs` — adds libpython's rpath (a library's `rustc-link-arg` does not
  reach a dependent binary).
