# Plan: finalize the Qwen3-TTS backend

Status: implemented 2026-08-26 on branch `python-cleanup` (steps 1–5; the
config move of step 4 landed with step 1). Kept as the record of the layout
and its reasons.

Originally: Makes `crates/qwen-tts` self-contained (Rust
and Python halves, own venv) and adds a top-level `qwen-tts-tester/` dev
tool built from copies of `poc-qwen-streaming`'s GUI and bench. **No `poc*`
directory is touched**: everything is copied, nothing moved or deleted.
Archiving the PoC dirs is a separate, later step once nothing depends on
them (this plan's end state is exactly that).

## Where things are today

| Piece | Location | Notes |
|---|---|---|
| Engine (mlx-audio load/LRU/warm, MLX worker thread, clone/custom/design, transcribe, voice discovery) | `poc-qwen/poc_qwen/engine.py` (396 lines) | Imported at runtime by the bridge via `sys.path` |
| Config loader, text chunking | `poc-qwen/poc_qwen/config.py`, `text.py` | Same |
| Gradio GUI (3 tabs, whole utterance, :8007) | `poc-qwen/poc_qwen/app.py` | Pulls `gradio` into the venv |
| Bench / smoke / stream spike | `poc-qwen/poc_qwen/bench.py`, `smoke.py`, `spike_stream.py` | Python-only, whole-utterance timings |
| Python venv the server links against | `poc-qwen/.venv` | `PYO3_PYTHON` in `Makefile:16`, `poc/platform.sh:324` |
| Engine unit tests (fake loader, GPU-free) | `poc-qwen/tests/` | `test_engine`, `test_text`, `test_config`, `test_app`, `test_bench` |
| PyO3 bridge (`Bridge`, `stream()`, `preload()`, `Seam`) | `crates/qwen-tts/python/poc_qwen_streaming/bridge.py` | Byte-identical copy of `poc-qwen-streaming/poc_qwen_streaming/bridge.py` |
| Rust engine (Python thread, `Cmd` channel, shutdown) | `crates/qwen-tts/src/{engine,config,pcm}.rs` | Diverged from the PoC copy (shutdown, config) |
| Streaming GUI: axum routes + browser UI (:8008, WS) | `poc-qwen-streaming/src/server.rs`, `ui/` | Exercises the *production* path: Rust engine → bridge → mlx |
| Headless TTFA bench | `poc-qwen-streaming/src/bench.rs` | `reports/rs_runs.jsonl` |
| Bridge tests (fake model) | `poc-qwen-streaming/tests/test_bridge.py`, `conftest.py` | `sys.path` hacks into both PoC dirs |
| Engine profiles | `poc-qwen-streaming/config{,.explore,.flowcat}.yaml`, `crates/server/config/qwen.yaml` | `qwen.yaml` has `python.paths` pointing at `../../../poc-qwen` |

Everything the server needs at runtime is ~500 lines of Python spread over
two PoC directories plus a venv that only exists because `make -C poc-qwen
setup` was run once.

## Decisions

**Q: Should all the Python live in `crates/qwen-tts/python/`?**
Yes — everything the *server* needs, as one package. The crate is the unit
that is built, tested, and shipped; its Python half is part of it exactly
like `build.rs` is. One package `qwen_tts` (engine, text, config, bridge),
one `requirements.txt`, one venv at `crates/qwen-tts/.venv`. No `sys.path`
entries outside the crate.

**Q: Where does the GUI for testing models live?**
It's a developer tool, not product code, but it must use the *same* engine
and bridge the server uses or it tests the wrong thing. So: a top-level
`qwen-tts-tester/` workspace crate that depends on `crates/qwen-tts` by
path — `serve` / `bench` / `info` subcommands, static UI, GUI profiles,
bench reports. That is `poc-qwen-streaming`'s GUI copied, not
rewritten. Keeping it out of `crates/` says "not shipped"; keeping it a
path dependency (not a copy) says "tests the real thing". The library
crate stays lean: no axum, no UI assets linked into the server.

Keep exactly one GUI: the streaming one. The Gradio app is not copied;
it stays usable in `poc-qwen` until that dir is archived. Reasons: the streaming UI runs the production path (Rust engine,
PyO3 bridge, chunking, seam crossfade) so what you hear is what the chatbot
says; it already has all three tabs, upload, transcribe, unload, and the
explore profile; and dropping Gradio removes a heavy dependency from the venv
the server embeds. The Gradio app's one feature the streaming UI lacks —
whole-utterance timings to `reports/runs.jsonl` — is covered by `bench`.

**Q: What about `poc-qwen/` and `poc-qwen-streaming/`?**
Untouched. Files are copied out, not moved, so both PoCs keep working
against their own venv and configs and their `reports/` remain the measured
baselines. After step 5 nothing outside the PoC dirs references them, which
is the precondition for moving all `poc*` dirs to `archive/` later.

## Target layout

```
crates/qwen-tts/                # the library the server embeds — Rust + Python halves
  Cargo.toml                    # unchanged deps
  build.rs                      # unchanged: rpath + POC_PYTHON from PYO3_PYTHON
  Makefile                      # setup, test-py, clean
  requirements.txt              # mlx-audio, mlx-whisper, soundfile, pyyaml, numpy, pytest
  mise.toml                     # python 3.12 (from poc-qwen)
  setup.sh                      # venv + deps, idempotent stamp (from poc-qwen)
  .venv/                        # gitignored; PYO3_PYTHON target
  config/
    server.yaml                 # what crates/server/config/qwen.yaml is today (moves here)
  python/
    qwen_tts/
      __init__.py
      engine.py                 # from poc_qwen/engine.py
      text.py                   # from poc_qwen/text.py
      config.py                 # from poc_qwen/config.py (POC_DIR → crate-relative)
      bridge.py                 # from poc_qwen_streaming/bridge.py; imports .engine
    tests/
      conftest.py               # fake loader/model; no sys.path hacks (package is installed -e)
      test_engine.py test_text.py test_config.py test_bridge.py
  src/
    lib.rs config.rs engine.rs pcm.rs          # unchanged
  README.md                     # setup, profiles, how the embed works (ADR from the PoC README)

qwen-tts-tester/                # dev tool: the streaming GUI + bench (copied from poc-qwen-streaming)
  Cargo.toml                    # workspace member; qwen-tts = { path = "../crates/qwen-tts" }, axum, tower-http
  Makefile                      # run, explore, bench, info (all `make -C ../crates/qwen-tts setup` first)
  gui.yaml                      # poc-qwen-streaming/config.yaml (flat: ui/, reports/, uploads/ resolve next to the profile)
  explore.yaml                  # poc-qwen-streaming/config.explore.yaml
  build.rs                      # libpython rpath (a dependency's rustc-link-arg is not transitive)
  src/
    main.rs                     # serve | bench | info
    server.rs bench.rs          # from poc-qwen-streaming/src
  ui/
    index.html app.js styles.css
  reports/                      # rs_runs.jsonl copied so the bench history is continuous; new rows gitignored
  README.md                     # GUI, bench, explore profile, measured numbers (bench-m4-max.md + streaming README)
```

`crates/server` changes: `config/qwen.yaml` → `POC_QWEN_CONFIG` default
becomes `crates/qwen-tts/config/server.yaml`; `python.paths` becomes
`[../.venv/lib/python3.12/site-packages, ../python]`. Root `Makefile`
`QWEN_PYTHON := crates/qwen-tts/.venv/bin/python`; `poc/platform.sh:324`
same path and message (`make -C crates/qwen-tts setup`).

## Steps

Each step builds, passes `make check`, and leaves the server runnable.

1. **Python package.** Copy `engine.py`, `text.py`, `config.py` into
   `crates/qwen-tts/python/qwen_tts/`; rename the crate's own
   `python/poc_qwen_streaming/bridge.py` → `qwen_tts/bridge.py` with
   `from .engine import …`. Rust `init_bridge` imports `qwen_tts.bridge`.
   Copy the four test files over; `conftest.py` drops its `sys.path` lines.
   `qwen.yaml` `python.paths` drops the `../../../poc-qwen` entry (the venv
   entry stays for now). Verify: `make server-build`, server starts, speaks.

2. **Own venv.** Copy `requirements.txt` (minus `gradio`, `httpx`),
   `mise.toml`, `setup.sh` from poc-qwen into the crate; venv at `crates/qwen-tts/.venv`,
   package installed editable (`pip install -e python`). Point `Makefile`,
   `poc/platform.sh`, and `python.paths` at it. Crate `Makefile` with
   `setup`, `test-py`, `clean`.
   Verify: `rm -rf` nothing — build against the new venv from a clean
   `cargo clean`, run the server, run `pytest`.

3. **Tester.** New `qwen-tts-tester/` with copies of
   `poc-qwen-streaming/{src/main.rs,src/server.rs,src/bench.rs,ui/,
   config.yaml,config.explore.yaml,Makefile,reports/rs_runs.jsonl}`. Not
   copied: `src/{engine,config,pcm,lib}.rs`, `build.rs`,
   `poc_qwen_streaming/`, `tests/`, `Cargo.lock` (all superseded by the
   crate). Root workspace member with
   `qwen-tts = { path = "../crates/qwen-tts" }`; `main.rs` uses
   `qwen_tts::{config::Config, engine::Engine}`. Profiles into `config/`,
   `--config` default `config/gui.yaml`. Verify: `make -C qwen-tts-tester run`, all three
   tabs generate in the browser; `bench` appends a row to `rs_runs.jsonl`.

4. **Server config.** `crates/server/config/qwen.yaml` →
   `crates/qwen-tts/config/server.yaml`; server default path updated;
   `python.paths` crate-relative. Verify: server start with no `POC_QWEN_*`
   overrides.

5. **Docs + cut-over check.** Crate README (setup, server profile, how
   the embed works) and tester README (GUI, bench, explore profile,
   measured numbers from `bench-m4-max.md` and the streaming README). Root
   README/`docs/` references updated; the PoC READMEs and
   `docs/superpowers/plans/*qwen*` are left as they are.
   Done when `grep -rn poc-qwen` outside `poc*/` and `docs/` returns
   nothing. Verify: clean clone → `make -C crates/qwen-tts setup && make
   server-build && make server`, then `make -C qwen-tts-tester run` — with
   no `poc-qwen*` venv present.

## Out of scope

- Rewriting the engine in Rust (mlx-audio has no Rust binding; the PyO3
  embed is the design, see ADR in `poc-qwen-streaming/README.md`).
- The Gradio GUI's persona/`reports/runs.jsonl` workflow — not carried over.
- Whisper transcription in the server profile stays `enabled: false`.
- Archiving any `poc*` dir (including `poc-qwen*`) — separate step once
  this plan's cut-over check passes.
