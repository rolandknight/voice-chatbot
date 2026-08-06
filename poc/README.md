# FlowCat PoC — Python side (harness + stubs)

See `CONTRACT.md` for the fixed integration contract and
`../docs/poc/flowcat-poc-plan.md` for the why. The Rust embedder lives in
`poc/flowcat/` (built in parallel; not touched here).

## Layout

- `stubs/skills.json` — the 8 tool schemas (single source of truth for the
  stub server and the Rust embedder).
- `stubs/stub_server.py` — stub skill services on :8790 (call log,
  latency/failure injection).
- `stubs/kokoro_shim.py` — OpenAI-speech Kokoro TTS shim on :8880
  (raw s16le 24 kHz PCM out; models auto-downloaded to `models/kokoro/`).
- `harness/` — pytest suite + `FlowCatAdapter` (WebRTC via aiortc,
  events over a side WebSocket) + WAV fixtures.

## Setup

```sh
cd poc
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
.venv/bin/python -m harness.make_fixtures   # no-op if fixtures are committed
```

## Run

```sh
./run_poc.sh up     # stubs + kokoro shim (+ flowcat-poc if built); waits on /health
./run_poc.sh test   # pytest harness -m smoke  (T1/T2)
./run_poc.sh down
```

Tool tests: `.venv/bin/pytest harness -m tools` (from `poc/`).
`FLOWCAT_URL` / `STUBS_URL` env vars override the default 127.0.0.1 ports.
The FlowCat server's wire details (endpoint paths, JSON field names) are
pinned in `FlowCatWire` in `harness/flowcat_adapter.py` — adjust there once
the Rust side confirms them.
