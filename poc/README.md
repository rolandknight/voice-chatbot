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

## Quickstart (this directory, Mac Studio profile)

`poc/` has its own `Makefile` in the `poc-qwen` style: the toolchain is
mise-pinned (`mise.toml`: python 3.12, rust 1.97.1) and everything is driven
from the untracked `.env` (`cp .env.example .env`). The Mac profile
used for the Nemotron test is local end to end: Ollama `gemma4:26b` through its
OpenAI-compatible `/v1` (the OpenRouter client honors `OPENROUTER_BASE_URL`,
any non-empty `OPENROUTER_API_KEY` satisfies the check), Nemotron STT on Metal,
Kokoro TTS shim.

```sh
cd poc
make                # setup (venv, models, Nemotron runtime) -> build -> ollama serve + model resident -> stack up on :6210
make client         # native mic/speaker call; INPUT_DEVICE='Jabra' OUTPUT_DEVICE='Jabra' to pick devices
                    # from another machine: POC_BIND=0.0.0.0:6210 make up, then make client FLOWCAT_URL=http://<server-lan-ip>:6210
make client-devices # list CoreAudio devices
make status         # listeners, Ollama resident model, flowcat healthz
make restart        # after `make build` or editing .env
make test POC_MARKER=smoke
make down
make help
```

`make` is idempotent: setup is stamped and `up` skips parts already running.
The LLM lifecycle is the chatbot's own (ADR-0007): at start-up `flowcat-poc`
spawns `ollama serve` if nothing answers on the base URL (`POC_OLLAMA_SUPERVISE`),
pulls the model if missing, warms the exact system-prompt + sorted-tools prefix
so the first turn hits the prompt cache, and verifies the model is pinned with
the requested context. Every turn goes through Ollama's native `/api/chat`
with `keep_alive: -1`, `num_ctx` and `think: false` on the request, so residency
no longer depends on how `serve` was started (the `/v1` endpoint resets
keep-alive to 5 min on each request — the cause of the old ~10 s first turns).
On exit (`make down` → SIGTERM) it unloads the model so its ~17 GB returns and
stops a serve it spawned. `make ollama` runs the same start-up path and exits
(`--warm-only`). `POC_OLLAMA_UNLOAD_ON_EXIT=false` keeps the model across dev
restarts.

## Remote callers (LAN)

The server binds `127.0.0.1:6210` by default. For a browser or the native
client on another machine, bind the LAN (`POC_BIND=0.0.0.0:6210`, or a specific
interface) — the media socket is already wildcard-bound. ICE uses host
candidates only (no STUN/TURN): the server advertises the interface that routes
back to each caller (loopback for same-machine peers, the LAN interface for
remote ones; `POC_ADVERTISE_IP` overrides), and the native client binds and
advertises the interface that routes to `--server-url`. Nothing on the offer
endpoint is authenticated, so bind to a trusted network only.

**Browsers need HTTPS.** `getUserMedia`/`enumerateDevices` exist only on secure
origins (`https://…` or `localhost`), so a browser on another machine opening
`http://<ip>:6210` loads the page but has no microphone API ("cannot enumerate
audio devices"). `make up-lan` adds an HTTPS listener on `:6443` with the repo's
self-signed dev cert (`make tls-cert`; SAN covers localhost and this Mac's LAN
IPs): open `https://<server-lan-ip>:6443`, accept the certificate warning once,
and the playground works (its events WebSocket switches to `wss` by itself). The
plain `:6210` listener stays for the harness and the native client. Verified
2026-08-25 with the native client pairing on the Mac's LAN address
(`local_addr=192.168.0.245`) while the loopback harness kept passing.

## TTS backends

`POC_TTS_BACKEND` selects one of three; the other two cost nothing at build or
run time.

| backend | what runs | first audio | notes |
|---|---|---|---|
| `kokoro` (default) | `stubs/kokoro_shim.py` sidecar on :8880 | after each whole sentence is synthesized (~0.4 s) | CPU ONNX; no GPU memory |
| `chatterbox` | external Chatterbox-TTS-Server on :8004 | whole sentence (1–3 s) | cloned Marvin voice; started outside `run_poc.sh` |
| `qwen` | **in-process** Qwen3-TTS via poc-qwen-streaming's PyO3 mlx-audio engine (Cargo feature `qwen-tts`) | **streamed**: ~0.2 s to the first chunk, then chunk-by-chunk | Apple Silicon; clones `voices/<POC_QWEN_VOICE>` (babel by default); 4.3 GB active / 6.4 GB peak measured |

`qwen` must be compiled in: `POC_TTS_BACKEND=qwen make build` links against
poc-qwen's venv interpreter (`make setup` creates it when the backend is qwen)
and enables the feature; a kokoro build carries no libpython. FlowCat loads and
warms the model before binding (~11 s from a warm HF cache, minutes on the first
download) and caches the `Ready.` greeting. Streaming reaches the caller through
`TtsService::run_tts_stream` in the vendored flowcat-core: the TTS processor
forwards frames as the engine yields them and drops the stream on barge-in,
which stops generation after the current chunk. Tunables: `POC_QWEN_SIZE`
(`1.7B`/`0.6B`), `POC_QWEN_INTERVAL_S` (chunk length), `POC_QWEN_CONFIG`
(engine profile, default `poc-qwen-streaming/config.flowcat.yaml`).

## Cross-platform profile

The supported validation profile uses local STT, OpenRouter Claude Haiku 4.5,
and a local Chatterbox cloned voice. STT is selected at runtime with
`POC_STT_BACKEND=whisper|moonshine|nemotron`; it does not use a cloud speech
service.
The scripts detect the host:

| Host | Whisper build | Moonshine | Nemotron sidecar | Chatterbox runtime |
|---|---|---|---|---|
| macOS (Apple Silicon) | Metal when available; otherwise CPU | CPU | Metal | CPU by default (override with `CHATTERBOX_DEVICE=mps`) |
| macOS (Intel) | CPU | CPU | CPU | CPU |
| Linux + NVIDIA | CUDA only with `nvcc`; otherwise CPU | CPU | CUDA from a prebuilt runtime; no `nvcc` needed | CUDA |
| Linux without NVIDIA | CPU | CPU | CPU | CPU |

Override Whisper acceleration deterministically with
`POC_STT_ACCELERATOR=cpu|metal|cuda make poc-build`. CUDA Whisper requires the
CUDA toolkit; CUDA Chatterbox only requires the NVIDIA driver and its CUDA
PyTorch environment. On a 6 GB GPU, start with CPU Whisper so Chatterbox has
the GPU to itself. The laptop profile runs `base.en` with eight CPU workers;
override that with `POC_WHISPER_THREADS` when testing concurrent calls.

Moonshine Medium Streaming is the low-latency alternative. It runs on CPU, so
the laptop's 6 GB GPU remains available to Chatterbox. It produces an updated
interim transcript about every 250 ms while speech is in progress. Interim
frames are display-only: FlowCat sends exactly one final transcription to
Claude Haiku after the existing Silero VAD endpoint. Selecting Moonshine does
not run Whisper in shadow mode or decode each utterance twice.

NVIDIA Nemotron Speech Streaming English 0.6B is the GPU-streaming option. A
pinned NeMo-Speech.cpp sidecar keeps the Q8 model resident and exposes only a
localhost WebSocket to Rust. FlowCat still owns VAD and turn boundaries:
partials update the playground, while the VAD-triggered commit produces the
single final transcript that can reach Haiku and tools. The default 560 ms
cache window is the accuracy/latency operating point; set
`POC_NEMOTRON_RIGHT_CONTEXT=13` for the 1120 ms maximum-context setting.
The API exposes phrase boosting, but NeMo-Speech.cpp v0.1.0 reports that the
pinned published Q8 lacks the tokenizer data needed to apply it, so the PoC
does not depend on boosting for its accuracy result.

## Setup

```sh
cp poc/.env.example poc/.env   # then fill in OPENROUTER_API_KEY
make poc-setup
make poc-doctor
make poc-build
```

Whisper is included in the normal build. To add Moonshine support, install its
pinned native runtime and Medium Streaming English model, then build the
optional Cargo feature:

```sh
./scripts/setup_moonshine.sh
POC_STT_BACKEND=moonshine make poc-build
```

The setup script installs the v0.1.3 native library under
`poc/.deps/moonshine/` and downloads the model to
`poc/models/moonshine/download.moonshine.ai/model/medium-streaming-en/quantized_26_07_30/`.
The platform build enables Cargo feature `moonshine` and supplies its native
library directory. To use a separately staged model, set
`POC_MOONSHINE_MODEL` to its absolute directory.

To add Nemotron, install the pinned v0.1.0 CUDA/Metal/CPU runtime and verified
English Q8 model. The Rust binary itself needs no NVIDIA-specific Cargo
feature:

```sh
./scripts/setup_nemotron.sh
# in poc/.env: POC_STT_BACKEND=nemotron
make poc-build
```

`make poc-up` starts the localhost sidecar on port 8178 when needed and waits
for `/ready` before starting FlowCat. Set `POC_NEMOTRON_DEVICE=cpu` to force a
CPU comparison, or `cuda:0`/`metal` to require that accelerator.

Select the backend in the untracked configuration file. The other two remain
available as one-line rollback choices:

```sh
# poc/.env
POC_STT_BACKEND=nemotron  # or moonshine / whisper
# POC_MOONSHINE_MODEL=/absolute/path/to/another/medium-streaming-model
```

Once the binary has been built with the Moonshine feature, switching among
`whisper`, `moonshine`, and `nemotron` is a runtime configuration change; it
does not require another build. Restart `make poc-up` after changing the value.

Build prerequisites:

- macOS: `brew install cmake pkg-config opus ffmpeg`
- Debian/Ubuntu/Pop!_OS: `sudo apt install build-essential cmake pkg-config libopus-dev libsndfile1 ffmpeg`

This checkout may also contain a PoC-local Opus build under `.deps/`; the
platform script uses it automatically before asking for a system package.

For a fresh Chatterbox install, upstream requires Python 3.10. If it is not
already installed, `uv python install 3.10` is the simplest portable setup.
The launcher reuses both `venv/` and legacy `.venv/` environments and both
capitalizations of the upstream checkout, so it will not duplicate an
existing multi-gigabyte installation.

## Run

Start Chatterbox in its own terminal so repeated test groups reuse the loaded
model:

```sh
make poc-chatterbox
```

The launcher stages the tracked Marvin reference clip as a mono 24 kHz
`marvin.wav` when it is missing. It never overwrites an existing reference.

For a live browser call, start the Rust stack in a second terminal and open
<http://127.0.0.1:6210>:

```sh
make poc-up
```

For a terminal-only native Rust call, no browser is needed. List the CoreAudio
(macOS) or ALSA (Linux) devices, then connect the selected microphone and
speaker directly to the same WebRTC endpoint:

```sh
make flowcat-client-devices
make flowcat-client-run INPUT_DEVICE='Jabra' OUTPUT_DEVICE='Jabra'
```

Omit either selector to use the operating-system default. See
[`flowcat-client/README.md`](flowcat-client/README.md) for stable device-ID
selection, platform build packages, and the hardware echo-cancellation note.

Use **Test microphone** before starting the call. After permission is granted,
the page lists the browser's available input devices and meters the exact track
that will be sent over WebRTC. Select the internal mic or Jabra explicitly and
confirm the meter moves; a moving meter with no transcript isolates the problem
to WebRTC/VAD/STT rather than browser capture. The selector can also replace the
microphone during a call without changing the operating-system default.

Then run the fast gates before the long matrix:

```sh
make poc-test POC_MARKER=smoke
make poc-test POC_MARKER=tools
make poc-test POC_MARKER=duplex
make poc-test POC_MARKER=voice
make poc-test-all
make poc-results
```

`run_poc.sh` verifies a real Chatterbox synthesis before reporting the stack
ready and reuses that WAV for the fixed `Ready.` connect greeting. The greeting
therefore does not call OpenRouter or regenerate speech on each reconnect.
Chatterbox remains an external sidecar and `poc-down` does not kill it.

A cold Chatterbox launch can take tens of seconds while Python/PyTorch imports
and the GPU model loads. Keep the sidecar running between FlowCat rebuilds and
test sessions; reconnects reuse the resident model. The browser playground uses
loopback ICE only and bounds candidate gathering, so an unavailable public STUN
server cannot leave a local call stuck at `negotiating`.

Speech turns use `POC_VAD_STOP_SECS=0.2`, matching the production Python
chatbot's `wake.vad_stop_secs`. Silero evaluates 32 ms windows, so the observed
endpoint is about 192 ms. This is configurable for A/B testing, but increasing
it directly delays the final STT result; reducing it can split natural
mid-sentence pauses. With Moonshine, partial text can appear before this edge,
but only the final result after the edge can start Haiku or invoke a tool.

For the wake test, run
`POC_WAKE_MODEL="$PWD/models/wakeword/hey_babel.onnx" make poc-test POC_MARKER=wake`
separately because ordinary fixtures do not contain a wake phrase.

`FLOWCAT_URL` / `STUBS_URL` env vars override the default 127.0.0.1 ports.
The FlowCat server's wire details (endpoint paths, JSON field names) are
pinned in `FlowCatWire` in `harness/flowcat_adapter.py` — adjust there once
the Rust side confirms them.
