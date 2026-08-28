# poc-qwen-streaming — Rust server embedding poc-qwen's mlx-audio engine (PyO3),
# streaming Qwen3-TTS to the browser over WebSocket. Bare `make` builds and
# serves on :8008. Run from this directory.

.DEFAULT_GOAL := run

# The engine's venv lives in poc-qwen (mise Python 3.12); PyO3 links against
# that interpreter, and config.yaml's python.paths puts its site-packages on
# sys.path at run time. cargo comes from the repo's hermit bin/.
QWEN := ../poc-qwen
PY := $(abspath $(QWEN)/.venv/bin/python)
export PYO3_PYTHON := $(PY)
CARGO := ../bin/cargo
BIN := target/release/poc-qwen-streaming

DOTENV := $(wildcard .env)
LOAD_ENV = $(if $(DOTENV),set -a; . ./.env; set +a;)

.PHONY: run setup build info bench test test-py test-rs clean help explore

run: build  ## Build if needed, then serve the UI on http://127.0.0.1:8008
	@$(LOAD_ENV) ./$(BIN) serve

explore: build  ## Serve :8008 with config.explore.yaml (CustomVoice preloaded, LRU 2) for voice comparisons
	@$(LOAD_ENV) ./$(BIN) --config config.explore.yaml serve

setup:  ## poc-qwen venv (mise python 3.12 + mlx-audio); idempotent
	@$(MAKE) -C $(QWEN) setup

build: setup  ## Release build against poc-qwen's Python
	@$(CARGO) build --release

info: build  ## Print model_info() from the embedded engine and exit
	@$(LOAD_ENV) ./$(BIN) info

bench: build  ## Headless TTFA bench, three sentences -> reports/rs_runs.jsonl + reports/bench_*.wav
	@$(LOAD_ENV) ./$(BIN) bench

test: test-py test-rs  ## GPU-free tests (bridge with a fake model; Rust unit tests)

test-py: setup
	@$(PY) -m pytest tests -q

test-rs:
	@$(CARGO) test --release

clean:  ## Drop build output and uploads. Leaves reports/ and poc-qwen's venv.
	rm -rf target uploads .pytest_cache poc_qwen_streaming/__pycache__ tests/__pycache__

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
