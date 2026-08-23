# poc-tts — Chatterbox Flash

Standalone Chatterbox Flash server on :8005, serving a copy of the Chatterbox
web GUI. Turbo keeps :8004, so both run side by side for A/B.

    make poc-tts-setup    # mise python 3.10, venv, deps, flashinfer probe
    make poc-tts          # server + GUI on http://127.0.0.1:8005
    make poc-tts-bench    # tuning sweep -> reports/runs.jsonl
    make poc-tts-test     # GPU-free unit tests

Python here is mise-pinned (`mise.toml`); the rest of the repo stays on hermit.

Never install into `vendor/chatterbox-tts-server/venv` — it is pinned at
torch 2.5.1+cu121 and Flash needs >= 2.6.

Design: `docs/superpowers/specs/2026-08-23-poc-tts-flash-design.md`
