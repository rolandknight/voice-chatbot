# poc-tts — Chatterbox Flash

Standalone Chatterbox Flash server on :8005, serving a copy of the Chatterbox
web GUI. Turbo keeps :8004, so both run side by side for A/B.

    make              # install anything missing, then serve on :8005
    make test         # GPU-free unit tests
    make bench        # tuning sweep -> reports/runs.jsonl
    make clean        # drop the venv

Run from this directory. The repo-root `make poc-tts*` targets delegate here.

Python here is mise-pinned (`mise.toml`); the rest of the repo stays on hermit.

If a `.env` file is present in this directory, the Makefile loads it (shell
`source`, not make's `include`, so `$HOME`-style shell expansions resolve
correctly) and exports its variables to every recipe. It's gitignored, so
create one locally for machine-specific settings such as `CUDA_HOME` — needed
for FlashInfer's `nvcc` probe on GPUs that support it (compute capability
>= 8.0); on this box's sm_75 card it's a no-op and setup falls back to the
torch SDPA backend.

Never install into `vendor/chatterbox-tts-server/venv` — it is pinned at
torch 2.5.1+cu121 and Flash needs >= 2.6.

Design: `docs/superpowers/specs/2026-08-23-poc-tts-flash-design.md`
