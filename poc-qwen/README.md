# poc-qwen — Qwen3-TTS voice cloning on Apple Silicon (mlx-audio)

Gradio app on :8007 mirroring the three tabs of
https://huggingface.co/spaces/Qwen/Qwen3-TTS (Voice Design / Voice Clone /
TTS with preset speakers), running Qwen3-TTS on the M4 Max GPU via mlx-audio.

    make              # install anything missing, then serve on :8007
    make smoke        # go/no-go: clone voices/one-one.mp3 -> reports/smoke.wav
    make bench        # latency/RTF sweep -> reports/runs.jsonl
    make test         # GPU-free unit tests
    make clean        # drop the venv

Run from this directory. The repo-root `make poc-qwen*` targets delegate here.
Python is mise-pinned (`mise.toml`, 3.12); the rest of the repo stays on hermit.

Plan: `docs/superpowers/plans/2026-08-24-poc-qwen3-tts.md`
