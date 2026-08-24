# poc-qwen — Qwen3-TTS voice cloning on Apple Silicon (mlx-audio)

Gradio app on :8007 mirroring the three tabs of
https://huggingface.co/spaces/Qwen/Qwen3-TTS — **Voice Design**, **Voice
Clone** (zero-shot from a reference clip + transcript), **TTS (CustomVoice)**
with preset speakers — running Qwen3-TTS on the M4 Max GPU via mlx-audio.

    make              # install anything missing, then serve on http://127.0.0.1:8007
    make smoke        # go/no-go: clone voices/one-one.mp3 with 0.6B -> reports/smoke.wav
    make bench        # latency/RTF sweep -> reports/runs.jsonl (see bench-m4-max.md)
    make models       # pre-download all selectable models (~14 GB)
    make bench-stream # stream=True TTFA spike -> reports/stream_spike.jsonl
    make test         # GPU-free unit tests (engine mocked)
    make clean        # drop the venv

Run from this directory. The repo-root `make poc-qwen*` targets delegate here.
Python is mise-pinned (`mise.toml`, 3.12); the rest of the repo stays on hermit.
`HOST=0.0.0.0 make` exposes the app on the LAN.

## Layout

- `poc_qwen/engine.py` — `Qwen3Engine`: lazy model registry (LRU of
  `models.max_resident`), warm-up incl. the ICL clone path, sentence-chunked
  generation with crossfaded seams, `stream_clone()` generator for iteration 2.
  Owns every `mlx_audio` import.
- `poc_qwen/app.py` — Gradio Blocks; never imports `mlx_audio`. Presets come
  from `../voices/*.{wav,mp3}`; a `<name>.txt` sidecar supplies the transcript,
  otherwise `Auto-transcribe` runs mlx-whisper. Each generation appends to
  `reports/ui_runs.jsonl`.
- `poc_qwen/text.py` — sentence split + chunk grouping (`max_chunk_chars`),
  keeps every Metal call short of the watchdog.
- `poc_qwen/bench.py`, `poc_qwen/spike_stream.py` — see `bench-m4-max.md`.
- `config.yaml` — models, sampling, voice dirs; override any scalar with
  `POC_QWEN_<SECTION>_<KEY>=...` (a gitignored `.env` is sourced by make).

## Models (HF cache, downloaded on first use)

| tab | model |
| --- | --- |
| Voice Clone | `mlx-community/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base-bf16` |
| TTS (CustomVoice) | `mlx-community/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice-bf16` |
| Voice Design | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` |

## Results (2026-08-24)

Warm whole-utterance: 1.7B 1.04 / 2.21 / 6.49 s for the short / medium / long
bench sentences (RTF ≈ 0.37); 0.6B 1.04 / 1.82 / 5.08 s (RTF ≈ 0.3). Streaming
time-to-first-chunk 0.18 s (1.7B) / 0.12 s (0.6B). Details and caveats in
[`bench-m4-max.md`](bench-m4-max.md).

Pinned versions that produced them: mlx 0.32.1, mlx-audio 0.5.0, gradio 5.50.0,
mlx-whisper 0.4.3.

Plan: `docs/superpowers/plans/2026-08-24-poc-qwen3-tts.md`
