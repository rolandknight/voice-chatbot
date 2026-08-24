# Qwen3-TTS on Apple M4 Max (mlx-audio) — voice-cloning bench

Measured 2026-08-24 on a Mac Studio, Apple M4 Max (14 cores), 36 GB unified
memory, macOS 26.4, Python 3.12.14, mlx 0.32.1, mlx-audio 0.5.0. Reference
clip `voices/one-one.mp3` (12 s) with its `one-one.txt` transcript, ICL
cloning, `language=English`, temperature 0.9 / top_p 0.9, whole-utterance
generation. Same three sentences as `poc-tts/poc_tts/bench.py`. Median of the
two warm repeats; the first repeat per model is recorded `cold: true` in
[`reports/runs.jsonl`](reports/runs.jsonl) and excluded.

## Whole-utterance latency vs Chatterbox Flash (same machine, same sentences)

| sentence | Qwen3-TTS 0.6B | Qwen3-TTS 1.7B | Chatterbox Flash MLX fp16 (`poc-tts/bench-m4-max.md`) |
| --- | --- | --- | --- |
| short (30 chars) | **1.04 s** / RTF 0.35 | 1.04 s / RTF 0.49 | 0.92 s |
| medium (104 chars) | **1.82 s** / RTF 0.29 | 2.21 s / RTF 0.38 | 1.37 s |
| long (317 chars) | **5.08 s** / RTF 0.28 | 6.49 s / RTF 0.36 | 4.20 s |

- Qwen3-TTS generates ~15–25 % more audio per sentence than Chatterbox (slower
  delivery, e.g. medium = 6.1 s vs ~5 s), which is most of the wall-clock gap;
  per second of audio the 0.6B model is the fastest engine measured on this box
  (RTF 0.28–0.35).
- Peak unified memory: 0.6B 8.0 GiB, 1.7B 9.9 GiB; with 1.7B-Base and
  1.7B-CustomVoice both resident (the app's LRU of 2) 10.4 GiB peak, 8.6 GiB
  active. Plenty of room for an LLM alongside.
- Load: 1.7B 4.0 s from the HF cache + 5.0 s warm-up (the warm-up now
  exercises the ICL path; before that the first real clone cost 5.9 s for the
  short sentence instead of ~1 s). First-ever download: 0.6B ≈1.5 GB in ~3 min,
  1.7B ≈3.5 GB.
- Plan's success bar was "1.7B medium ≤ 1.5 s warm, RTF ≤ 0.6". RTF is met;
  the absolute latency is not (2.21 s), and neither is it with 0.6B (1.82 s)
  — the model simply speaks more slowly. Whole-utterance is not the
  interesting number here anyway; see streaming below.

## Streaming spike (`make spike`, `stream=True`)

Medium sentence, 1.7B and 0.6B, mlx-audio's chunked decode
(`streaming_interval` seconds per chunk), warm run. Raw rows in
[`reports/stream_spike.jsonl`](reports/stream_spike.jsonl).

| model | interval | time-to-first-chunk | chunk cadence (median) | total | audio |
| --- | --- | --- | --- | --- | --- |
| 0.6B | 0.32 s | **0.12 s** | 83 ms per 320 ms chunk | 1.75 s | 6.5 s |
| 0.6B | 0.64 s | 0.20 s | 160 ms per 640 ms chunk | 1.56 s | 6.0 s |
| 1.7B | 0.32 s | **0.18 s** | 106 ms per 320 ms chunk | 1.96 s | 5.6 s |
| 1.7B | 0.64 s | 0.28 s | 205 ms per 640 ms chunk | 1.70 s | 5.0 s |

- That is time to the first *decoded audio chunk* inside the process, with
  the reference already encoded (mlx-audio's ICL cache). Cold first call
  (kernel compilation) was 2.6–4.4 s; the engine's warm-up absorbs that.
- Generation runs at ~3–4× real time, so playback never starves once the
  first chunk is out.
- Streamed output re-transcribes with whisper identically to the
  whole-utterance output. Sample-to-sample jumps at chunk seams are within the
  in-speech range (max |Δ| 0.11, p99.9 0.06); mlx-audio's decoder keeps
  streaming state across chunks, so no client-side crossfade was needed in the
  spike. Listen to `reports/stream_1.7B_0.32.wav` before trusting that.

**Go for iteration 2.** Expected browser TTFA ≈ 0.2 s model + transport, an
order of magnitude under the 0.9 s target in the research bar, using the
existing `Qwen3Engine.stream_clone()` generator — it already yields float32
24 kHz chunks and is the seam for `poc-tts-streaming`'s Realtime session
(`poc_tts_streaming/realtime/session.py` audio push).

## Clone quality notes (not measured — read `reports/*.wav`)

- `one-one` clone output re-transcribes verbatim with whisper for all bench
  sentences. In two of ~15 runs (0.6B, and the 0.6B streaming run) the output
  began with "Here you go" — the trailing phrase of the reference transcript
  leaking into the target. Likely the `one-one.txt` tail ("Ah, ah, here you
  go") is inaccurate; fix the transcript, or trim the clip, rather than the
  model.
- x-vector-only cloning (no transcript) works and is faster (0.73 s vs
  1.08 s for the short sentence on 0.6B) but is expected to be a weaker match;
  it is exposed in the UI as "Use x-vector only".
- `voices/babel.mp3` and `voices/marvin.mp3` have sidecar transcripts; no
  A/B against Chatterbox was done by ear in this iteration.

## Reproduce

    make bench          # both sizes, 3 repeats  -> reports/runs.jsonl
    make spike          # stream=True            -> reports/stream_spike.jsonl
    make                # Gradio app on :8007
