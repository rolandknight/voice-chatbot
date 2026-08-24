# poc-tts-streaming on an RTX 2060 — time-to-first-audio

Measured 2026-08-24 on `pop-os`, NVIDIA GeForce RTX 2060 (6 GB, compute
capability sm_75), driver 580.159.03, torch 2.6.0+cu124. Generation config
is `config.yaml`'s defaults (`num_steps: 4, n_cfm_timesteps: 1,
chunk_size: 120, split_text: true, split_on_clauses: true`), resolved
dtype `float16`, backend `torch` (SDPA — flashinfer is unavailable on this
card; see `poc-tts/bench-rtx-2060.md`). Voice: `one-one.mp3`.

Raw data: [`reports/stream_runs.jsonl`](reports/stream_runs.jsonl).
Reproduce the engine column with `make bench-stream`.

| sentence | chars | chunks | TTFA engine (s) | TTFA browser (s) | total gen (s) | audio (s) | poc-tts whole-utterance (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 30 | 1 | 0.524 | 1.136 | 0.52 | 2.32 | 0.59 |
| medium | 104 | 1 | 0.847 | 1.455 | 0.85 | 5.16 | 1.03 |
| long | 317 | 4 | 0.843 | 1.424 | 3.51 | 18.56 | 3.38 |

- **TTFA engine** — `t0` (call into `synthesize_stream`) to the first
  `(chunk_text, pcm)` yielded, from `reports/stream_runs.jsonl` (best-of-2,
  measured by this task's `make bench-stream` run). It excludes WebRTC/HTTP
  transport, encode, and the browser's jitter buffer — it is the engine's own
  floor.
- **TTFA browser** = `response.created` sent → first non-silent sample at the
  AnalyserNode, via headless Chrome driving the real UI at config-default
  knobs and the same voice. Includes Opus encode, jitter buffer, and decode.
  Measured by the controller, not by this bench script.
- **total gen (s)** / **audio (s)** are the engine bench's totals across all
  chunks of that sentence (`gen_s`, `audio_s` in the JSONL row).
- **poc-tts whole-utterance (s)** — `poc-tts/bench-rtx-2060.md`'s tuned row
  (`drf_block_size=32, num_steps=4, n_cfm_timesteps=1`); poc-tts has no
  streaming path, so this is the time until *any* audio exists at all —
  the number poc-tts-streaming's TTFA is meant to beat.

Engine TTFA already beats the poc-tts whole-utterance baseline on short and
medium (0.52 s vs 0.59 s; 0.85 s vs 1.03 s) and by a wide margin on long
(0.84 s vs 3.38 s), because only the first chunk has to finish before audio
starts, not the whole paragraph. Browser TTFA is roughly 0.6 s higher across
the board — that gap is WebRTC/session overhead (client-secret round trip,
SDP offer/answer, ICE, Opus encode, jitter buffer), not generation.

Long sentence: 4 chunks, matching the engine bench's own 4-chunk split.

## Server TTFA (response.created → output_audio_buffer.started)

A second controller measurement, taken at the server side of the WebRTC
session rather than the browser's AnalyserNode — narrower than browser TTFA
(no Opus decode / jitter buffer) but still inside the full session, unlike
the engine column above:

| sentence | server TTFA (s) |
|---|---:|
| short | 0.772 |
| medium | 1.228 |
| long | 1.187 |

Long, at the server: 4 chunks, total gen 5.153 s, audio 28.711 s (a
different run from the engine-bench row above — Flash's output length is
not deterministic between runs; see `poc-tts/bench-rtx-2060.md`'s
"Read gen_s, not RTF" section). Gaps between chunks: none observed — the
level meter stayed continuous, and total generation time is far below
total audio duration on every sentence, so the synthesis worker always
stays ahead of playback.

## HTTP chunked-PCM reference points (`/v1/audio/speech`, warm)

For a 319-char Dickens paragraph, at `chunk_size: 120` (config default):
TTFA 1.24 s. Reducing `chunk_size` to 50 (a smaller first chunk) drops that
to 0.58 s. Whole generation ranged 3.3–6.0 s for 21–24 s of audio across
these runs — consistent with the non-deterministic output length noted
above.

## Cold start

The very first generation after model load took about 14.6 s (CUDA
warm-up / kernel compilation). Every number above is warm — `make
bench-stream` runs an explicit warm-up request before measuring, and the
browser/server numbers were taken against an already-warm server.

## Gaps between chunks

Not observed, by ear or by the level meter, on any sentence: the worker's
per-chunk generation time is consistently far shorter than that chunk's
audio duration (e.g. long: 0.80–1.02 s to generate each ~4–5 s chunk), so
the next chunk is always ready well before the previous one finishes
playing.
