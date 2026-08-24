# poc-tts-streaming — Chatterbox Flash over the OpenAI Realtime API (WebRTC)

Copy of `poc-tts/` that streams audio sentence-by-sentence over WebRTC on
:8006, speaking the OpenAI Realtime API (`POST /v1/realtime/calls`,
`oai-events` data channel). poc-tts keeps :8005 so both run side by side.

    make              # install anything missing, then serve on :8006
    make test         # GPU-free unit + loopback WebRTC tests
    make bench-stream # TTFA per baseline sentence -> reports/stream_runs.jsonl
    make clean

Design: `docs/superpowers/specs/2026-08-23-poc-tts-streaming-design.md`
