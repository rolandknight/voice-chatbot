# Archived: the Pipecat Python app

The original voice chatbot (Pipecat pipeline, `app.py` for the always-on
Jabra loop, `server.py` for the WebRTC backend, `skills/` handlers,
`scripts/*.py` backends, `devices/rpi5` satellite client). Superseded by the
Rust server and native client in `crates/` (see the repo README), which were
ported from this code and checked against it (`docs/plans/skills-in-server.md`).

Kept for reference; not maintained. Paths inside these files still assume the
repo root (`config.yaml`, `scripts/`, `voices/`, `models/wakeword/`), so it
does not run from here without adjustment. Its `.venv` (if present) is the
old Hermit-managed virtualenv and is stale.
