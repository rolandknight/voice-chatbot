Vendored from https://github.com/skoky/oww_rs @ v0.3.3 (MIT).
Local modifications for the PoC (docs/poc/flowcat-poc-plan.md Phase 1a):
- cpal/mic modules stripped (audio arrives via WebRTC; no ALSA on the build box)
- `OwwModel::new_from_path` added (load arbitrary openWakeWord head models —
  upstream only ships four embedded models); upstream-PR candidate
- Arc wrap fix in the new constructor
- mic-loop leftovers removed so the build is warning-free: unused imports in
  lib.rs, `Models::new`/`frame_length`, `model::new_model`/`new_oww_model`,
  `Model::frame_length` (2026-08-26)
- `OwwModel::head_from_path` (frontend-less head; `audio` is now an `Option`) so
  the server's multi-wake-word bank shares one melspectrogram/embedding
  frontend across N heads via `OwwModel::detect(features)` (2026-08-26)
- `poc_probe`'s two test-only env vars renamed to `OWW_PROBE_MODEL` /
  `OWW_PROBE_WAV`. The repo-wide retirement of the old PoC-era variable prefix
  had swept them to bare `MODEL`/`WAV`, which are far too collision-prone to
  read out of a developer's environment; a distinctive prefix keeps the probe
  opt-in without reviving the retired one (2026-08-30)
