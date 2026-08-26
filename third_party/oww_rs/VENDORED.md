Vendored from https://github.com/skoky/oww_rs @ v0.3.3 (MIT).
Local modifications for the PoC (docs/poc/flowcat-poc-plan.md Phase 1a):
- cpal/mic modules stripped (audio arrives via WebRTC; no ALSA on the build box)
- `OwwModel::new_from_path` added (load arbitrary openWakeWord head models —
  upstream only ships four embedded models); upstream-PR candidate
- Arc wrap fix in the new constructor
- mic-loop leftovers removed so the build is warning-free: unused imports in
  lib.rs, `Models::new`/`frame_length`, `model::new_model`/`new_oww_model`,
  `Model::frame_length` (2026-08-26)
