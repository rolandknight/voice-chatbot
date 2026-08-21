Vendored from https://github.com/AreevAI/flowcat at merge commit
`4ff03f3ef8e179d988a20c6f46498dfb9419c1c1` (PR #61, Apache-2.0).

Local modifications for the PoC:

- `Cargo.toml` has standalone package metadata because this copy is outside
  FlowCat's Cargo workspace.
- `build_cascaded_call_duplex` accepts extra input processors between the VAD
  and speech gate. Babel uses this seam for its optional wake-word gate.
- `Cargo.lock` is retained for standalone validation of the vendored crate.
