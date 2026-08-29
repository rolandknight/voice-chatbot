Vendored from https://github.com/AreevAI/flowcat at merge commit
`4ff03f3ef8e179d988a20c6f46498dfb9419c1c1` (PR #61, Apache-2.0).

Local modifications for the PoC:

- `Cargo.toml` has standalone package metadata because this copy is outside
  FlowCat's Cargo workspace.
- `build_cascaded_call_duplex` accepts extra input processors between the VAD
  and speech gate. Babel uses this seam for its optional wake-word gate.
- The shared media-transport facade uses a single-owner command actor so a
  pending inbound receive cannot block bot-first audio or playback clears.
- `Cargo.lock` is retained for standalone validation of the vendored crate.
- `RollingContext` carries a user-turn generation; `CascadedToolBridge` skips
  the LLM re-run for a tool result whose call belongs to a superseded turn
  (a wake phrase and its command landing as two finals answered twice).
- `SpeechGate::on_interruption` no longer clears the pre-roll ring. The VAD
  broadcasts `Interruption` before the new turn's `UserStartedSpeaking`, so
  clearing there emptied the ring the rising edge was about to replay and the
  STT lost the first word of every utterance spoken over the bot's reply.
