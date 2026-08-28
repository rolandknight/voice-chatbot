//! `ask_claude` — route the rest of this call's turns to Claude instead of
//! the local model (port of skills/backends/ask_claude). The flag lives in
//! the call's [`CallState`](super::CallState) and the LLM stage checks it
//! every turn; it resets when the call ends.

use async_trait::async_trait;
use serde_json::Value;

use super::{CallCtx, LlmBackend, Skill};

/// The tool result, which by the time it is read is a prompt *to Claude* — the
/// backend has already flipped, so the continuation of this very turn runs on
/// Claude. The Python original returned a bare "Asking Claude. Go ahead."
/// confirmation, written for a world where the local model answered the
/// continuation; fed to Claude it just makes it re-announce the handover and
/// never answer, costing the caller a turn (and, if the wake session lapses
/// first, the backend flip too). The brevity clause keeps the reply inside one
/// spoken turn — unprompted, Claude writes paragraphs that take a minute to
/// speak.
pub(crate) const HANDOVER: &str =
    "You are now answering as Claude. Answer the user's request above yourself, \
directly — do not say you are handing over or ask them to repeat it. Keep it to one or two short \
spoken sentences; offer to go deeper if they want more.";

pub struct AskClaude;

#[async_trait]
impl Skill for AskClaude {
    fn name(&self) -> &str {
        "ask_claude"
    }

    async fn call(&self, _args: &Value, ctx: &CallCtx) -> String {
        match &ctx.state {
            Some(state) => {
                state.set_backend(LlmBackend::Claude);
                HANDOVER.to_string()
            }
            None => "Claude isn't available right now.".to_string(),
        }
    }
}
