//! `ask_claude` — route the rest of this call's turns to Claude instead of
//! the local model (port of skills/backends/ask_claude). The flag lives in
//! the call's [`CallState`](super::CallState) and the LLM stage checks it
//! every turn; it resets when the call ends.

use async_trait::async_trait;
use serde_json::Value;

use super::{CallCtx, LlmBackend, Skill};

/// What the tool returns. Since the handover instruction moved to
/// `call::CLAUDE_SYSTEM_SUFFIX`, this string is stripped from the rolling
/// context before either backend's next turn (`call::strip_ask_claude`) and is
/// never read by a model. It stays non-empty because the tool contract requires
/// a spoken-friendly string, and it shows up in the `tool return` log line.
const HANDOVER: &str = "Switched to Claude.";

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
