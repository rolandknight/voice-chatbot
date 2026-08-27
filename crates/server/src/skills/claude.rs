//! `ask_claude` — route the rest of this call's turns to Claude instead of
//! the local model (port of skills/backends/ask_claude). The flag lives in
//! the call's [`CallState`](super::CallState) and the LLM stage checks it
//! every turn; it resets when the call ends.

use async_trait::async_trait;
use serde_json::Value;

use super::{CallCtx, LlmBackend, Skill};

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
                "Asking Claude. Go ahead.".to_string()
            }
            None => "Claude isn't available right now.".to_string(),
        }
    }
}
