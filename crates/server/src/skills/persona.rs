//! `switch_persona` — change the voice for the rest of the call (port of
//! skills/persona/switch_persona). Personas are the Qwen preset voices the
//! server loaded (`QWEN_VOICES`); the TTS stage reads the call's chosen
//! voice on every utterance.

use async_trait::async_trait;
use serde_json::Value;

use super::{arg_str, CallCtx, Skill};

pub struct SwitchPersona {
    personas: Vec<String>,
}

impl SwitchPersona {
    pub fn new(personas: Vec<String>) -> Self {
        Self { personas }
    }
}

#[async_trait]
impl Skill for SwitchPersona {
    fn name(&self) -> &str {
        "switch_persona"
    }

    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        let requested = arg_str(args, "persona").to_lowercase();
        if requested.is_empty() {
            return "Tell me which persona to switch to.".to_string();
        }
        // Accept "one_one" for "one-one" and vice versa: the LLM sees ids.
        let matches = |p: &String| {
            p.eq_ignore_ascii_case(&requested) || p.replace('-', "_") == requested.replace('-', "_")
        };
        let Some(persona) = self.personas.iter().find(|p| matches(p)) else {
            let mut options = self.personas.clone();
            options.sort();
            return format!(
                "I don't have a persona called {requested}. I can be {}.",
                options.join(", ")
            );
        };
        match &ctx.state {
            Some(state) => {
                state.set_voice(persona);
                format!("Switched to {persona}.")
            }
            None => "I can't switch voices on this connection.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::CallState;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn switches_known_personas_and_lists_options() {
        let s = SwitchPersona::new(vec!["babel".into(), "marvin".into(), "one-one".into()]);
        let state = Arc::new(CallState::default());
        let ctx = CallCtx {
            state: Some(state.clone()),
            ..CallCtx::detached(1)
        };
        assert_eq!(
            s.call(&json!({"persona": "Marvin"}), &ctx).await,
            "Switched to marvin."
        );
        assert_eq!(state.voice().as_deref(), Some("marvin"));
        assert_eq!(
            s.call(&json!({"persona": "one_one"}), &ctx).await,
            "Switched to one-one."
        );
        assert_eq!(
            s.call(&json!({"persona": "jeeves"}), &ctx).await,
            "I don't have a persona called jeeves. I can be babel, marvin, one-one."
        );
        assert_eq!(
            s.call(&json!({}), &ctx).await,
            "Tell me which persona to switch to."
        );
    }
}
