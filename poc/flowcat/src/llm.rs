//! PoC-specific LLM policy.
//!
//! The cascaded pipeline asks the LLM for an opening turn on every connection,
//! even though Babel's prompt requires that turn to be the literal `Ready.`.
//! Keep normal user turns on OpenRouter, but make the deterministic greeting a
//! local frame sequence so reconnect does not depend on a network round trip.

use async_trait::async_trait;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};
use flowcat_core::Result;
use futures::stream::{self, BoxStream};
use futures::StreamExt;

pub struct StaticGreetingLlm<L> {
    inner: L,
    greeting: String,
    first_run: bool,
}

impl<L> StaticGreetingLlm<L> {
    pub fn new(inner: L, greeting: impl Into<String>) -> Self {
        Self {
            inner,
            greeting: greeting.into(),
            first_run: true,
        }
    }

    fn is_initial_context(ctx: &LlmContext) -> bool {
        !ctx.messages.iter().any(|message| {
            matches!(
                message.get("role").and_then(serde_json::Value::as_str),
                Some("user" | "assistant" | "tool")
            )
        })
    }
}

#[async_trait]
impl<L: LlmService> LlmService for StaticGreetingLlm<L> {
    fn name(&self) -> &str {
        "static-greeting-llm"
    }

    async fn start(&mut self, params: &StartParams) -> Result<()> {
        self.inner.start(params).await
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let use_local_greeting = self.first_run && Self::is_initial_context(ctx);
        self.first_run = false;

        if use_local_greeting {
            tracing::info!("serving deterministic greeting locally");
            return Ok(stream::iter([
                Frame::LlmResponseStart,
                Frame::LlmText(self.greeting.clone()),
                Frame::LlmResponseEnd,
            ])
            .boxed());
        }

        self.inner.run_llm(ctx).await
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.inner.set_tools(tools);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    struct CountingLlm {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LlmService for CountingLlm {
        fn name(&self) -> &str {
            "counting"
        }

        async fn start(&mut self, _params: &StartParams) -> Result<()> {
            Ok(())
        }

        async fn run_llm<'a>(&'a mut self, _ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(stream::iter([
                Frame::LlmResponseStart,
                Frame::LlmText("remote".to_string()),
                Frame::LlmResponseEnd,
            ])
            .boxed())
        }

        fn set_tools(&mut self, _tools: Vec<Tool>) {}
    }

    fn text(frames: &[Frame]) -> Vec<&str> {
        frames
            .iter()
            .filter_map(|frame| match frame {
                Frame::LlmText(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn initial_system_only_turn_is_local_then_user_turn_delegates() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut llm = StaticGreetingLlm::new(
            CountingLlm {
                calls: calls.clone(),
            },
            "Ready.",
        );
        let initial = LlmContext {
            messages: vec![serde_json::json!({"role": "system", "content": "prompt"})],
            tools: vec![],
        };

        let greeting: Vec<_> = llm
            .run_llm(&initial)
            .await
            .expect("local greeting")
            .collect()
            .await;
        assert_eq!(text(&greeting), vec!["Ready."]);
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let user = LlmContext {
            messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
            tools: vec![],
        };
        let response: Vec<_> = llm
            .run_llm(&user)
            .await
            .expect("remote user turn")
            .collect()
            .await;
        assert_eq!(text(&response), vec!["remote"]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn first_user_turn_never_gets_replaced_by_the_greeting() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut llm = StaticGreetingLlm::new(
            CountingLlm {
                calls: calls.clone(),
            },
            "Ready.",
        );
        let user = LlmContext {
            messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
            tools: vec![],
        };

        let response: Vec<_> = llm
            .run_llm(&user)
            .await
            .expect("remote user turn")
            .collect()
            .await;
        assert_eq!(text(&response), vec!["remote"]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
