//! Claude over the Messages API (`POST /v1/messages`, streaming) as a FlowCat
//! [`LlmService`] — the `ask_claude` backend. Raw HTTP: there is no official
//! Rust SDK, and the call shape is small (one streamed request per turn).
//!
//! The rolling context is OpenAI-shaped (`assistant.tool_calls`, `tool`
//! messages); it is translated to Messages-API content blocks here and the
//! stream is folded back into the same frames the Ollama adapter emits
//! (`LlmText`, `FunctionCallsStarted`, usage metrics).

use std::collections::VecDeque;
use std::time::Instant;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

use flowcat_core::processor::frame::{Frame, FunctionCall, LlmContext, StartParams};
use flowcat_core::processor::metrics::{LlmTokenUsage, MetricsData};
use flowcat_core::service::{LlmService, Tool};
use flowcat_core::{FlowcatError, Result};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
/// Ceiling for one streamed turn, shared by adaptive thinking and the spoken
/// reply. Opus 5 thinks by default (an omitted `thinking` field means adaptive),
/// and at the old 1024 a hard question could spend the entire budget thinking
/// and return `stop_reason: max_tokens` with no text at all — a silent turn.
/// Reply length is governed by the system prompt, not by this ceiling.
const MAX_TOKENS: u32 = 4096;

/// Default `output_config.effort`. Spoken one-or-two-sentence answers do not
/// need the API default (`high`); `low` measured ~0.7 s to first token against
/// ~1.0 s. `thinking.budget_tokens` is rejected with a 400 on Opus 5, so effort
/// is the only depth knob. Empty (`POC_CLAUDE_EFFORT=`) omits the field, for
/// models that reject it.
pub const DEFAULT_EFFORT: &str = "low";

/// Spoken when a turn ends with neither text nor a tool call, so the call never
/// just goes quiet on the caller.
const EMPTY_TURN_FALLBACK: &str = "Sorry, I lost that one. Say it again?";

/// Skills that must not reach Claude.
///
/// `web_search` because Anthropic's server-side tool carries the **same name**
/// — two tools called `web_search` in one request is a collision, and the
/// server-side one is the whole point of routing to Claude. `ask_claude`
/// because Claude is already answering: calling it re-sets a flag that is
/// already set and costs the caller a turn.
const HIDDEN_FROM_CLAUDE: [&str; 2] = ["web_search", "ask_claude"];

pub struct ClaudeLlm {
    http: reqwest::Client,
    api_key: String,
    model: String,
    /// `output_config.effort`; empty omits the field.
    effort: String,
    tools: Vec<Tool>,
}

impl ClaudeLlm {
    pub fn new(api_key: String, model: String, effort: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            model,
            effort,
            tools: Vec::new(),
        }
    }

    fn tools_json(&self, ctx: &LlmContext) -> Vec<Value> {
        let from_ctx: Vec<Tool> = ctx
            .tools
            .iter()
            .filter_map(|v| serde_json::from_value::<Tool>(v.clone()).ok())
            .collect();
        let tools = if from_ctx.is_empty() {
            &self.tools
        } else {
            &from_ctx
        };
        let mut out: Vec<Value> = tools
            .iter()
            .filter(|t| !HIDDEN_FROM_CLAUDE.contains(&t.name.as_str()))
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": if t.params.is_null() { json!({"type": "object", "properties": {}}) } else { t.params.clone() },
                })
            })
            .collect();
        out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        out
    }

    /// The request body (pure — the wire-fixture seam).
    pub fn request_body(&self, ctx: &LlmContext) -> Result<Value> {
        let (system, messages) = translate_messages(&ctx.messages)?;
        let mut body = json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "stream": true,
            "messages": messages,
        });
        if !system.is_empty() {
            body["system"] = json!(system);
        }
        // `thinking` is deliberately left unset: on Opus 5 that means adaptive.
        // `{"type": "disabled"}` is not an option here — with thinking off the
        // model sometimes writes a tool call into its visible text instead of a
        // `tool_use` block, which on a voice call means Babel reads the tool
        // name aloud instead of running it. Depth is tuned with effort instead.
        if !self.effort.is_empty() {
            body["output_config"] = json!({ "effort": self.effort });
        }
        let tools = self.tools_json(ctx);
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        Ok(body)
    }
}

fn content_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
    }
}

/// OpenAI-shaped rolling context → (system text, Messages-API messages).
/// Consecutive `tool` results merge into one user turn (parallel calls).
pub fn translate_messages(messages: &[Value]) -> Result<(String, Vec<Value>)> {
    let mut system = String::new();
    let mut out: Vec<Value> = Vec::new();
    for m in messages {
        match m["role"].as_str().unwrap_or("") {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&content_string(&m["content"]));
            }
            "user" => {
                let text = content_string(&m["content"]);
                if !text.trim().is_empty() {
                    out.push(json!({"role": "user", "content": text}));
                }
            }
            "assistant" => {
                let mut blocks = Vec::new();
                let text = content_string(&m["content"]);
                if !text.trim().is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for c in m["tool_calls"].as_array().into_iter().flatten() {
                    let name = c["function"]["name"].as_str().unwrap_or("");
                    if name.is_empty() {
                        return Err(FlowcatError::Other(
                            "assistant tool_call without a function name".into(),
                        ));
                    }
                    let input = match &c["function"]["arguments"] {
                        Value::String(s) if s.trim().is_empty() => json!({}),
                        Value::String(s) => {
                            serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({"_raw": s}))
                        }
                        Value::Null => json!({}),
                        other => other.clone(),
                    };
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": c["id"].as_str().unwrap_or("call_0"),
                        "name": name,
                        "input": input,
                    }));
                }
                if !blocks.is_empty() {
                    out.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"].as_str().unwrap_or(""),
                    "content": content_string(&m["content"]),
                });
                match out.last_mut() {
                    Some(last)
                        if last["role"] == "user"
                            && last["content"]
                                .as_array()
                                .map(|a| a.iter().all(|b| b["type"] == "tool_result"))
                                .unwrap_or(false) =>
                    {
                        last["content"].as_array_mut().unwrap().push(block);
                    }
                    _ => out.push(json!({"role": "user", "content": [block]})),
                }
            }
            other => {
                return Err(FlowcatError::Other(format!(
                    "unsupported message role {other:?} for the Messages API"
                )))
            }
        }
    }
    Ok((system, out))
}

/// Folds the SSE stream into frames.
struct Folder {
    model: String,
    buf: Vec<u8>,
    pending: VecDeque<Frame>,
    started: bool,
    finished: bool,
    /// Open `tool_use` blocks by index: (id, name, partial json).
    tool_blocks: std::collections::HashMap<u64, (String, String, String)>,
    calls: Vec<FunctionCall>,
    input_tokens: u64,
    output_tokens: u64,
    thinking_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    requested_at: Instant,
    /// First output of any kind, thinking included — the real time-to-first-token.
    first_output_at: Option<Instant>,
    /// First *speakable* token. Adaptive thinking can put seconds between the
    /// two, and only this one ends the caller's silence.
    first_text_at: Option<Instant>,
    stop_reason: Option<String>,
}

impl Folder {
    fn new(model: String) -> Self {
        Self {
            model,
            buf: Vec::new(),
            pending: VecDeque::new(),
            started: false,
            finished: false,
            tool_blocks: Default::default(),
            calls: Vec::new(),
            input_tokens: 0,
            output_tokens: 0,
            thinking_tokens: 0,
            cache_read: 0,
            cache_write: 0,
            requested_at: Instant::now(),
            first_output_at: None,
            first_text_at: None,
            stop_reason: None,
        }
    }

    fn start(&mut self) {
        if !self.started {
            self.started = true;
            self.pending.push_back(Frame::LlmResponseStart);
        }
    }

    /// One SSE `data:` payload (pure — the wire-fixture seam).
    fn event(&mut self, ev: &Value) {
        self.start();
        match ev["type"].as_str().unwrap_or("") {
            "message_start" => {
                let u = &ev["message"]["usage"];
                self.input_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                self.cache_read = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                self.cache_write = u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let idx = ev["index"].as_u64().unwrap_or(0);
                let block = &ev["content_block"];
                if block["type"] == "tool_use" {
                    self.tool_blocks.insert(
                        idx,
                        (
                            block["id"].as_str().unwrap_or("").to_string(),
                            block["name"].as_str().unwrap_or("").to_string(),
                            String::new(),
                        ),
                    );
                }
            }
            "content_block_delta" => {
                let idx = ev["index"].as_u64().unwrap_or(0);
                let delta = &ev["delta"];
                match delta["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        if let Some(t) = delta["text"].as_str().filter(|t| !t.is_empty()) {
                            let now = Instant::now();
                            self.first_output_at.get_or_insert(now);
                            self.first_text_at.get_or_insert(now);
                            self.pending.push_back(Frame::LlmText(t.to_string()));
                        }
                    }
                    // Adaptive thinking (the Opus 5 default). Never a spoken
                    // frame — it only moves the TTFT clock. `display` defaults
                    // to "omitted", so the text is usually empty anyway; the
                    // paired `signature_delta` needs nothing from us.
                    "thinking_delta" => {
                        self.first_output_at.get_or_insert_with(Instant::now);
                    }
                    "input_json_delta" => {
                        if let Some(b) = self.tool_blocks.get_mut(&idx) {
                            b.2.push_str(delta["partial_json"].as_str().unwrap_or(""));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let idx = ev["index"].as_u64().unwrap_or(0);
                if let Some((id, name, raw)) = self.tool_blocks.remove(&idx) {
                    if name.is_empty() {
                        return;
                    }
                    // Output, but not speech — `first_text_at` stays unset so
                    // `text_ms` only ever measures time to something spoken.
                    self.first_output_at.get_or_insert_with(Instant::now);
                    let arguments = if raw.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&raw).unwrap_or_else(|_| json!({"_raw": raw}))
                    };
                    self.calls.push(FunctionCall {
                        function_name: name,
                        tool_call_id: id,
                        arguments,
                    });
                }
            }
            "message_delta" => {
                if let Some(n) = ev["usage"]["output_tokens"].as_u64() {
                    self.output_tokens = n;
                }
                if let Some(n) = ev["usage"]["output_tokens_details"]["thinking_tokens"].as_u64() {
                    self.thinking_tokens = n;
                }
                if let Some(r) = ev["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(r.to_string());
                }
            }
            "message_stop" => self.finish(),
            "error" => {
                tracing::warn!(error = %ev["error"], "claude: stream error event");
                self.finish();
            }
            _ => {}
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.start();
        if self.stop_reason.as_deref() == Some("refusal") {
            tracing::warn!("claude: request was refused (stop_reason=refusal)");
        }
        // Nothing to say and nothing to run. The way this happens in practice is
        // adaptive thinking eating the whole `max_tokens` budget
        // (`stop_reason: max_tokens`, every output token a thinking token), which
        // used to leave the caller in silence with no clue why.
        if self.calls.is_empty() && self.first_text_at.is_none() {
            tracing::warn!(
                stop_reason = self.stop_reason.as_deref().unwrap_or(""),
                output_tokens = self.output_tokens,
                thinking_tokens = self.thinking_tokens,
                max_tokens = MAX_TOKENS,
                "claude: turn produced no speakable text; speaking the fallback"
            );
            self.pending
                .push_back(Frame::LlmText(EMPTY_TURN_FALLBACK.to_string()));
        }
        if !self.calls.is_empty() {
            self.pending
                .push_back(Frame::FunctionCallsStarted(std::mem::take(&mut self.calls)));
        }
        let since_request = |t: Instant| t.duration_since(self.requested_at).as_millis() as u64;
        let ttft_ms = self.first_output_at.map(since_request);
        let text_ms = self.first_text_at.map(since_request);
        tracing::info!(
            input_tokens = self.input_tokens,
            output_tokens = self.output_tokens,
            thinking_tokens = self.thinking_tokens,
            cache_read = self.cache_read,
            ttft_ms,
            text_ms,
            stop_reason = self.stop_reason.as_deref().unwrap_or(""),
            "claude turn"
        );
        self.pending
            .push_back(Frame::Metrics(vec![MetricsData::LlmUsage {
                processor: "claude".to_string(),
                model: Some(self.model.clone()),
                tokens: LlmTokenUsage {
                    prompt_tokens: self.input_tokens,
                    completion_tokens: self.output_tokens,
                    total_tokens: self.input_tokens + self.output_tokens,
                    cache_read_input_tokens: Some(self.cache_read),
                    cache_creation_input_tokens: Some(self.cache_write),
                    reasoning_tokens: None,
                },
            }]));
        self.pending.push_back(Frame::LlmResponseEnd);
    }

    /// Feed raw bytes; complete `data:` lines are parsed and folded.
    fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            if let Some(data) = line.strip_prefix("data:") {
                match serde_json::from_str::<Value>(data.trim()) {
                    Ok(ev) => self.event(&ev),
                    Err(e) => tracing::debug!(error = %e, "claude: unparsable SSE data line"),
                }
            }
        }
    }
}

#[async_trait]
impl LlmService for ClaudeLlm {
    fn name(&self) -> &str {
        "claude"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = self.request_body(ctx)?;
        let resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("claude send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("claude {status}: {text}")));
        }
        let folder = Folder::new(self.model.clone());
        Ok(sse_to_frames(resp.bytes_stream(), folder))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

/// Fold the SSE byte stream into frames as they arrive.
fn sse_to_frames<'a, S>(byte_stream: S, folder: Folder) -> BoxStream<'a, Frame>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send + 'a,
{
    futures::stream::unfold(
        (byte_stream.boxed(), folder),
        |(mut bytes, mut st)| async move {
            loop {
                if let Some(f) = st.pending.pop_front() {
                    return Some((f, (bytes, st)));
                }
                if st.finished {
                    return None;
                }
                match bytes.next().await {
                    Some(Ok(chunk)) => st.feed(chunk.as_ref()),
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "claude: stream read failed");
                        st.finish();
                    }
                    None => st.finish(),
                }
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_openai_context_to_messages_api() {
        let ctx = vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "time?"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1_0", "type": "function", "function": {"name": "get_current_time", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "call_1_0", "content": "It's noon."}),
            json!({"role": "assistant", "content": "It's noon."}),
        ];
        let (system, msgs) = translate_messages(&ctx).unwrap();
        assert_eq!(system, "Be brief.");
        assert_eq!(msgs[0], json!({"role": "user", "content": "time?"}));
        assert_eq!(msgs[1]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[1]["content"][0]["id"], "call_1_0");
        assert_eq!(msgs[1]["content"][0]["input"], json!({}));
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["tool_use_id"], "call_1_0");
        assert_eq!(msgs[3]["content"][0]["text"], "It's noon.");
    }

    #[test]
    fn folds_text_and_tool_use_stream_into_frames() {
        let mut f = Folder::new("m".into());
        let lines = [
            r#"event: message_start"#,
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
            "",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"One sec."}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"set_timer","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"min"}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"utes\": 5}"}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
            r#"data: {"type":"message_stop"}"#,
        ];
        f.feed((lines.join("\n") + "\n").as_bytes());
        let frames: Vec<Frame> = f.pending.drain(..).collect();
        assert!(matches!(frames[0], Frame::LlmResponseStart));
        assert!(matches!(&frames[1], Frame::LlmText(t) if t == "One sec."));
        match &frames[2] {
            Frame::FunctionCallsStarted(calls) => {
                assert_eq!(calls[0].function_name, "set_timer");
                assert_eq!(calls[0].tool_call_id, "toolu_1");
                assert_eq!(calls[0].arguments, json!({"minutes": 5}));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(frames[3], Frame::Metrics(_)));
        assert!(matches!(frames[4], Frame::LlmResponseEnd));
    }

    #[test]
    fn request_body_shape() {
        let mut llm = ClaudeLlm::new("k".into(), "claude-opus-5".into(), "low".into());
        llm.set_tools(vec![Tool {
            name: "b".into(),
            description: "B".into(),
            params: json!({"type": "object", "properties": {}}),
        }]);
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": "S"}),
                json!({"role": "user", "content": "hi"}),
            ],
            tools: vec![],
        };
        let body = llm.request_body(&ctx).unwrap();
        assert_eq!(body["model"], "claude-opus-5");
        assert_eq!(body["system"], "S");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], MAX_TOKENS);
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["messages"][0]["content"], "hi");
        // Thinking stays unset (= adaptive on Opus 5); depth is tuned by effort.
        assert_eq!(body["output_config"], json!({"effort": "low"}));
        assert!(body.get("thinking").is_none(), "{body}");
    }

    fn plain_ctx() -> LlmContext {
        LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![],
        }
    }

    #[test]
    fn claude_is_not_shown_web_search_or_ask_claude() {
        let mut llm = ClaudeLlm::new("k".into(), "claude-opus-5".into(), "low".into());
        llm.set_tools(vec![
            Tool {
                name: "web_search".into(),
                description: "local".into(),
                params: json!({"type": "object", "properties": {}}),
            },
            Tool {
                name: "ask_claude".into(),
                description: "local".into(),
                params: json!({"type": "object", "properties": {}}),
            },
            Tool {
                name: "get_weather".into(),
                description: "W".into(),
                params: json!({"type": "object", "properties": {}}),
            },
        ]);
        let body = llm.request_body(&plain_ctx()).unwrap();
        let names: Vec<&str> = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["get_weather"]);
    }

    #[test]
    fn empty_effort_omits_output_config() {
        let llm = ClaudeLlm::new("k".into(), "claude-haiku-4-5".into(), String::new());
        let body = llm.request_body(&plain_ctx()).unwrap();
        assert!(body.get("output_config").is_none(), "{body}");
    }

    fn frames_of(lines: &[&str]) -> Vec<Frame> {
        let mut f = Folder::new("m".into());
        f.feed((lines.join("\n") + "\n").as_bytes());
        f.pending.drain(..).collect()
    }

    fn spoken(frames: &[Frame]) -> String {
        frames
            .iter()
            .filter_map(|f| match f {
                Frame::LlmText(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn thinking_blocks_are_never_spoken() {
        // Opus 5 thinks by default and `display` defaults to "omitted", so the
        // thinking text is empty — but the blocks still arrive.
        let frames = frames_of(&[
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weighing it up"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Paris."}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40,"output_tokens_details":{"thinking_tokens":33}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(spoken(&frames), "Paris.", "thinking must not reach the TTS");
    }

    #[test]
    fn a_turn_that_only_thinks_still_says_something() {
        // The live failure: adaptive thinking spent the whole max_tokens budget,
        // so the stream carried no text and the call went silent.
        let frames = frames_of(&[
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":4000}}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4096,"output_tokens_details":{"thinking_tokens":4096}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(spoken(&frames), EMPTY_TURN_FALLBACK);
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
    }

    #[test]
    fn a_tool_only_turn_stays_silent() {
        // Tool calls speak through their result, so no fallback there.
        let frames = frames_of(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_current_time","input":{}}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(spoken(&frames), "");
        assert!(frames
            .iter()
            .any(|f| matches!(f, Frame::FunctionCallsStarted(_))));
    }
}

#[cfg(test)]
mod network_tests {
    //! One real streamed turn: `cargo test -p voice-chatbot-server -- --ignored network_claude`.
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn network_claude_streams_a_short_reply() {
        crate::env_file::load_if_unset(std::path::Path::new("../../.env"));
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY in .env");
        let mut llm = ClaudeLlm::new(key, "claude-opus-5".into(), DEFAULT_EFFORT.into());
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": "Reply with exactly the word: pong"}),
                json!({"role": "user", "content": "ping"}),
            ],
            tools: vec![],
        };
        let frames: Vec<Frame> = llm.run_llm(&ctx).await.expect("request").collect().await;
        let text: String = frames
            .iter()
            .filter_map(|f| match f {
                Frame::LlmText(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        eprintln!("claude said: {text:?} ({} frames)", frames.len());
        assert!(text.to_lowercase().contains("pong"), "{text}");
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
    }

    /// The turn that was failing live: `ask_claude` has flipped the backend, so
    /// the continuation of that same turn runs here, reading the skill's result
    /// as its prompt. It has to answer the question — the old result string had
    /// it re-announce the handover instead, burning the turn.
    #[tokio::test]
    #[ignore]
    async fn network_claude_answers_within_the_ask_claude_turn() {
        crate::env_file::load_if_unset(std::path::Path::new("../../.env"));
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY in .env");
        let mut llm = ClaudeLlm::new(key, "claude-opus-5".into(), DEFAULT_EFFORT.into());
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": include_str!("../prompt.babel.txt")}),
                json!({"role": "assistant", "content": "Ready."}),
                json!({"role": "user", "content": "Ask Claude what the capital of France is."}),
                json!({"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1_0", "type": "function",
                    "function": {"name": "ask_claude", "arguments": "{}"}}]}),
                json!({"role": "tool", "tool_call_id": "call_1_0",
                       "content": crate::skills::claude::HANDOVER}),
            ],
            tools: vec![],
        };
        let frames: Vec<Frame> = llm.run_llm(&ctx).await.expect("request").collect().await;
        let text: String = frames
            .iter()
            .filter_map(|f| match f {
                Frame::LlmText(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        eprintln!("claude said: {text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "answered the handover instead of the question: {text}"
        );
    }
}
