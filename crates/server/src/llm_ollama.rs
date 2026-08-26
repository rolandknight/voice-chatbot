//! Native Ollama `LlmService` (`POC_LLM_PROVIDER=ollama`) — ADR-0007 Layer 1.
//!
//! Streams `POST /api/chat` (NDJSON) instead of Ollama's OpenAI-compatible
//! `/v1`, because only the native endpoint lets a *request* carry what keeps
//! the model correct and resident: `keep_alive` (pin), `options.num_ctx`
//! (context — the wrong size gets the resident model evicted and reloaded),
//! `think: false`, and it reports `prompt_eval_duration` / `eval_duration`,
//! the only trustworthy prompt-cache evidence (`prompt_eval_count` includes
//! cached tokens since Ollama 0.32, PR #16428). The `/v1` endpoint applies
//! the serve's default keep-alive to every request (issue #2963), which is
//! how the pinned model kept unloading after five idle minutes.
//!
//! Frame contract is identical to flowcat-services' `OpenAiLlm`:
//! `LlmResponseStart`, `LlmText`*, one `FunctionCallsStarted`,
//! `Metrics(LlmUsage)`, `LlmResponseEnd`. Dropping the stream drops the HTTP
//! body, which aborts generation (barge-in).

use std::collections::VecDeque;
use std::time::Instant;

use async_trait::async_trait;
use flowcat_core::processor::frame::{Frame, FunctionCall, LlmContext, StartParams};
use flowcat_core::processor::metrics::{LlmTokenUsage, MetricsData};
use flowcat_core::service::{LlmService, Tool};
use flowcat_core::{FlowcatError, Result};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

pub struct OllamaLlm {
    base_url: String,
    model: String,
    num_ctx: u32,
    /// `-1` pins the model resident; `0` unloads. Sent on every request.
    keep_alive: i64,
    temperature: f64,
    num_predict: u32,
    http: reqwest::Client,
    tools: Vec<Tool>,
    /// Per-service counter so synthesized tool-call ids are unique per turn
    /// (Ollama's native API returns none; the tool bridge matches on id).
    turn: u64,
}

/// What a start-up warm learned; logged and asserted by the caller.
#[derive(Debug, Clone)]
pub struct WarmReport {
    pub load_ms: u64,
    pub prompt_eval_ms: u64,
    pub prompt_tokens: u64,
    pub total_ms: u64,
}

/// One row of `GET /api/ps`.
#[derive(Debug, Clone)]
pub struct Residency {
    pub context_length: u64,
    /// True when `expires_at` is effectively never (keep_alive -1 → far future).
    pub pinned: bool,
    pub size_vram: u64,
}

impl OllamaLlm {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base_url: String = base_url.into();
        Self {
            base_url: base_url
                .trim_end_matches('/')
                .trim_end_matches("/v1")
                .to_string(),
            model: model.into(),
            num_ctx: 8192,
            keep_alive: -1,
            temperature: 0.2,
            num_predict: 512,
            http: reqwest::Client::new(),
            tools: Vec::new(),
            turn: 0,
        }
    }

    pub fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = num_ctx;
        self
    }

    /// Ollama-native tool schema (same wrapping as OpenAI).
    fn tool_json(t: &Tool) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.params,
            }
        })
    }

    /// Tools for this request: the context's (as `ToolDecl` JSON) else the
    /// service-level set — **sorted by name** so the rendered prefix is
    /// byte-stable across turns (ADR-0003 prefix-cache discipline).
    fn tools_json(&self, ctx: &LlmContext) -> Vec<Value> {
        let mut tools: Vec<Value> = if !ctx.tools.is_empty() {
            ctx.tools
                .iter()
                .filter_map(|v| serde_json::from_value::<Tool>(v.clone()).ok())
                .map(|t| Self::tool_json(&t))
                .collect()
        } else {
            self.tools.iter().map(Self::tool_json).collect()
        };
        tools.sort_by(|a, b| {
            a["function"]["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["function"]["name"].as_str().unwrap_or(""))
        });
        tools
    }

    /// The `/api/chat` body (pure — the wire-fixture seam).
    pub fn request_body(&self, ctx: &LlmContext, stream: bool) -> Result<Value> {
        let messages = translate_messages(&ctx.messages)?;
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": false,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        });
        let tools = self.tools_json(ctx);
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        Ok(body)
    }

    async fn post(&self, path: &str, body: &Value) -> Result<reqwest::Response> {
        let resp = self
            .http
            .post(format!("{}{}", self.base_url, path))
            .json(body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama send {path}: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "ollama {path} {status}: {text}"
            )));
        }
        Ok(resp)
    }

    /// Prefill the exact production prefix (system prompt + sorted tools,
    /// nothing else) with a one-token completion so llama.cpp's prompt cache
    /// holds it before the first caller connects. Pins the model as a side
    /// effect (`keep_alive` is on the request).
    pub async fn warm(&self, system_prompt: &str, tools: &[Tool]) -> Result<WarmReport> {
        let ctx = LlmContext {
            messages: vec![json!({"role": "system", "content": system_prompt})],
            tools: tools
                .iter()
                .map(|t| serde_json::to_value(t).unwrap_or(Value::Null))
                .collect(),
        };
        let mut body = self.request_body(&ctx, false)?;
        body["options"]["num_predict"] = json!(1);
        let started = Instant::now();
        let resp = self.post("/api/chat", &body).await?;
        let v: Value = resp
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama warm body: {e}")))?;
        let ns = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0) / 1_000_000;
        Ok(WarmReport {
            load_ms: ns("load_duration"),
            prompt_eval_ms: ns("prompt_eval_duration"),
            prompt_tokens: v
                .get("prompt_eval_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            total_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// `GET /api/ps` row for our model, if resident.
    pub async fn residency(&self) -> Result<Option<Residency>> {
        let v: Value = self
            .http
            .get(format!("{}/api/ps", self.base_url))
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama ps: {e}")))?
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama ps body: {e}")))?;
        Ok(v["models"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|m| m["name"] == self.model.as_str() || m["model"] == self.model.as_str())
            .map(|m| Residency {
                context_length: m["context_length"].as_u64().unwrap_or(0),
                pinned: expires_far_future(m["expires_at"].as_str().unwrap_or("")),
                size_vram: m["size_vram"].as_u64().unwrap_or(0),
            }))
    }

    /// Release the model (`keep_alive: 0`). The serve stays up.
    pub async fn unload(&self) -> Result<()> {
        self.post(
            "/api/generate",
            &json!({"model": self.model, "keep_alive": 0}),
        )
        .await?;
        Ok(())
    }

    /// `GET /api/version` → e.g. "0.32.5".
    pub async fn version(&self) -> Result<String> {
        let v: Value = self
            .http
            .get(format!("{}/api/version", self.base_url))
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama version: {e}")))?
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama version body: {e}")))?;
        Ok(v["version"].as_str().unwrap_or("?").to_string())
    }

    /// Model present in `GET /api/tags`?
    pub async fn has_model(&self) -> Result<bool> {
        let v: Value = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama tags: {e}")))?
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("ollama tags body: {e}")))?;
        Ok(v["models"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|m| m["name"] == self.model.as_str() || m["model"] == self.model.as_str()))
    }
}

/// `expires_at` from `/api/ps`; `keep_alive: -1` yields a date ~300 years out.
fn expires_far_future(expires_at: &str) -> bool {
    expires_at
        .get(..4)
        .and_then(|y| y.parse::<i32>().ok())
        .map(|year| year >= 2100)
        .unwrap_or(false)
}

/// OpenAI-shaped context messages → Ollama `/api/chat` messages.
///
/// The rolling context stores `assistant.tool_calls[{id, type, function{name,
/// arguments: <JSON string>}}]` and `tool{tool_call_id, content}`. Ollama wants
/// `assistant.tool_calls[{function{name, arguments: <object>}}]` and
/// `tool{content, tool_name}` (no ids). Anything unrecognised is an error, not
/// a silent drop.
pub fn translate_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(messages.len());
    // tool_call_id → function name, for the `tool` messages that follow.
    let mut call_names: std::collections::HashMap<String, String> = Default::default();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("");
        match role {
            "system" | "user" => {
                out.push(json!({"role": role, "content": content_string(&m["content"])}));
            }
            "assistant" => {
                let mut msg =
                    json!({"role": "assistant", "content": content_string(&m["content"])});
                if let Some(calls) = m["tool_calls"].as_array() {
                    let mut tcs = Vec::with_capacity(calls.len());
                    for c in calls {
                        let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                        if name.is_empty() {
                            return Err(FlowcatError::Other(
                                "assistant tool_call without a function name".into(),
                            ));
                        }
                        let args = match &c["function"]["arguments"] {
                            Value::String(s) if s.trim().is_empty() => json!({}),
                            Value::String(s) => serde_json::from_str::<Value>(s)
                                .unwrap_or_else(|_| json!({"_raw": s})),
                            Value::Null => json!({}),
                            other => other.clone(),
                        };
                        if let Some(id) = c["id"].as_str() {
                            call_names.insert(id.to_string(), name.clone());
                        }
                        tcs.push(json!({"function": {"name": name, "arguments": args}}));
                    }
                    msg["tool_calls"] = Value::Array(tcs);
                }
                out.push(msg);
            }
            "tool" => {
                let id = m["tool_call_id"].as_str().unwrap_or("");
                let tool_name = m["tool_name"]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| call_names.get(id).cloned())
                    .unwrap_or_default();
                let mut msg = json!({"role": "tool", "content": content_string(&m["content"])});
                if !tool_name.is_empty() {
                    msg["tool_name"] = json!(tool_name);
                }
                out.push(msg);
            }
            other => {
                return Err(FlowcatError::Other(format!(
                    "unsupported message role {other:?} for ollama /api/chat"
                )))
            }
        }
    }
    Ok(out)
}

fn content_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        // OpenAI content parts → concatenated text.
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
    }
}

/// State carried across NDJSON lines by [`ndjson_to_frames`].
struct NdjsonState {
    buf: Vec<u8>,
    started: bool,
    finished: bool,
    pending: VecDeque<Frame>,
    calls: Vec<FunctionCall>,
    model: String,
    turn: u64,
    requested_at: Instant,
    first_token_at: Option<Instant>,
}

impl NdjsonState {
    fn start(&mut self) {
        if !self.started {
            self.started = true;
            self.pending.push_back(Frame::LlmResponseStart);
        }
    }

    /// Fold one `/api/chat` chunk (pure — the wire-fixture seam).
    fn accumulate(&mut self, chunk: &Value) {
        self.start();
        let msg = &chunk["message"];
        if let Some(text) = msg["content"].as_str() {
            if !text.is_empty() {
                if self.first_token_at.is_none() {
                    self.first_token_at = Some(Instant::now());
                }
                self.pending.push_back(Frame::LlmText(text.to_string()));
            }
        }
        if let Some(thinking) = msg["thinking"].as_str() {
            if !thinking.is_empty() {
                tracing::warn!(
                    chars = thinking.len(),
                    "ollama: thinking delta despite think=false (dropped)"
                );
            }
        }
        if let Some(calls) = msg["tool_calls"].as_array() {
            for c in calls {
                let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                if name.is_empty() {
                    continue;
                }
                let arguments = match &c["function"]["arguments"] {
                    Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
                    Value::Null => json!({}),
                    other => other.clone(),
                };
                if self.first_token_at.is_none() {
                    self.first_token_at = Some(Instant::now());
                }
                let n = self.calls.len();
                self.calls.push(FunctionCall {
                    function_name: name,
                    tool_call_id: format!("call_{}_{}", self.turn, n),
                    arguments,
                });
            }
        }
        if chunk["done"].as_bool() == Some(true) {
            self.finish(Some(chunk));
        }
    }

    fn finish(&mut self, done_chunk: Option<&Value>) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.start();
        if !self.calls.is_empty() {
            self.pending
                .push_back(Frame::FunctionCallsStarted(std::mem::take(&mut self.calls)));
        }
        if let Some(v) = done_chunk {
            let get = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
            let prompt_tokens = get("prompt_eval_count");
            let completion_tokens = get("eval_count");
            let prompt_eval_ms = get("prompt_eval_duration") / 1_000_000;
            let eval_ms = get("eval_duration") / 1_000_000;
            // Wall time from request to the first token, as the caller experiences it.
            let ttft_ms = self
                .first_token_at
                .map(|t| t.duration_since(self.requested_at).as_millis() as u64);
            // The cache-hit signal is prompt_eval_ms (durations), never the count.
            tracing::info!(
                prompt_tokens,
                prompt_eval_ms,
                completion_tokens,
                eval_ms,
                ttft_ms,
                load_ms = get("load_duration") / 1_000_000,
                "ollama turn"
            );
            self.pending
                .push_back(Frame::Metrics(vec![MetricsData::LlmUsage {
                    processor: "ollama".to_string(),
                    model: Some(self.model.clone()),
                    tokens: LlmTokenUsage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        cache_read_input_tokens: None,
                        cache_creation_input_tokens: None,
                        reasoning_tokens: None,
                    },
                }]));
        }
        self.pending.push_back(Frame::LlmResponseEnd);
    }

    /// Feed raw bytes; complete lines are parsed and folded.
    fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(line) {
                Ok(v) => {
                    if let Some(err) = v.get("error").and_then(Value::as_str) {
                        tracing::warn!(error = %err, "ollama stream error");
                        self.finish(None);
                    } else {
                        self.accumulate(&v);
                    }
                }
                Err(e) => tracing::warn!(error = %e, "ollama: unparseable NDJSON line (skipped)"),
            }
        }
    }
}

/// Turn an NDJSON byte stream into frames (owns the body, never borrows the service).
pub fn ndjson_to_frames<S, B, E>(
    byte_stream: S,
    model: String,
    turn: u64,
    requested_at: Instant,
) -> BoxStream<'static, Frame>
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let state = NdjsonState {
        buf: Vec::new(),
        started: false,
        finished: false,
        pending: VecDeque::new(),
        calls: Vec::new(),
        model,
        turn,
        requested_at,
        first_token_at: None,
    };
    futures::stream::unfold(
        (byte_stream.boxed(), state),
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
                        tracing::warn!(error = %e, "ollama stream read error");
                        st.finish(None);
                    }
                    None => st.finish(None),
                }
            }
        },
    )
    .boxed()
}

#[async_trait]
impl LlmService for OllamaLlm {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        self.turn += 1;
        let requested_at = Instant::now();
        let body = self.request_body(ctx, true)?;
        let resp = self.post("/api/chat", &body).await?;
        Ok(ndjson_to_frames(
            resp.bytes_stream(),
            self.model.clone(),
            self.turn,
            requested_at,
        ))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Tool {
        Tool {
            name: name.into(),
            description: format!("{name} desc"),
            params: json!({"type": "object", "properties": {}, "required": []}),
        }
    }

    #[test]
    fn request_body_pins_context_and_sorts_tools() {
        let mut llm = OllamaLlm::new("http://127.0.0.1:11434/v1", "gemma4:26b").num_ctx(8192);
        llm.set_tools(vec![tool("set_timer"), tool("get_current_time")]);
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": "sys"}),
                json!({"role": "user", "content": "hi"}),
            ],
            tools: vec![],
        };
        let body = llm.request_body(&ctx, true).unwrap();
        assert_eq!(
            llm.base_url, "http://127.0.0.1:11434",
            "/v1 suffix stripped"
        );
        assert_eq!(body["model"], "gemma4:26b");
        assert_eq!(body["stream"], true);
        assert_eq!(body["think"], false);
        assert_eq!(body["keep_alive"], -1);
        assert_eq!(body["options"]["num_ctx"], 8192);
        let names: Vec<&str> = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["get_current_time", "set_timer"]);
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn context_tools_win_over_service_tools_and_are_sorted() {
        let mut llm = OllamaLlm::new("http://x", "m");
        llm.set_tools(vec![tool("zzz")]);
        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![
                serde_json::to_value(tool("play_spotify")).unwrap(),
                serde_json::to_value(tool("get_weather")).unwrap(),
            ],
        };
        let body = llm.request_body(&ctx, true).unwrap();
        let names: Vec<&str> = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["get_weather", "play_spotify"]);
    }

    #[test]
    fn tool_round_trip_translates_openai_shapes() {
        // Exactly what RollingContext pushes (cascaded.rs push_assistant_tool_call / push_tool_result).
        let msgs = vec![
            json!({"role": "user", "content": "what time is it?"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1_0", "type": "function",
                "function": {"name": "get_current_time", "arguments": "{\"tz\":\"UTC\"}"}
            }]}),
            json!({"role": "tool", "tool_call_id": "call_1_0", "content": "{\"time\":\"14:05\"}"}),
        ];
        let out = translate_messages(&msgs).unwrap();
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"], "");
        assert_eq!(
            out[1]["tool_calls"][0]["function"]["name"],
            "get_current_time"
        );
        assert_eq!(
            out[1]["tool_calls"][0]["function"]["arguments"]["tz"], "UTC",
            "arguments string → object"
        );
        assert!(
            out[1]["tool_calls"][0].get("id").is_none(),
            "no ids on the native API"
        );
        assert_eq!(out[2]["role"], "tool");
        assert_eq!(
            out[2]["tool_name"], "get_current_time",
            "looked up from the preceding call"
        );
        assert_eq!(out[2]["content"], "{\"time\":\"14:05\"}");
        assert!(out[2].get("tool_call_id").is_none());
    }

    #[test]
    fn unknown_role_is_an_error() {
        assert!(translate_messages(&[json!({"role": "developer", "content": "x"})]).is_err());
    }

    async fn frames_from(lines: &[&str]) -> Vec<Frame> {
        let chunks: Vec<std::result::Result<Vec<u8>, String>> = lines
            .iter()
            .map(|l| Ok(format!("{l}\n").into_bytes()))
            .collect();
        ndjson_to_frames(futures::stream::iter(chunks), "m".into(), 7, Instant::now())
            .collect()
            .await
    }

    #[tokio::test]
    async fn text_deltas_then_usage_then_end() {
        let frames = frames_from(&[
            r#"{"message":{"role":"assistant","content":"It "},"done":false}"#,
            r#"{"message":{"role":"assistant","content":"is 2:05."},"done":false}"#,
            r#"{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":1100,"prompt_eval_duration":120000000,"eval_count":6,"eval_duration":70000000}"#,
        ])
        .await;
        assert!(matches!(frames[0], Frame::LlmResponseStart));
        let text: Vec<&str> = frames
            .iter()
            .filter_map(|f| match f {
                Frame::LlmText(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, vec!["It ", "is 2:05."]);
        let usage = frames.iter().find_map(|f| match f {
            Frame::Metrics(m) => m.iter().find_map(|d| match d {
                MetricsData::LlmUsage { tokens, .. } => Some(tokens.clone()),
                _ => None,
            }),
            _ => None,
        });
        let usage = usage.expect("usage metric");
        assert_eq!(usage.prompt_tokens, 1100);
        assert_eq!(usage.completion_tokens, 6);
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
    }

    #[tokio::test]
    async fn tool_call_chunk_becomes_one_function_calls_started_with_ids() {
        let frames = frames_from(&[
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"set_timer","arguments":{"minutes":5}}}]},"done":false}"#,
            r#"{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":10,"eval_count":3}"#,
        ])
        .await;
        let calls = frames
            .iter()
            .find_map(|f| match f {
                Frame::FunctionCallsStarted(c) => Some(c.clone()),
                _ => None,
            })
            .expect("FunctionCallsStarted");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "set_timer");
        assert_eq!(calls[0].tool_call_id, "call_7_0");
        assert_eq!(calls[0].arguments["minutes"], 5);
        // Tool call must come before End and after Start.
        let names: Vec<&str> = frames.iter().map(|f| f.name()).collect();
        assert_eq!(names.first().copied(), Some("LlmResponseStart"));
        assert_eq!(names.last().copied(), Some("LlmResponseEnd"));
    }

    #[tokio::test]
    async fn stream_error_line_closes_the_response() {
        let frames = frames_from(&[
            r#"{"message":{"role":"assistant","content":"par"},"done":false}"#,
            r#"{"error":"model runner has unexpectedly stopped"}"#,
        ])
        .await;
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
        assert_eq!(
            frames
                .iter()
                .filter(|f| matches!(f, Frame::LlmText(_)))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn truncated_stream_still_ends() {
        let frames =
            frames_from(&[r#"{"message":{"role":"assistant","content":"x"},"done":false}"#]).await;
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
    }

    #[test]
    fn far_future_expiry_means_pinned() {
        assert!(expires_far_future("2318-12-05T08:37:38.200477807-05:00"));
        assert!(!expires_far_future("2026-08-25T09:45:39.263565-04:00"));
        assert!(!expires_far_future(""));
    }
}
