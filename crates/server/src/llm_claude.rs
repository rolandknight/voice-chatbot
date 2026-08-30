//! Claude over the Messages API (`POST /v1/messages`, streaming) as a FlowCat
//! [`LlmService`] — the `ask_claude` backend. Raw HTTP: there is no official
//! Rust SDK, and the call shape is small (one streamed request per turn, plus
//! a resend of the same turn when a server-side search pauses it).
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

/// How many times a `pause_turn` may be resumed within one turn. The API pauses
/// a long server-side search loop rather than running it forever; each resume
/// is another request, so the cap bounds both spend and the caller's silence.
const MAX_RESUMES: u32 = 2;

/// Reaching api.anthropic.com at all.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Longest gap *between* reads on the streamed body — not a cap on the whole
/// response, which may legitimately run for the length of a searching turn.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Default `output_config.effort`. Spoken one-or-two-sentence answers do not
/// need the API default (`high`); `low` measured ~0.7 s to first token against
/// ~1.0 s. `thinking.budget_tokens` is rejected with a 400 on Opus 5, so effort
/// is the only depth knob. Empty (`CLAUDE_EFFORT=`) omits the field, for
/// models that reject it.
pub const DEFAULT_EFFORT: &str = "low";

/// Anthropic's web search tool type. `_20260209` adds dynamic filtering —
/// Claude writes and runs code that trims the results before they reach the
/// context window, which cuts input tokens on a search-heavy turn. It needs
/// Opus 4.6+ / Sonnet 4.6+; `claude_model` defaults to `claude-opus-5`. The
/// older `web_search_20250305` is the fallback for other models and is the
/// faster path (no code-execution step), at the cost of untrimmed results.
pub const DEFAULT_SEARCH_TOOL: &str = "web_search_20260209";

/// Searches allowed per turn. A cap is both a cost control ($10 per 1,000
/// searches) and a latency one: every search is dead air on a voice call.
pub const DEFAULT_SEARCH_MAX_USES: u32 = 3;

/// Spoken when a turn ends with neither text nor a tool call, so the call never
/// just goes quiet on the caller. Also the line for a turn that broke — a
/// failed request or a truncated stream really is one Babel lost.
const EMPTY_TURN_FALLBACK: &str = "Sorry, I lost that one. Say it again?";

/// Spoken when the turn is still paused on a server-side search after
/// `MAX_RESUMES`. Distinct from [`EMPTY_TURN_FALLBACK`] because nothing was
/// lost and repeating the question will not help: the search simply ran past
/// what a voice call can wait for.
const SEARCH_GAVE_UP_FALLBACK: &str = "That search is taking too long — ask me again in a moment.";

/// Always hidden from Claude: Claude is already answering, so `ask_claude`
/// re-sets a flag that is already set and costs the caller a turn.
const ASK_CLAUDE: &str = "ask_claude";

/// Hidden from Claude *only while* Anthropic's server-side tool is declared,
/// because that tool carries the **same name** — two tools called `web_search`
/// in one request is a collision, and the server-side one is the whole point of
/// routing to Claude. With `CLAUDE_WEB_SEARCH=false`, or once the org kill
/// switch has fired, there is no server tool and no collision, so Claude gets
/// the local (Brave-backed) skill instead of no search at all.
const LOCAL_WEB_SEARCH: &str = "web_search";

/// Server-side web search settings for the Claude turns. `None` on the
/// `ClaudeLlm` means the tool is not declared at all.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Anthropic tool `type`, e.g. `web_search_20260209`.
    pub tool: String,
    /// `max_uses`; 0 omits the field and takes the API default.
    pub max_uses: u32,
    /// `user_location` object, or `Value::Null` for none.
    pub user_location: Value,
}

pub struct ClaudeLlm {
    http: reqwest::Client,
    api_key: String,
    model: String,
    /// `output_config.effort`; empty omits the field.
    effort: String,
    tools: Vec<Tool>,
    /// Anthropic's server-side web search, when configured.
    search: Option<SearchConfig>,
    /// Set when the API rejects the request because web search is switched off
    /// for the organisation. Without it every Claude turn would fail; with it
    /// the turn is retried once without the tool and the process carries on
    /// unsearched. `AtomicBool` because the stream holds `&self`.
    search_disabled: std::sync::atomic::AtomicBool,
}

impl ClaudeLlm {
    pub fn new(
        api_key: String,
        model: String,
        effort: String,
        search: Option<SearchConfig>,
    ) -> Self {
        Self {
            // Deliberately no total `timeout`: the response is a streamed SSE
            // body that legitimately stays open for the length of the turn,
            // server-side searches included, and a total cap would kill working
            // turns. `connect_timeout` bounds reaching the API at all;
            // `read_timeout` bounds the gap *between* reads, which is what a
            // wedged stream looks like. The API pings during a search, so 60 s
            // of complete silence on an open body is already pathological —
            // while a live voice call must not hang on it forever.
            http: reqwest::Client::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .read_timeout(READ_TIMEOUT)
                .build()
                .expect("reqwest client"),
            api_key,
            model,
            effort,
            tools: Vec::new(),
            search,
            search_disabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Switches Anthropic's server-side search tool off for the rest of *this
    /// call* — called once, after the API refuses a request because web search
    /// is disabled for the organisation. The scope is the call, not the
    /// process: `call.rs` builds a fresh `ClaudeLlm` per WebRTC call, so a
    /// later call re-pays one refused request plus its retry before latching
    /// again. Set `CLAUDE_WEB_SEARCH=false` to stop paying it at all.
    pub fn disable_search(&self) {
        self.search_disabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
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
        // True exactly when Anthropic's server-side `web_search` is going into
        // this request — the only reason to hide the identically-named local
        // skill. When it is not, Claude keeps the local one.
        let searching = self.search.is_some()
            && !self
                .search_disabled
                .load(std::sync::atomic::Ordering::Relaxed);
        let mut out: Vec<Value> = tools
            .iter()
            .filter(|t| t.name != ASK_CLAUDE && !(searching && t.name == LOCAL_WEB_SEARCH))
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": if t.params.is_null() { json!({"type": "object", "properties": {}}) } else { t.params.clone() },
                })
            })
            .collect();
        if let (Some(s), true) = (&self.search, searching) {
            let mut tool = json!({ "type": s.tool, "name": LOCAL_WEB_SEARCH });
            if s.max_uses > 0 {
                tool["max_uses"] = json!(s.max_uses);
            }
            if !s.user_location.is_null() {
                tool["user_location"] = s.user_location.clone();
            }
            out.push(tool);
        }
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

    /// Issue one streamed request. Split out of `run_llm` so a `pause_turn` can
    /// issue the next one from inside the returned stream.
    async fn send(
        &self,
        body: &Value,
    ) -> std::result::Result<
        impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send,
        SendFail,
    > {
        let resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| SendFail::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(SendFail::Http { status, body });
        }
        Ok(resp.bytes_stream())
    }
}

/// The follow-up request for a `pause_turn`: the same body with the paused
/// assistant turn appended verbatim. Deliberately no "continue" user message —
/// the API resumes from the trailing `server_tool_use` block on its own, and an
/// extra message would confuse it.
fn resume_body(body: &Value, blocks: Vec<Value>) -> Value {
    let mut next = body.clone();
    if let Some(msgs) = next["messages"].as_array_mut() {
        msgs.push(json!({ "role": "assistant", "content": blocks }));
    }
    next
}

/// The 400 the API returns when web search is disabled for the organisation in
/// the Console. Matched on text because there is no distinct error code, so it
/// is deliberately narrow: a 400 that names web search and says it is off.
fn is_web_search_disabled(status: u16, body: &str) -> bool {
    if status != 400 {
        return false;
    }
    let body = body.to_ascii_lowercase();
    body.contains("web search") && (body.contains("not enabled") || body.contains("disabled"))
}

/// Whether an org-level search refusal may be answered by replaying the turn
/// without the tool.
///
/// Only before the first resume. The retry re-sends request 1, and from resume
/// 1 on the TTS has already spoken request 1's frames — the caller would hear
/// the opening line and the answer twice — while `Folder::blocks` still holds
/// the first pass at its old offset. Once per turn, too: the stripped request
/// cannot name web search in a 400, but a loop here would be unbounded.
fn may_replay_without_search(resumes: u32, already_retried: bool) -> bool {
    resumes == 0 && !already_retried
}

/// The same request with Anthropic's server-side search tool removed, for the
/// one retry after an org-level refusal.
fn strip_server_search(body: &Value) -> Value {
    let mut next = body.clone();
    let remaining: Vec<Value> = next["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t.get("type").is_none())
        .cloned()
        .collect();
    if remaining.is_empty() {
        if let Some(o) = next.as_object_mut() {
            o.remove("tools");
        }
    } else {
        next["tools"] = Value::Array(remaining);
    }
    next
}

/// How the request loop left the turn. Only [`TurnEnd::Completed`] may end in
/// silence; the other two always speak, because by then the caller has usually
/// already heard the "let me check" preamble the system suffix asks for and
/// would otherwise be left waiting forever.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TurnEnd {
    /// `message_stop` arrived with a terminal `stop_reason`.
    Completed,
    /// Still `pause_turn` after `MAX_RESUMES` resumes — the search outran the
    /// call.
    SearchCapped,
    /// No 2xx, or the stream ended/broke before `message_stop`.
    Failed,
}

/// What the loop does once one request of the turn has ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Step {
    /// Send the next request of this turn (a `pause_turn` resume).
    Resume,
    /// Stop, and let `Folder::finish` speak for this outcome.
    Stop(TurnEnd),
}

/// The loop's exit decision, pure so every branch is testable without a
/// request. `request_done` is `message_stop`: without it the body ended early
/// (a read error, or a server that just closed), which is a failure however
/// much text arrived first.
fn after_request(request_done: bool, stop_reason: Option<&str>, resumes: u32) -> Step {
    if !request_done {
        return Step::Stop(TurnEnd::Failed);
    }
    match stop_reason {
        Some("pause_turn") if resumes < MAX_RESUMES => Step::Resume,
        Some("pause_turn") => Step::Stop(TurnEnd::SearchCapped),
        _ => Step::Stop(TurnEnd::Completed),
    }
}

/// Why one request failed, kept apart so the caller can react to the status.
enum SendFail {
    Http { status: u16, body: String },
    Transport(String),
}

impl std::fmt::Display for SendFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendFail::Http { status, body } => write!(f, "claude {status}: {body}"),
            SendFail::Transport(e) => write!(f, "claude send: {e}"),
        }
    }
}

/// Append to a string field of a rebuilt content block, creating it if absent.
fn append_str(
    blocks: &mut std::collections::BTreeMap<u64, Value>,
    abs: u64,
    field: &str,
    text: &str,
) {
    let Some(block) = blocks.get_mut(&abs) else {
        return;
    };
    let mut current = block[field].as_str().unwrap_or("").to_string();
    current.push_str(text);
    block[field] = json!(current);
}

/// An OpenAI-shaped `content` field as plain text: a bare string, or the
/// concatenated `text` of a block array. `pub(crate)` because `call.rs` reads
/// the same field when it appends the Claude system suffix, and must flatten
/// the array shape the same way rather than dropping it.
pub(crate) fn content_string(v: &Value) -> String {
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
    /// Output totals from the requests of this turn that have already ended.
    /// `message_delta` reports `output_tokens` cumulative *within* a request
    /// and may repeat, so the running total is this base plus the latest
    /// per-request figure — never a sum of the deltas. Advanced by
    /// `begin_request`.
    output_tokens_before: u64,
    thinking_tokens_before: u64,
    cache_read: u64,
    cache_write: u64,
    /// Server-side searches billed so far this turn, from
    /// `usage.server_tool_use.web_search_requests`. Same shape as
    /// `output_tokens`: cumulative *within* a request and possibly repeated,
    /// so it is assigned as `search_requests_before + n`, never summed.
    search_requests: u64,
    search_requests_before: u64,
    requested_at: Instant,
    /// First output of any kind, thinking included — the real time-to-first-token.
    first_output_at: Option<Instant>,
    /// First *speakable* token. Adaptive thinking can put seconds between the
    /// two, and only this one ends the caller's silence.
    first_text_at: Option<Instant>,
    stop_reason: Option<String>,
    /// The assistant turn's raw content blocks, keyed by absolute index.
    ///
    /// A `pause_turn` is resumed by sending this turn back *verbatim*: the API
    /// rejects a `web_search_tool_result` whose `encrypted_content` is missing
    /// or altered. `index` restarts at 0 in every request of the turn, hence
    /// the offset.
    blocks: std::collections::BTreeMap<u64, Value>,
    block_index_offset: u64,
    /// Partial `input` JSON by absolute index, for `tool_use` and
    /// `server_tool_use` alike.
    block_input: std::collections::HashMap<u64, String>,
    /// Set by `message_stop`; the stream driver decides whether that ends the
    /// turn or starts a resume.
    request_done: bool,
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
            output_tokens_before: 0,
            thinking_tokens_before: 0,
            cache_read: 0,
            cache_write: 0,
            search_requests: 0,
            search_requests_before: 0,
            requested_at: Instant::now(),
            first_output_at: None,
            first_text_at: None,
            stop_reason: None,
            blocks: Default::default(),
            block_index_offset: 0,
            block_input: Default::default(),
            request_done: false,
        }
    }

    fn start(&mut self) {
        if !self.started {
            self.started = true;
            self.pending.push_back(Frame::LlmResponseStart);
        }
    }

    fn abs(&self, idx: u64) -> u64 {
        self.block_index_offset + idx
    }

    /// Start another request in the same logical turn (a `pause_turn` resume).
    /// Blocks and timings carry over; per-request state does not.
    fn begin_request(&mut self) {
        self.output_tokens_before = self.output_tokens;
        self.thinking_tokens_before = self.thinking_tokens;
        self.search_requests_before = self.search_requests;
        self.block_index_offset = self.blocks.keys().last().map_or(0, |k| k + 1);
        self.block_input.clear();
        self.tool_blocks.clear();
        self.buf.clear();
        self.stop_reason = None;
        self.request_done = false;
    }

    /// The assistant turn so far, in wire order.
    fn assistant_blocks(&self) -> Vec<Value> {
        self.blocks.values().cloned().collect()
    }

    /// One SSE `data:` payload (pure — the wire-fixture seam).
    fn event(&mut self, ev: &Value) {
        self.start();
        match ev["type"].as_str().unwrap_or("") {
            "message_start" => {
                let u = &ev["message"]["usage"];
                // One `message_start` per request, and every request of a
                // paused turn is billed its own input — so these add up over
                // the turn rather than the last request replacing the rest.
                self.input_tokens += u["input_tokens"].as_u64().unwrap_or(0);
                self.cache_read += u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                self.cache_write += u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let idx = ev["index"].as_u64().unwrap_or(0);
                let block = &ev["content_block"];
                self.blocks.insert(self.abs(idx), block.clone());
                if block["type"] == "web_search_tool_result" && block["content"].is_object() {
                    tracing::warn!(
                        error_code = %block["content"]["error_code"],
                        "claude: web search returned an error"
                    );
                }
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
                            append_str(&mut self.blocks, self.block_index_offset + idx, "text", t);
                        }
                    }
                    // Adaptive thinking (the Opus 5 default). Never a spoken
                    // frame — it only moves the TTFT clock. `display` defaults
                    // to "omitted", so the text is usually empty anyway; the
                    // paired `signature_delta` needs nothing from us.
                    "thinking_delta" => {
                        self.first_output_at.get_or_insert_with(Instant::now);
                        if let Some(t) = delta["thinking"].as_str() {
                            append_str(
                                &mut self.blocks,
                                self.block_index_offset + idx,
                                "thinking",
                                t,
                            );
                        }
                    }
                    // Thinking blocks are only replayable with their signature.
                    "signature_delta" => {
                        if let Some(b) = self.blocks.get_mut(&(self.block_index_offset + idx)) {
                            b["signature"] = delta["signature"].clone();
                        }
                    }
                    "citations_delta" => {
                        if let Some(b) = self.blocks.get_mut(&(self.block_index_offset + idx)) {
                            if !b["citations"].is_array() {
                                b["citations"] = json!([]);
                            }
                            if let Some(a) = b["citations"].as_array_mut() {
                                a.push(delta["citation"].clone());
                            }
                        }
                    }
                    "input_json_delta" => {
                        let partial = delta["partial_json"].as_str().unwrap_or("");
                        if let Some(b) = self.tool_blocks.get_mut(&idx) {
                            b.2.push_str(partial);
                        }
                        self.block_input
                            .entry(self.block_index_offset + idx)
                            .or_default()
                            .push_str(partial);
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let idx = ev["index"].as_u64().unwrap_or(0);
                let abs = self.abs(idx);
                if let Some(raw) = self.block_input.remove(&abs) {
                    if let Some(b) = self.blocks.get_mut(&abs) {
                        b["input"] = if raw.trim().is_empty() {
                            json!({})
                        } else {
                            serde_json::from_str(&raw).unwrap_or_else(|_| json!({}))
                        };
                    }
                }
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
                // Cumulative within the request and possibly repeated, so `n`
                // replaces this request's share instead of adding to it; the
                // earlier requests of the turn are in `*_before`.
                if let Some(n) = ev["usage"]["output_tokens"].as_u64() {
                    self.output_tokens = self.output_tokens_before + n;
                }
                if let Some(n) = ev["usage"]["output_tokens_details"]["thinking_tokens"].as_u64() {
                    self.thinking_tokens = self.thinking_tokens_before + n;
                }
                if let Some(r) = ev["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(r.to_string());
                }
                if let Some(n) = ev["usage"]["server_tool_use"]["web_search_requests"].as_u64() {
                    self.search_requests = self.search_requests_before + n;
                }
            }
            "message_stop" => self.request_done = true,
            // Deliberately does NOT set `request_done`: an `overloaded_error`
            // arriving after the spoken preamble would otherwise be read as a
            // completed turn, so `finish()` would skip the fallback and leave
            // the caller in silence. The read loop exits on stream close
            // anyway, and `TurnEnd::Failed` is then the correct verdict.
            "error" => {
                tracing::warn!(error = %ev["error"], "claude: stream error event");
            }
            _ => {}
        }
    }

    fn finish(&mut self, end: TurnEnd) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.start();
        if self.stop_reason.as_deref() == Some("refusal") {
            tracing::warn!("claude: request was refused (stop_reason=refusal)");
        }
        // A turn that did not complete always speaks, even when it already
        // spoke. `first_text_at` is `Some` on exactly the paths that hurt most:
        // the system suffix asks Claude to say "let me check" *before*
        // searching, so a turn abandoned mid-search has already spoken a
        // preamble and the empty-turn branch below would never fire — leaving
        // the caller with an opening line and then permanent silence.
        let interrupted = match end {
            TurnEnd::Completed => None,
            TurnEnd::SearchCapped => Some(SEARCH_GAVE_UP_FALLBACK),
            TurnEnd::Failed => Some(EMPTY_TURN_FALLBACK),
        };
        if let Some(line) = interrupted {
            tracing::warn!(
                end = ?end,
                spoke_first = self.first_text_at.is_some(),
                search_requests = self.search_requests,
                stop_reason = self.stop_reason.as_deref().unwrap_or(""),
                "claude: turn did not complete; speaking the fallback"
            );
            self.pending.push_back(Frame::LlmText(line.to_string()));
        } else if self.calls.is_empty() && self.first_text_at.is_none() {
            // A completed turn with nothing to say and nothing to run. The way
            // this happens in practice is adaptive thinking eating the whole
            // `max_tokens` budget (`stop_reason: max_tokens`, every output
            // token a thinking token), which used to leave the caller in
            // silence with no clue why.
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
            search_requests = self.search_requests,
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
        // The base request is not mutated by a resume: every resume is the base
        // plus the *whole* paused turn. `assistant_blocks()` is cumulative
        // across the turn, so appending it to the previous resume's body
        // instead would re-send request 1's blocks — duplicated
        // `server_tool_use` ids and `encrypted_content`, in two consecutive
        // assistant messages. The one exception is an org-level search
        // refusal: `base` is then rewritten in place (tool stripped) so every
        // later resume of this turn inherits the stripped form too.
        let base = self.request_body(ctx)?;
        let this: &ClaudeLlm = self;
        Ok(async_stream::stream! {
            let mut base = base;
            let mut body = base.clone();
            let mut folder = Folder::new(this.model.clone());
            let mut resumes = 0u32;
            let mut retried_without_search = false;
            // `Failed` is the right default: the send arms below break out of
            // the loop without ever reaching `after_request`, and a turn whose
            // request never got a stream is exactly a failed one.
            let mut end = TurnEnd::Failed;
            loop {
                let stream = match this.send(&body).await {
                    Err(SendFail::Http { status, body: err })
                        if is_web_search_disabled(status, &err)
                            && may_replay_without_search(resumes, retried_without_search) =>
                    {
                        tracing::error!(
                            "claude: web search is disabled for this organisation — enable it at \
https://platform.claude.com/settings/privacy, or set CLAUDE_WEB_SEARCH=false to stop asking. \
Retrying this turn without it."
                        );
                        this.disable_search();
                        retried_without_search = true;
                        base = strip_server_search(&base);
                        body = base.clone();
                        continue;
                    }
                    // The same refusal arriving on a resume. Latch the tool off
                    // so the rest of the call stops paying for it, but give up
                    // on this turn rather than replaying spoken frames.
                    Err(SendFail::Http { status, body: err })
                        if is_web_search_disabled(status, &err) =>
                    {
                        tracing::error!(
                            resumes,
                            "claude: web search refused for this organisation part-way through a \
turn — search is off for the rest of this call, and this turn is abandoned"
                        );
                        this.disable_search();
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, resumes, "claude: request failed");
                        break;
                    }
                    Ok(s) => s,
                };
                futures::pin_mut!(stream);
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(b) => folder.feed(b.as_ref()),
                        // `request_done` stays false unless `message_stop` had
                        // already arrived, so `after_request` turns this into
                        // `TurnEnd::Failed` on its own.
                        Err(e) => {
                            tracing::warn!(error = %e, resumes, "claude: stream read failed");
                            break;
                        }
                    }
                    while let Some(f) = folder.pending.pop_front() {
                        yield f;
                    }
                    // `message_stop` ends this request's read. The old
                    // `sse_to_frames` returned `None` on the same condition;
                    // reading on until the HTTP body closes instead leaves the
                    // generator parked after the last spoken frame, so
                    // `finish()` never runs, `LlmResponseEnd` is never emitted,
                    // and the pipeline's turn stays open — swallowing the
                    // caller's next utterance until the wake session expires.
                    if folder.request_done {
                        break;
                    }
                }
                while let Some(f) = folder.pending.pop_front() {
                    yield f;
                }
                match after_request(
                    folder.request_done,
                    folder.stop_reason.as_deref(),
                    resumes,
                ) {
                    Step::Resume => {
                        resumes += 1;
                        tracing::info!(resumes, "claude: resuming a paused search turn");
                        body = resume_body(&base, folder.assistant_blocks());
                        folder.begin_request();
                    }
                    Step::Stop(TurnEnd::SearchCapped) => {
                        tracing::warn!(
                            resumes,
                            max_resumes = MAX_RESUMES,
                            search_requests = folder.search_requests,
                            "claude: search still paused after the resume cap; giving up on this turn"
                        );
                        end = TurnEnd::SearchCapped;
                        break;
                    }
                    Step::Stop(e) => {
                        end = e;
                        break;
                    }
                }
            }
            folder.finish(end);
            while let Some(f) = folder.pending.pop_front() {
                yield f;
            }
        }
        .boxed())
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
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
        f.finish(TurnEnd::Completed);
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
        let mut llm = ClaudeLlm::new("k".into(), "claude-opus-5".into(), "low".into(), None);
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

    fn search_cfg() -> SearchConfig {
        SearchConfig {
            tool: DEFAULT_SEARCH_TOOL.to_string(),
            max_uses: 3,
            user_location: crate::location::SearchLocation::parse(crate::location::DEFAULT)
                .unwrap()
                .unwrap()
                .user_location(),
        }
    }

    #[test]
    fn the_server_side_search_tool_is_declared_and_localized() {
        let mut llm = ClaudeLlm::new(
            "k".into(),
            "claude-opus-5".into(),
            "low".into(),
            Some(search_cfg()),
        );
        llm.set_tools(vec![Tool {
            name: "web_search".into(),
            description: "local".into(),
            params: json!({"type": "object", "properties": {}}),
        }]);
        let body = llm.request_body(&plain_ctx()).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(
            tools.len(),
            1,
            "the local skill must not survive: {tools:?}"
        );
        assert_eq!(tools[0]["type"], "web_search_20260209");
        assert_eq!(tools[0]["name"], "web_search");
        assert_eq!(tools[0]["max_uses"], 3);
        assert_eq!(tools[0]["user_location"]["city"], "Toronto");
        assert_eq!(tools[0]["user_location"]["country"], "CA");
    }

    #[test]
    fn no_search_config_means_no_server_tool() {
        let llm = ClaudeLlm::new("k".into(), "claude-opus-5".into(), "low".into(), None);
        let body = llm.request_body(&plain_ctx()).unwrap();
        assert!(body.get("tools").is_none(), "{body}");
    }

    /// The local skill list every one of these cases starts from.
    fn local_tools() -> Vec<Tool> {
        ["web_search", "ask_claude", "get_weather"]
            .into_iter()
            .map(|name| Tool {
                name: name.into(),
                description: "local".into(),
                params: json!({"type": "object", "properties": {}}),
            })
            .collect()
    }

    /// Declared tool names, and whether a server-side tool (one with a `type`)
    /// is among them.
    fn declared(llm: &ClaudeLlm) -> (Vec<String>, bool) {
        let body = llm.request_body(&plain_ctx()).unwrap();
        let tools = body["tools"].as_array().cloned().unwrap_or_default();
        let server = tools.iter().any(|t| t.get("type").is_some());
        (
            tools
                .iter()
                .map(|t| t["name"].as_str().unwrap_or("").to_string())
                .collect(),
            server,
        )
    }

    /// Claude is already answering, so `ask_claude` is hidden whatever search
    /// is doing — its reason has nothing to do with the name collision.
    #[test]
    fn ask_claude_is_hidden_from_claude_either_way() {
        for search in [None, Some(search_cfg())] {
            let mut llm = ClaudeLlm::new("k".into(), "claude-opus-5".into(), "low".into(), search);
            llm.set_tools(local_tools());
            let (names, _) = declared(&llm);
            assert!(!names.iter().any(|n| n == "ask_claude"), "{names:?}");
        }
    }

    /// Search on: the server-side tool takes the `web_search` name, so the
    /// local skill of the same name must not also be declared.
    #[test]
    fn with_search_on_the_server_tool_replaces_the_local_skill() {
        let mut llm = ClaudeLlm::new(
            "k".into(),
            "claude-opus-5".into(),
            "low".into(),
            Some(search_cfg()),
        );
        llm.set_tools(local_tools());
        let (names, server) = declared(&llm);
        assert!(server, "the server-side search tool must be declared");
        assert_eq!(
            names.iter().filter(|n| *n == "web_search").count(),
            1,
            "exactly one web_search — the server-side one: {names:?}"
        );
        let body = llm.request_body(&plain_ctx()).unwrap();
        let local = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "web_search")
            .unwrap();
        assert_eq!(local["type"], DEFAULT_SEARCH_TOOL, "{local}");
    }

    /// `CLAUDE_WEB_SEARCH=false`: no server tool, so no collision — and hiding
    /// the local skill anyway would leave Claude with no search at all.
    #[test]
    fn with_search_off_claude_keeps_the_local_web_search_skill() {
        let mut llm = ClaudeLlm::new("k".into(), "claude-opus-5".into(), "low".into(), None);
        llm.set_tools(local_tools());
        let (names, server) = declared(&llm);
        assert!(!server, "no server-side tool may be declared: {names:?}");
        assert_eq!(names, vec!["get_weather", "web_search"]);
    }

    /// Same, reached the other way: the org kill switch fired mid-call.
    #[test]
    fn the_org_kill_switch_hands_the_local_skill_back_to_claude() {
        let mut llm = ClaudeLlm::new(
            "k".into(),
            "claude-opus-5".into(),
            "low".into(),
            Some(search_cfg()),
        );
        llm.set_tools(local_tools());
        llm.disable_search();
        let (names, server) = declared(&llm);
        assert!(!server, "the refused tool must not come back: {names:?}");
        assert_eq!(names, vec!["get_weather", "web_search"]);
    }

    #[test]
    fn empty_effort_omits_output_config() {
        let llm = ClaudeLlm::new("k".into(), "claude-haiku-4-5".into(), String::new(), None);
        let body = llm.request_body(&plain_ctx()).unwrap();
        assert!(body.get("output_config").is_none(), "{body}");
    }

    fn frames_of(lines: &[&str]) -> Vec<Frame> {
        let mut f = Folder::new("m".into());
        f.feed((lines.join("\n") + "\n").as_bytes());
        f.finish(TurnEnd::Completed);
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

    /// The doc's own streaming example, trimmed: text, the search query, the
    /// results, then the cited answer.
    const SEARCH_STREAM: [&str; 8] = [
        r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}"#,
        r#"data: {"type":"content_block_stop","index":0}"#,
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search"}}"#,
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"cineplex etobicoke showtimes\"}"}}"#,
        r#"data: {"type":"content_block_stop","index":1}"#,
        r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://example.com","title":"Showtimes","encrypted_content":"EqgfCioIA"}]}}"#,
    ];

    #[test]
    fn a_server_tool_use_block_is_never_dispatched_as_a_skill() {
        let frames = frames_of(&SEARCH_STREAM);
        assert_eq!(spoken(&frames), "Let me check.");
        assert!(
            !frames
                .iter()
                .any(|f| matches!(f, Frame::FunctionCallsStarted(_))),
            "server_tool_use must never become a client-side tool call: {frames:?}"
        );
    }

    #[test]
    fn assistant_blocks_are_rebuilt_verbatim_for_a_resume() {
        let mut f = Folder::new("m".into());
        f.feed((SEARCH_STREAM.join("\n") + "\n").as_bytes());
        let blocks = f.assistant_blocks();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], json!({"type": "text", "text": "Let me check."}));
        assert_eq!(blocks[1]["type"], "server_tool_use");
        assert_eq!(
            blocks[1]["input"],
            json!({"query": "cineplex etobicoke showtimes"})
        );
        assert_eq!(
            blocks[2]["content"][0]["encrypted_content"], "EqgfCioIA",
            "encrypted_content must survive verbatim or the resume 400s"
        );
    }

    #[test]
    fn a_resumed_request_appends_its_blocks_after_the_paused_ones() {
        let mut f = Folder::new("m".into());
        f.feed((SEARCH_STREAM.join("\n") + "\n").as_bytes());
        f.begin_request();
        f.feed(
            ([
                r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Nothing showing."}}"#,
                r#"data: {"type":"content_block_stop","index":0}"#,
            ]
            .join("\n")
                + "\n")
                .as_bytes(),
        );
        let blocks = f.assistant_blocks();
        assert_eq!(
            blocks.len(),
            4,
            "index restarts at 0 per request: {blocks:?}"
        );
        assert_eq!(
            blocks[3],
            json!({"type": "text", "text": "Nothing showing."})
        );
    }

    #[test]
    fn a_resume_body_appends_the_paused_turn_and_nothing_else() {
        let body = json!({
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "showtimes?"}]
        });
        let blocks =
            vec![json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"})];
        let next = resume_body(&body, blocks.clone());
        let msgs = next["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1], json!({"role": "assistant", "content": blocks}));
        assert_eq!(
            next["model"], body["model"],
            "everything else must be unchanged"
        );
    }

    /// `assistant_blocks()` is cumulative across the requests of a turn, so
    /// each resume body must be derived from the *unmodified* base request.
    /// Deriving the second one from the first would send request 1's blocks
    /// twice — duplicated `server_tool_use` ids and `encrypted_content`, in two
    /// consecutive assistant messages — which is not a shape the API accepts.
    #[test]
    fn a_second_resume_is_built_from_the_base_body_not_the_first_resume() {
        let base = json!({
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "showtimes?"}]
        });
        let b1 = vec![json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"})];
        let mut b1_plus_b2 = b1.clone();
        b1_plus_b2
            .push(json!({"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search"}));

        let first = resume_body(&base, b1.clone());
        assert_eq!(first["messages"].as_array().unwrap().len(), 2);

        let second = resume_body(&base, b1_plus_b2.clone());
        let msgs = second["messages"].as_array().unwrap();
        assert_eq!(
            msgs.len(),
            2,
            "the base's user message and exactly one assistant message: {msgs:?}"
        );
        assert_eq!(msgs[0], base["messages"][0]);
        assert_eq!(msgs[1], json!({"role": "assistant", "content": b1_plus_b2}));
        assert_eq!(
            base["messages"].as_array().unwrap().len(),
            1,
            "the base must not have been mutated"
        );
    }

    /// Every request of a paused turn is billed, so the turn's one `Metrics`
    /// frame has to sum them. `output_tokens` is cumulative *within* a request
    /// and can arrive on several `message_delta` events, so it must not be
    /// summed there.
    #[test]
    fn a_resumed_turn_bills_every_request_it_made() {
        let mut f = Folder::new("m".into());
        f.feed(
            (SEARCH_STREAM.join("\n")
                + "\n"
                + r#"data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":9}}"#
                + "\n"
                + r#"data: {"type":"message_stop"}"#
                + "\n")
                .as_bytes(),
        );
        f.begin_request();
        f.feed(
            ([
                r#"data: {"type":"message_start","message":{"usage":{"input_tokens":2000,"cache_read_input_tokens":5,"cache_creation_input_tokens":7}}}"#,
                r#"data: {"type":"message_delta","delta":{},"usage":{"output_tokens":20,"output_tokens_details":{"thinking_tokens":2}}}"#,
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":31,"output_tokens_details":{"thinking_tokens":3}}}"#,
                r#"data: {"type":"message_stop"}"#,
            ]
            .join("\n")
                + "\n")
                .as_bytes(),
        );
        assert_eq!(f.input_tokens, 2010, "10 from the paused request + 2000");
        assert_eq!(f.cache_read, 5);
        assert_eq!(f.cache_write, 7);
        assert_eq!(
            f.output_tokens, 40,
            "9 from the paused request + 31 (not 20 + 31) from the resume"
        );
        assert_eq!(f.thinking_tokens, 3, "0 + 3, not 2 + 3");
    }

    /// `web_search_requests` lives in the same `usage.server_tool_use` object
    /// as `output_tokens`, and inherits the same trap: cumulative *within* a
    /// request, and the request can emit `message_delta` more than once.
    #[test]
    fn search_requests_are_not_double_counted_across_a_resumed_turn() {
        let mut f = Folder::new("m".into());
        f.feed(
            ([
                r#"data: {"type":"message_delta","delta":{},"usage":{"server_tool_use":{"web_search_requests":1}}}"#,
                r#"data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"server_tool_use":{"web_search_requests":2}}}"#,
                r#"data: {"type":"message_stop"}"#,
            ]
            .join("\n")
                + "\n")
                .as_bytes(),
        );
        assert_eq!(
            f.search_requests, 2,
            "the second delta replaces the first within a request, it does not add to it"
        );
        f.begin_request();
        f.feed(
            ([
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"server_tool_use":{"web_search_requests":1}}}"#,
                r#"data: {"type":"message_stop"}"#,
            ]
            .join("\n")
                + "\n")
                .as_bytes(),
        );
        assert_eq!(
            f.search_requests, 3,
            "2 from the paused request + 1 from the resume (not 1+2+1=4, not just 1)"
        );
    }

    #[test]
    fn pause_turn_is_detected_from_the_stream() {
        let mut f = Folder::new("m".into());
        f.feed(
            (SEARCH_STREAM.join("\n")
                + "\n"
                + r#"data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":9}}"#
                + "\n"
                + r#"data: {"type":"message_stop"}"#
                + "\n")
                .as_bytes(),
        );
        assert_eq!(f.stop_reason.as_deref(), Some("pause_turn"));
        assert!(f.request_done);
    }

    #[test]
    fn an_org_level_web_search_refusal_is_recognised() {
        assert!(is_web_search_disabled(
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"Web search is not enabled for this organization."}}"#
        ));
        assert!(!is_web_search_disabled(
            400,
            r#"{"error":{"message":"max_tokens: must be >= 1"}}"#
        ));
        assert!(!is_web_search_disabled(
            429,
            r#"{"error":{"message":"web search is not enabled"}}"#
        ));
    }

    #[test]
    fn disabling_search_drops_the_tool_from_the_next_request() {
        let llm = ClaudeLlm::new(
            "k".into(),
            "claude-opus-5".into(),
            "low".into(),
            Some(search_cfg()),
        );
        assert!(llm.request_body(&plain_ctx()).unwrap()["tools"].is_array());
        llm.disable_search();
        assert!(
            llm.request_body(&plain_ctx())
                .unwrap()
                .get("tools")
                .is_none(),
            "the tool must not come back on the retry"
        );
    }

    #[test]
    fn stripping_the_server_tool_leaves_the_client_tools() {
        let body = json!({"tools": [
            {"type": "web_search_20260209", "name": "web_search"},
            {"name": "get_weather", "description": "W", "input_schema": {}}
        ]});
        assert_eq!(
            strip_server_search(&body)["tools"],
            json!([{"name": "get_weather", "description": "W", "input_schema": {}}])
        );
        let only_server = json!({"tools": [{"type": "web_search_20260209", "name": "web_search"}]});
        assert!(strip_server_search(&only_server).get("tools").is_none());
    }

    #[test]
    fn a_failed_search_result_block_is_not_mistaken_for_results() {
        let frames = frames_of(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}}}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"I could not look that up."}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(spoken(&frames), "I could not look that up.");
        assert!(!frames
            .iter()
            .any(|f| matches!(f, Frame::FunctionCallsStarted(_))));
    }

    /// A paused turn that has already spoken the "let me check" preamble the
    /// system suffix asks for, folded up to (but not including) the resume.
    fn paused_after_a_preamble() -> Folder {
        let mut f = Folder::new("m".into());
        f.feed(
            (SEARCH_STREAM.join("\n")
                + "\n"
                + r#"data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":9}}"#
                + "\n"
                + r#"data: {"type":"message_stop"}"#
                + "\n")
                .as_bytes(),
        );
        assert!(f.first_text_at.is_some(), "the preamble must have spoken");
        f
    }

    /// The silence bug: `CLAUDE_SYSTEM_SUFFIX` makes Claude say a filler line
    /// *before* searching, so `first_text_at` is `Some` and the empty-turn
    /// fallback never fires. Exhausting `MAX_RESUMES` then left the caller with
    /// "Let me check." and nothing else, forever.
    #[test]
    fn a_capped_search_speaks_even_though_the_preamble_already_did() {
        let mut f = paused_after_a_preamble();
        f.finish(TurnEnd::SearchCapped);
        let frames: Vec<Frame> = f.pending.drain(..).collect();
        assert_eq!(
            spoken(&frames),
            format!("Let me check.{SEARCH_GAVE_UP_FALLBACK}"),
            "the caller must not be left on the preamble alone"
        );
        assert_ne!(
            SEARCH_GAVE_UP_FALLBACK, EMPTY_TURN_FALLBACK,
            "a search that ran long is not a turn Babel lost"
        );
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
    }

    /// Same for a turn abandoned because a request failed or the stream broke
    /// — including one that failed on a resume, after frames were spoken.
    #[test]
    fn a_failed_turn_speaks_even_though_the_preamble_already_did() {
        let mut f = paused_after_a_preamble();
        f.finish(TurnEnd::Failed);
        let frames: Vec<Frame> = f.pending.drain(..).collect();
        assert_eq!(
            spoken(&frames),
            format!("Let me check.{EMPTY_TURN_FALLBACK}")
        );
        assert!(matches!(frames.last(), Some(Frame::LlmResponseEnd)));
    }

    /// ...and a turn that completed normally must not have a fallback bolted
    /// onto the end of a perfectly good answer.
    #[test]
    fn a_completed_turn_adds_no_fallback() {
        let frames = frames_of(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Paris."}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(spoken(&frames), "Paris.");
    }

    /// A turn with nothing spoken speaks the fallback exactly once, not twice
    /// (the interrupted branch and the empty-turn branch are exclusive).
    #[test]
    fn a_failed_turn_that_never_spoke_says_the_line_once() {
        let mut f = Folder::new("m".into());
        f.finish(TurnEnd::Failed);
        let frames: Vec<Frame> = f.pending.drain(..).collect();
        assert_eq!(spoken(&frames), EMPTY_TURN_FALLBACK);
    }

    /// The loop's exit decision. Every branch that is not `Completed` speaks,
    /// so this is what keeps a paused or truncated turn from going quiet.
    #[test]
    fn the_loop_gives_up_on_a_search_still_paused_at_the_cap() {
        assert_eq!(
            after_request(true, Some("pause_turn"), 0),
            Step::Resume,
            "under the cap, resume"
        );
        assert_eq!(
            after_request(true, Some("pause_turn"), MAX_RESUMES - 1),
            Step::Resume
        );
        assert_eq!(
            after_request(true, Some("pause_turn"), MAX_RESUMES),
            Step::Stop(TurnEnd::SearchCapped),
            "the cap must end the turn as a give-up, not as a normal end"
        );
        assert_eq!(
            after_request(true, Some("end_turn"), 0),
            Step::Stop(TurnEnd::Completed)
        );
        assert_eq!(
            after_request(true, Some("tool_use"), 1),
            Step::Stop(TurnEnd::Completed)
        );
    }

    /// A stream that broke (a read error) or was closed early never reached
    /// `message_stop`, whatever it had already said.
    #[test]
    fn a_stream_that_never_reached_message_stop_is_a_failed_turn() {
        assert_eq!(after_request(false, None, 0), Step::Stop(TurnEnd::Failed));
        assert_eq!(
            after_request(false, Some("pause_turn"), 0),
            Step::Stop(TurnEnd::Failed),
            "a broken resume is a failure, not another resume"
        );
        assert_eq!(
            after_request(false, Some("end_turn"), 0),
            Step::Stop(TurnEnd::Failed)
        );
    }

    /// Replaying the turn without the tool is only safe before the first
    /// resume: after one, the TTS has already spoken request 1's frames and
    /// the caller would hear the opening line and the answer twice.
    #[test]
    fn an_org_refusal_is_only_replayed_before_the_first_resume() {
        assert!(may_replay_without_search(0, false));
        assert!(
            !may_replay_without_search(1, false),
            "a refusal on a resume must not replay spoken frames"
        );
        assert!(!may_replay_without_search(MAX_RESUMES, false));
        assert!(
            !may_replay_without_search(0, true),
            "once per turn — the stripped request must not loop"
        );
    }
}

#[cfg(test)]
mod network_tests {
    //! One real streamed turn: `cargo test -p voice-chatbot-server -- --ignored network_claude`.
    use super::*;

    fn search_cfg() -> SearchConfig {
        SearchConfig {
            tool: DEFAULT_SEARCH_TOOL.to_string(),
            max_uses: 3,
            user_location: crate::location::SearchLocation::parse(crate::location::DEFAULT)
                .unwrap()
                .unwrap()
                .user_location(),
        }
    }

    #[tokio::test]
    #[ignore]
    async fn network_claude_streams_a_short_reply() {
        crate::env_file::load_if_unset(std::path::Path::new("../../.env"));
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY in .env");
        let mut llm = ClaudeLlm::new(
            key,
            "claude-opus-5".into(),
            DEFAULT_EFFORT.into(),
            Some(search_cfg()),
        );
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

    /// The handover instruction now arrives as a suffix on the system prompt
    /// (`call::CLAUDE_SYSTEM_SUFFIX`) rather than as `ask_claude`'s tool
    /// result, and there is no `ask_claude` tool call in the context at all —
    /// Claude is never shown that tool. This checks the suffix alone is enough
    /// to make Claude answer the question directly instead of announcing a
    /// handover that, from Claude's point of view, never happened.
    #[tokio::test]
    #[ignore]
    async fn network_claude_answers_directly_with_the_handover_suffix() {
        crate::env_file::load_if_unset(std::path::Path::new("../../.env"));
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY in .env");
        let mut llm = ClaudeLlm::new(
            key,
            "claude-opus-5".into(),
            DEFAULT_EFFORT.into(),
            Some(search_cfg()),
        );
        let system = format!(
            "{}{}",
            include_str!("../prompt.babel.txt"),
            crate::call::CLAUDE_SYSTEM_SUFFIX
        );
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": system}),
                json!({"role": "assistant", "content": "Ready."}),
                json!({"role": "user", "content": "What is the capital of France?"}),
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
