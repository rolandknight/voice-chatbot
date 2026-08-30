# web-search — real search on both backends

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-08-29
**Status:** Not started. Decisions marked **[assumption]** are the author's calls and should be confirmed by the requester.
**Diagnosis this plan fixes:** "Use Claude to tell me what movies are playing at Cineplex Etobicoke" fired `ask_claude`, flipped the backend, and then Claude — handed Babel's own `web_search` skill — called it. That skill is DuckDuckGo's Instant Answer API, which returns `AbstractText: ""` and `RelatedTopics: []` for that query (verified live), so Claude apologised. Two tool calls was the handover working as designed; the defect is *which* search tool Claude was given, and that Babel hands one flat tool list to both backends.

**Goal:** Claude answers web questions with Anthropic's server-side `web_search`, and the local model's own `web_search` skill returns real results instead of nothing.

**Architecture:** Three seams, no new subsystems. (1) `ClaudeLlm::tools_json` becomes the point where Claude's advertised tool list diverges from the local model's — it drops `web_search` and `ask_claude` and adds Anthropic's server tool. (2) `ask_claude`'s handover prompt moves out of the tool-result channel and into a system-prompt suffix on the Claude branch, so Claude never carries history for a tool it can no longer call. (3) The local `web_search` skill defaults to Brave instead of DuckDuckGo.

**Tech stack:** Rust 2021, `reqwest` (raw HTTP against `POST /v1/messages` — there is no official Anthropic Rust SDK, and `claude-sdk-rs` is a Claude Code CLI subprocess wrapper with no token streaming, so it cannot feed the TTS stage). One new dependency: `async-stream`.

**Spec:** none — the design was agreed in conversation and is restated here in full. This document is self-contained.

## Global constraints

- **No environment variable may be prefixed `POC_`.** New and renamed variables use bare names, matching the repo's existing non-PoC convention (`SERVER_URL`, `LOG_LEVEL`, `INPUT_DEVICE`). Task 9 removes the remaining 56.
- **Brave is the default search provider.** `BRAVE_API_KEY` lives in `.env`. DuckDuckGo and Tavily stay selectable.
- **Default location:** `Toronto,Ontario,CA,America/Toronto`.
- **Claude search tool:** `web_search_20260209` (dynamic filtering), `max_uses` 3. Requires Opus 4.6+ / Sonnet 4.6+; `claude_model` already defaults to `claude-opus-5`.
- **Server-tool billing:** $10 per 1,000 searches plus token cost. Search results count as input tokens.
- **No test may hit the network** except the existing `#[ignore]`d ones.
- Every task ends green on `make check` (`cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`) — confirm the exact recipe with `make -n check` before relying on it.

## Facts the implementer will need

Verified against `https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool`:

- The server tool's `name` is literally `web_search` — the **same name** as Babel's skill. Both in one `tools` array is a collision; one must go.
- Response blocks are `server_tool_use` (the query) then `web_search_tool_result` (the results). Neither is a `tool_use` block and neither is ever dispatched client-side.
- Search results carry `encrypted_content`. **If** you echo the assistant turn back, it must be verbatim or the request 400s. Dropping the blocks entirely is legal — which is what Babel's OpenAI-shaped rolling context already does, since it stores only text and `tool_calls`. No context-shape change is needed, and the cost is that a follow-up question may re-search.
- A long search turn can end with `stop_reason: "pause_turn"`. Resuming means re-sending with the paused assistant turn appended verbatim and **no** extra "continue" user message.
- Search failures return **HTTP 200** with `content` as a single error object instead of a list. The one hard failure is org-level: if web search is disabled in the Console the whole request returns 400.
- If Claude emits a client tool call and a search in the same turn, the API returns `stop_reason: "tool_use"` and defers the search. Babel's context drops the `server_tool_use` block, so that deferred search silently never runs and Claude re-decides next turn. Accepted limitation.

---

### Task 1: `SearchLocation` — one household location, two consumers

Claude's `user_location` needs structured fields; Brave takes a `country`. `POC_WEATHER_DEFAULT_LOCATION` is a free-text string with its own IP-geolocation fallback and is left alone.

**Files:**
- Create: `crates/server/src/location.rs`
- Modify: `crates/server/src/main.rs` (add `mod location;` beside the other `mod` lines near the top)

**Interfaces:**
- Produces: `location::SearchLocation { city, region, country, timezone }`, `SearchLocation::parse(&str) -> Result<Option<SearchLocation>, String>`, `SearchLocation::user_location(&self) -> serde_json::Value`, `location::DEFAULT: &str`.

- [ ] **Step 1: Write the failing test**

Create `crates/server/src/location.rs` with only the test module and the `use` line, so the test compiles against nothing:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_four_fields() {
        let l = SearchLocation::parse("Toronto,Ontario,CA,America/Toronto")
            .unwrap()
            .unwrap();
        assert_eq!(l.city, "Toronto");
        assert_eq!(l.region, "Ontario");
        assert_eq!(l.country, "CA");
        assert_eq!(l.timezone, "America/Toronto");
    }

    #[test]
    fn the_default_parses() {
        assert!(SearchLocation::parse(DEFAULT).unwrap().is_some());
    }

    #[test]
    fn trims_fields_and_uppercases_the_country() {
        let l = SearchLocation::parse(" Toronto , Ontario , ca , America/Toronto ")
            .unwrap()
            .unwrap();
        assert_eq!(l.city, "Toronto");
        assert_eq!(l.country, "CA");
    }

    #[test]
    fn empty_means_no_location() {
        assert_eq!(SearchLocation::parse("").unwrap(), None);
        assert_eq!(SearchLocation::parse("   ").unwrap(), None);
    }

    #[test]
    fn rejects_the_wrong_field_count_and_a_bad_country() {
        assert!(SearchLocation::parse("Toronto,Ontario,CA").is_err());
        assert!(SearchLocation::parse("Toronto,Ontario,CA,America/Toronto,extra").is_err());
        assert!(SearchLocation::parse("Toronto,Ontario,Canada,America/Toronto").is_err());
    }

    #[test]
    fn renders_the_anthropic_user_location_object() {
        let l = SearchLocation::parse(DEFAULT).unwrap().unwrap();
        assert_eq!(
            l.user_location(),
            serde_json::json!({
                "type": "approximate",
                "city": "Toronto",
                "region": "Ontario",
                "country": "CA",
                "timezone": "America/Toronto"
            })
        );
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p voice-chatbot-server location::`
Expected: FAIL — `cannot find type SearchLocation in this scope` (the module is not yet declared in `main.rs`, so first add `mod location;` and re-run to get the real error).

- [ ] **Step 3: Write the implementation**

Put this above the test module in `crates/server/src/location.rs`:

```rust
//! `SEARCH_LOCATION` — where the household is, for search that depends on it.
//!
//! One setting feeds two consumers: Claude's server-side web search takes it as
//! a structured `user_location` object, and Brave takes its `country`. Without
//! it "what's on at Cineplex Etobicoke" is a query with no place attached, which
//! is how the DuckDuckGo era failed. `POC_WEATHER_DEFAULT_LOCATION` is
//! deliberately untouched: it is free text with its own geolocation fallback.

use serde_json::{json, Value};

/// `city,region,country,timezone`. Country is ISO 3166-1 alpha-2; timezone is
/// an IANA id. An empty setting means "send no location".
pub const DEFAULT: &str = "Toronto,Ontario,CA,America/Toronto";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchLocation {
    pub city: String,
    pub region: String,
    /// ISO 3166-1 alpha-2, uppercase. The API rejects anything else with a 400.
    pub country: String,
    /// IANA timezone id.
    pub timezone: String,
}

impl SearchLocation {
    /// Parsed once at startup, like every other config value here, so a typo is
    /// a boot failure and not a puzzling search result months later.
    pub fn parse(value: &str) -> Result<Option<Self>, String> {
        let value = value.trim();
        if value.is_empty() {
            return Ok(None);
        }
        let parts: Vec<&str> = value.split(',').map(str::trim).collect();
        let [city, region, country, timezone] = parts.as_slice() else {
            return Err(format!(
                "SEARCH_LOCATION must be city,region,country,timezone (got {} field(s) in {value:?}; e.g. {DEFAULT})",
                parts.len()
            ));
        };
        if country.len() != 2 || !country.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(format!(
                "SEARCH_LOCATION country must be a two-letter ISO 3166-1 alpha-2 code (got {country:?}; Canada is CA)"
            ));
        }
        Ok(Some(Self {
            city: city.to_string(),
            region: region.to_string(),
            country: country.to_ascii_uppercase(),
            timezone: timezone.to_string(),
        }))
    }

    /// The Messages API `user_location` object for the web search tool.
    pub fn user_location(&self) -> Value {
        json!({
            "type": "approximate",
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "timezone": self.timezone,
        })
    }
}
```

Add `mod location;` to `crates/server/src/main.rs` alongside the existing `mod` declarations (they sit around lines 15–22).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server location::`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/location.rs crates/server/src/main.rs
git commit -m "feat(server): parse SEARCH_LOCATION into a shared household location"
```

---

### Task 2: Brave becomes the default provider

`Provider::parse("")` currently yields DuckDuckGo, whose Instant Answer API is a disambiguation endpoint, not a web index. Brave replaces it as the default, gains the `country` filter, and returns titles as well as descriptions — three bare description fragments joined by spaces is thin material for a small model to read aloud.

**Files:**
- Modify: `crates/server/src/skills/web_search.rs`
- Modify: `crates/server/src/main.rs:1021-1026` (the `build_skills` construction)

**Interfaces:**
- Consumes: `location::SearchLocation` (Task 1).
- Produces: `WebSearch::new(provider: Provider, brave_key: String, tavily_key: String, country: Option<String>)` — a fourth parameter on an existing constructor.

- [ ] **Step 1: Write the failing tests**

In `crates/server/src/skills/web_search.rs`, replace the `provider_parsing` test and extend `brave_and_tavily_extraction`:

```rust
    #[test]
    fn provider_parsing_defaults_to_brave() {
        assert_eq!(Provider::parse("").unwrap(), Provider::Brave);
        assert_eq!(Provider::parse(" Brave ").unwrap(), Provider::Brave);
        assert_eq!(Provider::parse("duckduckgo").unwrap(), Provider::DuckDuckGo);
        assert_eq!(Provider::parse("tavily").unwrap(), Provider::Tavily);
        assert!(Provider::parse("bing").is_err());
    }

    #[test]
    fn brave_extraction_keeps_titles() {
        let brave = json!({"web": {"results": [
            {"title": "Showtimes", "description": "a"},
            {"title": "", "description": ""},
            {"description": "b"},
            {"title": "T", "description": "c"}
        ]}});
        assert_eq!(extract_brave(&brave), "Showtimes: a b");
    }
```

Note the `take(3)` in `extract_brave` means the fourth entry never appears; the second contributes nothing because its description is empty.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server web_search::`
Expected: FAIL — `provider_parsing_defaults_to_brave` asserts Brave where DuckDuckGo is returned; `brave_extraction_keeps_titles` expects `"Showtimes: a b"` and gets `"a b"`.

- [ ] **Step 3: Write the implementation**

In `crates/server/src/skills/web_search.rs`:

Change the doc comment at the top of the file:

```rust
//! `web_search` — Brave by default, DuckDuckGo or Tavily on request.
//!
//! Advertised to the **local** model only. Claude gets Anthropic's server-side
//! web search instead, which carries the same tool name (see `llm_claude.rs`).
```

Change the default arm of `Provider::parse`:

```rust
            "" | "brave" => Ok(Self::Brave),
            "duckduckgo" => Ok(Self::DuckDuckGo),
```

(keep the `"tavily"` arm and the error arm; update the error text to `"(expected brave, duckduckgo, or tavily)"`.)

Add the field and constructor parameter:

```rust
pub struct WebSearch {
    http: reqwest::Client,
    provider: Provider,
    brave_key: String,
    tavily_key: String,
    /// ISO 3166-1 alpha-2 from `SEARCH_LOCATION`, passed to Brave so results
    /// are local. `None` sends no country and lets Brave guess from the egress
    /// IP, which on a home server is usually right but not always.
    country: Option<String>,
}
```

```rust
    pub fn new(
        provider: Provider,
        brave_key: String,
        tavily_key: String,
        country: Option<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(HTTP_TIMEOUT)
                .build()
                .expect("reqwest client"),
            provider,
            brave_key,
            tavily_key,
            country,
        }
    }
```

Replace the Brave request's query construction:

```rust
                let mut params: Vec<(&str, &str)> = vec![("q", query), ("count", "3")];
                if let Some(c) = &self.country {
                    params.push(("country", c));
                }
                let data: Value = self
                    .http
                    .get("https://api.search.brave.com/res/v1/web/search")
                    .query(&params)
                    .header("Accept", "application/json")
                    .header("X-Subscription-Token", key)
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                extract_brave(&data)
```

Replace `extract_brave`:

```rust
/// Brave: up to three results as `Title: description`. The title carries the
/// venue or publication, which is often the part that answers the question.
fn extract_brave(data: &Value) -> String {
    data.pointer("/web/results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(3)
        .filter_map(|r| {
            let desc = text_field(r, "description")?;
            Some(match text_field(r, "title") {
                Some(title) => format!("{title}: {desc}"),
                None => desc.to_string(),
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}
```

Update the existing `#[ignore]`d network test to the new constructor and provider:

```rust
    #[tokio::test]
    #[ignore]
    async fn network_duckduckgo_instant_answer() {
        let s = WebSearch::new(
            Provider::DuckDuckGo,
            String::new(),
            String::new(),
            Some("CA".to_string()),
        );
```

In `crates/server/src/main.rs`, replace the `build_skills` block at lines 1021–1026:

```rust
    let provider = skills::web_search::Provider::parse(&env_or("WEB_SEARCH_PROVIDER", ""))?;
    let brave_key = env_or("BRAVE_API_KEY", "");
    if provider == skills::web_search::Provider::Brave && brave_key.trim().is_empty() {
        return Err("WEB_SEARCH_PROVIDER=brave (the default) needs BRAVE_API_KEY in .env — \
free tier at https://brave.com/search/api/. Set WEB_SEARCH_PROVIDER=duckduckgo to run without a key."
            .into());
    }
    list.push(Arc::new(skills::web_search::WebSearch::new(
        provider,
        brave_key,
        env_or("TAVILY_API_KEY", ""),
        search_location.as_ref().map(|l| l.country.clone()),
    )));
```

`build_skills` needs the location. It is called once, from `main`. Add this as its last parameter:

```rust
    search_location: Option<&location::SearchLocation>,
```

and at the call site pass `cfg.search_location.as_ref()` (the `PocConfig` field arrives in Task 5; until then, resolve it inline at the call site with `location::SearchLocation::parse(&env_or("SEARCH_LOCATION", location::DEFAULT))?.as_ref()` bound to a `let` first, since a temporary cannot be borrowed across the call). With a borrowed parameter the closure in the block above is `search_location.map(|l| l.country.clone())` — drop the `.as_ref()` shown there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server web_search::`
Expected: PASS. Then `cargo build --release -p voice-chatbot-server` to confirm the `main.rs` wiring compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/skills/web_search.rs crates/server/src/main.rs
git commit -m "feat(server): default web_search to Brave, localized by SEARCH_LOCATION"
```

---

### Task 3: Claude's tool list stops being the local model's

`node_tools` hands every skill to both backends (`session.rs:92` → `SwitchingLlm::set_tools`, `call.rs:294`). `ClaudeLlm::tools_json` already owns Claude's view of that list, so the divergence belongs there and nothing upstream changes.

**Files:**
- Modify: `crates/server/src/llm_claude.rs:63-84` (`tools_json`)

**Interfaces:**
- Produces: `HIDDEN_FROM_CLAUDE: [&str; 2]`.

- [ ] **Step 1: Write the failing test**

Add to the test module in `crates/server/src/llm_claude.rs`:

```rust
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p voice-chatbot-server claude_is_not_shown -- --nocapture`
Expected: FAIL — `assertion failed: left == right`, left is `["ask_claude", "get_weather", "web_search"]`.

- [ ] **Step 3: Write the implementation**

Add above `impl ClaudeLlm` in `crates/server/src/llm_claude.rs`:

```rust
/// Skills that must not reach Claude.
///
/// `web_search` because Anthropic's server-side tool carries the **same name**
/// — two tools called `web_search` in one request is a collision, and the
/// server-side one is the whole point of routing to Claude. `ask_claude`
/// because Claude is already answering: calling it re-sets a flag that is
/// already set and costs the caller a turn.
const HIDDEN_FROM_CLAUDE: [&str; 2] = ["web_search", "ask_claude"];
```

In `tools_json`, filter before mapping:

```rust
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
```

Leave the trailing `out.sort_by(...)` alone — a stable tool order is what keeps the prompt-cache prefix intact across turns.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server --lib llm_claude`
Expected: PASS, including the pre-existing `request_body_shape`.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/llm_claude.rs
git commit -m "feat(server): hide web_search and ask_claude from Claude's tool list"
```

---

### Task 4: the handover moves from a tool result to the system prompt

`HANDOVER` currently reaches Claude as `ask_claude`'s tool *result* (`skills/claude.rs:20`), which leaves a `tool_use` block in history for a tool Task 3 just stopped advertising. Moving it to a system-prompt suffix removes that mismatch, lets `strip_ask_claude` run unconditionally, and makes the brevity rule hold for every Claude turn rather than only the first. It also carries the preamble instruction that keeps the caller from sitting in silence while a search runs.

**Files:**
- Modify: `crates/server/src/call.rs:150-160` (`SwitchingLlm`, helpers), `:198-240` (`strip_ask_claude`), `:258-294` (`run_llm`)
- Modify: `crates/server/src/skills/claude.rs:20-42`

**Interfaces:**
- Produces: `call::CLAUDE_SYSTEM_SUFFIX`, `call::append_system_suffix(&mut LlmContext, &str)`.
- Consumes: `HIDDEN_FROM_CLAUDE` from Task 3 (conceptually — no code dependency).

- [ ] **Step 1: Write the failing tests**

Add to the test module in `crates/server/src/call.rs` (create one at the end of the file if absent, with `use super::*;`):

```rust
    fn ask_claude_ctx() -> flowcat_core::processor::frame::LlmContext {
        flowcat_core::processor::frame::LlmContext {
            messages: vec![
                serde_json::json!({"role": "system", "content": "Be Babel."}),
                serde_json::json!({"role": "user", "content": "Use Claude to find showtimes"}),
                serde_json::json!({"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "ask_claude", "arguments": "{}"}}
                ]}),
                serde_json::json!({"role": "tool", "tool_call_id": "c1", "content": "handover"}),
            ],
            tools: vec![],
        }
    }

    #[test]
    fn the_ask_claude_exchange_is_stripped_for_both_backends() {
        let mut messages = ask_claude_ctx().messages;
        strip_ask_claude(&mut messages);
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["system", "user"]);
    }

    #[test]
    fn the_claude_suffix_appends_to_the_existing_system_prompt() {
        let mut ctx = ask_claude_ctx();
        append_system_suffix(&mut ctx, CLAUDE_SYSTEM_SUFFIX);
        let system = ctx.messages[0]["content"].as_str().unwrap();
        assert!(system.starts_with("Be Babel."), "{system}");
        assert!(system.contains("answering as Claude"), "{system}");
        assert!(system.contains("let me check"), "{system}");
    }

    #[test]
    fn the_claude_suffix_creates_a_system_message_when_there_is_none() {
        let mut ctx = flowcat_core::processor::frame::LlmContext {
            messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
            tools: vec![],
        };
        append_system_suffix(&mut ctx, CLAUDE_SYSTEM_SUFFIX);
        assert_eq!(ctx.messages[0]["role"], "system");
        assert_eq!(ctx.messages[1]["role"], "user");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server --lib call::`
Expected: FAIL — `cannot find function append_system_suffix` and `cannot find value CLAUDE_SYSTEM_SUFFIX`.

- [ ] **Step 3: Write the implementation**

In `crates/server/src/call.rs`, add beside `with_system_prompt`:

```rust
/// Appended to the system prompt on the Claude branch.
///
/// This used to arrive as `ask_claude`'s tool *result*, which meant Claude's
/// history carried a call to a tool Claude is no longer shown — and meant the
/// brevity rule applied only to the turn immediately after the flip. As a
/// system suffix it holds for every Claude turn, and `strip_ask_claude` can run
/// unconditionally.
const CLAUDE_SYSTEM_SUFFIX: &str = "\n\nYou are now answering as Claude, on a live voice call. \
Answer the caller directly — never say you are handing over, and never ask them to repeat \
themselves. Keep replies to one or two short spoken sentences, and offer to go deeper if they \
want more. If you are going to search the web, say a short line first — \"let me check\" — so the \
caller is not sitting in silence while the search runs.";

/// Append `suffix` to `ctx`'s system message, creating one if there is none.
fn append_system_suffix(ctx: &mut flowcat_core::processor::frame::LlmContext, suffix: &str) {
    match ctx.messages.first_mut() {
        Some(first) if first.get("role").and_then(|r| r.as_str()) == Some("system") => {
            let base = first
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string();
            *first = serde_json::json!({"role": "system", "content": format!("{base}{suffix}")});
        }
        _ => ctx.messages.insert(
            0,
            serde_json::json!({"role": "system", "content": suffix.trim_start()}),
        ),
    }
}
```

Replace the context-rewrite block in `SwitchingLlm::run_llm` (currently `call.rs:274-291`) with:

```rust
        let backend = state.backend();
        let on_claude = claude.is_some() && backend == crate::skills::LlmBackend::Claude;
        let prompt = state.prompt();
        // The ask_claude exchange is hidden from *both* backends now: from the
        // local model because reading it back convinces it that it already is
        // Claude, and from Claude because it names a tool Claude is not shown.
        let rewrite = prompt.is_some() || on_claude || has_ask_claude(&ctx.messages);
        let ctx: &'a flowcat_core::processor::frame::LlmContext = if rewrite {
            *scratch = match &prompt {
                Some(p) => with_system_prompt(ctx, p),
                None => ctx.clone(),
            };
            if on_claude {
                append_system_suffix(scratch, CLAUDE_SYSTEM_SUFFIX);
            }
            strip_ask_claude(&mut scratch.messages);
            scratch
        } else {
            ctx
        };
```

Update the doc comment above `strip_ask_claude` (`call.rs:198-207`): delete the final sentence "Claude's spoken answers stay — only the call and its result go, and only on the branch that runs the local model." and replace it with:

```rust
/// Claude's spoken answers stay — only the call and its result go. It runs on
/// both branches: Claude receives the handover as a system-prompt suffix
/// (`CLAUDE_SYSTEM_SUFFIX`) instead, so it never sees a call to a tool that is
/// not in its own tool list.
```

In `crates/server/src/skills/claude.rs`, replace the `HANDOVER` constant and its doc comment. The tool result is now only ever read by the *local* model in the window before the flip takes effect, and by nothing at all once `strip_ask_claude` removes it — so it becomes a plain confirmation again:

```rust
/// What the tool returns. Since the handover instruction moved to
/// `call::CLAUDE_SYSTEM_SUFFIX`, this string is stripped from the rolling
/// context before either backend's next turn (`call::strip_ask_claude`) and is
/// never read by a model. It stays non-empty because the tool contract requires
/// a spoken-friendly string, and it shows up in the `tool return` log line.
const HANDOVER: &str = "Switched to Claude.";
```

Change its visibility from `pub(crate)` to private and drop the now-unused import in `call.rs` if one exists (`grep -n 'llm_claude::HANDOVER\|claude::HANDOVER' crates/server/src` to check).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server --lib call::`
Expected: PASS, including any pre-existing `strip_ask_claude` tests.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/call.rs crates/server/src/skills/claude.rs
git commit -m "feat(server): deliver the Claude handover as a system-prompt suffix"
```

---

### Task 5: declare Anthropic's server-side web search

**Files:**
- Modify: `crates/server/src/llm_claude.rs` (`ClaudeLlm`, `new`, `tools_json`)
- Modify: `crates/server/src/call.rs:464-470` (the `ClaudeLlm::new` call site)
- Modify: `crates/server/src/main.rs` (config fields and defaults)

**Interfaces:**
- Consumes: `location::SearchLocation` (Task 1), `HIDDEN_FROM_CLAUDE` (Task 3).
- Produces: `llm_claude::SearchConfig { tool: String, max_uses: u32, user_location: Value }`, `ClaudeLlm::new(api_key, model, effort, search: Option<SearchConfig>)`.

- [ ] **Step 1: Write the failing test**

```rust
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
        assert_eq!(tools.len(), 1, "the local skill must not survive: {tools:?}");
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server --lib llm_claude`
Expected: FAIL — `ClaudeLlm::new` takes 3 arguments, `SearchConfig` and `DEFAULT_SEARCH_TOOL` are undefined.

- [ ] **Step 3: Write the implementation**

In `crates/server/src/llm_claude.rs`, add near `DEFAULT_EFFORT`:

```rust
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
```

Add the field to `ClaudeLlm` and the parameter to `new`:

```rust
pub struct ClaudeLlm {
    http: reqwest::Client,
    api_key: String,
    model: String,
    /// `output_config.effort`; empty omits the field.
    effort: String,
    tools: Vec<Tool>,
    /// Anthropic's server-side web search, when configured.
    search: Option<SearchConfig>,
}

impl ClaudeLlm {
    pub fn new(
        api_key: String,
        model: String,
        effort: String,
        search: Option<SearchConfig>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            model,
            effort,
            tools: Vec::new(),
            search,
        }
    }
```

In `tools_json`, push the server tool into `out` **before** the existing `out.sort_by(...)`, so the array stays deterministic for the prompt cache:

```rust
        if let Some(s) = &self.search {
            let mut tool = json!({ "type": s.tool, "name": "web_search" });
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
```

Update every other `ClaudeLlm::new` call site to pass a fourth argument: `None` in `request_body_shape` (~line 552), `empty_effort_omits_output_config` (~586) and `claude_is_not_shown_web_search_or_ask_claude` (added in Task 3), and `Some(search_cfg())` in the two `#[ignore]`d network tests (~667, ~697) so a manual run exercises the real thing. `grep -n 'ClaudeLlm::new' crates/server/src` to catch them all.

In `crates/server/src/main.rs`, add to `PocConfig` beside `claude_effort`:

```rust
    /// Anthropic's server-side web search on the Claude turns: tool type, per-turn
    /// cap, and whether it is declared at all.
    pub claude_web_search: bool,
    pub claude_search_tool: String,
    pub claude_search_max_uses: u32,
    /// `SEARCH_LOCATION`, shared with the Brave provider.
    pub search_location: Option<crate::location::SearchLocation>,
```

and to the `PocConfig` literal:

```rust
        claude_web_search: env_or("CLAUDE_WEB_SEARCH", "true").trim() != "false",
        claude_search_tool: env_or("CLAUDE_SEARCH_TOOL", llm_claude::DEFAULT_SEARCH_TOOL),
        claude_search_max_uses: env_or(
            "CLAUDE_SEARCH_MAX_USES",
            &llm_claude::DEFAULT_SEARCH_MAX_USES.to_string(),
        )
        .parse()
        .map_err(|e| format!("invalid CLAUDE_SEARCH_MAX_USES: {e}"))?,
        search_location: location::SearchLocation::parse(&env_or(
            "SEARCH_LOCATION",
            location::DEFAULT,
        ))?,
```

**Remove Task 2's stop-gap.** Task 2 resolved `SEARCH_LOCATION` inline at the `build_skills` call site because `PocConfig` had nowhere to put it yet. That parse is now duplicated — delete the inline `SearchLocation::parse(...)` binding and pass `cfg.search_location.as_ref()` instead. `SEARCH_LOCATION` must be read in exactly one place.

In `crates/server/src/call.rs`, extend the `ClaudeLlm::new` call at line 464:

```rust
    let claude = (!cfg.anthropic_key.trim().is_empty()).then(|| {
        let search = cfg.claude_web_search.then(|| crate::llm_claude::SearchConfig {
            tool: cfg.claude_search_tool.clone(),
            max_uses: cfg.claude_search_max_uses,
            user_location: cfg
                .search_location
                .as_ref()
                .map(|l| l.user_location())
                .unwrap_or(serde_json::Value::Null),
        });
        crate::llm_claude::ClaudeLlm::new(
            cfg.anthropic_key.clone(),
            cfg.claude_model.clone(),
            cfg.claude_effort.clone(),
            search,
        )
    });
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server --lib` then `cargo build --release -p voice-chatbot-server`
Expected: PASS and a clean build.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/llm_claude.rs crates/server/src/call.rs crates/server/src/main.rs
git commit -m "feat(server): give the Claude backend Anthropic's server-side web search"
```

---

### Task 6: rebuild the assistant's raw content blocks from the stream

Two things need this. `server_tool_use` must be proven never to become a dispatchable `FunctionCall` — if it ever did, Babel would try to run a nonexistent local skill. And `pause_turn` (Task 7) can only be resumed by echoing the assistant turn back verbatim, `encrypted_content` intact.

**Files:**
- Modify: `crates/server/src/llm_claude.rs` (`Folder` struct, `new`, `event`, and the `frames_of` test helper)

**Interfaces:**
- Produces: `Folder::assistant_blocks(&self) -> Vec<Value>`, `Folder.block_index_offset: u64`.

- [ ] **Step 1: Write the failing test**

```rust
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
        assert_eq!(blocks[1]["input"], json!({"query": "cineplex etobicoke showtimes"}));
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
        assert_eq!(blocks.len(), 4, "index restarts at 0 per request: {blocks:?}");
        assert_eq!(blocks[3], json!({"type": "text", "text": "Nothing showing."}));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server --lib llm_claude`
Expected: FAIL — `no method named assistant_blocks`, `no method named begin_request`.

- [ ] **Step 3: Write the implementation**

Add to the `Folder` struct in `crates/server/src/llm_claude.rs`:

```rust
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
```

Initialise them in `Folder::new` (`blocks: Default::default(), block_index_offset: 0, block_input: Default::default(), request_done: false`).

Add the methods:

```rust
    fn abs(&self, idx: u64) -> u64 {
        self.block_index_offset + idx
    }

    /// Start another request in the same logical turn (a `pause_turn` resume).
    /// Blocks and timings carry over; per-request state does not.
    fn begin_request(&mut self) {
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
```

In `event`, extend the three block arms. `content_block_start`:

```rust
            "content_block_start" => {
                let idx = ev["index"].as_u64().unwrap_or(0);
                let block = &ev["content_block"];
                self.blocks.insert(self.abs(idx), block.clone());
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
```

Note the `tool_blocks` key stays the **relative** index — it is per-request state cleared by `begin_request`, and the existing `content_block_stop` arm looks it up by relative index.

In `content_block_delta`, mirror each delta into `blocks` and add the two new delta types:

```rust
                    "text_delta" => {
                        if let Some(t) = delta["text"].as_str().filter(|t| !t.is_empty()) {
                            let now = Instant::now();
                            self.first_output_at.get_or_insert(now);
                            self.first_text_at.get_or_insert(now);
                            self.pending.push_back(Frame::LlmText(t.to_string()));
                            append_str(&mut self.blocks, self.block_index_offset + idx, "text", t);
                        }
                    }
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
```

with this free function beside `content_string`:

```rust
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
```

At the top of the `content_block_stop` arm, materialise the accumulated input:

```rust
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
```

That `if let` block is the existing code, moved down unchanged; only the `block_input` materialisation above it is new.

Change `message_stop` from `self.finish()` to `self.request_done = true`, and the `"error"` arm to set both `self.request_done = true` and keep its warn (the driver in Task 7 calls `finish()`). Update the `frames_of` test helper so the existing four tests keep seeing the end-of-turn frames:

```rust
    fn frames_of(lines: &[&str]) -> Vec<Frame> {
        let mut f = Folder::new("m".into());
        f.feed((lines.join("\n") + "\n").as_bytes());
        f.finish();
        f.pending.drain(..).collect()
    }
```

and add the same `f.finish();` before the drain in `folds_text_and_tool_use_stream_into_frames`, which builds its `Folder` inline.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server --lib llm_claude`
Expected: PASS, all pre-existing fold tests included.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/llm_claude.rs
git commit -m "feat(server): rebuild Claude's raw content blocks from the stream"
```

---

### Task 7: resume a paused search turn

Without this, a `pause_turn` ends the turn wherever the search happened to be — after the spoken preamble, with no answer behind it.

**Files:**
- Modify: `crates/server/Cargo.toml` (add `async-stream`)
- Modify: `crates/server/src/llm_claude.rs` (`run_llm`, `sse_to_frames` removal, `send`, `resume_body`)

**Interfaces:**
- Consumes: `Folder::assistant_blocks`, `Folder::begin_request` (Task 6).
- Produces: `resume_body(&Value, Vec<Value>) -> Value`, `MAX_RESUMES: u32`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn a_resume_body_appends_the_paused_turn_and_nothing_else() {
        let body = json!({
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "showtimes?"}]
        });
        let blocks = vec![json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"})];
        let next = resume_body(&body, blocks.clone());
        let msgs = next["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1], json!({"role": "assistant", "content": blocks}));
        assert_eq!(
            next["model"], body["model"],
            "everything else must be unchanged"
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server --lib llm_claude`
Expected: FAIL — `cannot find function resume_body`.

- [ ] **Step 3: Write the implementation**

Add to `crates/server/Cargo.toml` under `[dependencies]`:

```toml
# A `pause_turn` makes one logical turn span several HTTP requests; a plain
# `stream::unfold` over that becomes a hand-rolled state machine.
async-stream = "0.3"
```

In `crates/server/src/llm_claude.rs`, add near `MAX_TOKENS`:

```rust
/// How many times a `pause_turn` may be resumed within one turn. The API pauses
/// a long server-side search loop rather than running it forever; each resume
/// is another request, so the cap bounds both spend and the caller's silence.
const MAX_RESUMES: u32 = 2;
```

Add beside the other free functions:

```rust
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
```

Add the request helper to `impl ClaudeLlm` — it takes `&self` so the returned stream can call it again for a resume:

```rust
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
```

Replace `run_llm` and delete `sse_to_frames` entirely (nothing else calls it):

```rust
    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let mut body = self.request_body(ctx)?;
        let this: &ClaudeLlm = self;
        Ok(async_stream::stream! {
            let mut folder = Folder::new(this.model.clone());
            let mut resumes = 0u32;
            loop {
                let stream = match this.send(&body).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "claude: request failed");
                        break;
                    }
                };
                futures::pin_mut!(stream);
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(b) => folder.feed(b.as_ref()),
                        Err(e) => {
                            tracing::warn!(error = %e, "claude: stream read failed");
                            break;
                        }
                    }
                    while let Some(f) = folder.pending.pop_front() {
                        yield f;
                    }
                }
                while let Some(f) = folder.pending.pop_front() {
                    yield f;
                }
                if folder.stop_reason.as_deref() == Some("pause_turn") && resumes < MAX_RESUMES {
                    resumes += 1;
                    tracing::info!(resumes, "claude: resuming a paused search turn");
                    body = resume_body(&body, folder.assistant_blocks());
                    folder.begin_request();
                    continue;
                }
                break;
            }
            folder.finish();
            while let Some(f) = folder.pending.pop_front() {
                yield f;
            }
        }
        .boxed())
    }
```

`finish()` already handles the case where a paused turn produced no speakable text: it warns and speaks `EMPTY_TURN_FALLBACK`, so exhausting `MAX_RESUMES` never leaves the caller in silence.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server --lib llm_claude` then `cargo clippy -p voice-chatbot-server --all-targets -- -D warnings`
Expected: PASS and no clippy warnings (watch for an unused-import warning where `sse_to_frames` used `StreamExt`).

- [ ] **Step 5: Commit**

```bash
git add crates/server/Cargo.toml Cargo.lock crates/server/src/llm_claude.rs
git commit -m "feat(server): resume a Claude turn paused mid-search"
```

---

### Task 8: search errors, an org-level kill switch, and search spend in the log

Two failure modes the caller must not hit. A search that fails returns **HTTP 200** with `content` as an error object where a list belongs — Claude reads it and speaks to it, so all Babel owes it is a log line. But if web search is disabled for the organisation in the Console, the whole request 400s and every Claude turn dies until someone notices.

**Files:**
- Modify: `crates/server/src/llm_claude.rs` (`ClaudeLlm`, `tools_json`, `event`, `finish`, `run_llm`)

**Interfaces:**
- Consumes: `SendFail` (Task 7), `SearchConfig` (Task 5).
- Produces: `is_web_search_disabled(status: u16, body: &str) -> bool`, `ClaudeLlm.search_disabled: AtomicBool`.

- [ ] **Step 1: Write the failing tests**

```rust
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
            llm.request_body(&plain_ctx()).unwrap().get("tools").is_none(),
            "the tool must not come back on the retry"
        );
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server --lib llm_claude`
Expected: FAIL — `cannot find function is_web_search_disabled`, `no method named disable_search`.

- [ ] **Step 3: Write the implementation**

Add the field to `ClaudeLlm` (initialise to `false` in `new`):

```rust
    /// Set when the API rejects the request because web search is switched off
    /// for the organisation. Without it every Claude turn would fail; with it
    /// the turn is retried once without the tool and the process carries on
    /// unsearched. `AtomicBool` because the stream holds `&self`.
    search_disabled: std::sync::atomic::AtomicBool,
```

```rust
    pub fn disable_search(&self) {
        self.search_disabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
```

Guard the server tool in `tools_json`:

```rust
        let searching = !self
            .search_disabled
            .load(std::sync::atomic::Ordering::Relaxed);
        if let (Some(s), true) = (&self.search, searching) {
```

Add the predicate:

```rust
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
```

Add the body rewriter. `request_body` cannot be called from inside the stream (it returns a `Result` and the stream cannot use `?`), so the retry edits the body it already has. A server tool is the only entry in `tools` carrying a `type` field — client tools carry `name`/`description`/`input_schema` — which makes the filter a one-liner:

```rust
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
```

Now wire the retry into `run_llm`. Declare `let mut retried_without_search = false;` beside `let mut resumes = 0u32;`, and replace the single `Err(e)` arm of the `this.send(&body).await` match with two arms:

```rust
                    Err(SendFail::Http { status, body: err })
                        if is_web_search_disabled(status, &err) && !retried_without_search =>
                    {
                        tracing::error!(
                            "claude: web search is disabled for this organisation — enable it at \
https://platform.claude.com/settings/privacy, or set CLAUDE_WEB_SEARCH=false to stop asking. \
Retrying this turn without it."
                        );
                        this.disable_search();
                        retried_without_search = true;
                        body = strip_server_search(&body);
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "claude: request failed");
                        break;
                    }
```

The `body:` binding is renamed to `err` because `body` is already the request being sent. `disable_search()` keeps the tool off for the rest of the process, so the next turn skips straight past this.

Add a test for the rewriter alongside the others:

```rust
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
```

Finally, count searches. Add `search_requests: u64` to `Folder` (initialised to 0), read it in the `message_delta` arm:

```rust
                if let Some(n) = ev["usage"]["server_tool_use"]["web_search_requests"].as_u64() {
                    self.search_requests += n;
                }
```

and add `search_requests = self.search_requests,` to the existing `"claude turn"` `tracing::info!` in `finish()`, so search spend is visible in the same line as tokens and TTFT.

Log a failed result block in the `content_block_start` arm:

```rust
                if block["type"] == "web_search_tool_result" && block["content"].is_object() {
                    tracing::warn!(
                        error_code = %block["content"]["error_code"],
                        "claude: web search returned an error"
                    );
                }
```

(On success `content` is a list; on failure it is a single object — branch on that, never on indexing.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server --lib llm_claude` then `cargo clippy -p voice-chatbot-server --all-targets -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/llm_claude.rs
git commit -m "feat(server): survive an org-level web search refusal and log search spend"
```

---

### Task 9: retire the `POC_` prefix

56 variables still carry a prefix from PoC trees that were archived. A half-renamed config is worse than either end state, so this sweeps the rest and adds a guard that turns a stale `.env` into a boot failure instead of a silent revert to defaults.

**Files:**
- Modify: `crates/server/src/main.rs`, `crates/server/src/llm_claude.rs`, `crates/server/src/ollama_serve.rs`, `crates/server/src/tts_qwen.rs`, `crates/server/src/skills/sfx.rs`, `crates/server/src/skills/web_search.rs`, `crates/wake/src/lib.rs`, `crates/qwen-tts/build.rs`, `crates/qwen-tts/src/engine.rs`, `crates/qwen-tts/python/qwen_tts/config.py`, `crates/qwen-tts/python/tests/test_config.py`, `qwen-tts-tester/tests/e2e_ws.py`
- Modify: `.env.example`, `README.md`, `crates/client/README.md`, `qwen-tts-tester/README.md`, `docs/adr/0005-nemotron-streaming-stt.md`, `docs/research/speaker-recognition.md`
- Do **not** touch: `archive/**` (frozen PoC trees), `docs/superpowers/plans/**` and `docs/poc/**` (historical records of what the names were at the time)

- [ ] **Step 1: Write the failing test**

Add **only this test module** to `crates/server/src/env_file.rs` — the implementation it calls arrives in Step 3, which is what makes Step 2 fail:

```rust
#[cfg(test)]
mod retired_tests {
    use super::*;

    #[test]
    fn flags_only_the_retired_prefix() {
        let keys = [
            "POC_STT_BACKEND",
            "SERVER_URL",
            "POC_LLM_MODEL",
            "BRAVE_API_KEY",
        ]
        .into_iter()
        .map(String::from);
        assert_eq!(
            retired_names(keys),
            vec!["POC_LLM_MODEL".to_string(), "POC_STT_BACKEND".to_string()]
        );
    }

    #[test]
    fn an_environment_without_them_is_clean() {
        assert!(retired_names(["SERVER_URL".to_string()].into_iter()).is_empty());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p voice-chatbot-server --lib env_file`
Expected: FAIL — `cannot find function retired_names`.

- [ ] **Step 3: Do the rename**

First add the implementation the test calls, to `crates/server/src/env_file.rs`:

```rust
/// Names dropped when the `POC_` prefix was retired. A stale `.env` would
/// otherwise be read as "unset" and silently take the defaults — on a home
/// server that surfaces days later as the wrong STT model or a missing skill.
pub const RETIRED_PREFIX: &str = "POC_";

/// Every `POC_*` name found in the environment, so startup can refuse to run.
pub fn retired_names<I: Iterator<Item = String>>(keys: I) -> Vec<String> {
    let mut found: Vec<String> = keys.filter(|k| k.starts_with(RETIRED_PREFIX)).collect();
    found.sort();
    found
}
```

Then apply this mapping. Every name loses `POC_`; the six that would collide with something generic or already-used keep a qualifier.

| Old | New | | Old | New |
|---|---|---|---|---|
| `POC_ADVERTISE_IP` | `ADVERTISE_IP` | | `POC_QWEN_CONFIG` | `QWEN_CONFIG` |
| `POC_BIND` | `BIND` | | `POC_QWEN_INTERVAL_S` | `QWEN_INTERVAL_S` |
| `POC_CHATTERBOX_URL` | `CHATTERBOX_URL` | | `POC_QWEN_SERVER_PORT` | `QWEN_SERVER_PORT` |
| `POC_CHATTERBOX_VOICE` | `CHATTERBOX_VOICE` | | `POC_QWEN_SIZE` | `QWEN_SIZE` |
| `POC_CLAUDE_EFFORT` | `CLAUDE_EFFORT` | | `POC_QWEN_VOICE` | `QWEN_VOICE` |
| `POC_CLAUDE_MODEL` | `CLAUDE_MODEL` | | `POC_QWEN_VOICES` | `QWEN_VOICES` |
| `POC_GREETING_WAV` | `GREETING_WAV` | | `POC_SFX_BACKEND` | `SFX_BACKEND` |
| `POC_KOKORO_URL` | `KOKORO_URL` | | `POC_SFX_SAO_URL` | `SFX_SAO_URL` |
| `POC_KOKORO_VOICE` | `KOKORO_VOICE` | | `POC_SFX_WOOSH_URL` | `SFX_WOOSH_URL` |
| `POC_LLM_MODEL` | `LLM_MODEL` | | `POC_SKILLS_RADIO` | `SKILLS_RADIO` |
| `POC_LLM_NUM_CTX` | `LLM_NUM_CTX` | | `POC_SKILLS_SFX` | `SKILLS_SFX` |
| `POC_LLM_PROVIDER` | `LLM_PROVIDER` | | `POC_SKILLS_SHOWS` | `SKILLS_SHOWS` |
| `POC_MOONSHINE_KEYTERMS` | `MOONSHINE_KEYTERMS` | | `POC_SKILLS_SPOTIFY` | `SKILLS_SPOTIFY` |
| `POC_MOONSHINE_MODEL` | `MOONSHINE_MODEL` | | `POC_STT_BACKEND` | `STT_BACKEND` |
| `POC_MOONSHINE_UPDATE_INTERVAL_MS` | `MOONSHINE_UPDATE_INTERVAL_MS` | | `POC_TLS_BIND` | `TLS_BIND` |
| `POC_NEMOTRON_DEVICE` | `NEMOTRON_DEVICE` | | `POC_TLS_CERT` | `TLS_CERT` |
| `POC_NEMOTRON_MODEL` | `NEMOTRON_MODEL` | | `POC_TLS_KEY` | `TLS_KEY` |
| `POC_NEMOTRON_RIGHT_CONTEXT` | `NEMOTRON_RIGHT_CONTEXT` | | `POC_TTS_BACKEND` | `TTS_BACKEND` |
| `POC_NEMOTRON_SPEECH_CONTEXTS` | `NEMOTRON_SPEECH_CONTEXTS` | | `POC_VAD_MODEL` | `VAD_MODEL` |
| `POC_NEMOTRON_URL` | `NEMOTRON_URL` | | `POC_VAD_STOP_SECS` | `VAD_STOP_SECS` |
| `POC_OLLAMA_BIN` | `OLLAMA_BIN` | | `POC_WAKE_DIR` | `WAKE_DIR` |
| `POC_OLLAMA_HOST` | `OLLAMA_HOST` | | `POC_WAKE_GRACE_SECS` | `WAKE_GRACE_SECS` |
| `POC_OLLAMA_KEEPWARM_SECS` | `OLLAMA_KEEPWARM_SECS` | | `POC_WAKE_MODEL` | `WAKE_MODEL` |
| `POC_OLLAMA_SUPERVISE` | `OLLAMA_SUPERVISE` | | `POC_WAKE_SESSION_SECS` | `WAKE_SESSION_SECS` |
| `POC_OLLAMA_UNLOAD_ON_EXIT` | `OLLAMA_UNLOAD_ON_EXIT` | | `POC_WAKE_THRESHOLD` | `WAKE_THRESHOLD` |
| `POC_PROMPT` | `PROMPT_FILE` | | `POC_WEATHER_DEFAULT_LOCATION` | `WEATHER_DEFAULT_LOCATION` |
| `POC_PYTHON` | `QWEN_PYTHON` | | `POC_WHISPER_MODEL` | `WHISPER_MODEL` |
| | | | `POC_WHISPER_THREADS` | `WHISPER_THREADS` |

`POC_WEB_SEARCH_PROVIDER` → `WEB_SEARCH_PROVIDER` was already done in Task 2. `POC_PROMPT` becomes `PROMPT_FILE` because a bare `PROMPT` is too easy to collide with, and `POC_PYTHON` becomes `QWEN_PYTHON` to match the Makefile variable of that name.

Mechanical pass, then read the diff:

```bash
git ls-files -z \
  ':!archive/**' ':!docs/superpowers/plans/**' ':!docs/poc/**' \
  ':!crates/server/src/env_file.rs' ':!docs/plans/web-search.md' \
  | xargs -0 grep -lZ 'POC_' \
  | xargs -0 sed -i \
      -e 's/POC_PROMPT\b/PROMPT_FILE/g' \
      -e 's/POC_PYTHON\b/QWEN_PYTHON/g' \
      -e 's/\bPOC_//g'
git diff --stat
```

Two things about that command. The order of the `-e` expressions matters: the two renamed-with-a-qualifier substitutions must run before the blanket strip, or `POC_PROMPT` becomes `PROMPT` and never reaches the first rule. And `env_file.rs` is excluded deliberately — it is the one file that must keep the literal string `POC_`, in `RETIRED_PREFIX` and in the test's fixture names; letting the sed through it would rewrite the guard into checking for a prefix that no longer exists. This plan file is excluded for the same reason.

Then check nothing outside the intended set moved:

```bash
git grep -n 'POC_' -- ':!archive/**' ':!docs/superpowers/plans/**' ':!docs/poc/**' ':!docs/plans/web-search.md'
```

Expected: only `crates/server/src/env_file.rs`.

Add the guard to `main.rs`, immediately after the `.env` load and before `PocConfig` is built:

```rust
    let retired = env_file::retired_names(std::env::vars().map(|(k, _)| k));
    if !retired.is_empty() {
        return Err(format!(
            "these environment variables lost their POC_ prefix and are no longer read: {}. \
Rename them in .env (drop POC_; POC_PROMPT is now PROMPT_FILE and POC_PYTHON is now QWEN_PYTHON) \
— leaving them set means silently running on the defaults.",
            retired.join(", ")
        )
        .into());
    }
```

Update `.env.example` in the same pass, and add the three new settings to it:

```bash
# Web search. brave is the default and needs a key (free tier at
# https://brave.com/search/api/); duckduckgo needs none but its Instant Answer
# API returns nothing for most real questions; tavily needs TAVILY_API_KEY.
# WEB_SEARCH_PROVIDER=brave
BRAVE_API_KEY=
# Where the household is: city,region,ISO-3166-1-alpha-2,IANA-timezone.
# Feeds Claude's server-side web search and Brave's country filter.
# SEARCH_LOCATION=Toronto,Ontario,CA,America/Toronto
# Anthropic's server-side web search on the Claude turns ($10 per 1,000
# searches plus tokens). CLAUDE_SEARCH_TOOL=web_search_20250305 is the faster
# path for models older than Opus 4.6.
# CLAUDE_WEB_SEARCH=true
# CLAUDE_SEARCH_TOOL=web_search_20260209
# CLAUDE_SEARCH_MAX_USES=3
```

- [ ] **Step 4: Run the full suite**

Run: `cargo test -p voice-chatbot-server && cargo build --release -p voice-chatbot-server && make check`
Then start the server with a deliberately stale variable and confirm it refuses:

```bash
POC_STT_BACKEND=nemotron ./target/release/voice-chatbot-server
```

Expected: exits with the rename message naming `POC_STT_BACKEND`.

- [ ] **Step 5: Commit**

Stage only tracked files that actually changed — the working tree carries untracked wakeword models and an unrelated `Makefile` edit that must not ride along:

```bash
git add -u                              # tracked files only — never -A
git status --short                      # review what is staged
git restore --staged Makefile           # only if it still carries the unrelated edit
git commit -m "refactor: drop the POC_ prefix from every environment variable"
```

---

## Verification

Unit tests cover the pure seams only. Before calling this done, make one real call:

1. `make server` with `BRAVE_API_KEY` and `ANTHROPIC_API_KEY` in `.env`. Confirm the startup log's `skills loaded` line still advertises `web_search`.
2. `make call`, then: *"What's the weather in Toronto?"* — the local model calls `web_search`, and the answer should contain real content, not "didn't get useful results".
3. *"Use Claude to tell me what movies are playing at Cineplex Etobicoke."* Expect, in the server log: `tool invoke tool=ask_claude` → **one** `claude turn` line with `search_requests=1` or more → a spoken answer naming actual films. There should be **no** `tool invoke tool=web_search` after the handover — that is the bug this plan closes.
4. Ask a follow-up in the same wake session (*"what time is the first one?"*) and confirm Claude answers rather than erroring — this is the path where the search blocks were dropped from the rolling context.

## Known limitations, accepted

- **Search results do not persist across turns.** The rolling context stores text and `tool_calls` only, so `web_search_tool_result` blocks are dropped between turns. Legal (the verbatim-echo rule binds only if you include them) and it keeps the context shape unchanged, but a follow-up may re-search at $0.01 a time.
- **A client tool and a search in the same turn loses the search.** The API returns `stop_reason: "tool_use"` and defers the search to the next request, which needs the `server_tool_use` block echoed back — and Babel drops it. Claude re-decides on the next turn. Not worth a context-shape change until it is seen in practice.
- **`is_web_search_disabled` matches on error text.** There is no distinct error code for the org-level switch. If Anthropic rewords the message the retry stops firing and Claude turns fail outright — the `tracing::error!` is what will point at this.
- **Three LLM turns for "Use Claude to…"** (local decides → Claude decides → Claude answers) is untouched. It is real latency and a separate change.
