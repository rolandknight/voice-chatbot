# FlowCat PoC: own the LLM lifecycle (native Ollama service + serve supervisor)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **2026-08-25 — ADR-0007 decision: Ollama stays for this phase; `llama-server` is deferred.** This plan is executed as written against Ollama (native `/api/chat` service so residency and context are per-request and independent of serve env; in-binary supervisor). When the llama-server phase opens, Layer 2 and the startup warm retarget to it and Layer 1's native service remains for the `ollama` profile.

**Goal:** The Rust chatbot (`poc/flowcat`, the future implementation) makes its LLM correct and resident by itself: at start-up it ensures an Ollama serve, loads gemma4, warms the exact prompt prefix and pins the model; while it runs the model stays resident regardless of how serve was started; on exit the model is unloaded so the ~17 GB returns. No Makefile/Python glue in the loop (`ollama_ctl.py` retires), and the Ollama.app / env / context-length failure modes found on 2026-08-25 become impossible by construction.

**Architecture:** Two layers behind one config block (`POC_LLM_*` env, mirroring the existing `POC_QWEN_*` pattern).

1. **Layer 1 — native Ollama `LlmService` (`poc/flowcat/src/llm_ollama.rs`).** Streams `POST /api/chat` (NDJSON) instead of `/v1/chat/completions`, so every request carries `keep_alive`, `options.num_ctx`, `think: false`, and the sorted tool list. The service also exposes `warm()` (the same request body with only the system message, `num_predict: 1`) and `unload()` (`keep_alive: 0`), and reads `prompt_eval_count/duration` + `eval_count/duration` from the final chunk into `Frame::Metrics(LlmUsage)` so the run records a prompt-cache hit the way `poc-gemma4` measures it. The OpenRouter/OpenAI path stays selectable (`POC_LLM_PROVIDER=openrouter`) for the cloud profile.
2. **Layer 2 — serve supervisor (`poc/flowcat/src/ollama_serve.rs`).** At start-up: if nothing answers on the base URL, spawn `ollama serve` as a child (env `OLLAMA_KEEP_ALIVE=-1`, logs to `poc/logs/ollama.log`), wait for `/api/tags`; if something answers, verify it after the warm (`/api/ps`: model present, `context_length == num_ctx`) and just use it — with the native API the serve's env no longer matters. On shutdown (SIGTERM/SIGINT/ctrl-c): unload (Layer 1), then terminate the child if we spawned it. Config: `POC_OLLAMA_SUPERVISE=auto|never|always`, `POC_OLLAMA_UNLOAD_ON_EXIT=true|false` (dev convenience: keep the model across chatbot restarts).

**Tech Stack:** Rust (mise 1.97.1), reqwest (already a dependency; NDJSON via `bytes_stream` + line splitting), tokio `process` + `signal`, serde_json. Ollama 0.32.5 (brew) — `/api/chat` streaming with tools, `keep_alive`, `options.num_ctx`, `think` (all present since 0.9/0.12; `prompt_eval_count` includes cached tokens since 0.32, so cache hits are judged on `prompt_eval_duration`).

**Prior art in this repo:** `poc-gemma4/poc_gemma4/ollama.py` (native `/api/chat` client: request shape incl. `keep_alive` as an int, `think`, `options`, the tool-call/tool-result message forms Ollama accepts, TTFT/prefill metrics — port it), `.hermit/rust/git/checkouts/flowcat-*/flowcat-services/src/llm/openai.rs` (`sse_to_frames`/`accumulate`: the frame contract a streaming LLM service must produce — `LlmResponseStart`, `LlmText`*, one `FunctionCallsStarted(Vec<FunctionCall>)`, `Metrics(LlmUsage)`, `LlmResponseEnd`; its unit tests are the template), `poc/ollama_ctl.py` (every lifecycle failure mode and its check, in Python — the behaviour to reproduce, then delete), `poc/flowcat/src/main.rs::start_qwen` (start-up dependency pattern: block readiness before bind, log timings), `poc/vendor/flowcat-core/src/pipeline/cascaded.rs` L145-180 (the OpenAI-shaped `tool_calls` / `role: tool` messages the context holds — the native service translates these).

## Global Constraints

- **Frame contract unchanged**: the pipeline (`LlmProcessor`, tool bridge, aggregators, `StaticGreetingLlm`) must not know which provider is underneath. Same frames, same order, same `set_tools` semantics, same barge-in cancel (dropping the stream aborts the HTTP request).
- **Context message translation is total**: the `RollingContext` stores OpenAI-shaped messages (`assistant.tool_calls[{id, type, function{name, arguments: <JSON string>}}]`, `tool{tool_call_id, content}`). Ollama's `/api/chat` wants `assistant.tool_calls[{function{name, arguments: <object>}}]` and `tool{content, tool_name}` (no ids). Translate on the way out; never mutate the context. A message the translator does not understand is an error, not a silent drop.
- **Prompt-cache discipline** (from `docs/…/ollama-prompt-cache` findings, 2026-08-24): identical system prefix every turn; tools sorted by name every request; `num_ctx` ≥ 8192 and constant; `think: false`. Gate any cache assertion on `prompt_eval_duration`, never on `prompt_eval_count`.
- **Residency is explicit**: `keep_alive: -1` on every chat/warm request while running; `keep_alive: 0` on unload. No dependence on serve env (works against the brew serve, a `launchd` serve, or the Ollama.app one).
- **Zero behaviour change for the cloud profile**: `POC_LLM_PROVIDER=openrouter` keeps `OpenAiLlmBuilder` + `OPENROUTER_*` exactly as today; Layer 2 is skipped.
- **No new sidecar scripts**: the supervisor lives in the binary. `make ollama` becomes `flowcat-poc --warm-only` (or is deleted); `run_poc.sh` stops referencing Ollama.
- Branch `poc-flowcat-ollama-lifecycle` off `poc-flowcat-qwen-tts`; commit per task with `poc(ollama): …`. All `make`/`pytest` from `poc/`.

---

### Task 1: Native Ollama `LlmService` (request/response, offline-testable)

**Files:**
- Create: `poc/flowcat/src/llm_ollama.rs`
- Modify: `poc/flowcat/src/main.rs` (`mod llm_ollama;`)

**Interfaces:**
- `pub struct OllamaLlm { base_url, model, num_ctx, keep_alive: i64, http, tools: Vec<Tool> }`
- `OllamaLlm::new(base_url, model) -> Self`, `.num_ctx(u32)`, `.keep_alive(i64)`
- `fn request_body(&self, ctx: &LlmContext, stream: bool) -> Value` — pure; `messages` translated, `tools` from `ctx.tools` else `self.tools`, sorted by name, wrapped `{type: function, function{name, description, parameters}}`; `options: {num_ctx}`; `keep_alive`; `think: false`; `stream`.
- `fn translate_messages(msgs: &[Value]) -> Result<Vec<Value>>` — OpenAI → Ollama shapes (assistant `tool_calls` arguments string → object; `tool` message gets `tool_name` looked up from the preceding assistant call by `tool_call_id`).
- `fn ndjson_to_frames(byte_stream, model) -> BoxStream<'static, Frame>` — per line: `message.content` → `LlmText`; `message.tool_calls[]` → accumulate; `done: true` → `Metrics(LlmUsage{prompt/eval tokens})` + tracing of `prompt_eval_duration`/`eval_duration` (ms) + `FunctionCallsStarted` (if any; synthesize `tool_call_id = "call_{n}"`) + `LlmResponseEnd`. `LlmResponseStart` before the first output.
- `impl LlmService for OllamaLlm` — `run_llm` POSTs `/api/chat` and returns `ndjson_to_frames`; `set_tools` stores.

**Steps:**
- [ ] Port the request shape from `poc-gemma4/poc_gemma4/ollama.py` (`keep_alive` as a number, `think`, `options`); write `request_body` + `translate_messages` with unit tests: system-only prefix, user turn, assistant-with-tool_calls + tool result round trip (from cascaded.rs L145-180 shapes), tools sorted, `num_ctx` present.
- [ ] `ndjson_to_frames` unit tests with canned Ollama chunks (mirror `openai.rs` tests): text deltas in order; a streamed tool call becomes one `FunctionCallsStarted` with parsed `arguments` object and a synthesized id; `done` emits usage then `LlmResponseEnd`; malformed line → `Frame::Error`-free `Err` on the stream (log + end).
- [ ] `cargo test` green; no feature flag needed (reqwest/serde already linked).

### Task 2: Provider selection + warm/unload API

**Files:**
- Modify: `poc/flowcat/src/main.rs` (`PocConfig`: `llm_provider: String` (`ollama` default when `OPENROUTER_BASE_URL` points at :11434 or `POC_LLM_PROVIDER=ollama`; `openrouter` otherwise), `llm_num_ctx: u32` (`POC_LLM_NUM_CTX`, default 8192), `llm_unload_on_exit: bool`), `poc/flowcat/src/call.rs` (`PocLlm` enum: `Ollama(OllamaLlm)` | `OpenAi(OpenAiLlm)`, like `PocTts`), `poc/flowcat/src/llm_ollama.rs`
- Modify: `poc/.env.example`

**Interfaces:**
- `OllamaLlm::warm(&self, system_prompt: &str, tools: &[Tool]) -> Result<WarmReport { load_ms, prompt_eval_ms, prompt_tokens }>` — `request_body` with only the system message, `num_predict: 1`, `stream: false`; then `GET /api/ps` → assert model listed with `context_length == num_ctx` and `expires_at` > 1 year (pinned) — else `Err` naming the problem.
- `OllamaLlm::unload(&self) -> Result<()>` — `POST /api/generate {model, keep_alive: 0}`.
- `PocLlm` forwards `LlmService`; `StaticGreetingLlm` wraps it as today.

**Steps:**
- [ ] Wire `PocLlm` in `call.rs`; startup validation accepts both providers (`openrouter` still requires the key; `ollama` requires base URL + model).
- [ ] `main.rs`: for `ollama`, after STT/TTS preload, call `warm()` with the brain's system prompt + the session's `node_tools` (the same tools every call advertises — this is what makes the prefix byte-identical) and log the report at info.
- [ ] Smoke against the running serve: `pytest harness -m smoke` green on the native path; log shows warm `prompt_eval_ms` ≈ 2 s once, then first real turn `prompt_eval_ms` < 200 ms (cache hit). Record numbers in the README.

### Task 3: Serve supervisor + graceful shutdown

**Files:**
- Create: `poc/flowcat/src/ollama_serve.rs`
- Modify: `poc/flowcat/src/main.rs` (start-up order; shutdown), `poc/.env.example`, `poc/run_poc.sh` (drop Ollama mentions), `poc/Makefile` (`ollama` target → `flowcat-poc --warm-only`, or remove), `poc/README.md`

**Interfaces:**
- `pub enum Supervise { Auto, Never, Always }` from `POC_OLLAMA_SUPERVISE` (default `auto`).
- `pub struct OllamaServe { child: Option<tokio::process::Child>, base_url }`
- `OllamaServe::ensure(base_url, supervise, log_path) -> Result<Self>`: `Never` → probe only; `Auto` → spawn if `/api/tags` fails (binary from `POC_OLLAMA_BIN` or `ollama` on PATH; env `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_CONTEXT_LENGTH=<num_ctx>` for good measure; stdout/stderr → log), wait ≤ 60 s; `Always` → refuse if the port is busy and we didn't start it (clear error: "quit the Ollama.app or use auto").
- `OllamaServe::shutdown(self, unload: impl Future)`: run the unload first (model released even when serve stays), then `child.kill()`/wait if owned.
- `main.rs`: install `tokio::signal::ctrl_c` + SIGTERM handler; `axum::serve(...).with_graceful_shutdown(...)`; after serve returns, call `shutdown`. `--warm-only` flag: run Layer 2 + `warm()` and exit 0 (for `make ollama`/CI).

**Steps:**
- [ ] Implement + unit-test the decision table (port free/busy × supervise mode × owned/not) with a fake prober.
- [ ] Manual matrix on the Mac: (a) nothing running → spawns, warms, `make down` (SIGTERM) unloads and kills the child — `/api/ps` empty, wired memory back; (b) brew serve already running without env → used as-is, model pinned by requests, unload on exit leaves serve up; (c) Ollama.app serve running (start it manually) → same as (b) — the native API makes its env irrelevant; note the 0.24.0 version in the log as a warning.
- [ ] `run_poc.sh down` sends SIGTERM (not SIGKILL) so the unload runs; verify via `/api/ps` in the `down` output.

### Task 4: Retire the glue, document, measure

**Files:**
- Delete: `poc/ollama_ctl.py`; Modify: `poc/Makefile`, `poc/README.md`, `poc/harness/results.py` (snapshot `llm_provider`, `num_ctx`, cache-hit `prompt_eval_ms` from the run log if cheap), memory note in `~/.claude` (the ollama-prompt-cache memory: native API path)

**Steps:**
- [ ] `make` from a cold machine (Ollama not running) brings the whole stack up with one command; first user turn LLM round ≤ 0.6 s (cache warm), `make down` returns the memory.
- [ ] `pytest harness -m "smoke or tools or duplex"` green on both TTS backends with the native LLM path; `poc-results` rows show `llm=gemma4:26b` `provider=ollama`.
- [ ] README: lifecycle section (what happens at start/stop, the three config knobs, how to keep the model across dev restarts with `POC_OLLAMA_UNLOAD_ON_EXIT=false`).

## Risks / decisions to confirm

- **Ollama tool-call fidelity on `/api/chat` vs `/v1`**: both go through the same Go template/parser, but ids are absent on the native path — the synthesized `call_n` ids only need to be consistent within a turn (the tool bridge matches on id). Covered by the T4 round-trip test.
- **Thinking**: `think: false` must be sent explicitly; gemma4 has no CoT but a future model swap would silently pay a thinking tax otherwise.
- **Dev ergonomics**: with `unload_on_exit=true` every flowcat restart reloads the model (10–40 s). Default it to `true` for `make up`/`down` symmetry but document `false` for rebuild-heavy sessions; `--warm-only` covers CI/prewarm.
- **Two Ollama installs** (brew 0.32.5 vs app 0.24.0): the supervisor logs `GET /api/version` and warns below 0.32 (prompt-cache accounting differs); it does not try to pick one.
