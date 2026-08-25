# Research: Qwen3.8-27B via MTPLX for the chatbot LLM (function calling, streaming latency, residency)

> **Status: PARKED (2026-08-24).** Not started. Reason: the model peaks at
> ~23.6 GB and the recomputed residency budget leaves ≈ 0–0.5 GB slack on the
> 36 GB Mac Studio next to TTS (4.3 GB) + Nemotron (< 1 GB) — too little free
> memory to run the PoC alongside the rest of the stack right now. Revisit if
> memory frees up (TTS trimmed further, or a smaller Qwen3.8 build such as
> `mlx-community/Qwen3.8-27B-MTP-4bit` at 16.1 GB), or if Task 0's cheap
> prefix-cache check on Qwen3.5-9B is worth answering on its own.
> The PoC plan below is kept intact so it can be picked up as written.

**Date:** 2026-08-24 · **Branch:** `poc-qwen3-tts` (or new `poc-qwen-3.8-mtplx`) · **Dir:** `poc-qwen-3.8-mtplx/`

## Goal

Run `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` on this box (Mac Studio M4
Max, 36 GB unified) behind `mtplx serve`, and answer two questions with
evidence:

1. **Function calling** — given the exact OpenAI `tools` payload the pipeline
   sends today (top-15 skill schemas from `skills/*/SKILL.md`, streamed
   `/v1/chat/completions`, `temperature 0.2`, `max_tokens 512`, no thinking),
   does the model pick the right tool, emit valid arguments, stay silent on
   chit-chat, and never leak reasoning tokens? Scored by an automated test
   suite, compared with the current `gemma4:12b-mlx` default and the
   `gemma4:26b` reference from ADR-0001.
2. **Streaming latency** — the pipeline streams LLM text into TTS while the
   user waits, so two numbers are hard gates: **TTFT** (silence before the
   voice starts; ADR-0001 target ≤ 0.4 s warm) and **decode rate vs speech
   rate** (spoken English ≈ 150 wpm ≈ 2.5 words/s ≈ 3.5–4 tok/s; TTS on
   this box runs at RTF ≈ 0.3–0.37, so it consumes text at ≈ 10 tok/s
   equivalent). The LLM must stay ahead of that with margin, in the
   **worst** case — long reply, full 15-tool prefix, tool-call second pass —
   not the median. Measured per token from the stream, not from `usage`.
3. **Feel** — a small Gradio GUI to try prompts by hand: chat with tools on,
   see the raw `tool_calls`, TTFT and tok/s per turn, toggle
   thinking/effort, edit the system prompt.

Deliberately **out of scope** (owned by
`2026-08-24-poc-qwen-llm-server.md`): engine comparisons and wiring the
pipeline. Prefix-cache verification (Task 0) and a co-residency confirmation
with the trimmed TTS + Nemotron (Task 8) are **in** scope, since they decide
whether this model is viable at all.

## What we are running

| | |
| --- | --- |
| Model | `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` — base `Qwen/Qwen3.8-27B`, MLX, dynamic 4-bit (embeddings / lm_head / sensitive layers 8–16-bit), MTP head kept for speculative decoding |
| Size | **20.4 GB download, 23.6 GB peak unified** (card). 4–6 GB more than `mlx-community/Qwen3.8-27B-4bit` (16.1 GB) — the price of the higher-precision layers |
| Server | `pip install mtplx` (2.9.1, py ≥ 3.11), `mtplx serve --port <p>`; OpenAI-compatible `/v1/chat/completions`, streaming, tool calls in OpenAI and Anthropic styles; SSD session cache on by default; `mtplx pull / inspect / tune / doctor / stop` |
| Speed (card, M5 Max) | 58.7 tok/s coding, 35–37 tok/s long reasoning, draft acceptance 0.80–0.95. Expect ~50–70 % of that on the binned M4 Max — measure |
| Thinking | Qwen3.8 thinks by default at `xhigh`. Must be **off** for the pipeline (`chat_template_kwargs: {enable_thinking: false}` / `reasoning_effort`). Whether MTPLX honours that per request is the first thing to check |
| Tool format | Qwen3-Coder XML `<tool_call><function=…><parameter=…>`; MTPLX converts to OpenAI `tool_calls` — verify it does so **in streamed deltas**, not just in the non-streamed body |

### Residency budget (recomputed 2026-08-24: 36 GB box, TTS trimmed to 4.3 GB, Nemotron < 1 GB)

| Resident component | GB | Note |
| --- | --- | --- |
| macOS + desktop + compressor headroom | 5.0 | reserve |
| Qwen3-TTS (trimmed) | 4.3 | down from 9.9 GiB in `poc-qwen/bench-m4-max.md` |
| Nemotron Speech Streaming 0.6B (NeMo-Speech.cpp, Metal) | 1.0 | upper bound |
| Pipecat server, whisper-tiny fallback, mpv, openWakeWord | 1.5 | current `run.sh` stack |
| **Left for LLM** | **≈ 24** | vs ≈ 17–18 in the sibling plan's old table |
| This model, peak (card) | 23.6 | + ~0.5 GB KV at 8 K ctx (≈ 64 KB/token, hybrid) |
| **Slack** | **≈ 0–0.5** | fits, but with no room for a second variant, a long context, or the Gradio/browser on the same box |

Wired-memory check: MLX's ceiling here is `max_recommended_working_set_size`
= 28.1 GiB ≈ 30.2 GB. GPU-resident set = 4.3 + 1.0 + 24.1 = 29.4 GB ≈
27.4 GiB → **under the ceiling by ~0.7 GiB**. Fits without touching
`iogpu.wired_limit_mb`, but barely; if a run shows wired pages dropping or
compressor growth, raise it to 30720 MB (leaves 6 GB for the OS) or use
`mlx-community/Qwen3.8-27B-MTP-4bit` (16.1 GB, ~7 GB slack) as the fallback
row. The eval in this PoC runs the LLM alone; Task 8 repeats the bench with
TTS + Nemotron loaded to confirm.

### Can the prefix be cached and reused across turns? (researched 2026-08-24)

Not confirmable from documentation alone — it has to be measured — but the
structural picture is better than the sibling plan's "broken for hybrids":

- Qwen3.8-27B is a Gated-DeltaNet + attention hybrid. Its recurrent state
  is not trimmable at arbitrary token boundaries, which is why naive
  longest-prefix KV reuse fails (mlx-lm #980, filed 2026-03-11).
- **mlx-lm #911 (merged 2026-03-06, shipped in 0.31.x)** works around that
  for the server: requests that end in a user message become *checkpoints*;
  the cache is snapshotted at message boundaries, so a byte-identical
  system+tools prefix and a growing multi-turn history are reused as long
  as each turn extends the previous checkpoint exactly. Release notes for
  0.31.2: "Caching system prompt and user messages for non-trimmable
  caches." Caveat from the author: "only apply to the batched generation
  for now"; matching is at message boundaries, not arbitrary prefixes.
- **MTPLX 2.9.1 pins `mlx-lm >=0.31,<0.32`** — so the mechanism exists in
  its dependency — but MTPLX runs its own FastAPI server with a "warm-prefix
  session bank" (RAM tier) + SSD cold tier, and nothing in its README says
  whether that bank checkpoints hybrid recurrent state or just KV. Issue
  #323 (2026-08-22, open) shows the session code marks tool-bearing
  sessions `live_ref_only` and skips the SSD tier for them; the RAM tier is
  unaffected, so within a server lifetime our tool-bearing turns should
  still hit — across restarts they will re-prefill. PR #335 (Qwen3.8, draft)
  is decode-speed work and says nothing about cache.
- Implication for the pipeline: the chatbot's request shape already suits
  boundary caching (system + tools fixed, history appended, each request
  ends in a user message). Two things break it and must be avoided:
  a top-K tool set that changes between turns (the prefix changes →
  full re-prefill), and mid-conversation `system` messages.

**So: plausible, unverified — and cheap to verify before the 20 GB pull.**
Task 0 below does it in about an hour with `Qwen3.5-9B` (same hybrid
architecture, ~6 GB): if turn 2 TTFT is ≥ 5× lower than turn 1 with the
real prefix and stays flat through turn 10, the mechanism works on this
architecture in MTPLX; then the 27B measurement is only about prefill
speed, not about whether caching exists.

## The contract under test (from the pipeline)

- Tool schemas: `skills/_loader.py` turns each `SKILL.md` frontmatter
  (`name`, `description`, `parameters.{type,required,description}`) into a
  pipecat `FunctionSchema` → OpenAI `{"type":"function","function":{…}}`.
  17 skills today: core (`get_current_time`, `get_current_date`,
  `get_weather`, `set_timer`, `web_search`), radio (`play_bbc_radio`,
  `stop_bbc_radio`), shows (`play_bbc_show`), spotify (7), sfx
  (`generate_sound_effect`), persona (`switch_persona`), backends
  (`ask_claude`).
- Per turn the model sees the always-available set + top-K (`filter_k: 15`)
  by trigger-substring score (`SkillRegistry.filter_for_turn`).
- System prompt: `server.py:_ollama_system_prompt()` ("fast local voice
  assistant… one or two short sentences") plus the tool meta-hint from
  `app.py` (~line 476: "Call tools whenever the user asks for the time…
  After a tool returns, repeat its result back in one short spoken sentence").
- Request: streaming, `temperature 0.2` (0.0 on the tool-result second pass),
  `max_tokens 512`; a truncated or malformed tool-call JSON is dropped
  downstream (`app.py:256`), so **parse validity is a hard requirement**.
- Two-pass flow: tool call → handler result appended as `role: tool` →
  second completion must produce a short spoken sentence (and no second
  tool call unless warranted).

## Decision

- **Thin wrapper, no server code.** `mtplx serve` is the engine; the PoC
  adds a launcher, a schema builder that mirrors `_loader.py` without
  importing pipecat (poc venv stays small), the eval suite, the GUI, and
  reports. Same shape as `poc-qwen`: mise Python 3.12, `.venv`, stamp file,
  `config.yaml` + `POC_MTPLX_*` env overrides, `reports/*.jsonl` gitignored.
- **Port 8009** for the GUI, **8012** for `mtplx serve` (8007/8008 are the
  TTS PoCs, 8010/8011 reserved by the LLM-server plan, 8000 is MTPLX's
  default and may collide with other tools).
- Eval cases are **data, not code** (`tests/cases.yaml`), so adding a
  phrasing is a one-line change and the same file runs against any
  OpenAI-compatible endpoint (`make eval BASE_URL=http://localhost:11434/v1
  MODEL=gemma4:12b-mlx` gives the baseline).

## Layout

```
poc-qwen-3.8-mtplx/
  Makefile  mise.toml  requirements.txt  setup.sh  config.yaml  README.md
  poc_mtplx/
    __init__.py
    config.py        # load config.yaml + POC_MTPLX_* overrides (copy of poc-qwen/config.py)
    schemas.py       # SKILL.md frontmatter -> OpenAI tool dicts; filter_for_turn clone
    prompt.py        # system prompt (+ tool hint) identical to server.py/app.py
    client.py        # streaming chat call: TTFT, tok/s, tool_call delta assembly, think-leak detection
    server.py        # launch/stop/health for `mtplx serve`; warm-up request
    eval.py          # runs tests/cases.yaml -> reports/eval.jsonl + summary table
    bench.py         # latency/throughput sweep -> reports/bench.jsonl
    app.py           # Gradio GUI on :8009
    smoke.py         # go/no-go: one tool call end to end
  tests/
    cases.yaml       # the function-calling suite (data)
    conftest.py  test_schemas.py  test_prompt.py  test_client.py  test_eval.py
    fixtures/        # recorded SSE streams (tool call, plain text, think-leak, truncated)
  reports/           # gitignored: eval.jsonl, bench.jsonl, ui_runs.jsonl, env_probe.json
```

## Makefile (mirrors `poc-qwen/Makefile`)

```make
.DEFAULT_GOAL := run
VENV := .venv ; PY := $(VENV)/bin/python ; STAMP := $(VENV)/.setup-stamp
MODEL ?= Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
LOAD_ENV = $(if $(wildcard .env),set -a; . ./.env; set +a;)

run: setup        ## Install if needed, start mtplx serve (:8012) if not up, serve GUI on :8009
setup: $(STAMP)   ## mise python 3.12, venv, deps (mtplx, gradio, openai, pyyaml, pytest), env probe
build: setup      ## Fetch the model (mtplx pull) + `mtplx inspect` compat check + `mtplx tune` -> reports/build.json
models: build     ## alias kept for parity with poc-qwen
serve: setup      ## Foreground `mtplx serve --port 8012` with config-driven flags
stop:             ## `mtplx stop`
smoke: setup      ## One streamed "what time is it" -> expect get_current_time tool_call; exit 1 otherwise
eval: setup       ## Function-calling suite (tests/cases.yaml) vs BASE_URL/MODEL -> reports/eval.jsonl + table
bench: setup      ## TTFT / tok/s with the 15-tool prefix, 3 repeats -> reports/bench.jsonl
test: setup       ## GPU-free unit tests (server mocked, recorded SSE fixtures)
clean:            ## rm venv + caches; leaves reports/ and HF cache
help:
```

`run` = `setup` → `server.ensure_running()` (starts `mtplx serve` detached
if `/v1/models` doesn't answer, waits for health, one warm-up request) →
Gradio. `build` is the heavy step (~20 GB) and is **not** a prerequisite of
`run`; `run` prints the `make build` hint if the model isn't cached. Root
`Makefile` gets `poc-mtplx`, `poc-mtplx-build`, `poc-mtplx-eval`,
`poc-mtplx-test` delegations next to the `poc-qwen-*` block.

## Tasks

### Task 0: Prefix-cache pre-check on a small hybrid (1 hour, before `make build`)
**Files:** `scripts/cache_precheck.py`, `reports/cache_precheck.jsonl`
- [ ] `pip install mtplx` in a scratch venv; `mtplx pull mlx-community/Qwen3.5-9B-4bit`
      (same GDN+attention hybrid as Qwen3.8, ~6 GB, already the sibling
      plan's fallback model).
- [ ] Build the real prefix (system prompt + the 15 tools `filter_for_turn`
      picks for "what time is it", sorted by name) and run a 10-turn
      scripted chat, streamed, each request = full history ending in a user
      message. Record TTFT and, if MTPLX exposes it, `usage.prompt_tokens`
      / any `cached_tokens` field / server log lines about prefill count.
- [ ] Variants: (a) MTPLX default; (b) `--ssd-session-cache off`; (c) same
      10 turns against `mlx_lm.server --prompt-cache-size 8` as the
      reference implementation of #911; (d) one turn with the tool set
      shuffled (expect a full re-prefill — proves we are measuring the
      cache, not the model warming up); (e) one turn after `mtplx stop` +
      restart (SSD tier; expect re-prefill per #323 if tools are present).
- [ ] Pass = turn-2 TTFT ≤ ⅕ of turn-1 and turns 3–10 TTFT within 1.5× of
      turn 2 in (a) or (b). Record which; if only (c) passes, MTPLX's session
      bank is the problem and the sibling plan's `mlx_lm.server` route is
      the one to take for Qwen3.8 (MTP via `--draft-model` there instead).
- [ ] Also note decode tok/s and whether tool calls arrive as streamed
      OpenAI deltas with thinking off — the same three plumbing questions
      answered on the small model first.

### Task 1: Skeleton + env (½ day)
**Files:** `Makefile`, `mise.toml`, `requirements.txt`, `setup.sh`, `config.yaml`, `poc_mtplx/config.py`, `.gitignore` (+ `reports/`)
- [ ] Copy `poc-qwen/{setup.sh,mise.toml,config.py}`; requirements:
      `mtplx>=2.9,<3`, `gradio>=5,<6`, `openai>=1.50`, `httpx`, `pyyaml`,
      `pytest`. Env probe records `mlx`, `mtplx`, `mlx-lm` versions.
- [ ] `config.yaml`: `server.{host,port:8012,model,extra_args:[]}`,
      `gui.{host,port:8009}`, `request.{temperature:0.2,tool_temperature:0.0,max_tokens:512,enable_thinking:false,reasoning_effort:low}`,
      `skills.{root:../skills,filter_k:15}`, `eval.{repeats:1,base_url,model}`,
      `bench.{repeats:3}`.
- [ ] `make setup` green; `make test` runs 0 tests green.

### Task 2: Schemas + prompt parity (½ day)
**Files:** `poc_mtplx/schemas.py`, `poc_mtplx/prompt.py`, `tests/test_schemas.py`, `tests/test_prompt.py`
- [ ] `load_skills(root)` parses every `SKILL.md` frontmatter → OpenAI tool
      dicts (`parameters` → JSON-schema `properties` + `required`; honour
      `enabled_when` against a config dict; `always_available`, `triggers`).
- [ ] `filter_for_turn(text, k)` — port of `_loader.py`'s tokenizer,
      `_contains_subseq`, and tie-break so the eval sees the **same 15 tools**
      the pipeline would. Unit test against hand-picked transcripts
      ("play radio 4" → `play_bbc_radio` in set; "pause the music" does not
      match trigger "use the").
- [ ] Parity test: expected schema count and names for the checked-in
      `skills/` tree; one snapshot of `get_weather`'s full dict. (Drift
      guard: if `_loader.py` changes shape, this test tells us.)
- [ ] `prompt.system_prompt()` reproduces `_ollama_system_prompt()` + the
      `app.py` tool hint verbatim; test compares against the strings
      imported from repo `server.py`/`app.py` **only if** pipecat is
      importable, else against a copied constant (skip marker).

### Task 3: Client + server launcher (½ day)
**Files:** `poc_mtplx/client.py`, `poc_mtplx/server.py`, `poc_mtplx/smoke.py`, `tests/test_client.py`, `tests/fixtures/*.sse`
- [ ] `chat(messages, tools, stream=True, …)` via `openai` client to
      `BASE_URL`; sends `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`
      and `reasoning_effort` per config. Returns `Turn(text, tool_calls,
      ttft_s, decode_tps, prompt_tokens, completion_tokens, raw_reasoning,
      finish_reason, error)`. Assembles `tool_calls` from streamed deltas
      (index/id/name/arguments fragments); tolerates servers that send the
      whole call in one delta.
- [ ] Think-leak detection: any `reasoning`/`reasoning_content` delta, or
      `<think>` in content, or `completion_tokens` ≫ visible tokens.
- [ ] Recorded fixtures: normal tool call, multi-tool call, plain text,
      think leak, truncated JSON (`finish_reason: length`). Tests assert the
      parser's verdicts without a GPU.
- [ ] `server.py`: `ensure_running()` (probe `/v1/models`, else spawn
      `mtplx serve --port … <extra_args>` with logs to `reports/mtplx.log`,
      poll ≤ 180 s), `warmup()` (system prompt + 15 tools + "ping"),
      `stop()`. Records first-request vs second-request TTFT (a free
      cache-hit data point; not a gate here).
- [ ] `make smoke`: "what time is it" → exactly one `get_current_time`
      call, `{}` args, no leak, TTFT printed. **First live checkpoint** —
      if this fails on thinking/format, stop and fix the request shape
      before writing more cases.

### Task 4: Function-calling suite (1 day)
**Files:** `tests/cases.yaml`, `poc_mtplx/eval.py`, `tests/test_eval.py`
- [ ] ~60 cases, each: `id`, `user` (or a short `messages` list for
      multi-turn), `expect` (`tool: name | none`, `args` exact / `args_match`
      regex per key / `args_absent`), `tags`. Coverage, per skill at least
      one direct + one indirect phrasing:
      - direct: "set a timer for ten minutes", "play radio 4", "what's the
        weather in Bristol", "pause spotify", "switch to marvin"
      - indirect (the Gemma-E4B failure class): "remind me in 10 minutes
        to check the oven", "put the Archers on", "I can't hear myself
        think" (→ `stop_bbc_radio`), "what's on right now" (→
        `whats_playing`), "is it going to rain"
      - argument fidelity: station verbatim ("BBC Radio 4 Extra"), ISO
        date resolution ("yesterday's Today programme" → `date` set),
        timer duration units, `switch_persona` enum values
      - disambiguation: live station vs on-demand show; spotify
        `play_spotify` vs `play_spotify_playlist`; `stop` with nothing
        specified
      - no-tool: 10 chit-chat / knowledge turns ("tell me a joke", "how are
        you", "what's the capital of Peru") → must answer in ≤ 2 sentences
        with **zero** tool calls
      - multi-tool: "what's the time and the date" → both calls (accept
        either order, or two rounds)
      - second pass: tool result appended (`role: tool`, e.g. time
        `"14:05"`) → reply ≤ 25 words, contains the value, no new call
      - robustness: mid-conversation `system` message (persona switch) →
        request must not 400; a 6-turn history → still calls correctly;
        tool set of 15 that **excludes** the right tool → must not
        hallucinate a name outside the list
- [ ] Scoring per case: `selected_ok`, `args_ok` (JSON-schema validate +
      expectations), `no_leak`, `parsed_ok`, `false_tool`, `ttft_s`,
      `tps`. Summary: select-accuracy, arg-validity, false-tool rate on
      no-tool cases, leak count, malformed count, TTFT p50/p95, tok/s
      median. Written to `reports/eval.jsonl` (one row per case per
      run, with `model`, `base_url`, `temperature`, git sha) and printed as
      a markdown table.
- [ ] `make eval` runs against MTPLX; `make eval BASE_URL=http://localhost:11434/v1 MODEL=gemma4:12b-mlx`
      and `MODEL=gemma4:26b` produce the two comparison rows for the README.
- [ ] `test_eval.py`: scoring logic on canned `Turn`s (no server).

### Task 5: Bench (½ day)
**Files:** `poc_mtplx/bench.py`
- [ ] Same prefix (system + 15 tools) + 5 representative turns × 3 repeats:
      TTFT p50/p95, decode tok/s, with `reasoning_effort` low vs thinking
      off, and MTP acceptance if MTPLX exposes it (`/v1/…usage` extras or
      `mtplx tune` output). Cold-start (server just launched) row tagged.
- [ ] **Streaming profile per turn**: timestamp every SSE delta →
      `ttft_s`, `time_to_first_sentence_s` (first `.?!` — when TTS can
      start), `tps_first_2s`, `tps_sustained`, `tps_min_1s_window` (the
      stall metric — MTP draft rejections and hybrid-attention recompute
      show up as stalls, not as a low mean), and `inter_token_p95_ms`.
      Reply lengths 20 / 60 / 200 tokens.
- [ ] **Speech-pace check**: simulate the TTS consumer — a reader that
      drains the stream at 10 tok/s (RTF-adjusted TTS pace from
      `poc-qwen/bench-m4-max.md`) and logs `underrun` if the LLM ever falls
      behind after the first sentence. Also report `lead_s` = how far ahead
      the LLM is at end of utterance.
- [ ] **Tool-turn latency**: TTFT to the first `tool_calls` delta, total
      time to a complete parsed call, then TTFT of the second pass with the
      tool result appended (the user hears nothing until *this* starts) —
      report the **sum** as `tool_round_trip_s`, the number the wake/idle
      tracker in `app.py:228` actually waits on.
- [ ] Warm vs cold prefix: turn 1 vs turn 2 with an identical prefix, and
      turn N with growing history. If TTFT scales with history length the
      MTPLX session cache isn't hitting — record, flag for the sibling plan.
- [ ] Peak memory during a turn: `footprint -p <pid>` / `ps rss` sampled
      at 1 Hz → `reports/bench.jsonl`. Confirms or refutes the card's 23.6 GB.

### Task 6: GUI (1 day)
**Files:** `poc_mtplx/app.py`, `tests/test_app.py`
- [ ] Gradio Blocks on :8009, never imports mlx/mtplx (talks HTTP only):
      - **Chat** tab: chatbot, system-prompt textbox (prefilled with the
        pipeline prompt), tool checklist (all 17, default = what
        `filter_for_turn` picks for the current message, with an "auto
        top-15" toggle), temperature, thinking on/off + effort, max_tokens.
        Each assistant turn shows text, a collapsible raw `tool_calls`
        JSON, and a stats line (TTFT, tok/s, prompt/completion tokens, leak
        flag). "Send tool result" box to hand-feed a `role: tool` message
        and run the second pass.
      - **Cases** tab: dropdown of `tests/cases.yaml` ids → run one, show
        verdict; "Run all" streams the summary table.
      - **Server** tab: status/health, model, `mtplx` version, start/stop
        buttons, tail of `reports/mtplx.log`.
- [ ] Every turn appended to `reports/ui_runs.jsonl` (like poc-qwen).
- [ ] Tests: schema-to-checklist mapping, stats formatting, history
      round-trip with a fake client.

### Task 8: Co-residency confirmation (½ day, after Task 5)
**Files:** `scripts/coresidency.sh`
- [ ] Load the trimmed TTS (4.3 GB) and Nemotron, idle; repeat `make bench`.
      Record `vm_stat` wired pages before/after, `footprint -p` per process,
      swap and compressor deltas, TTFT delta vs standalone. Then once more
      with TTS streaming a 300-char utterance during the LLM turn.
- [ ] Pass = wired pages stable, no swap growth, TTFT within 1.2× of
      standalone. If not, retry with `iogpu.wired_limit_mb=30720` and with
      `mlx-community/Qwen3.8-27B-MTP-4bit`; report both rows.

### Task 7: README + results (½ day)
- [ ] `README.md`: quick start (`make`, `make build`, `make eval`,
      `make test`), layout, results table (MTPLX vs gemma4:12b-mlx vs
      gemma4:26b: select-acc, arg-valid, false-tool, leaks, malformed,
      TTFT p50/p95, tok/s, peak GB), pinned versions, go/no-go.
- [ ] Root `Makefile` delegations + `.PHONY`; tick this plan.

## Exit criteria

| Gate | Pass |
| --- | --- |
| Thinking off | 0 leaked reasoning across the suite with the request shape the pipeline can send (`extra_body` only — no server-side patching) |
| Streamed tool calls | 100 % parse into valid OpenAI `tool_calls`; 0 truncated at `max_tokens 512` |
| Select accuracy | ≥ `gemma4:26b` − 2 pts on the same cases; ≥ `gemma4:12b-mlx` outright |
| Indirect phrasings | ≥ 80 % correct |
| Args | ≥ 95 % schema-valid; verbatim-station and ISO-date cases pass |
| False tool on chit-chat | ≤ 1 / 10 |
| Second pass | ≤ 25 words, contains the tool value, no spurious call |
| TTFT (standalone, warm prefix) | p50 ≤ 0.4 s, p95 ≤ 0.6 s with the 15-tool prefix (ADR-0001 target; the pipeline adds STT + TTS on top) |
| Tool round trip | first `tool_calls` delta ≤ 0.5 s; second-pass TTFT ≤ 0.4 s; `tool_round_trip_s` p50 ≤ 1.2 s |
| Decode vs speech | `tps_sustained` ≥ 40 tok/s **and** `tps_min_1s_window` ≥ 15 tok/s (≥ 1.5× TTS pace) on the 200-token reply; 0 underruns in the speech-pace check |
| Stream smoothness | `inter_token_p95_ms` ≤ 100 ms (no MTP-rejection stalls audible as pauses) |
| Prefix cache (Task 0, then 27B) | turn-2 TTFT ≤ ⅕ turn-1; turns 3–10 flat (≤ 1.5× turn 2); shuffled-tools turn re-prefills (proves the measurement) |
| Peak memory / co-residency (Task 8) | LLM ≤ 24 GB peak; with TTS 4.3 GB + Nemotron loaded: wired pages stable, no swap growth, TTFT ≤ 1.2× standalone |

Expected outcome, stated so it can be wrong: tool selection and args will be
strong (Qwen3.8's agentic scores) and decode will clear the speech-pace gate
easily with MTP (card: 58 tok/s on M5 Max; even 30 tok/s is 3× TTS pace).
**TTFT is the gate most likely to fail**: dense-27B prefill of the ~2 K-token
prefix is ~7 s cold, so it only passes if MTPLX's session cache reuses the
prefix across turns. `make bench` answers that in the first hour; if it
fails, the tool suite still tells us whether the model is worth the
prefix-cache work in the sibling plan.

## Risks

| Risk | Signal | Fallback |
| --- | --- | --- |
| MTPLX ignores `enable_thinking:false` / `reasoning_effort` per request | `reasoning` deltas or multi-second silent TTFT | `mtplx settings set` server-side default; `--extra_args`; if only global, note it (pipeline would need a dedicated instance) |
| Tool calls not streamed as OpenAI deltas | `tool_calls` only in final message or raw `<tool_call>` XML in content | client-side XML→tool_calls converter (≤ 100 lines) in `client.py`, flagged in results as "needs shim" |
| Qwen3.8 template rejects mid-conversation `system` messages | 400 on persona-switch case | fold switch into a user-role note; record as a pipeline change requirement |
| 23.6 GB peak + GUI + browser near the 28.1 GiB wired ceiling | `mtplx doctor` / swap growth / TTFT spikes | run eval headless (no GUI); `mtplx settings` to cap KV; try `mlx-community/Qwen3.8-27B-MTP-4bit` for a size-vs-accuracy row |
| MTPLX SSD session cache interferes with repeat runs (stale hits, oMLX-#825-class corruption on cache hit) | eval passes cold, fails warm, or vice-versa | `--ssd-session-cache` off for the eval; run both and report |
| Dense-27B prefill makes TTFT scale with the ~2 K-token prefix (≈ 7 s cold at ~270 tok/s) and MTPLX's session cache doesn't cover it | TTFT ≫ 0.4 s and grows with history | this is disqualifying for the chatbot regardless of tool scores — report it as the headline; the sibling plan's prefix-cache work is the only fix |
| MTP speculative decoding wins on mean tok/s but stalls on rejections | high `tps_sustained`, low `tps_min_1s_window`, `inter_token_p95_ms` spikes | reduce draft depth via `mtplx settings`; compare with MTP off; judge on the min-window metric |
| `mtplx` CLI flags differ from the README (fast-moving, 2.9.1 on 2026-08-22) | launcher fails | pin the version; `mtplx serve --help` output saved to `reports/`; `server.extra_args` in config |
| Trigger filter hides the right tool for an indirect phrasing (a pipeline bug, not a model bug) | case fails with tool absent from the 15 | eval records `tool_in_set`; report those separately as filter gaps to fix in `skills/*/SKILL.md` triggers |

## Sources

[Model card](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed), [mtplx on PyPI](https://pypi.org/project/mtplx/), [MTPLX repo](https://github.com/youssofal/MTPLX) ([pyproject: mlx-lm >=0.31,<0.32](https://raw.githubusercontent.com/youssofal/MTPLX/main/pyproject.toml), [#323 session cache skips tool sessions](https://github.com/youssofal/MTPLX/issues/323), [#335 Qwen3.8 draft PR](https://github.com/youssofal/MTPLX/pull/335)), [mlx-lm #911 boundary checkpoints for non-trimmable caches](https://github.com/ml-explore/mlx-lm/pull/911), [mlx-lm #980 hybrid prefix cache](https://github.com/ml-explore/mlx-lm/issues/980), [mlx-lm #923 (closed in favour of #911)](https://github.com/ml-explore/mlx-lm/pull/923), [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), sibling plan `docs/superpowers/plans/2026-08-24-poc-qwen-llm-server.md` (Qwen3.8 research, memory budget, prefix-cache caveats), `docs/adr/0001-core-llm-model-selection.md`, `skills/_loader.py`, `skills/_filter.py`, `app.py` (tool hint, `max_tokens` note), `server.py:_ollama_system_prompt`, `config.yaml` `llm:`/`skills:`, `poc-qwen/Makefile`.
