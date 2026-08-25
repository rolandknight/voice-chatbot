# ADR-0003: Core LLM serving — Gemma 4 26B-A4B on Ollama with a cached, byte-stable prefix

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-24 |
| **Decision** | Serve **Gemma 4 26B-A4B** (`gemma4:26b`, GGUF Q4_K_M) resident in **Ollama's llama.cpp runner**, thinking off, and make the prompt prefix (system prompt + the **complete, name-sorted** tool list) byte-identical on every turn so Ollama's prompt cache serves it. Do **not** move to the MLX runner (`gemma4:26b-mlx`) and do **not** adopt Qwen3.8-27B; both investigated and rejected for time-to-first-token (TTFT) reasons detailed below. |
| **Related** | ADR-0001 (core LLM model selection: keeps Gemma 4 26B-A4B as the model); ADR-0002 (Pipecat orchestration). PRD latency budget: warm TTFT ≤ 0.4 s; decode must outrun speech. |

---

## Context

The chatbot streams the LLM's text into text-to-speech while the user waits.
Two latency properties therefore dominate model *serving* choices, independent
of model *quality*:

1. **Time to first token (TTFT).** Silence between the end of the user's
   utterance and the first spoken word. Budget: **≤ 0.4 s warm** (p50), ≤ 0.6 s
   p95, measured with the production system prompt and tool schemas in
   context.
2. **Decode rate vs speech rate.** Spoken English is ≈ 150 words/min ≈
   3.5–4 tokens/s. Local TTS on the host (Qwen3-TTS, real-time factor ≈ 0.3–0.37)
   consumes text at ≈ 10 tokens/s equivalent. The LLM must sustain **≥ 40
   tokens/s** so the stream never underruns, including the worst 1-second
   window, not just the mean.

Every request the pipeline sends carries a large, mostly constant prefix:
the system prompt plus the JSON schemas of the voice skills the model may
call (18 tools ≈ 2,200 prompt tokens with Gemma 4's tokenizer). Prefilling
2,200 tokens costs ~2 s on this host. **Meeting a 0.4 s TTFT is therefore
impossible unless the engine reuses the previous turn's KV cache for that
prefix.** Whether it does — and what the pipeline must do so that it does —
was the open question this ADR closes.

### Host

Mac Studio, Apple M4 Max (binned 32-core GPU, 410 GB/s), **36 GB unified
memory**, macOS 26.4. MLX's default wired-memory ceiling is 28.1 GiB. Other
resident components: Qwen3-TTS (trimmed to 4.3 GB), Nemotron streaming STT
(< 1 GB), the Pipecat server plus wake-word/VAD/mpv (~1.5 GB), OS reserve
~5 GB. **Budget left for the LLM (weights + KV + activations): ≈ 24 GB.**

### Model under decision: Gemma 4 26B-A4B

- Mixture-of-experts: 25.2 B total / **3.8 B active** (8 of 128 experts + 1
  shared). Cheap per-token prefill and decode relative to its quality.
- Attention: interleaved **1024-token sliding-window** layers and global
  layers (last layer global). Sliding-window KV is a ring buffer and cannot
  be trimmed back to an arbitrary earlier position, so prefix reuse requires
  engine support (checkpoints), which is why this had to be verified rather
  than assumed.
- Thinking is opt-in via a `<|think|>` token in the system prompt; absent it,
  the model answers directly. Native function calling.
- Ollama tag `gemma4:26b`: GGUF Q4_K_M, 19 GB on disk, ~17–19 GB resident;
  256 K context nominal. Selected as the core model in ADR-0001 on tool-call
  reliability (τ²-bench 85.5 % measured in-pipeline) and a measured 0.37 s
  warm TTFT.

## Investigations (August 2026)

### A. Qwen3.8-27B via MTPLX — investigated and parked

The candidate was `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`: Qwen3.8-27B
(dense, hybrid Gated-DeltaNet + gated attention, Apache 2.0) in MLX dynamic
4-bit with its multi-token-prediction head retained, served by `mtplx serve`
(OpenAI-compatible, speculative decoding via the native MTP head; 58.7 tok/s
reported on an M5 Max).

Findings:

| Aspect | Finding |
| --- | --- |
| Memory | 20.4 GB download, **23.6 GB peak** unified. Against the 24 GB LLM budget that is ≈ 0–0.5 GB slack; GPU-resident total with TTS + STT ≈ 27.4 GiB against a 28.1 GiB wired ceiling. Fits only barely and standalone. |
| Prefill | Dense 27B prefills at ~270 tok/s on this class of GPU: the ~2 K-token prefix costs **~7 s cold**. Meeting 0.4 s TTFT depends entirely on cross-turn prefix reuse. |
| Prefix reuse | The hybrid's recurrent state is non-trimmable (mlx-lm issue #980). mlx-lm PR #911 (merged 2026-03-06, in 0.31.x) adds message-boundary checkpoints for non-trimmable caches; MTPLX 2.9.1 pins mlx-lm 0.31.x but runs its own server with its own "session bank", whose behaviour on hybrid state is undocumented; MTPLX issue #323 shows tool-bearing sessions are excluded from its persistent tier. Plausible, unverified, and only verifiable empirically. |
| Thinking | On by default at `xhigh` effort; must be disabled per request (`enable_thinking: false`); whether MTPLX honours that per request is unverified. |
| Tool format | Qwen3-Coder XML `<tool_call><function=…>`; MTPLX's conversion to streamed OpenAI `tool_calls` deltas unverified. |

Verdict: **parked** on 2026-08-24. It enters with two strikes (slow dense
prefill, unproven cache on a hybrid architecture) and no memory headroom;
it could only win on tool-call quality, which was never measured because the
box lacked the free memory to run the trial. Revisit only if a ~16 GB build
(`mlx-community/Qwen3.8-27B-MTP-4bit`) is in play and a one-hour cache
pre-check on a small model of the same architecture (Qwen3.5-9B) passes.

### B. Engines for Gemma 4 26B-A4B — prefix-cache support

| Engine | Prefix reuse for a sliding-window model | Verdict |
| --- | --- | --- |
| **Ollama, `gemma4:26b` (llama.cpp runner)** | Yes: llama.cpp prompt cache with SWA checkpoints; cross-conversation reuse and checkpointing added in Ollama 0.19+. Loses the cache silently when a prompt exceeds `num_ctx` (Ollama truncates). Known cold-prefill-after-idle bug (Ollama #16051), mitigated by a periodic keepalive request. | **Chosen — measured below** |
| Ollama, `gemma4:26b-mlx` (MLX runner, multi-token prediction) | **No** as of 0.32.14: Ollama #17829 (2026-08-17, open) — no prompt/prefix caching between requests; TTFT degrades as history grows. | Rejected until #17829 closes. Faster decode is irrelevant: decode already exceeds speech pace 8×. |
| `llama-server` direct (b10280) | Yes: `--cache-prompt` (default), `--cache-reuse 256`, `--swa-checkpoints N` (small: only the SWA part is stored) or `--swa-full`; verifiable per request via `timings.prompt_n`. 1.7–2× faster cold prefill than MLX. | Fallback if Ollama's p95 ever misses 0.6 s. Flags: `-np 1 -fa on -ctk q8_0 -ctv q8_0 --load-mode mlock --jinja` (q4 KV degrades tool calls). |
| `mlx_lm.server` | Yes via PR #911 message-boundary checkpoints; but `RotatingKVCache` cannot be quantized (mlx-lm #1573) and cold prefill is slower. | No advantage here. |

### C. What determines whether the cache hits

The chat template renders the tool list into the **front** of the prompt,
before the conversation. The cache is a longest-prefix match, so a turn hits
only if everything before the new user message is byte-identical to the
previous request:

1. **Same tool set, same order, every turn of a session.** The pipeline's
   per-turn relevance filter (top-K tools by trigger-phrase score) rewrites
   the front of the prompt whenever the selected set changes — a full
   re-prefill. With 18 skills in total, filtering saves nothing and costs
   ~2 s.
2. **No mid-conversation `system` messages** unless a re-prefill on that turn
   is acceptable (persona switches).
3. **History appended, never rewritten.**
4. **Thinking off**: never emit `<|think|>` into the system prompt; pass
   `think: false` explicitly (Ollama's renderer injects thinking for some
   Gemma 4 variants).
5. **`num_ctx` large enough** that Ollama never truncates (8192; the
   conversation is wiped at the session idle timeout anyway).

## Test strategy

A standalone probe against Ollama's native `/api/chat` endpoint (chosen over
the OpenAI-compatible endpoint the pipeline uses because the native response
carries `prompt_eval_count`, `prompt_eval_duration`, `eval_count`,
`eval_duration`). Streaming; TTFT measured at the first content, tool-call or
thinking delta.

**Cache-hit evidence is `prompt_eval_duration`, not `prompt_eval_count`.**
On Ollama ≥ 0.32 the token count includes cached tokens (Ollama PR #16428,
merged 2026-06-02), so it reads ~2,200 on every turn whether or not the
cache hit. A **cache miss baseline** is measured explicitly with a
never-seen prefix: a nonce prepended to the system prompt *and* a nonce'd
dummy tool sorted first, so the prefix differs from its first byte whichever
the template renders first. Each run tags its first user turn with a run id
so a previous run's cached history (prompts are deterministic at
temperature 0.2) cannot mask results; the system+tools prefix itself stays
identical to production.

Scenarios:

| Scenario | Purpose | Gate |
| --- | --- | --- |
| Cache miss baseline | cost of a full prefill | informational |
| Stable session, 6 turns, byte-identical prefix, history appended | the recommendation | turns 2–6: `prompt_eval_duration` ≤ 15 % of the miss; TTFT p50 ≤ 0.4 s, max ≤ 0.6 s; decode ≥ 40 tok/s |
| Reordered tools on turn 3 (same history) | reproduces the per-turn filter | prefill ≥ 3× the warm median (proves the measurement sees misses) |
| Mid-conversation `system` message on turn 3 | persona switch | request accepted, reply produced |
| "what time is it?" with all tools | tool selection, thinking leakage | exactly `get_current_time`, no `thinking` deltas and no `<\|channel>` in content |
| Second pass with tool result `14:05` appended as a `tool` message | spoken reply | reply contains 14:05 or 2:05, no further tool call |
| Unit tests (no GPU) | schema builder and gate logic | fake cache with longest-common-prefix semantics; gates pass with a cache and fail without one |

## Results — 2026-08-24

Ollama 0.32.5, `gemma4:26b`, 18 tools (≈ 2,200 prompt tokens), `num_ctx`
8192, `think: false`, temperature 0.2, `keep_alive: -1`. Host as above,
LLM running alone (TTS/STT not loaded).

| Case | Prefill | TTFT |
| --- | --- | --- |
| Cache miss (never-seen prefix), 3 repeats | 2,195 tokens in **2.15 s** (~1,000 tok/s) | **2.4–2.6 s** |
| Identical request repeated | 0.017 s | 0.23–0.24 s |
| Same prefix, one turn appended | 0.11–0.12 s | 0.32–0.34 s |
| Stable 6-turn session, turns 2–6 | 0.12–0.16 s | **0.33–0.37 s** (p50 0.36) |
| Same history, **tool list reordered** | 0.67–2.15 s | **0.95–2.6 s** |
| Mid-conversation `system` message | accepted; 0.19 s | 0.41 s |
| "what time is it?" | `get_current_time` called, no thinking | 0.42–0.48 s |
| Second pass with tool result `14:05` | "It's 2:05 PM.", no extra call | 0.40 s |
| Decode, all turns | **80–89 tok/s** | |

All eight gates pass; the seven live assertions pass; the nine unit tests
pass. Reordering the tools was sometimes a partial miss (0.67 s — llama.cpp's
cache-reuse shifts matching KV chunks) and sometimes a full one (2.15 s);
either is over budget.

Interpretation:

- The prompt cache **works** for Gemma 4's sliding-window layers on Ollama's
  llama.cpp runner: a hit is 15–100× cheaper than a miss and lands warm TTFT
  at 0.33–0.37 s. No engine work is needed.
- **The per-turn tool filter is the TTFT defect.** Any change in the tool
  list costs 0.7–2.2 s. This is a pipeline change, not a model change.
- Tool-call turns land at 0.42–0.48 s, marginally over the 0.4 s target; the
  full tool round trip (call + spoken second pass) is ≈ 0.9 s. Shorter tool
  descriptions are the lever if this matters (the prefix is ~120 tokens per
  tool).
- Decode is ~8× the TTS consumption pace; multi-token prediction and
  speculative decoding are unnecessary.

## Decision

1. **Model and engine.** `gemma4:26b` (GGUF Q4_K_M) on Ollama's llama.cpp
   runner, pinned resident. Not the MLX tag; not Qwen3.8.
2. **Ollama request settings** (native or OpenAI-compatible endpoint):
   - `keep_alive: -1` (numeric on the native endpoint; the string `"-1"` is
     rejected there with `missing unit in duration`). A keepalive request
     every 60 s keeps the runner hot.
   - `options.num_ctx: 8192` — never let a prompt reach it.
   - `think: false`; system prompt must not contain `<|think|>`.
   - `temperature 0.2` for chat turns, `0.0` for the pass after a tool
     result; `num_predict` (max tokens) 512 — a truncated tool-call JSON is
     dropped downstream, so this must not be lowered.
   - Streaming on.
3. **Prompt layout, fixed for the whole session:**
   - System prompt, verbatim: *"You are a fast local voice assistant. Keep
     replies brief and conversational. Prefer one or two short sentences.
     Call tools whenever the user asks for the time, the date, a timer, the
     weather, radio, music, or a sound effect. After a tool returns, repeat
     its result back in one short spoken sentence."*
   - Then **all enabled tools, sorted by name**, as OpenAI function schemas:
     `{"type":"function","function":{"name","description","parameters":{"type":"object","properties":{…},"required":[…]}}}`.
     The current set (18): `ask_claude`, `generate_sound_effect`,
     `get_current_date`, `get_current_time`, `get_weather`, `pause_spotify`,
     `play_bbc_radio`, `play_bbc_show`, `play_spotify`,
     `play_spotify_playlist`, `resume_spotify`, `set_timer`, `skip_spotify`,
     `stop_bbc_radio`, `stop_spotify`, `switch_persona`, `web_search`,
     `whats_playing`. Descriptions are whitespace-normalised single lines.
   - Then the conversation, appended only. Persona switches go into a user-role
     note, or accept one re-prefill on that turn.
4. **Remove the per-turn top-K tool filtering from the request path.** The
   tool set may change only at session start (when a skill is disabled by
   configuration). If the skill count ever grows to where prompt length
   matters, select the set **once per session**, not per turn.
5. **Latency gates for the pipeline's own measurement** (in-pipeline, with
   STT and TTS loaded): warm TTFT p50 ≤ 0.4 s / p95 ≤ 0.6 s over ≥ 20 turns;
   tool round trip p50 ≤ 1.2 s; sustained decode ≥ 40 tok/s with no 1-second
   window below 15 tok/s.

## Consequences

- TTFT drops from up to ~2.5 s (whenever the filtered tool set changed) to
  ~0.35 s on ordinary turns, with no new component.
- Prompt length is fixed at ~2.2 K tokens per request regardless of the
  user's words. At 8 K context this leaves ~5.8 K tokens for a session's
  conversation, ample given the idle-timeout wipe.
- Memory: ~19 GB resident LLM + small KV under a 24 GB budget; ~4–5 GB slack
  next to the trimmed TTS and STT.
- The Qwen3.8 line of work stays parked; the memory and cache findings above
  are the entry conditions for reopening it.
- Verification is cheap and repeatable: the probe takes ~30 s once the model
  is loaded and reads only response metadata; it should be re-run after any
  Ollama upgrade, tool-set change, or system-prompt edit.

## Re-evaluation triggers

- Ollama #17829 closes (prefix cache on the MLX runner): re-measure
  `gemma4:26b-mlx`; adopt only if warm TTFT p95 ≤ 0.6 s holds.
- In-pipeline warm TTFT p95 > 0.6 s after the tool-set fix: move to direct
  `llama-server` with the flags in §B and compare `timings.prompt_n`.
- Tool-call selection accuracy regresses below ADR-0001's bar, or the skill
  count exceeds ~30 (prefix > 4 K tokens): revisit once-per-session tool
  selection and shorter descriptions before revisiting the model.
- A ≤ 16 GB Qwen3.8 build with verified hybrid-cache support: reopen
  investigation A with a cache pre-check first.

## Sources

- Gemma 4: [google/gemma-4-26b-a4b-it model card](https://huggingface.co/google/gemma-4-26b-a4b-it); [Ollama `gemma4` library page](https://ollama.com/library/gemma4).
- Ollama caching and runner: [Ollama MLX runner announcement](https://ollama.com/blog/mlx); [#17829 MLX engine: no prompt/prefix caching between requests](https://github.com/ollama/ollama/issues/17829); [#16428 include cached prompt tokens in llama-server counts](https://github.com/ollama/ollama/pull/16428); [#16051 cold prefill after idle](https://github.com/ollama/ollama/issues/16051).
- llama.cpp: [server README (cache-reuse, swa-checkpoints, swa-full, slots)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md); [#15293 SWA checkpoints](https://github.com/ggml-org/llama.cpp/pull/15293).
- mlx-lm: [#911 better caching for non-trimmable caches](https://github.com/ml-explore/mlx-lm/pull/911); [#980 prefix cache only works for pure attention models](https://github.com/ml-explore/mlx-lm/issues/980); [#1573 RotatingKVCache cannot be quantized](https://github.com/ml-explore/mlx-lm/issues/1573).
- Qwen3.8 / MTPLX: [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed); [mtplx on PyPI](https://pypi.org/project/mtplx/); [MTPLX repository](https://github.com/youssofal/MTPLX); [MTPLX #323 session cache and tool sessions](https://github.com/youssofal/MTPLX/issues/323); [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
