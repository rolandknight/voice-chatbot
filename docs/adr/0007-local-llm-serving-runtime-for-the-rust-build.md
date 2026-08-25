# ADR-0007: Local LLM serving runtime for the Rust build — `llama-server` supervised by the chatbot, Ollama demoted to a dev profile

| | |
|---|---|
| **Status** | Accepted in direction, **deferred** (2026-08-25): Ollama stays the local runtime for the current phase; the Rust chatbot takes over Ollama's lifecycle now (lifecycle plan, native `/api/chat` service + supervisor). The `llama-server` switch is a future phase, entered via the gates in "Test strategy". ADR-0003's model choice, prompt-layout rules and prefix-cache discipline are unchanged and carried forward. |
| **Date** | 2026-08-25 |
| **Decision (proposed)** | Serve **Gemma 4 26B-A4B (GGUF Q4_K_M)** on **llama.cpp `llama-server`**, started and stopped by the Rust chatbot as a **child process** (`-np 1 --jinja -c 8192 --cache-reuse 256 …`), so that the model is resident exactly as long as the chatbot runs, the prompt cache is verifiable per request (`timings.cache_n`), and none of Ollama's scheduler/keep-alive semantics are in the path. Keep **Ollama** as a *dev profile* (`POC_LLM_PROVIDER=ollama`) for continuity with ADR-0003's numbers. Time-box a **Rapid-MLX** A/B for TTFT. **vLLM / vllm-metal, mlx_lm.server, LM Studio, mistral.rs, in-process bindings: not adopted** (reasons below). |
| **Related** | ADR-0001 (model), ADR-0003 (Ollama serving + prefix cache — partially superseded), ADR-0006 (FlowCat Rust runtime), `docs/poc/flowcat-poc-plan.md` §5 Phase 2 (already named llama-server primary), `docs/superpowers/plans/2026-08-25-poc-flowcat-ollama-lifecycle.md` (the lifecycle work this ADR redirects). |

---

## Context

ADR-0003 (2026-08-24) chose Ollama's llama.cpp runner for `gemma4:26b` and
proved the prompt cache works when the prefix is byte-stable: warm TTFT
0.33–0.37 s, decode 80–89 tok/s. Its stated fallback was "direct
`llama-server` if in-pipeline warm TTFT p95 ever misses 0.6 s". That trigger
did not fire on inference grounds. What fired instead, during the first two
days of running the **Rust build** (ADR-0006) against Ollama on the Mac
Studio, was a run of **lifecycle** failures — none of them about model
quality or raw speed, all of them about *who owns the model*:

| Observed (poc/logs, 2026-08-24/25) | Cause | Effect on a caller |
|---|---|---|
| First turn after ~5 idle minutes: 10–12 s | Ollama's OpenAI-compatible `/v1` applies the **server-default keep-alive (5 min) to every request**, overwriting a `keep_alive: -1` pin sent via `/api/generate` ([#2963](https://github.com/ollama/ollama/issues/2963), open since 2024; [#11458](https://github.com/ollama/ollama/issues/11458) closed as duplicate). Only the serve process's `OLLAMA_KEEP_ALIVE` env pins durably. | cold load + full prefill |
| First turn: 7.9 s even with the model "resident" | A residency load via bare `/api/generate` created a runner with a different config from what the first `/v1` request needed; Ollama **swapped runners** (model reload) on that request. | cold load |
| Mid-session 10–17 s turns after Qwen3-TTS started | Ollama's scheduler re-predicts fit on each request; when Metal "available" shrank by the TTS engine's 4–6 GB it decided the *already-resident* 26b "predicted to exceed available memory" (18.0 GB at 32K ctx vs 15.6 GB) and **evicted + reloaded it, now with `--no-mmap`** (17.7 GB RSS, fully wired). Mitigated by `OLLAMA_CONTEXT_LENGTH=8192` — again serve-process env, not a request setting. | reload; then paging |
| Another client's `ollama run gemma4` evicted 26b | one shared scheduler, newest request wins | 10 s reload on the next turn |
| After a reboot, `:11434` belonged to **Ollama.app** (0.24.0, no env, ignores `launchctl setenv`, ignores AppleScript quit) rather than the brew 0.32.5 serve | two Ollama installs | keep-alive and context length silently wrong |

Every one of these was worked around in `poc/ollama_ctl.py` (218 lines of
Python: detect the serve, restart it with env, warm the prefix via `/v1`,
verify `expires_at` and `context_length`, stop the app). That helper is the
symptom: the chatbot cannot make Ollama behave through the API it uses, only
by controlling Ollama's *process environment*. The Rust build will own its
LLM lifecycle (start, warm, pin, unload — see the lifecycle plan), and the
question this ADR answers is which runtime makes that ownership simple and
which makes it a fight.

### What changed since ADR-0003

- **The caller is a Rust binary that owns its sidecars.** It already embeds
  Qwen3-TTS in-process and waits on the Nemotron sidecar's readiness. A
  runtime that must be configured through env vars of a daemon started by
  someone else does not fit; a runtime that is a child process with flags
  does.
- **Memory is shared with a GPU TTS engine.** Qwen3-TTS (2.4 GB at 0.6B,
  4.3 GB at 1.7B, peaks +2 GB) sits in the same unified memory. Any runtime
  that re-plans residency dynamically ("does it still fit?") will thrash
  against it; a runtime that allocates once at start and never re-plans
  will not. Measured baseline after reboot: 36 GB total, ~13 GB desktop
  baseline, 26b = 17 GB wired at 8K ctx (19 GB at 32K).
- **The app needs per-request truth.** Warm-turn gates need cache-hit
  evidence from the response, not from a side probe: llama-server returns
  `timings.cache_n`/`prompt_n` on `/v1/chat/completions`; Ollama's `/v1`
  returns nothing usable ([#16428](https://github.com/ollama/ollama/pull/16428)
  made `prompt_eval_count` include cached tokens, so only durations tell).

### Requirements (unchanged latency budget, new lifecycle requirements)

1. Warm TTFT p50 ≤ 0.4 s / p95 ≤ 0.6 s with the ~1.1 K-token PoC prefix
   (2.2 K with the full 18-tool production set); decode ≥ 40 tok/s.
2. **Streaming tool calls in OpenAI form** (`tool_calls` deltas), since the
   FlowCat `OpenAiLlm` adapter and the harness's T3/T4 tests consume them.
3. **Residency = chatbot lifetime**: loaded before the first call, never
   unloaded while running, released on exit; immune to other clients.
4. Memory planning fixed at start (no runtime re-planning/eviction).
5. Per-request cache evidence; context size as a flag.
6. No Python in the runtime path (the harness stays Python).

## Candidates (August 2026)

| Runtime | Engine / weights | Gemma 4 tool calls (streaming) | Prefix cache | Residency & memory behaviour | Fit |
|---|---|---|---|---|---|
| **Ollama 0.32.5** (incumbent) | llama.cpp runner, GGUF Q4_K_M 17 GB | mature built-in renderer/parser (`renderer=gemma4 parser=gemma4`) | works (ADR-0003); loses it on truncation | keep-alive via env only; scheduler may evict/reload the resident model when GPU "available" changes; shared with any client | **dev profile** |
| **`llama-server` b10280** (brew `llama.cpp`, installed) | same llama.cpp, same GGUF (`ggml-org/gemma-4-26B-A4B-it-GGUF`, 14 GB Q4_0 cached; Q4_K_M 17 GB) | `--jinja` + PEG parser for Gemma 4 ([PR #21326](https://github.com/ggml-org/llama.cpp/pull/21326), tokenizer [PR #21343](https://github.com/ggml-org/llama.cpp/pull/21343)); open edge cases: array args with `{}` in strings ([#21384](https://github.com/ggml-org/llama.cpp/issues/21384)), malformed `<tool_call\|>` leaking as text ([#21882](https://github.com/ggml-org/llama.cpp/issues/21882)) | `--cache-prompt` (default), `--cache-reuse N`, `--ctx-checkpoints`/`--swa-checkpoints`, `--cache-ram`, `/slots` save/restore; **`timings.cache_n` per response** | model loaded once for the process; no keep-alive concept; `-c`, `-np`, KV types are flags; nothing re-plans | **primary** |
| **Rapid-MLX** (Apache-2.0, Python/MLX) | MLX 4-bit (~14 GB download; ~14–18 GB resident) | own Gemma 4 parser incl. the bare-numeric-argument quirk; "100 % tool calling" claimed | radix prompt cache; 0.26 s TTFT / 85 tok/s reported on M3 Ultra; "2–10× on follow-ups" | resident for process life (server) | **A/B**, not primary: Python runtime, young project, MLX weights ≠ ADR-0001's evaluated GGUF |
| `mlx_lm.server` | MLX | Gemma 4 tool parser bug [#1125](https://github.com/ml-explore/mlx-lm/issues/1125) | message-boundary checkpoints (PR #911); rotating KV cannot be quantized | resident | no |
| LM Studio (`lms`/llmster headless) | MLX or GGUF | Gemma 4 26B tool use reported broken in 0.4.13 GUI ([lmstudio-bug-tracker #1927](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1927)) | yes | closed-source app; JIT load/auto-unload TTL semantics like Ollama's | no |
| vLLM / **vllm-metal** 0.2 | MLX plugin | mainline Gemma 4 parser has open **streaming** bugs ([#42696](https://github.com/vllm-project/vllm/issues/42696), [#44522](https://github.com/vllm-project/vllm/issues/44522), [#39089](https://github.com/vllm-project/vllm/issues/39089)); Gemma 4 on the Metal backend in progress, MoE routing unconfirmed | automatic prefix caching (excellent on CUDA) | pre-allocated KV pool (`gpu_memory_utilization`) competes with TTS; heavy PyTorch process | no on Apple Silicon; **the right answer if the target becomes NVIDIA** |
| mistral.rs (Rust) | safetensors/UQFF; Gemma 4 **GGUF unsupported** ("Unknown GGUF architecture", [#2171](https://github.com/EricLBuehler/mistral.rs/issues/2171)) | OpenAI tool calling | prefix cache | could be in-process | later bake-off; not now |
| in-process llama.cpp bindings (`llama-cpp-2`) | same GGUF | **no tool-call parser** in the C API (it lives in the server's C++ `common/chat`) | manual | in-binary | no — would re-implement the parser |
| SGLang | — | no macOS backend | — | — | no |

### Evidence from this repo

- **llama-server already ran this stack.** 2026-08-07, b10280,
  `ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0`, `-c 32768`: FlowCat T5 barge-in
  passed (`reply_start_latency` 4.96 s — LLM-bound long reply, same as
  Ollama's 4.6–6.3 s this week), prefill **~1,000 tok/s** (938 tokens in
  0.93 s), decode **86–97 tok/s**, later turns 27–43 prompt tokens in
  0.13–0.15 s (cache hits) — `poc/logs/llama-server.log`,
  `poc/reports/runs.jsonl`.
- **Ollama prompt-cache numbers** (ADR-0003): warm TTFT 0.33–0.37 s, miss
  2.4–2.6 s, decode 80–89 tok/s. Same engine → expect parity from
  llama-server; the difference is control, not speed.
- **Lifecycle failures** above: all reproduced in logs on 2026-08-25.

## Decision (proposed)

1. **Runtime:** `llama-server` from brew `llama.cpp` (pin the build in the
   README; b10280 is installed and known-good with this GGUF), serving
   `ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M` (ADR-0001's evaluated quant;
   the Q4_0 already cached is acceptable for bring-up, not for gates).
   Flags, from ADR-0003 §B plus this week's findings:
   `-ngl 99 -np 1 -c 8192 --jinja -fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256
   --swa-checkpoints 8 --load-mode mlock --port 8089 --no-webui`
   (`-c 8192`: the prompt is ~1.1–2.2 K tokens and the session is wiped at
   idle timeout; KV at 8K keeps ~2 GB off the 32K figure; q4 KV degrades tool
   calls per ADR-0003).
2. **Ownership:** the Rust chatbot spawns and supervises it (lifecycle plan
   Layer 2), warms the exact prefix once at startup (Layer 1), and kills it
   on exit. Residency is the process; "pin" and "unload" cease to exist as
   API concerns. `POC_LLM_PROVIDER=llama-server` becomes the default local
   profile; `ollama` stays selectable for ADR-0003 continuity;
   `openrouter` stays for the cloud profile.
3. **Adapter:** the existing FlowCat `OpenAiLlm` (`/v1/chat/completions`,
   streaming, `tool_calls` deltas) — no new LLM service needed. Read
   `timings.cache_n`/`prompt_n` from the final chunk into `Metrics(LlmUsage)`
   so the harness can gate cache hits per turn. (This makes the native-Ollama
   service in the lifecycle plan's Task 1 *optional*: it is only needed if
   the `ollama` profile must be immune to serve env, which a dev profile need
   not be.)
4. **Not adopted:** vLLM/vllm-metal (Python, immature Gemma 4 on Metal,
   streaming-parser bugs, KV pre-allocation vs TTS); mlx_lm.server and LM
   Studio (tool-call defects, closed app); mistral.rs and in-process bindings
   (no Gemma 4 GGUF / no parser) — re-check at the triggers below.
5. **A/B, time-boxed to one day:** Rapid-MLX serving
   `mlx-community/gemma-4-26b-a4b-it-4bit`, same harness, same gates; adopt
   only if it beats llama-server on warm TTFT p95 *and* passes T3/T4 tool
   tests *and* fits the memory budget with Qwen resident. Its Python runtime
   is the standing objection; a win must be large to overrule it.

## Consequences

- The `ollama_ctl.py` failure taxonomy (unpinned serve, wrong context
  length, foreign app on the port, runner swap, scheduler eviction) is
  eliminated rather than handled: a child process with flags has none of
  those states. The lifecycle plan shrinks to spawn/health/warm/kill.
- Model files move from Ollama's blob store to the HF cache
  (`llama-server -hf …` downloads on first run; `setup.sh` pre-fetches).
  Disk: +17 GB until the Ollama blob is removed.
- Two engines, one model: Ollama's `gemma4:26b` and the GGUF are the same
  quantization family but different files; results must name the file.
- **Risk accepted:** llama.cpp's Gemma 4 PEG parser has open edge cases
  (#21384, #21882). The harness's T3/T4 tests plus a new streaming
  tool-call fuzz (string args containing `{}`, multi-tool turns) are the
  gate, and Ollama remains one env var away if a regression lands.
- Multi-client sharing is gone by design (the chatbot owns the server on a
  private port); anything else wanting the LLM talks to the chatbot or runs
  its own.

## Test strategy — gates before Status → Accepted

| Gate | Method | Pass |
|---|---|---|
| G1 tool calls, streaming | `pytest harness -m tools` against llama-server (`--jinja`), plus 20 fuzzed tool turns (braces in strings, two tools in one reply) | all pass; zero raw `<|tool_call>` text in `LlmText` |
| G2 warm TTFT | 20-turn stable session via the harness latency marker; `timings.cache_n ≥ prefix tokens` on turns 2+ | p50 ≤ 0.4 s, p95 ≤ 0.6 s |
| G3 lifecycle | `make` from a cold machine → first turn; `make down` → `/health` gone, wired memory back; start Qwen 1.7B mid-session → no reload, no TTFT change | as stated |
| G4 barge-in / duplex | `pytest -m duplex` | T5 stop ≤ 400 ms |
| G5 parity | same session on `ollama` profile | llama-server TTFT ≤ Ollama's, decode ≥ 40 tok/s |

## Re-evaluation triggers

- llama.cpp Gemma 4 parser regression that G1 cannot be made to pass on a
  pinned build → fall back to the `ollama` profile (env-pinned) while it is
  fixed.
- Rapid-MLX A/B wins by ≥ 30 % on warm TTFT p95 with G1 green and the
  memory budget met → reopen for the MLX path.
- mistral.rs gains Gemma 4 GGUF + a Gemma 4 tool parser → in-process
  bake-off (same gates as the Qwen3-TTS Rust port).
- Deployment target moves to NVIDIA → vLLM, with automatic prefix caching
  and `cached_tokens` in usage, becomes the default runtime.

## Sources

- llama.cpp: [server README (cache-reuse, ctx/SWA checkpoints, slots, `timings.cache_n`, `--jinja`)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md); [PR #21326 Gemma 4 template/PEG parser](https://github.com/ggml-org/llama.cpp/pull/21326); [PR #21343 tokenizer fix](https://github.com/ggml-org/llama.cpp/pull/21343); [#21384 array args with braces](https://github.com/ggml-org/llama.cpp/issues/21384); [#21882 Gemma 4 edge cases](https://github.com/ggml-org/llama.cpp/issues/21882); [Gemma 4 26B tool calling on macOS walkthrough](https://gist.github.com/daniel-farina/87dc1c394b94e45bb700d27e9ea03193); [`--cache-ram` explainer](https://jessequinn.info/blog/llama-cpp-cache-ram-prompt-caching).
- Ollama: [#2963 options/keep_alive on OpenAI endpoints](https://github.com/ollama/ollama/issues/2963); [#11458 keep_alive ignored via OpenAI SDK](https://github.com/ollama/ollama/issues/11458); [#10263 prevent keep_alive overwrite](https://github.com/ollama/ollama/issues/10263); [#16428 cached tokens in counts](https://github.com/ollama/ollama/pull/16428); [#17829 MLX runner has no prefix cache](https://github.com/ollama/ollama/issues/17829); this repo's `poc/logs/ollama.log` (2026-08-25 eviction/reload lines).
- Rapid-MLX: [repository](https://github.com/raullenchai/Rapid-MLX); [Gemma 4 on Apple Silicon post](https://dev.to/raullen_chai_76e18e9705b0/gemma-4-on-apple-silicon-85-toks-with-a-pip-install-299a); [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit).
- vLLM: [vllm-metal](https://github.com/vllm-project/vllm-metal); [Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html); [#42696](https://github.com/vllm-project/vllm/issues/42696), [#44522](https://github.com/vllm-project/vllm/issues/44522), [#39089](https://github.com/vllm-project/vllm/issues/39089); [SGLang Gemma 4 MLX issue](https://github.com/sgl-project/sglang/issues/32101).
- Others: [mlx-lm #1125](https://github.com/ml-explore/mlx-lm/issues/1125); [LM Studio bug #1927](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1927); [mistral.rs #2171 Gemma 4 GGUF](https://github.com/EricLBuehler/mistral.rs/issues/2171); [mistral.rs tool calling](https://ericlbuehler.github.io/mistral.rs/TOOL_CALLING.html).
