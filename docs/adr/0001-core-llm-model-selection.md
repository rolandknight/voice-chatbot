# ADR-0001: Core (always-resident) LLM model selection

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-06 |
| **Decision** | Keep **Gemma 4 26B-A4B** (`gemma4:26b`) as the core always-resident model |
| **Related** | PRD §4.1 (CONV-2/3/4, latency targets), `config.yaml` `llm:`, `README.md` "Latency optimization knobs" |

---

## Context

The assistant uses a two-tier LLM design (PRD CONV-2/3): a **fast tier** — a local, always-resident model that handles common requests (timers, weather, time, media control) and fires skill tool-calls — and a **slow tier** (Claude via API) for complex prompts. This ADR covers the fast-tier core model. The cloud escalation model and the low-RAM fallbacks are noted at the end but are not the decision here.

### Constraints

1. **Memory budget: ≤ 24 GB resident.** Host is a Mac Studio with 36 GB unified memory. The core model is pinned resident (`ollama_keep_alive: "-1"`) so it never pays a cold-load penalty (~6–9 s). The remaining ~12 GB must hold Whisper MLX STT, Kokoro ONNX, the Chatterbox-Turbo server (MPS), optional SFX backends, mpv, the FastAPI/Pipecat server, and the OS. The 24 GB figure is a hard ceiling *including KV cache at our operating context length*, not just weights on disk.
2. **Tool-calling quality is the primary capability.** Skills are LLM tool calls (PRD §4.2). The model must reliably select and correctly parameterize tools from ~15 schemas per turn (post-`SkillFilterProcessor`), including indirect phrasings. Benchmarks of record: τ²-bench and BFCL.
3. **Latency: warm TTFB ≤ 0.4 s** with the babel system prompt + tool schemas in context (PRD latency table). This effectively **disqualifies models that emit reasoning/thinking tokens before acting** — a "thinking tax" of even a few hundred tokens adds seconds before the first tool call or spoken word.
4. **Ollama-served, OpenAI-compatible.** The pipeline talks to `http://localhost:11434/v1`; the model must be mature on Ollama (correct tool-call parsing in the chat template, stable GGUF).
5. **License** must permit personal/self-hosted use without friction; open weights strongly preferred (Apache 2.0 ideal).
6. **Nice-to-have:** vision input (future multimodal web UI, PRD CLI-4) and long context.

### Incumbent

`gemma4:26b` (Gemma 4 26B-A4B): 25.8 B total / ~3.8 B active MoE (8 of 128 experts + 1 shared), Q4_K_M, **18 GB on disk / ~17 GB resident**, 256 K context, tools + vision, Apache 2.0.

Measured in this repo (M4 Max, babel system prompt + tool schemas, after pre-warm):

- **Warm TTFB ~0.37 s**; fires tools without emitting chain-of-thought first.
- **τ²-bench 85.5 %** (project measurement; published figures range 68.2–88.3 depending on harness — see "Benchmark caveats").
- Reliable on indirect phrasings, unlike the smaller E4B variant which flips into thinking mode and can spike to 4 s+.

---

## Research: candidates (August 2026)

Deep-research pass over current leaderboards, vendor releases, and local-LLM roundups. Candidates evaluated against the constraints above.

### Fits the 24 GB budget

| Model | Total/active params | Resident (Q4-class) | Tool-calling signal | Thinking tax | Ollama maturity | Notes |
|---|---|---|---|---|---|---|
| **Gemma 4 26B-A4B** (incumbent) | 25.8 B / 3.8 B | **~17 GB** | τ² 85.5 % measured in-situ; "best tool-call reliability" in our testing | **None** (no CoT before tool calls) | Mature (official `gemma4:26b`) | Vision + tools; Apache 2.0; 256 K ctx |
| **Qwen3.6-35B-A3B** (Apr 2026) | 35 B / 3 B | ~18 GB weights IQ4_XS; **~20–21 GB** with 16 K ctx + q4 KV | Best-in-class agentic gen: SWE-bench Verified 73.4; strong BFCL/τ² (2-bit quant reportedly matches bf16 teacher) | **Thinking ON by default**; must set `enable_thinking: False`; non-thinking tool-call quality not yet benchmarked in our harness | Conflicting reports: native support claimed at launch vs. llama.cpp-from-source required; community GGUFs on ollama.com | ~30 % slower decode than Qwen3.5; tightest memory fit of the viable set |
| **Qwen3-30B-A3B-Instruct-2507** | 30 B / 3.3 B | ~18 GB | "Significant improvements in tool usage" (vendor); non-thinking-only variant | **None** (instruct-only, no thinking blocks) | Mature | A generation older than Qwen3.6; the credible like-for-like alternative to the incumbent |
| **gpt-oss-20b** | 21 B / 3.6 B | **~13–14 GB** (MXFP4) | 97.0 % accuracy in a 2026 voice-agent tool-calling bench — top of that cohort | **Reasoning model**: adjustable effort (low/med/high) but emits reasoning tokens even at low; measured **1.16 s** per turn on Ollama (0.46 s on vLLM — a 2.5× runtime gap we'd inherit) | Mature | Best memory headroom; Harmony-format tool-call quirks reported; Apache 2.0 |
| Gemma 4 31B (dense) | 31 B / 31 B | ~24 GB — no headroom | τ² 86.4 % (below the 26B MoE's published 88.3 %) | None | Mature | Dense → ~8× more active params per token than the MoE → slower decode; strictly dominated by 26B-A4B for this use |
| Qwen 3.5 9B / Gemma 4 E4B | 9 B / 4.5 B class | 6–10 GB | Good-not-great; E4B spikes on indirect phrasings (thinking mode) | E4B: yes, intermittently | Mature | Fallback tier only (already documented in `config.yaml`); see "Gemma 4 edge (E-series)" below |

### Gemma 4 edge (E-series) models — assessed, not core candidates

The E-series ("effective"-parameter edge models) was evaluated for three possible roles:

| Model | Params (total/effective) | Resident | Measured latency | Verdict |
|---|---|---|---|---|
| **Gemma 4 E4B** (`gemma4:latest`) | 8.0 B / 4 B | ~9.6 GB | TTFT ~0.3–0.35 s direct; **spikes to 4 s+ on indirect phrasings** (reasoning tokens before the tool call — measured in this repo) | Remains the documented **low-RAM fallback**, not the core model. On a 36 GB host it saves ~7 GB we don't need while giving up the incumbent's tool-call reliability and latency consistency. |
| **Gemma 4 E2B** (`gemma4:e2b`) | 5.1 B / 2 B | ~5–6 GB | **TTFT ~5.8 s on Ollama by default** — the `gemma4` renderer quietly injects `<|think|>` for E2B, generating 500+ hidden reasoning tokens; drops to ~0.73 s with `think=False` | **Rejected for every role.** Even de-thinked, 0.73 s TTFT misses the ≤ 0.4 s budget, and 2 B effective params is below the reliability bar for 15-schema tool selection. |
| E-series on the **RPi 5 satellite** (on-device LLM) | — | Pi 5 8 GB | E4B impractical (~9.6 GB > 8 GB; ~2–4 tok/s); E2B TTFT 3–4 s, 8–12 tok/s | **Rejected.** Satellites stay thin clients (wake word + WebRTC); the LLM stays on the server. No E-series model meets the latency budget on Pi-class hardware. |

Two E-series facts worth recording:

1. **Thinking-mode behavior differs by variant and harness.** Third-party benching reports E4B *ignores* thinking parameters (TTFT ~0.3 s) while E2B hidden-thinks by default; this repo's own measurement found E4B emitting reasoning tokens on *indirect* phrasings (4 s+ spikes). The reconciliation is likely prompt-dependent thinking. Practical rule if the fallback is ever used: pass `think=False` explicitly and re-measure in-pipeline — do not trust variant-level claims.
2. **E-series models take native audio input** (E2B/E4B/12B). Irrelevant to this decision, but a possible future path to a speech-native pipeline stage (e.g., audio-in classification or diarization assist, PRD SPKR-1) without a separate encoder. Not scored here.

### Ruled out on memory alone

- **Llama 4 Scout** — 109 B total / 17 B active; ~80 GB class. Unique 10 M context is irrelevant to a voice turn loop.
- **Mistral Small 4** — 119 B total / 6 B active MoE; ~48 GB at Q4. Excellent latency profile per vendor, but doesn't fit.
- **Command-R 35B, Llama-3-Groq-70B** — over budget and/or a generation stale on agentic benchmarks.

### Benchmark caveats

Published τ²-bench numbers for the same model vary widely by harness (e.g. 68.2 on the Ollama model card vs. 88.3 in vendor-adjacent coverage vs. **85.5 measured here** with the Sierra harness convention). Cross-vendor comparisons from blog roundups are directional at best. **The project's own measurement protocol is authoritative**: τ²-bench base tasks with thinking off + warm-TTFB measured in-pipeline with the babel system prompt and top-K tool schemas loaded (see Re-evaluation §).

---

## Decision

**Keep Gemma 4 26B-A4B (`gemma4:26b`) as the core model.** No challenger offers a latency-adjusted win today:

1. **It is the only candidate with a proven ≤ 0.4 s warm TTFB in this pipeline.** gpt-oss-20b's measured 1.16 s/turn on Ollama is ~3× our budget (its 0.46 s vLLM number would require abandoning Ollama — out of scope). Qwen3.6's thinking-by-default needs to be disabled and re-validated before it can even be scored.
2. **Tool-calling is excellent in-situ** (85.5 % τ², reliable indirect phrasings) — the metric that matters more than any leaderboard delta.
3. **Memory fit is comfortable**: ~17 GB resident leaves ~7 GB of the 24 GB LLM budget for KV cache growth and Ollama overhead. Qwen3.6 at ~20–21 GB with only 16 K context is a knife-edge fit under a hard 24 GB ceiling.
4. **MoE economics match the workload**: ~3.8 B active params per token gives small-model decode speed with large-model routing quality — the same reason Qwen3.6/gpt-oss are shaped this way.
5. **Operationally boring**: official Ollama model, Apache 2.0, and vision support banked for the future multimodal UI.

**Runner-up (documented fallback if the incumbent regresses): Qwen3-30B-A3B-Instruct-2507** — same memory class, non-thinking by design, mature Ollama support. It is the model we would bench first if Gemma 4 26B misbehaves, ahead of Qwen3.6.

**Watch item: Qwen3.6-35B-A3B** is the strongest agentic model that fits, and the one most likely to displace the incumbent — *if* (a) non-thinking mode preserves its tool-calling scores, (b) Ollama integration stabilizes, and (c) resident-with-KV stays under 24 GB at our operating context.

## Consequences

- `config.yaml` keeps `ollama_model: gemma4:26b`, `ollama_keep_alive: "-1"`; no changes required.
- Context length must stay capped (conversation context is wiped at `idle_timeout_secs` anyway) so KV cache cannot push residency past 24 GB. Do not raise the operating context without re-measuring residency.
- The MCP-client question (PRD CONV-7, `todo.md`) stays open: if `gemma4:26b` proves weak as an MCP client, that becomes a re-evaluation trigger for Qwen3.6 (whose agentic-coding pedigree suggests stronger MCP behavior) rather than a reason to route more turns to Claude by default.
- We accept a possible raw-capability gap vs. Qwen3.6 on complex agentic tasks — by design, those escalate to the slow tier (Claude) anyway.

## Re-evaluation triggers and protocol

Re-open this ADR when any of:

1. A new open-weights release claims ≥ 5-point τ²/BFCL improvement in a ≤ 20 GB-resident, non-thinking (or zero-cost-thinking) configuration.
2. Qwen3.6 lands officially in Ollama with a tool-call parser and a documented non-thinking mode.
3. Fast-tier misfires (wrong tool / bad args) become noticeable in daily use, or MCP-client support (CONV-7) is blocked on model quality.
4. The memory budget changes (host upgrade, or voice-model footprint growth squeezing the 24 GB allocation).

**Bench protocol for any challenger** (all three must pass before switching):

1. τ²-bench base tasks, thinking off, ≥ incumbent −2 points.
2. In-pipeline warm TTFB ≤ 0.4 s with babel system prompt + 15 tool schemas (measure p50/p95 over ≥ 20 turns after pre-warm).
3. Resident memory (weights + KV at operating context) ≤ 24 GB under `ollama ps` after a 50-turn session, plus a week of daily-driver use with no regression in indirect-phrasing tool selection.

## Related decisions (out of scope here, candidate future ADRs)

- **Slow-tier cloud model** — currently `claude-sonnet-4-6` with server-side web search/fetch (`config.yaml claude:`).
- **Low-RAM fallback ladder** — `gemma4:latest` (E4B) → `qwen2.5:3b`, already documented in `config.yaml`.
- **STT/TTS model choices** — Whisper MLX variants, Kokoro vs. Chatterbox (see `personas.yaml`, PRD §4.7).

## Sources

- Project measurements: `README.md` ("Latency optimization knobs", skills section), `config.yaml` comments.
- [Gemma 4 26B on Ollama](https://ollama.com/library/gemma4:26b) — size, architecture, model-card benchmarks.
- [google/gemma-4-26B-A4B-it (Hugging Face)](https://huggingface.co/google/gemma-4-26B-A4B-it); [Gemma 4 technical overview (Labellerr)](https://www.labellerr.com/blog/gemma-4-open-weight-ai-model-overview/) — τ² 88.3 % (26B MoE) vs. 86.4 % (31B dense).
- [Qwen3.6-35B-A3B (Hugging Face)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B); [release blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b); [24 GB deployment writeup (aminrj.com)](https://aminrj.com/posts/llamacpp-qwen36-35b/); [2-bit quant agentic evals (SyzygyResearch)](https://huggingface.co/SyzygyResearch/Qwen3.6-35B-A3B-2bit).
- [Qwen3-30B-A3B-Instruct-2507 (Hugging Face)](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) — non-thinking-only, improved tool use.
- [gpt-oss-20b (Ollama)](https://ollama.com/library/gpt-oss:20b); [openai/gpt-oss (GitHub)](https://github.com/openai/gpt-oss); voice-agent latency comparison (petronellatech.com, gpt-oss-20b voice-agent bench: 97.0 % @ 1.16 s Ollama vs 0.46 s vLLM).
- Landscape roundups: [Local Ollama tool-calling ranking](https://localaimaster.com/blog/best-ollama-models-tool-calling), [Gemma 4 vs Llama 4 vs Mistral Small 4](https://www.digitalapplied.com/blog/gemma-4-vs-llama-4-vs-mistral-small-4-comparison), [Best local LLMs 2026 (StationX)](https://app.stationx.net/articles/best-local-llm).
- E-series: [gemma-4-E4B-it (HF)](https://huggingface.co/google/gemma-4-E4B-it), [gemma4:e2b (Ollama)](https://ollama.com/library/gemma4:e2b), [E2B vs E4B hidden-thinking benchmark (theKodeLab)](https://thekodelab.com/en/posts/gemma4-e2b-vs-e4b-benchmark/), [Gemma 4 on Raspberry Pi (alanwest)](https://dev.to/alanwest/gemma-4-runs-on-a-raspberry-pi-i-tested-it-56c5), [E2B/E4B edge deployment (MindStudio)](https://www.mindstudio.ai/blog/gemma-4-edge-deployment-e2b-e4b-models).
