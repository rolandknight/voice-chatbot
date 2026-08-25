# poc-qwen-llm — OpenAI-compatible LLM endpoint for the chatbot on the M4 Max: Qwen3.8 (and the models it must beat), resident weights, lowest TTFT, tool calling

**Date:** 2026-08-24 · **Branch:** `poc-qwen3-tts` (or a new `poc-qwen-llm`) · **Dir:** `poc-qwen-llm/`

## Goal

Stand up **only** the OpenAI-compatible `/v1/chat/completions` server that the
voice pipeline (`server.py` → `llm.ollama_base_url`) will talk to, and answer
with numbers on this box (Mac Studio M4 Max, binned 32-core GPU, 410 GB/s,
**36 GB** unified, macOS 26.4):

1. **TTFT** with the babel system prompt + top-15 tool schemas in context —
   warm (prefix cached), and cold-prefix — p50/p95 over ≥ 20 turns.
2. **Decode tok/s** (with and without speculative/MTP decoding).
3. **Tool calling**: `tools=[…]` in, structured `tool_calls` out, streamed,
   no thinking tokens — across the ~15-schema skill set, incl. indirect
   phrasings (the failure mode that ruled out Gemma 4 E4B in ADR-0001).
4. **Residency**: weights + KV pinned (wired) so TTFT never includes a page-in,
   with TTS and STT resident alongside; proven by `vm_stat`/`footprint`, not
   by hope.

Targets carried from the PRD/ADR-0001: **warm TTFT ≤ 0.4 s**, tool-call
reliability ≥ `gemma4:26b`'s (85.5 % τ² in-situ), residency ≤ the LLM budget
below.

## Research summary (2026-08-24)

### What "Qwen3.8" actually is

Qwen3.8 (Aug 2026) ships **two** open checkpoints: **Qwen3.8-27B** (dense,
hybrid Gated-DeltaNet + gated attention, multimodal, Apache 2.0, 262 K ctx)
and Qwen3.8-2.4T-A95B (irrelevant here). **There is no 8–9 B or 30B-A3B
Qwen3.8.** Implications for this box:

| | Qwen3.8-27B |
| --- | --- |
| Weights | MLX 4-bit **16.1 GB** (`mlx-community/Qwen3.8-27B-4bit`, `-MTP-4bit`); GGUF UD-Q4_K_M 16.5 / UD-Q3_K_XL 13.1 / Q4_K_M 19.0 GB; Ollama `qwen3.8:27b-mlx` 18 GB |
| KV cache | ~64 KB/token (hybrid linear attention) → 0.5 GB at 8 K |
| Thinking | **on by default, `xhigh` effort**; must send `chat_template_kwargs: {enable_thinking:false}` (no `/no_think`); a Simon Willison run took 21 min with thinking on |
| Tool format | Qwen3-Coder XML `<tool_call><function=…><parameter=…>`; parser `qwen3_coder` (vLLM/mlx-lm); llama.cpp auto-derives it (`--jinja`); stock template rejects mid-conversation system messages |
| Speed, M4 Max | decode ~20–30 tok/s plain 4-bit; **50–72 tok/s with MTP** (oMLX/MTPLX/llama.cpp `--spec-type draft-mtp`); prefill ~270 tok/s (dense 27B) → a 2 K-token prompt costs ~7 s **cold** |
| Prefix caching | **Broken/fragile for hybrid GDN models** in mlx-lm (#980), llama.cpp (#20225, #24055 — checkpoints invalidated), oMLX (#825 — cache hit breaks tool calling). Only Ollama's MLX runner and vllm-metal claim it works; unverified |
| Tool-call benches | none published (no BFCL/τ²); Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7 — agentic-strong, tool-call reliability for our schema set unknown |

The two problems compound: a dense 27B prefills slowly **and** the thing that
makes TTFT independent of prompt length (prefix reuse) is unreliable for its
architecture. Qwen3.8-27B can only meet ≤ 0.4 s TTFT if the engine's
cross-turn cache verifiably hits. That is the central experiment.

### Memory budget on 36 GB (this is why the ADR's 24 GB LLM budget no longer holds)

| Resident component | Measured / expected | Source |
| --- | --- | --- |
| macOS + desktop + compressor headroom | ~5 GB | reserve |
| Qwen3-TTS 1.7B-Base (mlx-audio) | **9.9 GiB peak**, 10.4 GiB with 2 variants (LRU) | `poc-qwen/bench-m4-max.md` |
| Nemotron Speech Streaming en 0.6B q8_0 (NeMo-Speech.cpp, Metal) | 0.7 GB GGUF; **~1–1.5 GB resident** (no published RSS; MLX INT8 analog peaks 0.8 GB, fp16 MLX port 3.4 GB) — measure in Task 0 | HF model card, `speech-swift`, `nemotron-asr-mlx` |
| Pipecat server, whisper-tiny fallback, mpv, openWakeWord | ~1.5 GB | current `run.sh` stack |
| **Left for LLM (weights + KV + activations)** | **≈ 17–18 GB** | |

Constraints on pinning: MLX's default wired ceiling here is
`max_recommended_working_set_size` = **28.1 GiB** (`mx.device_info()`),
`iogpu.wired_limit_mb=0` (default). TTS (10.4) + LLM (≤ 18) = 28.4 → at the
edge; either run TTS with the 0.6B model (8.0 GiB) or raise the ceiling
(`sudo sysctl iogpu.wired_limit_mb=30720`, leave ≥ 5 GB for the OS,
non-persistent → LaunchDaemon). `ulimit -l` is unlimited on macOS, so
`mlock` needs no entitlement.

Fit verdict: Qwen3.8-27B 4-bit (16.1 GB + ~1.5 GB KV/activations) fits with
**zero headroom** next to 1.7B TTS; 3-bit (13.1 GB GGUF) or TTS-0.6B buys
~3 GB. Anything 8-bit is out.

### Comparators that must be in the same harness

Same tool format / same parser, so one bench script covers all:

| Model | Why it is in the bench | Resident (4-bit) | Prefix cache | Decode (M4 Max class) |
| --- | --- | --- | --- | --- |
| **Qwen3.8-27B** 4-bit (+MTP) | the ask | ~17.5 GB | hybrid → verify | 20–30 / 50–70 (MTP) |
| **Qwen3.5-9B** 4-bit (`qwen3.5:9b` is already pulled; `mlx-community/Qwen3.5-9B-4bit`) | only small Qwen with published tool numbers: **BFCL-V4 66.1, τ²-bench 79.1**; ~6 GB → 11 GB headroom; same hybrid arch (tests the cache question cheaply) | ~6 GB | hybrid → verify | ~60 |
| **Qwen3-30B-A3B-Instruct-2507** 4-bit | ADR-0001 runner-up; pure GQA → prefix cache works everywhere; non-thinking by design; 100–130 tok/s on mlx-lm | ~17–18 GB | works | 100–130 |
| `gemma4:12b-mlx` via Ollama (current default) | baseline the pipeline runs today | 7.7 GB | Ollama MLX runner | measured ~0.3–0.37 s TTFT in-pipeline |

### Server choice

| Server | Prefix cache | Tool parsing | Spec decode | Residency | Notes |
| --- | --- | --- | --- | --- | --- |
| **`mlx_lm.server`** (primary) | in-memory LRU trie (`--prompt-cache-size`), longest-prefix match; broken for hybrids as of #980 — **re-verify on current release** | `json_tools` (Hermes) and `qwen3_coder` parsers built in | `--draft-model`; no MTP head | calls `mx.set_wired_limit(max_recommended_working_set_size)` itself → weights+cache wired up to 28.1 GiB | fastest decode on Apple Silicon (126–132 tok/s on 35B-A3B); warm TTFT ~0.5 s on a 2 K chat |
| **`llama-server`** b10280 (already on PATH; runner-up) | per-slot prefix, `-np 1`, `--cache-reuse`, `/slots/0?action=save|restore`; hybrid checkpoints broken (#24055) | PEG auto-parser from template, handles Qwen XML; `-ctk q8_0` max (q4 KV degrades tool calls) | `--spec-type draft-mtp` with `ggml-org/…-mtp-Q4_0.gguf` (1.7 GB) — +75 % on dense 27B | `--load-mode mlock` (`--mlock` deprecated) | 1.7–2× faster **cold** prefill than MLX; 30–50 % slower decode on MoE |
| Ollama 0.32.5 (baseline only) | MLX runner has prefix trie + snapshots (claims hybrid support) | template-driven | automatic MTP for supported tags | `keep_alive:-1`; `use_mlock` undocumented; wired limit set by MLX runner | 25–50 % decode penalty; sequential runner; open bugs: KV accumulation (#16698), cold prefill 60–400× slower after idle (#16051) — the exact symptom `config.yaml`'s 60 s keepalive ping works around |
| vllm-mlx / oMLX / Rapid-MLX | trie + paged / SSD-tiered / radix | yes | MTP (oMLX, Rapid) | wired via MLX | oMLX: 1.2–1.5 s TTFT and #825; Rapid-MLX: hybrid cache stores but never serves (#163). Bench oMLX only if MTP on 27B is the deciding factor |

## Decision

- **Build the PoC as a thin launcher + bench harness around `mlx_lm.server`**,
  with `llama-server` as the second engine behind the same config, rather than
  writing our own server. Both already speak OpenAI chat + tools + SSE; the
  value is in the residency wrapper, the warm-up, the measurement, and the
  go/no-go — not in another HTTP layer. If a model needs a custom tool parser
  or a prompt-cache workaround, that becomes a small shim in front of
  `mlx_lm.server` (Task 6), not a fork.
- **Qwen3.8-27B is benched as asked, but the plan does not presume it wins.**
  It enters with two strikes (slow dense prefill, unreliable hybrid prefix
  cache) and zero memory headroom. The exit criteria decide.
- **Prefix caching is the load-bearing TTFT technique**, so the prompt is
  laid out for it: system prompt + tool schemas first and **byte-identical
  every turn** (the `SkillFilterProcessor` top-K set must be stable within a
  session, or sorted), conversation appended after. Warm-up sends the exact
  prefix at boot.
- **Residency is enforced three ways and verified**: (1) MLX wired limit /
  `--load-mode mlock`; (2) `iogpu.disable_wired_collector=1` during the bench
  so the wired collector can't evict GPU buffers under pressure; (3) a
  co-residency test with the real TTS and STT processes loaded, checking
  `vm_stat` "Pages wired down", `footprint -p`, swap and compressor deltas.
- Thinking off always (`enable_thinking:false`); sampling per Qwen's
  non-thinking recipe (temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5) for
  chat, `temperature 0` for tool turns as `server.py` does today.

## Tasks

### Task 0: Memory baseline of the neighbours (½ day)
**Files:** `poc-qwen-llm/scripts/residency.sh`, `reports/residency.jsonl`
- [ ] Start Qwen3-TTS 1.7B via `poc-qwen-streaming` (warm) and Nemotron via
      `nemo-speech serve --device metal` (`make poc-nemotron-setup`); record
      per-process `footprint -p`, `vm_stat` wired pages, swap. Fills the one
      unknown in the budget table (Nemotron RSS on Metal).
- [ ] Record `mx.device_info()` and `sysctl iogpu.wired_limit_mb`; decide
      whether the bench raises the wired limit (only if TTS 1.7B + LLM > 28 GiB).

### Task 1: Skeleton, env, model fetch (½ day)
**Files:** `poc-qwen-llm/{Makefile,mise.toml,requirements.txt,config.yaml,README.md}`
- [ ] mise-pinned Python 3.12 venv like `poc-qwen`; `mlx-lm` (current),
      `openai` client, `httpx`. Pin versions in the README when results land.
- [ ] `make models`: `mlx-community/Qwen3.8-27B-4bit`, `…-MTP-4bit`,
      `mlx-community/Qwen3.5-9B-4bit`, `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`;
      GGUFs `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M + `ggml-org` `mtp-Q4_0`,
      `Qwen3-30B-A3B-Instruct-2507` Q4_K_M. (~75 GB; `make models MODELS=…` to subset.)
- [ ] `config.yaml`: `engine: mlx|llama`, `model`, `port: 8010`, `ctx: 8192`,
      `wired_limit_mb`, `disable_wired_collector`, `warmup.prefix_file`,
      `bench:` section (turns, repeats). `POC_LLM_*` env overrides as in poc-qwen.

### Task 2: Launcher with residency guarantees (1 day)
**Files:** `poc_qwen_llm/launch.py`, `poc_qwen_llm/prefix.py`
- [ ] `make serve`: applies sysctls if configured (prints the `sudo` line
      rather than running it silently), then execs the engine:
      - mlx: `mlx_lm.server --model … --port 8010 --prompt-cache-size 8 --prefill-step-size 2048 --chat-template-args '{"enable_thinking":false}' [--draft-model …]`
      - llama: `llama-server -m … --load-mode mlock -fa on -np 1 -c 8192 --cache-reuse 256 -ctk q8_0 -ctv q8_0 --jinja --chat-template-kwargs '{"enable_thinking":false}' [--spec-type draft-mtp -md …-mtp-Q4_0.gguf] --slot-save-path reports/slots`
- [ ] `prefix.py` builds the babel system prompt + the top-15 skill schemas
      from `personas.yaml` / `skills/` exactly as `server.py` sends them
      (import `skills._loader`, sort tools by name) and dumps
      `reports/prefix.json` — the **byte-identical** prefix used by warm-up and
      every bench turn. Token count is reported (expect ~1.5–2.5 K).
- [ ] Warm-up: after `/health`, one non-streamed request with the prefix +
      "ping", then a second identical request; assert the second's TTFT is
      ≥ 5× lower than the first (**this is the prefix-cache-works check**; on
      the hybrid models it is expected to fail for some engines — record,
      don't crash).
- [ ] Residency proof at the end of warm-up: log `footprint -p <pid>`,
      wired-page delta vs. pre-launch, and refuse to report "resident" if
      swap or compressor grew by > 256 MB during load.

### Task 3: Bench harness (1 day)
**Files:** `poc_qwen_llm/bench.py`, `poc_qwen_llm/turns.py`, `reports/llm_runs.jsonl`
- [ ] 20-turn scripted session mirroring real use (`turns.py`): timers,
      weather, time, media control, two indirect phrasings per tool
      (“it's getting dark in here” → lights), three chit-chat turns, one
      multi-tool turn, one 400-token reply. Each turn is a streamed
      `/v1/chat/completions` call with `tools`, `stream=True`,
      conversation history appended (so the cache hit is the whole
      transcript, as in production).
- [ ] Per turn record: TTFT (first SSE delta with content **or** tool-call
      start), decode tok/s (from `usage` / token deltas), whether a tool call
      was emitted, parsed OK, correct tool, correct args; engine cache-hit
      evidence (mlx-lm: prompt tokens processed from server log; llama:
      `/slots` `n_prompt_tokens_processed`); `footprint` after the turn.
- [ ] Modes: `--cold-prefix` (restart server between turns, TTFT with a
      cold prefix — the number the pipeline sees after a context wipe),
      `--idle 300` (sleep 5 min then one turn — the "slow after idle" bug
      class), `--concurrent 2` (STT/TTS pipelines can overlap one turn with
      the next).
- [ ] Matrix (`make bench`): {Qwen3.8-27B, +MTP} × {mlx, llama}; Qwen3.5-9B
      × mlx; Qwen3-30B-A3B-2507 × {mlx, llama}; `gemma4:12b-mlx` × Ollama
      (control, current config). 3 repeats, median; cold rows tagged.

### Task 4: Co-residency run (½ day)
**Files:** `scripts/coresidency.sh`
- [ ] Repeat the warm bench for the top two candidates **with TTS 1.7B and
      Nemotron loaded and idle**, then with TTS actively streaming a 300-char
      utterance during the LLM turn (the real pipeline overlaps them at the
      turn boundary). Report TTFT delta and any wired-page eviction
      (wired count drop) or compressor growth. If eviction shows, rerun with
      `iogpu.disable_wired_collector=1` and note the difference.

### Task 5: Tool-calling fidelity (½ day, overlaps 3)
**Files:** `poc_qwen_llm/tools_eval.py`, `reports/tools.jsonl`
- [ ] 40 prompts across the 15 schemas (direct/indirect/no-tool-needed/two
      tools); score select-accuracy, arg-validity (JSON schema), false-tool
      rate on chit-chat, and **thinking leakage** (any `<think>` content or
      >0 reasoning tokens before the call). Run on each candidate; compare
      with `gemma4:26b` on Ollama as the ADR reference (it is pulled).
- [ ] Check the Qwen3.8 template issue with mid-conversation system messages
      (`server.py` may inject persona switches as system turns); if it bites,
      test `froggeric/Qwen-Fixed-Chat-Templates` via `--chat-template-file`.

### Task 6: Shim only if needed (optional, ½ day)
**Files:** `poc_qwen_llm/shim.py`
- [ ] If the winning engine lacks a parser for the winning model, or the
      cache needs a stable-prefix guard, put a ~150-line FastAPI proxy on
      :8011 in front: normalises `tools` ordering, strips `<think>` blocks,
      converts Qwen XML tool calls to OpenAI `tool_calls` deltas. Nothing
      else — this is not a server rewrite.

### Task 7: README + decision (½ day)
- [ ] `poc-qwen-llm/README.md` with the results table (warm TTFT p50/p95,
      cold TTFT, tok/s, tool-eval scores, resident GB with neighbours) and a
      go/no-go per model. Feed the ADR-0001 re-evaluation protocol: the
      three gates (τ²-class tool score, ≤ 0.4 s warm TTFT in-pipeline,
      residency under the new budget). Propose the `config.yaml` change
      (`ollama_base_url` → `http://127.0.0.1:8010/v1`) for a follow-up PoC that
      wires the pipeline; not done here.

## Exit criteria (per model × engine)

| Gate | Pass |
| --- | --- |
| Warm TTFT with prefix + 15 tools, 20-turn session | p50 ≤ 0.4 s, p95 ≤ 0.6 s |
| Prefix cache | ≥ 5× TTFT drop turn 1 → turn 2 **and** engine logs show only the new tokens prefilled on turns 3–20 |
| After 5 min idle | TTFT within 1.5× warm (no re-paging) |
| Decode | ≥ 40 tok/s (spoken reply of 60 tokens must finish before TTS needs it) |
| Tool calls | select-accuracy ≥ gemma4:26b − 2 pts on the 40-prompt set; 0 thinking leakage; 0 malformed calls |
| Residency | LLM footprint ≤ 18 GB with TTS 1.7B + Nemotron loaded; wired pages stable across the run; no swap growth |

Expected outcome (state it so it can be wrong): Qwen3.8-27B fails the TTFT
gate unless prefix caching works on its hybrid architecture in the current
mlx-lm/llama.cpp; Qwen3-30B-A3B-2507 on `mlx_lm.server` is the likely
winner on TTFT + decode; Qwen3.5-9B is the memory-comfortable fallback with
known tool numbers. If Qwen3.8-27B *does* cache correctly, its MTP variant on
llama-server (dense → +75 % decode) is the configuration to keep.

## Risks

| Risk | Signal | Fallback |
| --- | --- | --- |
| mlx-lm prefix cache silently no-ops on hybrid models | warm TTFT ≈ cold TTFT; server log prefills full prompt each turn | llama-server `-np 1` slot (per-slot cache); try Ollama `qwen3.8:27b-mlx` (claims hybrid snapshots) as a data point; else drop to a GQA model |
| Cache hit corrupts tool calling on hybrids (oMLX #825 class) | tool eval passes cold, fails warm | same as above; do not ship a hybrid until fixed upstream |
| 27B + 1.7B TTS exceeds the 28.1 GiB wired ceiling | `set_wired_limit` error, or wired pages < model size, compressor growth | 3-bit GGUF (13.1 GB), TTS 0.6B (−2 GB), or `iogpu.wired_limit_mb=30720` |
| Thinking not actually disabled through a given engine's template path | `<think>` in stream / long TTFT with no visible tokens | pass `reasoning_effort:"none"` too; `--reasoning-budget 0` on llama; fixed template file |
| Qwen3.8 template rejects mid-conversation system messages | 400 from server on persona-switch turns | fixed template (froggeric); or fold persona switches into a user-role note in the shim |
| MTP draft mismatch / slower on MoE | tok/s ≤ baseline with `--spec-type draft-mtp` | disable spec decode for MoE (measured +12 % only on 35B-A3B; +75 % on dense) |
| Wired collector evicts GPU buffers under pressure during TTS bursts | wired pages drop mid-run; TTFT spike | `iogpu.disable_wired_collector=1` (document as a LaunchDaemon sysctl if kept) |

## Sources

Qwen3.8: [HF collection](https://huggingface.co/collections/Qwen/qwen38), [Qwen3.8-27B card + template](https://huggingface.co/Qwen/Qwen3.8-27B), [GitHub](https://github.com/QwenLM/Qwen3.8), [Unsloth guide](https://unsloth.ai/docs/models/qwen3.8), [Willison on thinking cost](https://simonwillison.net/2026/Aug/16/qwen-38-27b/), [orcarouter MLX memory](https://www.orcarouter.ai/blog/qwen-3-8-27b-mlx), [M4 Max oMLX+MTP numbers](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac), [mlx-community/qwen38](https://huggingface.co/collections/mlx-community/qwen38), [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [ggml-org GGUF + MTP](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), [Ollama qwen3.8](https://ollama.com/library/qwen3.8), [fixed templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), [Qwen3.5-9B card (BFCL/τ²)](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).
Serving: [mlx_lm.server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py), [tool parsers](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers), [hybrid prefix-cache bug #980](https://github.com/ml-explore/mlx-lm/issues/980), [mx.set_wired_limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html), [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md), [llama.cpp hybrid checkpoints #24055](https://github.com/ggml-org/llama.cpp/issues/24055), [#20225](https://github.com/ggml-org/llama.cpp/issues/20225), [spec-decode bench](https://hiesch.eu/blog/llamacpp-benchmarks-speculative-decoding/), [Apple Silicon bench (stared)](https://github.com/stared/benching-local-llms-on-apple-silicon), [engine comparison M4 Max](https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/), [mlx-lm vs oMLX](https://medium.com/macoclock/mlx-lm-vs-omlx-i-was-wrong-about-the-winner-8f36be328069), [oMLX #825](https://github.com/jundot/omlx/issues/825), [Rapid-MLX #163](https://github.com/raullenchai/Rapid-MLX/issues/163), [Ollama MLX](https://ollama.com/blog/mlx), [Ollama #16051 cold prefill](https://github.com/ollama/ollama/issues/16051), [#16698 KV accumulation](https://github.com/ollama/ollama/issues/16698), [iogpu.wired_limit_mb](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap), [wired collector](https://ranranhaoranzhang.com/blog/2026/llm-inference-memory-allocation-apple-silicon/).
STT: [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) ([server.md](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/server.md)), [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), [speech-swift footprint](https://github.com/soniqo/speech-swift/blob/main/docs/models/nemotron-asr-streaming.md).
Local: `poc-qwen/bench-m4-max.md`, `docs/adr/0001-core-llm-model-selection.md`, `config.yaml` `llm:`, `mx.device_info()` on this host.
