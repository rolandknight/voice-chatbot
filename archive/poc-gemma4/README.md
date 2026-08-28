# poc-gemma4 — does Ollama reuse the prefill for `gemma4:26b` across turns?

Tests the recommendation from the 2026-08-24 LLM discussion: keep
`gemma4:26b` (GGUF) resident in Ollama, send a byte-identical prefix (system
prompt + **all** tool schemas, sorted) every turn, append history — and the
prompt cache makes warm TTFT fit the ≤ 0.4 s budget. Also checks that the
things which break the cache really do (reordered tools = today's per-turn
top-K filter).

    make            # install if needed, start Ollama if needed, run the probe -> reports/probe.jsonl
    make test       # GPU-free unit tests (Ollama mocked)
    make test-live  # live assertions against Ollama + gemma4:26b
    make build      # ollama pull gemma4:26b
    make help

Run from this directory, or `make poc-gemma4*` from the repo root.

## Layout

- `poc_gemma4/schemas.py` — `skills/*/*/SKILL.md` frontmatter → OpenAI `tools`,
  mirroring `skills/_loader.py` without importing pipecat; always sorted by name.
- `poc_gemma4/prompt.py` — the system prompt `server.py`/`app.py` send.
- `poc_gemma4/ollama.py` — native `/api/chat` client (streams; records TTFT,
  `prompt_eval_duration`, `eval_count`, tool calls, thinking leakage).
- `poc_gemma4/probe.py` — scenarios: stable 6-turn session, reordered tools on
  turn 3, mid-conversation system message, tool call + second pass. Gates in
  `config.yaml`.
- `tests/test_live.py` — the same scenarios as pytest assertions (`-m live`).

`prompt_eval_count` includes cached tokens on Ollama ≥ 0.32 (PR #16428), so
the cache-hit evidence is `prompt_eval_duration`, compared against a measured
miss (nonce'd system prompt + nonce'd dummy tool).

## Results — 2026-08-24, Mac Studio M4 Max 36 GB, Ollama 0.32.5, `gemma4:26b` (Q4_K_M, 18 tools ≈ 2.2 K prompt tokens, `num_ctx` 8192, `think: false`)

| Case | prefill | TTFT |
| --- | --- | --- |
| Cache miss (never-seen prefix) | 2,195 tok in **2.15 s** (~1,000 tok/s) | **2.5 s** |
| Identical request repeated | 0.017 s | 0.24 s |
| Same prefix, one turn appended | 0.11 s | 0.32 s |
| Stable 6-turn session, turns 2–6 | 0.12–0.16 s | **0.33–0.37 s** (p50 0.36) |
| Same history, **tool list reordered** | 0.67–2.15 s | **0.95–2.6 s** |
| Mid-conversation `system` message | accepted; 0.19 s | 0.41 s |
| "what time is it?" | `get_current_time` called, no thinking | 0.42–0.48 s |
| Second pass with tool result `14:05` | "It's 2:05 PM.", no extra call | 0.40 s |
| Decode | 80–89 tok/s (≈ 8× TTS consumption pace) | |

Gates (`make`): all pass. `make test-live`: 7/7.

## What this means for the pipeline

1. The prompt cache works for Gemma 4's sliding-window layers on Ollama's
   llama.cpp runner; nothing to build on the engine side.
2. **The per-turn `SkillFilterProcessor` top-K swap is the TTFT bug**: any
   change in the tool list re-prefills the ~2.2 K-token prefix (0.7–2.2 s).
   With 18 skills total, send all of them, sorted by name, every turn.
3. Tool-call turns land at ~0.42–0.48 s TTFT, slightly over budget; the
   tool round trip (call + second pass) is ~0.9 s. Trimming tool
   descriptions is the lever if that matters.
4. Keep `keep_alive: -1`, `num_ctx` ≥ 8192 (Ollama drops the cache when it
   truncates), `think: false`. Do not move to `gemma4:26b-mlx` until Ollama
   #17829 (no prefix cache on the MLX runner) is closed.
