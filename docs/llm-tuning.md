# LLM memory tuning — reducing Gemma 4 26B-A4B's footprint without giving up TTFT or tool calling

Review date: 2026-08-26. A review of [ADR-0003](adr/0003-gemma4-26b-serving-and-prefix-cache.md)
from the memory angle, read alongside [ADR-0001](adr/0001-core-llm-model-selection.md)
(model selection), [ADR-0007](adr/0007-local-llm-serving-runtime-for-the-rust-build.md)
(serving runtime for the Rust build), the Rust PoC's Ollama supervisor
(`poc/flowcat/src/ollama_serve.rs`) and the memory measurements in the 2026-08-25 plans.
**Options only — nothing here has been applied or measured yet.**

Host: Mac Studio M4 Max, 36 GB unified memory. Model: `gemma4:26b` (Gemma 4 26B-A4B,
GGUF Q4_K_M), ~17 GB resident at `num_ctx` 8192, resident alongside Qwen3-TTS
(4.3 GB active / 6.4 GB peak) and Nemotron STT (~1 GB).

## 1. Review findings (what bears on memory)

The serving and prefix-cache reasoning in ADR-0003 is sound and well-evidenced; the
memory section is the weak part.

1. **The 24 GB LLM budget is built on an optimistic baseline.** ADR-0003 assumes a ~5 GB
   OS reserve and concludes "~4–5 GB slack". ADR-0007 measured a **~13 GB desktop
   baseline** after reboot, and
   [`2026-08-25-poc-flowcat-qwen-streaming-tts.md`](superpowers/plans/2026-08-25-poc-flowcat-qwen-streaming-tts.md)
   records **33 GB used / 19.6 GB swap** with 26b + Qwen + Nemotron loaded. The slack is
   in fact a deficit of several GB. That, not LLM growth, is why the LLM needs to
   shrink; ADR-0003's Host section should be restated from measurement.
2. **Residency figures disagree across docs**: 17 GB (ADR-0001; ADR-0007 at 8K),
   "~19 GB" (ADR-0003 Consequences), 17.7 GB RSS after Ollama's `--no-mmap` reload.
   Pick one measured number (`footprint -p` / `ollama ps`) and cite it everywhere.
3. **ADR-0003 §B recommends q8 KV** (`-ctk q8_0 -ctv q8_0`; "q4 degrades tool calls"),
   but the Rust supervisor passes only `OLLAMA_KEEP_ALIVE` and `OLLAMA_CONTEXT_LENGTH`
   (`poc/flowcat/src/ollama_serve.rs:109-110`) — no `OLLAMA_FLASH_ATTENTION` /
   `OLLAMA_KV_CACHE_TYPE`. Small at 8K, but the ADR states it as the configuration.
4. **ADR-0007's `llama-server` flag string omits `--cache-ram`.** Recent llama-server
   builds default to an **8 GiB host-RAM prompt cache** for evicted slot states. On
   unified memory that is 8 GB of potential growth over a long session with persona
   switches. Unset = a latent memory sink.
5. **Config drift.** `config.yaml:34` and `config/schema.py:46` default to
   `gemma4:12b-mlx` (commit `598a615`), contradicting ADR-0001/0003 — and the MLX runner
   has no prefix cache (Ollama #17829), so that default cannot meet the TTFT budget as
   configured. Not a memory issue, but an ADR-compliance gap for the Python pipeline.
6. **The vision projector is loaded for a voice-only pipeline** — unmeasured and never
   mentioned in the ADRs.

## 2. Why TTFT is safe to hold constant

Warm TTFT on this pipeline is a prefix-cache hit: ~30–100 new tokens of prefill plus one
decode step (ADR-0003: 0.12–0.16 s prefill, 0.33–0.37 s TTFT). None of the options below
touch the prefix-cache discipline (byte-stable system prompt + name-sorted full tool
list, history appended, `think: false`, `num_ctx` never reached). So:

- **Weight options** are neutral-to-better for warm TTFT (less memory traffic per token
  on a bandwidth-bound decode step). Cold-miss prefill is compute-bound → neutral.
- **KV/buffer options** are neutral except where noted (batch size affects cold prefill
  only).
- **Tool-call quality is the only thing at risk**, and the repo already has the gate:
  ADR-0001's bench protocol (τ²-bench thinking-off ≥ incumbent −2; in-pipeline warm TTFT
  p50 ≤ 0.4 s / p95 ≤ 0.6 s over ≥ 20 turns; resident ≤ budget after a 50-turn
  session), the `poc-gemma4` probe (`make poc-gemma4`), the FlowCat T3/T4 tool tests, and
  the select-acc / arg-valid / false-tool / leaks / malformed eval shape in
  [`research/qwen3.8-mtplx.md`](research/qwen3.8-mtplx.md).

## 3. Options, ordered by GB saved

Savings are estimates scaled from the measured 17 GB Q4_K_M residency; every one must
be re-measured.

| # | Option | Saving (approx.) | TTFT | Tool-call risk | Verify with |
|---|---|---|---|---|---|
| 1 | **Drop the multimodal projector.** llama-server: omit `--mmproj`. Ollama: `ollama create` from the text-only GGUF with a Modelfile (the `gemma4:26b` tag bundles the projector). | The mmproj blob — typically several hundred MB at f16 (check `ollama show` / manifest blob sizes). | none | none | blob size; `ollama ps` before/after |
| 2 | **q8 KV.** Serve env `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`; llama-server `-fa on -ctk q8_0 -ctv q8_0`. | ~half of KV. At 8K KV ≈ 0.5 GB (19 GB @ 32K − 17 GB @ 8K ≈ 0.08 GB per 1K tokens) → ~0.25 GB. | none | none measured (ADR-0003 flags only q4) | runner log KV buffer size |
| 2b | **Confirm the SWA layers are not in `swa_full` mode** on Ollama's runner. If they are, the 1024-token sliding-window layers hold the full 8K and KV is several × larger than needed. | 0 to ~1 GB | none | none | runner log (`swa_full`, KV lines) |
| 3 | **Cap `--cache-ram`** on llama-server: `0`, or ~1024 MiB if persona-switch prefix *restores* are wanted (a saved 8K q8 state for this model is a few hundred MB). | Prevents up to 8 GB of growth. | a small non-zero value *helps* TTFT on persona switch (restore instead of re-prefill) | none | `/slots`; RSS over a long session |
| 4 | **Smaller prefill batch.** `num_batch` / `-b` / `-ub` 512 → 256 or 128. The compute buffer scales with batch (262K-vocab logits row, MoE intermediates). | Expect a few hundred MB. | warm: none (warm prefill is < 128 tokens anyway). Cold miss 2.15 s → roughly 2.5–3 s. | none | "compute buffer size" in runner log |
| 5 | **`llama-server` child instead of Ollama** (ADR-0007, already planned). | Ollama server + Go runner shim, a few hundred MB; and it makes 1–4 plain flags. | parity (same engine) | parser edge cases already listed in ADR-0007 | ADR-0007 gates G1–G5 |
| 6 | **IQ4_XS re-quant** of the same model. | 17 → ~13 GB (−4) | neutral/better | **low** — IQ4_XS sits within ~0.01–0.02 KLD of Q4_K_M in llama.cpp's tables; Metal has native kernels | ADR-0001 protocol |
| 7 | **Mixed-precision ("dynamic") quant.** Keep attention, shared expert and embeddings at Q5–Q8; shrink only the routed experts (`llama-quantize --tensor-type 'ffn_.*_exps=IQ3_S' …`, or an Unsloth UD-Q3_K_XL-class build if one is published). | ~11–12 GB (−5 to −6) | neutral/better | **low–medium; best quality per GB.** Routed experts are ~90 % of the bytes but each token uses 8 of 128; the always-on shared expert and attention are where quantisation error turns into malformed JSON, and they stay high-precision. | ADR-0001 protocol + arg-valid / malformed counts |
| 8 | **Q4_0** (already in the HF cache per ADR-0007, 14 GB). | −3 | neutral | **medium** — legacy quant, worse than Q4_K_M; ADR-0007 already says "bring-up only". Dominated by 6/7. | — |
| 9 | **IQ3_M / IQ2_M across the board.** | ~9–11 GB (−6 to −8) | decode slower on IQ Metal kernels, but 3.8 B active params leave ~2× headroom over the 40 tok/s floor | **high** for argument fidelity | ADR-0001 protocol; expect failures |
| 10 | **Re-measure Gemma 4 E4B under the ADR-0003 protocol** (explicit `think: false`, byte-stable prefix). | ~9.6 GB resident → −7.5 | ~0.3 s measured (ADR-0001) | **high but unproven.** ADR-0001 rejected E4B on thinking spikes on indirect phrasings, measured *before* ADR-0003 established that Ollama's renderer injects thinking for some variants and `think: false` must be explicit. If the spikes vanish, what remains is pure tool-selection quality. Cheap to test with the existing harness. | ADR-0001 protocol; `poc-gemma4` probe |
| 11 | **Expert-pruned (REAP-style) build** of 26B-A4B, if one is published (not verified). | −25–50 % of weights | identical (same architecture, same cache behaviour) | **high**, prompt-distribution dependent | ADR-0001 protocol |

On `gemma4:12b-mlx` (the `config.yaml` default, ~7.7 GB): not a candidate as
configured — no prefix cache on the MLX runner. A GGUF of the same model would be the
same experiment as #10, with the caveat that a dense 12 B's decode on this GPU sits near
the 40 tok/s floor.

## 4. Non-levers on unified memory

Listed so nobody reaches for them:

- **`--cpu-moe` / expert offload** — same memory pool on the Mac; only helps on the CUDA
  target in [`poc/flowcat-poc-plan.md`](poc/flowcat-poc-plan.md).
- **mmap instead of mlock** — lowers *wired* memory, not footprint. Under the swap
  pressure already seen, unwired weights get paged out and the first touch stalls (the
  72 s stall in the TTS plan). Keep `--load-mode mlock`; shrink the footprint instead.
- **`iogpu.wired_limit_mb`** — moves the ceiling, not the usage.
- **`num_ctx` below 8192** — ~0.15 GB per 2K tokens, and one silent truncation kills the
  cache (0.4 s → 2.5 s TTFT). Not worth it.
- **q4 KV** — ADR-0003 already rejects it for tool calls; agreed.
- **Speculative / draft models** — add memory; decode is already ~8× the TTS pace.

## 5. Suggested sequence

1. **Free, zero-risk** — #1, #2/#2b, #3, #4 (and #5 when ADR-0007 lands): roughly
   0.5–1.5 GB, one afternoon, no quality gate needed.
2. **#7** (or #6 as the conservative version) under the ADR-0001 protocol: 17 → ~11–13 GB
   with a measured quality gate. This is where the real saving is.
3. **Only if still short**: #10 (largest saving, cheapest to test, most likely to fail
   the quality gate) before #9 / #11.

Realistic outcome without touching TTFT: **~17.5 GB → ~11–12 GB**, all verifiable with
gates the repo already has.

The larger lever on the box is not the LLM at all: the ~13 GB desktop baseline measured
in ADR-0007. That is outside this document's scope but should be measured before any
re-quant is judged insufficient.

## 6. Bookkeeping

ADR edits implied by §1 (not made): restate ADR-0003's Host budget from measurement;
reconcile the 17 / 19 GB residency figures; add `--cache-ram` to ADR-0007's flag list;
either move `config.yaml` / `config/schema.py` back to `gemma4:26b` or record why the
Python pipeline is exempt from ADR-0003.
