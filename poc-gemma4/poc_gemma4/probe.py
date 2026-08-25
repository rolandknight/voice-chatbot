"""Prefix-cache + TTFT probe for gemma4 on Ollama.

Scenarios (each a fresh conversation unless stated):
  stable    N turns with a byte-identical prefix (system + all tools sorted),
            history appended. Expect prompt_eval_count on turns 2..N to be
            ~the new message only (cache hit) and warm TTFT within budget.
  shuffled  Turn 3 of a stable session but with the tool list reordered —
            the per-turn top-K filter's effect. Expect a full re-prefill.
  system    Turn 3 with a mid-conversation system message (persona switch).
  tool      One tool-call turn: right tool, args, no thinking leakage.
Writes reports/probe.jsonl; prints a table and the gate verdicts.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from poc_gemma4.config import ROOT, load_config
from poc_gemma4.ollama import OllamaClient, ensure_running
from poc_gemma4.prompt import SYSTEM_PROMPT
from poc_gemma4.schemas import load_skills, to_openai_tools

CHAT_TURNS = [
    "hi there, how's it going?",
    "tell me something interesting about octopuses",
    "what's the capital of Peru?",
    "give me a two line poem about rain",
    "do you prefer tea or coffee?",
    "what's a good name for a goldfish?",
    "thanks, that's all",
    "one more: what year did the moon landing happen?",
]


def build_tools(cfg: dict) -> list[dict]:
    root = (ROOT / cfg["skills"]["root"]).resolve()
    return to_openai_tools(load_skills(root, cfg["skills"]["enabled"]))


def measure_full_prefix(client: OllamaClient, tools: list[dict], nonce: str = "nonce-0") -> tuple[int, float]:
    """(prompt tokens, prefill seconds) for a never-seen prefix: the cost a cache miss pays.

    Ollama >= 0.32 counts cached tokens in prompt_eval_count (PR #16428), so
    prompt_eval_duration is the only field that distinguishes a hit from a miss.

    Both the system text and the tool list carry the nonce so the prefix
    differs from its first byte whichever the chat template renders first.
    """
    marker = {"type": "function", "function": {"name": "aaa_" + nonce.replace(":", "_").replace(".", "_"),
                                               "description": f"probe marker {nonce}",
                                               "parameters": {"type": "object", "properties": {}}}}
    t = client.chat([{"role": "system", "content": f"({nonce}) " + SYSTEM_PROMPT},
                     {"role": "user", "content": "ping"}], [marker] + tools)
    return t.prompt_eval_count, t.prompt_eval_s


def _turn(i: int, run_id: str) -> str:
    """First user turn carries the run id so a previous run's cached history
    (deterministic prompts at temperature 0.2) cannot mask this run's results.
    The system+tools prefix stays byte-identical to production."""
    text = CHAT_TURNS[i % len(CHAT_TURNS)]
    return f"{text} (run {run_id})" if i == 0 else text


def run_stable(client: OllamaClient, tools: list[dict], n: int, log, run_id: str = "0") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows = []
    for i in range(n):
        messages.append({"role": "user", "content": _turn(i, run_id)})
        t = client.chat(messages, tools)
        messages.append({"role": "assistant", "content": t.content})
        rows.append(log("stable", i + 1, t))
    return rows


def run_shuffled(client: OllamaClient, tools: list[dict], log, run_id: str = "0") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows = []
    for i in range(3):
        messages.append({"role": "user", "content": _turn(i, run_id + "s")})
        ts = tools if i < 2 else list(reversed(tools))
        t = client.chat(messages, ts)
        messages.append({"role": "assistant", "content": t.content})
        rows.append(log("shuffled", i + 1, t))
    return rows


def run_system(client: OllamaClient, tools: list[dict], log, run_id: str = "0") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows = []
    for i in range(3):
        if i == 2:
            messages.append({"role": "system", "content": "You are now Marvin, a gloomy robot. Stay brief."})
        messages.append({"role": "user", "content": _turn(i, run_id + "y")})
        t = client.chat(messages, tools)
        messages.append({"role": "assistant", "content": t.content})
        rows.append(log("system", i + 1, t))
    return rows


def run_tool(client: OllamaClient, tools: list[dict], log, run_id: str = "0") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"what time is it? (run {run_id})"}]
    t = client.chat(messages, tools)
    rows = [log("tool", 1, t)]
    if t.tool_calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": tc} for tc in t.tool_calls]})
        messages.append({"role": "tool", "content": "14:05", "tool_name": t.tool_calls[0].get("name", "")})
        t2 = client.chat(messages, tools)
        rows.append(log("tool", 2, t2))
    return rows


def spoken_time_ok(text: str) -> bool:
    """The tool returned "14:05"; the model may speak it as 14:05 or 2:05 (PM)."""
    return "14:05" in text or "2:05" in text


def gates(rows: list[dict], g: dict, miss_prefill_s: float) -> dict[str, bool]:
    stable = [r for r in rows if r["scenario"] == "stable"]
    warm = stable[1:]
    shuffled = [r for r in rows if r["scenario"] == "shuffled"]
    tool = [r for r in rows if r["scenario"] == "tool"]
    out = {}
    if stable:
        out["warm_prefill_small"] = all(
            r["prompt_eval_s"] <= g["warm_prefill_ratio_max"] * miss_prefill_s for r in warm)
        ttfts = [r["ttft_s"] for r in warm]
        out["warm_ttft_p50"] = statistics.median(ttfts) <= g["warm_ttft_p50_max_s"]
        out["warm_ttft_p95"] = max(ttfts) <= g["warm_ttft_p95_max_s"]
        out["decode_tps"] = statistics.median(r["decode_tps"] for r in stable) >= g["decode_tps_min"]
        out["no_thinking_leak"] = not any(r["leaked_thinking"] for r in rows)
    if len(shuffled) == 3 and warm:
        # Ollama's cache-reuse can shift matching KV chunks, so a reordered
        # tool list is sometimes a partial miss (~0.7 s) rather than a full
        # one (~2.1 s); either is several times a warm turn and over budget.
        warm_median = statistics.median(r["prompt_eval_s"] for r in warm)
        out["shuffled_tools_reprefill"] = shuffled[2]["prompt_eval_s"] > 3 * warm_median
    if tool:
        out["tool_call_emitted"] = bool(tool[0]["tool_calls"]) and tool[0]["tool_calls"][0]["name"] == "get_current_time"
        if len(tool) == 2:
            out["tool_second_pass_spoken"] = spoken_time_ok(tool[1]["content"]) and not tool[1]["tool_calls"]
    return out


def main() -> int:
    cfg = load_config()
    o = cfg["ollama"]
    ensure_running(o["base_url"])
    client = OllamaClient(o["base_url"], o["model"], o["keep_alive"], o["num_ctx"], o["temperature"], o["max_tokens"])
    if not client.has_model():
        print(f"{o['model']} is not pulled; run `make build`", file=sys.stderr)
        return 2
    tools = build_tools(cfg)
    print(f"model={o['model']} tools={len(tools)} num_ctx={o['num_ctx']}")
    out = Path("reports/probe.jsonl")
    out.parent.mkdir(exist_ok=True)
    rows: list[dict] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    def log(scenario: str, turn: int, t) -> dict:
        row = {"ts": started, "model": o["model"], "scenario": scenario, "turn": turn,
               "prompt_eval_count": t.prompt_eval_count, "prompt_eval_s": round(t.prompt_eval_s, 4),
               "eval_count": t.eval_count, "decode_tps": round(t.decode_tps, 1),
               "ttft_s": round(t.ttft_s, 3), "total_s": round(t.total_s, 3),
               "tool_calls": t.tool_calls, "leaked_thinking": t.leaked_thinking,
               "content": t.content[:120]}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {scenario:8s} t{turn}  prefill={t.prompt_eval_count:5d} tok ({t.prompt_eval_s:.2f}s)  "
              f"ttft={t.ttft_s:.3f}s  decode={t.decode_tps:5.1f} tok/s  tools={[c.get('name') for c in t.tool_calls]}")
        rows.append(row)
        return row

    # Warm-up loads the model; the second call measures a guaranteed cache miss.
    client.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "ping"}], tools)
    full_tok, miss_s = measure_full_prefix(client, tools, nonce=started)
    print(f"cache miss: {full_tok} prompt tokens prefilled in {miss_s:.2f}s")
    run_stable(client, tools, cfg["gates"]["turns"], log, started)
    run_shuffled(client, tools, log, started)
    run_system(client, tools, log, started)
    run_tool(client, tools, log, started)
    verdict = gates(rows, cfg["gates"], miss_s)
    print("\ngates:")
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
