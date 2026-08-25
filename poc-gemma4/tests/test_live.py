"""Live assertions of the recommendation against Ollama + gemma4:26b.

Run with `make test-live`. Skipped unless Ollama answers and the model is pulled.
"""
import statistics
import time

import pytest

from poc_gemma4 import probe
from poc_gemma4.config import load_config
from poc_gemma4.ollama import OllamaClient
from poc_gemma4.prompt import SYSTEM_PROMPT

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live():
    cfg = load_config()
    o = cfg["ollama"]
    client = OllamaClient(o["base_url"], o["model"], o["keep_alive"], o["num_ctx"], o["temperature"], o["max_tokens"])
    if not client.is_up() or not client.has_model():
        pytest.skip("ollama not running or model not pulled")
    tools = probe.build_tools(cfg)
    client.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "ping"}], tools)
    run_id = str(time.time())
    full_tok, miss_s = probe.measure_full_prefix(client, tools, nonce=run_id)
    return cfg, client, tools, {"tokens": full_tok, "miss_s": miss_s, "run_id": run_id}


def _rows(live, fn, *a):
    rows = []

    def log(scenario, turn, t):
        row = {"scenario": scenario, "turn": turn, "prompt_eval_count": t.prompt_eval_count,
               "prompt_eval_s": t.prompt_eval_s, "ttft_s": t.ttft_s, "decode_tps": t.decode_tps, "tool_calls": t.tool_calls,
               "leaked_thinking": t.leaked_thinking, "content": t.content}
        rows.append(row)
        return row
    fn(live[1], live[2], *a, log, live[3]["run_id"])
    return rows


def test_all_tools_render_under_budget(live):
    cfg, _, tools, full = live
    assert 10 <= len(tools) <= 25
    assert 500 < full["tokens"] < cfg["ollama"]["num_ctx"] // 2, full


def test_prefix_cache_hits_across_turns(live):
    cfg, _, _, full = live
    rows = _rows(live, probe.run_stable, cfg["gates"]["turns"])
    for r in rows[1:]:
        assert r["prompt_eval_s"] <= cfg["gates"]["warm_prefill_ratio_max"] * full["miss_s"], (full, rows)


def test_warm_ttft_within_budget(live):
    cfg, _, _, _ = live
    rows = _rows(live, probe.run_stable, cfg["gates"]["turns"])
    ttfts = [r["ttft_s"] for r in rows[1:]]
    assert statistics.median(ttfts) <= cfg["gates"]["warm_ttft_p50_max_s"], ttfts
    assert max(ttfts) <= cfg["gates"]["warm_ttft_p95_max_s"], ttfts


def test_decode_faster_than_speech(live):
    cfg, _, _, _ = live
    rows = _rows(live, probe.run_stable, 3)
    assert statistics.median(r["decode_tps"] for r in rows) >= cfg["gates"]["decode_tps_min"]


def test_reordered_tools_invalidate_cache(live):
    miss = live[3]["miss_s"]
    rows = _rows(live, probe.run_shuffled)
    assert rows[1]["prompt_eval_s"] < 0.15 * miss, (miss, rows)
    assert rows[2]["prompt_eval_s"] > 3 * rows[1]["prompt_eval_s"], (miss, rows)


def test_mid_conversation_system_message_is_accepted(live):
    rows = _rows(live, probe.run_system)
    assert len(rows) == 3 and rows[2]["content"]


def test_tool_call_without_thinking(live):
    rows = _rows(live, probe.run_tool)
    assert rows[0]["tool_calls"] and rows[0]["tool_calls"][0]["name"] == "get_current_time"
    assert not any(r["leaked_thinking"] for r in rows)
    assert len(rows) == 2 and probe.spoken_time_ok(rows[1]["content"]) and not rows[1]["tool_calls"], rows
