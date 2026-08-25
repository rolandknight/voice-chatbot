from poc_gemma4 import probe
from poc_gemma4.ollama import Turn


class FakeClient:
    """Prefill = tokens beyond the longest cached prefix, keyed on the
    rendered (tools, messages) prefix like a real prompt cache."""

    def __init__(self):
        self.cache = ""

    def chat(self, messages, tools=None, think=False):
        rendered = str(tools) + "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        common = 0
        for c, r in zip(self.cache, rendered):
            if c != r:
                break
            common += 1
        self.cache = rendered
        prefill = len(rendered) - common
        t = Turn(content="ok", prompt_eval_count=prefill, eval_count=10, eval_s=0.2,
                 prompt_eval_s=prefill / 1000, ttft_s=0.05 + prefill / 1000)
        if messages[-1]["role"] == "user" and "time" in messages[-1]["content"]:
            t.tool_calls = [{"name": "get_current_time", "arguments": {}}]
        if messages[-1]["role"] == "tool":
            t.content = "It's 14:05."
        return t


def _log(rows):
    def log(scenario, turn, t):
        row = {"scenario": scenario, "turn": turn, "prompt_eval_count": t.prompt_eval_count,
               "prompt_eval_s": t.prompt_eval_s, "ttft_s": t.ttft_s, "decode_tps": t.decode_tps, "tool_calls": t.tool_calls,
               "leaked_thinking": t.leaked_thinking, "content": t.content}
        rows.append(row)
        return row
    return log


def _run_all(client, tools, turns=4):
    rows = []
    log = _log(rows)
    client.chat([{"role": "system", "content": "warm"}, {"role": "user", "content": "ping"}], tools)
    _, full = probe.measure_full_prefix(client, tools)
    probe.run_stable(client, tools, turns, log)
    probe.run_shuffled(client, tools, log)
    probe.run_system(client, tools, log)
    probe.run_tool(client, tools, log)
    return rows, full


GATES = {"warm_prefill_ratio_max": 0.15, "warm_ttft_p50_max_s": 0.4,
         "warm_ttft_p95_max_s": 0.6, "decode_tps_min": 40, "turns": 4}
TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "x" * 200}} for i in range(15)]


def test_gates_pass_when_cache_behaves():
    rows, full = _run_all(FakeClient(), TOOLS)
    v = probe.gates(rows, GATES, full)
    assert v == {k: True for k in v}, v
    assert set(v) == {"warm_prefill_small", "warm_ttft_p50", "warm_ttft_p95", "decode_tps",
                      "no_thinking_leak", "shuffled_tools_reprefill", "tool_call_emitted",
                      "tool_second_pass_spoken"}


def test_shuffled_tools_cause_full_reprefill():
    rows, full = _run_all(FakeClient(), TOOLS)
    sh = [r for r in rows if r["scenario"] == "shuffled"]
    assert sh[1]["prompt_eval_s"] < 0.1 * full
    assert sh[2]["prompt_eval_s"] > 0.9 * full


def test_gates_fail_without_cache():
    class NoCache(FakeClient):
        def chat(self, messages, tools=None, think=False):
            self.cache = ""
            return super().chat(messages, tools, think)

    rows, full = _run_all(NoCache(), TOOLS)
    v = probe.gates(rows, GATES, full)
    assert v["warm_prefill_small"] is False
    assert v["warm_ttft_p50"] is False


def test_spoken_time_accepts_12h_and_24h():
    assert probe.spoken_time_ok("It's 2:05 PM.")
    assert probe.spoken_time_ok("The time is 14:05")
    assert not probe.spoken_time_ok("I don't know")


def test_thinking_leak_detected():
    t = Turn(content="x", thinking="let me think")
    assert t.leaked_thinking
    assert not Turn(content="plain").leaked_thinking
