"""Ollama native /api/chat client with per-turn cache and latency metrics.

The pipeline talks to Ollama's OpenAI-compatible /v1 endpoint; the native
endpoint is used here because it reports `prompt_eval_count` (tokens actually
prefilled this request) and `prompt_eval_duration`, which is the direct
evidence of a prompt-cache hit: a warm turn should prefill only the new
message, not the whole prefix.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class Turn:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    thinking: str = ""
    ttft_s: float = 0.0
    total_s: float = 0.0
    prompt_eval_count: int = 0
    eval_count: int = 0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    done_reason: str = ""

    @property
    def decode_tps(self) -> float:
        return self.eval_count / self.eval_s if self.eval_s else 0.0

    @property
    def leaked_thinking(self) -> bool:
        return bool(self.thinking.strip()) or "<|channel>" in self.content


class OllamaClient:
    def __init__(self, base_url: str, model: str, keep_alive: str = "-1",
                 num_ctx: int = 8192, temperature: float = 0.2, max_tokens: int = 512,
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Native /api/chat wants a number or a Go duration; "-1" (pipeline
        # config form) only parses on the OpenAI-compatible path.
        self.keep_alive = int(keep_alive) if str(keep_alive).lstrip("-").isdigit() else keep_alive
        self.options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens}
        self.http = httpx.Client(timeout=timeout)

    def is_up(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/api/tags", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self) -> bool:
        tags = self.http.get(f"{self.base_url}/api/tags").json().get("models", [])
        return any(m["name"] == self.model or m["name"] == f"{self.model}:latest" for m in tags)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, think: bool = False) -> Turn:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        if tools:
            body["tools"] = tools
        return self._stream(body)

    def _stream(self, body: dict) -> Turn:
        turn = Turn()
        t0 = time.perf_counter()
        first = None
        with self.http.stream("POST", f"{self.base_url}/api/chat", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                if first is None and (msg.get("content") or msg.get("tool_calls") or msg.get("thinking")):
                    first = time.perf_counter()
                turn.content += msg.get("content", "") or ""
                turn.thinking += msg.get("thinking", "") or ""
                for tc in msg.get("tool_calls") or []:
                    turn.tool_calls.append(tc.get("function", tc))
                if chunk.get("done"):
                    turn.prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    turn.eval_count = chunk.get("eval_count", 0)
                    turn.prompt_eval_s = chunk.get("prompt_eval_duration", 0) / 1e9
                    turn.eval_s = chunk.get("eval_duration", 0) / 1e9
                    turn.done_reason = chunk.get("done_reason", "")
        end = time.perf_counter()
        turn.ttft_s = (first or end) - t0
        turn.total_s = end - t0
        return turn


def ensure_running(base_url: str, wait_s: float = 60.0) -> None:
    c = OllamaClient(base_url, "")
    if c.is_up():
        return
    print("ollama not answering; starting `ollama serve` in the background ...", file=sys.stderr)
    subprocess.Popen(["ollama", "serve"], stdout=open("reports/ollama.log", "ab"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if c.is_up():
            return
        time.sleep(0.5)
    raise SystemExit("ollama did not come up")


if __name__ == "__main__":
    from poc_gemma4.config import load_config

    cfg = load_config()["ollama"]
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ensure"
    if cmd == "ensure":
        ensure_running(cfg["base_url"])
        print("ollama up at", cfg["base_url"])
    elif cmd == "pull":
        ensure_running(cfg["base_url"])
        subprocess.check_call(["ollama", "pull", cfg["model"]])
    else:
        raise SystemExit(f"unknown command {cmd}")
