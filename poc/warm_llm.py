"""Prewarm Ollama for the FlowCat PoC so the first user turn is not 8-12 s.

Two costs hide behind the first /v1/chat/completions request of a call:

1. Runner swap. Ollama tears down the resident runner and starts a new
   llama-server when the request needs a different runner config than the
   instance that was preloaded (observed: a bare `/api/generate` residency
   load, then the first /v1 chat forced a fresh runner: ~5-6 s of model
   load inside a 7.9 s request). Warming through the SAME /v1 endpoint
   FlowCat uses establishes the right runner up front.
2. Prefix prefill. The system prompt + 8 tool schemas are ~2 K tokens
   (~2 s prefill). llama-server's prompt cache keeps the longest common
   token prefix across requests, so warming with the byte-identical
   system message and tool list (same order and OpenAI wrapping as
   flowcat-services' OpenAiLlm) drops the first real turn to warm-turn
   latency (~1 s).

After the /v1 warm, `/api/generate {"keep_alive": -1}` pins the runner
resident; measured on Ollama 0.32.5 this does NOT swap the warmed runner.

Usage: .venv/bin/python warm_llm.py   (reads .env / env for base URL+model)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

POC_DIR = os.path.dirname(os.path.abspath(__file__))


def post(url: str, body: dict, timeout: float) -> None:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def main() -> int:
    base = os.environ.get("OPENROUTER_BASE_URL", "").rstrip("/")
    model = os.environ.get("POC_LLM_MODEL", "gemma4:26b")
    if "11434" not in base:
        print(f"warm_llm: {base or '<unset>'} is not Ollama; nothing to warm")
        return 0
    root = base.removesuffix("/v1")

    with open(os.path.join(POC_DIR, "flowcat", "prompt.txt")) as fh:
        system_prompt = fh.read()
    with open(os.path.join(POC_DIR, "stubs", "skills.json")) as fh:
        skills = json.load(fh)
    # Mirror flowcat-services OpenAiLlm::tool_to_openai exactly (same order).
    tools = [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in skills
    ]

    t0 = time.time()
    post(
        f"{base}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}],
            "tools": tools,
            "max_tokens": 1,
        },
        timeout=300,
    )
    t1 = time.time()
    # Pin residency. Runs after the /v1 warm on purpose: this order reuses the
    # warmed runner instead of establishing a differently-configured one.
    post(f"{root}/api/generate", {"model": model, "keep_alive": -1}, timeout=60)
    print(
        f"warm_llm: {model} warmed via /v1 in {t1 - t0:.1f}s "
        f"(system prompt + {len(tools)} tools prefilled), residency pinned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
