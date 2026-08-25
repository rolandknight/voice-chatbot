"""Ollama control for the FlowCat PoC: `ensure` makes the LLM ready for a call.

    .venv/bin/python ollama_ctl.py ensure   (reads .env / env: OPENROUTER_BASE_URL, POC_LLM_MODEL)

What "ready" means, and why each step exists (all measured on Ollama 0.32.5):

1. `ollama serve` running WITH `OLLAMA_KEEP_ALIVE=-1`. A per-request
   keep_alive pin (`/api/generate {"keep_alive": -1}`) is overwritten by the
   very next /v1/chat/completions, because the OpenAI-compatible endpoint
   applies the server-default keep-alive (5 min) to every request (ollama
   issue #2963). After five idle minutes the model unloads and the next
   call pays a ~10 s cold start. Only the serve process's environment makes
   residency stick. A running plain `ollama serve` without it (e.g. the
   orphaned one poc-gemma4 started) is restarted; the Ollama.app is not
   touched — instructions are printed instead.
2. The model pulled.
3. The FlowCat prefix warmed THROUGH /v1 (the endpoint FlowCat uses). Warming
   via /api/generate established a differently-configured runner, and the
   first /v1 request then swapped runners (~5-6 s inside a 7.9 s request).
   The warm carries the byte-identical system prompt + tool list (same order
   and OpenAI wrapping as flowcat-services' OpenAiLlm) so llama-server's
   prompt cache holds the ~2 K-token prefix: first real turn ~0.55 s.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

POC_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(POC_DIR, "logs", "ollama.log")


def http(method: str, url: str, body: dict | None = None, timeout: float = 30.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def is_up(root: str) -> bool:
    try:
        http("GET", f"{root}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_up(root: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if is_up(root):
            return
        time.sleep(0.5)
    sys.exit("ERROR: ollama did not come up")


def start_serve(root: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # Context length bounds the runner's KV budget, which is what Ollama's
    # scheduler counts when deciding whether the resident model still "fits"
    # once another Metal user (Qwen3-TTS in-process, ~4.3 GB) appears. 32K
    # (the model default) predicted 18.0 GB and got the resident 26b evicted
    # and reloaded mid-call; the PoC prompt is ~2-3 K tokens, so 8192 keeps
    # the prompt cache (memory: never let Ollama truncate) with ~1.1 GB less.
    num_ctx = os.environ.get("POC_OLLAMA_NUM_CTX", "8192")
    env = dict(os.environ, OLLAMA_KEEP_ALIVE="-1", OLLAMA_CONTEXT_LENGTH=num_ctx)
    print(f"ollama: starting `ollama serve` with OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH={num_ctx} (logs/ollama.log)")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open(LOG_PATH, "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    wait_up(root, 60)


def plain_serve_pids() -> list[int]:
    """PIDs of bare `ollama serve` processes (not the Ollama.app bundle)."""
    out = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True).stdout
    pids = []
    for pid in out.split():
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        if cmd.strip() == "ollama serve":
            pids.append(int(pid))
    return pids


def resident(root: str, model: str) -> dict | None:
    for m in http("GET", f"{root}/api/ps")["models"]:
        if m["name"] == model:
            return m
    return None


def pinned(entry: dict | None) -> bool:
    """True when expires_at is effectively never (keep_alive -1 → far future)."""
    if not entry or not entry.get("expires_at"):
        return False
    exp = dt.datetime.fromisoformat(entry["expires_at"].split(".")[0] + entry["expires_at"][-6:]
                                    if "." in entry["expires_at"] else entry["expires_at"])
    return exp - dt.datetime.now(exp.tzinfo) > dt.timedelta(days=365)


def warm(base: str, model: str) -> float:
    with open(os.path.join(POC_DIR, "flowcat", "prompt.txt")) as fh:
        system_prompt = fh.read()
    with open(os.path.join(POC_DIR, "stubs", "skills.json")) as fh:
        skills = json.load(fh)
    tools = [  # mirror flowcat-services OpenAiLlm::tool_to_openai, same order
        {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
        for s in skills
    ]
    t0 = time.time()
    http("POST", f"{base}/chat/completions",
         {"model": model, "messages": [{"role": "system", "content": system_prompt}], "tools": tools, "max_tokens": 1},
         timeout=600)
    return time.time() - t0


def ensure() -> int:
    base = os.environ.get("OPENROUTER_BASE_URL", "").rstrip("/")
    model = os.environ.get("POC_LLM_MODEL", "gemma4:26b")
    if "11434" not in base:
        print(f"ollama: {base or '<unset>'} is not Ollama; nothing to do")
        return 0
    root = base.removesuffix("/v1")

    if not is_up(root):
        start_serve(root)

    names = {m["name"] for m in http("GET", f"{root}/api/tags")["models"]}
    if model not in names:
        print(f"ollama: pulling {model} ...")
        subprocess.run(["ollama", "pull", model], check=True)

    secs = warm(base, model)
    entry = resident(root, model)
    want_ctx = int(os.environ.get("POC_OLLAMA_NUM_CTX", "8192"))
    ctx_ok = entry is not None and int(entry.get("context_length") or 0) == want_ctx
    if pinned(entry) and ctx_ok:
        print(f"ollama: {model} warmed via /v1 in {secs:.1f}s; resident (keep-alive pinned, ctx {want_ctx})")
        return 0
    if entry is not None and not ctx_ok:
        print(f"ollama: resident context_length is {entry.get('context_length')}, want {want_ctx}")

    # Either the serve process lacks OLLAMA_KEEP_ALIVE=-1 (the /v1 request set a
    # finite expiry) or its context length differs: restart it with our env.
    pids = plain_serve_pids()
    if pids:
        print(f"ollama: restarting serve (pid {pids}) with OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH={want_ctx}")
        for pid in pids:
            subprocess.run(["kill", str(pid)])
        deadline = time.time() + 30
        while is_up(root) and time.time() < deadline:
            time.sleep(0.5)
        start_serve(root)
        secs = warm(base, model)
        entry = resident(root, model)
        if pinned(entry) and int(entry.get("context_length") or 0) == want_ctx:
            print(f"ollama: restarted; {model} warmed via /v1 in {secs:.1f}s; resident (keep-alive pinned, ctx {want_ctx})")
            return 0
        sys.exit("ERROR: keep-alive still not pinned after restart; check logs/ollama.log")

    print(
        "WARNING: ollama is running without OLLAMA_KEEP_ALIVE=-1 and is not a plain "
        "`ollama serve` we can restart (Ollama.app?). The model will unload after 5 idle "
        "minutes and the next call pays a ~10 s cold start. Fix: "
        "`launchctl setenv OLLAMA_KEEP_ALIVE -1`, then quit and relaunch Ollama."
    )
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ensure"
    if cmd != "ensure":
        sys.exit(f"usage: {sys.argv[0]} ensure")
    sys.exit(ensure())
