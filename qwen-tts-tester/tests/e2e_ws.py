"""End-to-end check over the real server: start the binary, clone a preset
voice over /ws exactly as the browser does, and report client-side TTFA.

    ../crates/qwen-tts/.venv/bin/python tests/e2e_ws.py [--size 0.6B] [--text "..."]   (or: make e2e)

Needs the models (GPU); not part of `make test`. Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent.parent
BIN = HERE.parent / "target" / "release" / "qwen-tts-tester"


async def wait_ready(base: str, timeout=300.0) -> dict:
    """Up, and preload finished (state 'done' or 'idle' when disabled)."""
    t0 = time.time()
    async with httpx.AsyncClient() as c:
        while time.time() - t0 < timeout:
            try:
                r = await c.get(f"{base}/api/info", timeout=5)
                if r.status_code == 200:
                    info = r.json()
                    st = info.get("preload", {}).get("state", "idle")
                    if st in ("done", "idle"):
                        return info
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("server did not come up / preload did not finish")


async def run(base: str, ws_url: str, size: str, text: str) -> dict:
    async with httpx.AsyncClient() as c:
        cat = (await c.get(f"{base}/api/catalog")).json()
    voice = next(v for v in cat["voices"] if v["name"] == "one-one")
    assert "Auto" in cat["languages"] and cat["speakers"] and cat["sizes"] == ["0.6B", "1.7B"]

    async with websockets.connect(ws_url, max_size=None) as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "generate", "tab": "clone", "preset": "one-one", "ref_text": voice["transcript"], "text": text, "language": "English", "size": size}))
        ttfa = None
        nbytes = 0
        frames = 0
        start = done = None
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                nbytes += len(msg)
                frames += 1
                continue
            m = json.loads(msg)
            if m["type"] == "start":
                start = m
            elif m["type"] == "done":
                done = m["timings"]
                break
            elif m["type"] == "error":
                raise RuntimeError(m["message"])
        total = time.perf_counter() - t0
    assert start and done and frames > 0
    audio_s = nbytes / 2 / start["sample_rate"]
    assert abs(audio_s - done["audio_s"]) < 0.05, (audio_s, done["audio_s"])
    return {"client_ttfa_s": round(ttfa, 3), "server_ttfa_s": done["ttfa_s"], "client_total_s": round(total, 3), "gen_s": done["gen_s"], "audio_s": done["audio_s"], "rtf": done["rtf"], "frames": frames, "model": done["model"]}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.7B")
    ap.add_argument("--text", default="I checked the calendar for tomorrow and you have three meetings, the first one starting at nine fifteen.")
    ap.add_argument("--port", type=int, default=8019)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    if not BIN.exists():
        print(f"build first: make build ({BIN} missing)", file=sys.stderr)
        return 1
    env = {**os.environ, "QWEN_SERVER_PORT": str(a.port), "RUST_LOG": "warn"}
    proc = subprocess.Popen([str(BIN), "--config", str(HERE / "gui.yaml"), "serve"], cwd=HERE, env=env)
    base = f"http://127.0.0.1:{a.port}"
    try:
        info = await wait_ready(base)
        print("server up:", info.get("chip"), "mlx", info.get("mlx"), "preload:", json.dumps(info.get("preload")))
        for i in range(a.repeats):
            r = await run(base, f"ws://127.0.0.1:{a.port}/ws", a.size, a.text)
            print(json.dumps({"repeat": i, "first_request_of_process": i == 0, **r}))
        return 0
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
