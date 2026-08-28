"""Stub skill services for the FlowCat PoC (poc/CONTRACT.md, port 8790).

Single entry point `POST /tool/{name}`: validates args against
`skills.json`, records the call, returns a canned result. Admin endpoints
inject per-tool latency (T9) and failures (T11). Run with:

    uvicorn stub_server:app --host 127.0.0.1 --port 8790
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import jsonschema
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

SKILLS_PATH = Path(__file__).parent / "skills.json"

app = FastAPI(title="poc-stub-skills")

SCHEMAS: dict[str, dict] = {
    s["name"]: s["parameters"] for s in json.loads(SKILLS_PATH.read_text())
}

# Call log + fault injection state (single-process, no locking needed beyond GIL).
CALLS: list[dict[str, Any]] = []
LATENCY: dict[str, float] = {}  # tool -> seconds to sleep before answering
FAIL: dict[str, int] = {}  # tool -> HTTP status to return


def _canned_result(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Fixed-format results per CONTRACT.md."""
    now = datetime.now()
    if name == "get_current_time":
        return {"time": now.strftime("%-I:%M %p")}
    if name == "get_current_date":
        return {"date": now.strftime("%A, %B %-d, %Y")}
    if name == "set_timer":
        return {
            "minutes": args.get("minutes"),
            "label": args.get("label"),
            "status": "set",
        }
    if name == "get_weather":
        return {
            "location": args.get("location") or "here",
            "forecast": "18 degrees and cloudy",
        }
    if name == "play_bbc_radio":
        return {"status": "playing", "station": args.get("station")}
    if name == "stop_bbc_radio":
        return {"status": "stopped"}
    if name == "play_spotify":
        return {
            "status": "playing",
            "query": args.get("query"),
            "kind": args.get("kind", "any"),
        }
    if name == "pause_spotify":
        return {"status": "paused"}
    raise HTTPException(status_code=404, detail=f"unknown tool: {name}")


@app.post("/tool/{name}")
async def call_tool(name: str, request: Request) -> dict[str, Any]:
    if name not in SCHEMAS:
        raise HTTPException(status_code=404, detail=f"unknown tool: {name}")
    body = await request.body()
    args = json.loads(body) if body else {}
    if not isinstance(args, dict):
        raise HTTPException(status_code=422, detail="arguments must be an object")
    try:
        jsonschema.validate(args, SCHEMAS[name])
    except jsonschema.ValidationError as e:
        raise HTTPException(status_code=422, detail=f"invalid args: {e.message}")
    CALLS.append({"tool": name, "args": args, "ts": time.monotonic(), "wall_ts": time.time()})
    if name in LATENCY:
        await asyncio.sleep(LATENCY[name])
    if name in FAIL:
        raise HTTPException(status_code=FAIL[name], detail="injected failure")
    return {"result": _canned_result(name, args)}


@app.get("/calls")
def get_calls() -> dict[str, Any]:
    return {"calls": CALLS}


@app.delete("/calls")
def clear_calls() -> dict[str, Any]:
    CALLS.clear()
    return {"ok": True}


class LatencyReq(BaseModel):
    tool: str
    seconds: float


@app.post("/admin/latency")
def set_latency(req: LatencyReq) -> dict[str, Any]:
    if req.seconds <= 0:
        LATENCY.pop(req.tool, None)
    else:
        LATENCY[req.tool] = req.seconds
    return {"ok": True}


class FailReq(BaseModel):
    tool: str
    status: Optional[int] = None


@app.post("/admin/fail")
def set_fail(req: FailReq) -> dict[str, Any]:
    if req.status is None:
        FAIL.pop(req.tool, None)
    else:
        FAIL[req.tool] = req.status
    return {"ok": True}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8790)
