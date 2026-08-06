"""httpx client for the stub skill server API (poc/CONTRACT.md, :8790)."""

from __future__ import annotations

from typing import Any, Optional

import httpx

DEFAULT_URL = "http://127.0.0.1:8790"


class StubsClient:
    """Thin sync wrapper over the stub server's harness-facing endpoints."""

    def __init__(self, base_url: str = DEFAULT_URL):
        self._client = httpx.Client(base_url=base_url, timeout=10.0)

    def health(self) -> bool:
        try:
            return self._client.get("/health").json().get("ok", False)
        except httpx.HTTPError:
            return False

    def get_calls(self) -> list[dict[str, Any]]:
        r = self._client.get("/calls")
        r.raise_for_status()
        return r.json()["calls"]

    def calls_for(self, tool: str) -> list[dict[str, Any]]:
        return [c for c in self.get_calls() if c["tool"] == tool]

    def clear(self) -> None:
        self._client.delete("/calls").raise_for_status()

    def set_latency(self, tool: str, seconds: float) -> None:
        r = self._client.post("/admin/latency", json={"tool": tool, "seconds": seconds})
        r.raise_for_status()

    def set_fail(self, tool: str, status: Optional[int]) -> None:
        r = self._client.post("/admin/fail", json={"tool": tool, "status": status})
        r.raise_for_status()

    def close(self) -> None:
        self._client.close()
