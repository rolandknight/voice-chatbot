"""pytest fixtures for the PoC harness.

Process lifecycle (stubs, shim, flowcat) is run_poc.sh's job, not ours:
fixtures only connect to already-running services.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from . import audio
from .flowcat_adapter import FlowCatAdapter
from .stubs_client import StubsClient

POC_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

GREETING_TIMEOUT_S = 30.0  # bot speaks first on connect (CONTRACT.md known-fact 2)


def _load_env(path: Path) -> None:
    """Minimal .env loader; existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def pytest_configure(config: pytest.Config) -> None:
    _load_env(POC_DIR / ".env")


@pytest.fixture
def stubs() -> Any:
    client = StubsClient(os.environ.get("STUBS_URL", "http://127.0.0.1:8790"))
    if not client.health():
        pytest.fail("stub server not reachable — run `poc/run_poc.sh up` first")
    client.clear()
    yield client
    client.close()


@pytest.fixture
def adapter_factory() -> Callable[[], FlowCatAdapter]:
    url = os.environ.get("FLOWCAT_URL", "http://127.0.0.1:6210")
    return lambda: FlowCatAdapter(base_url=url)


class TurnRunner:
    """One connected session: send fixtures, await bot speech, poll stub calls."""

    def __init__(self, adapter: FlowCatAdapter, stubs: StubsClient):
        self.adapter = adapter
        self.stubs = stubs
        self.greeting_pcm: bytes = b""

    async def send_fixture(self, name: str) -> None:
        pcm, rate = audio.load_wav(FIXTURES_DIR / name)
        assert rate == 16000, f"{name}: expected 16 kHz fixture, got {rate}"
        await self.adapter.send_audio(pcm)  # outbound track paces itself

    async def await_bot_speech(self, timeout: float = 30.0) -> bytes:
        """Capture one bot utterance (turn boundary = trailing silence)."""
        return await audio.collect_speech(self.adapter.audio_out(), rate=16000, timeout=timeout)

    async def wait_for_tool(self, name: str, timeout: float = 60.0) -> dict[str, Any]:
        """Poll the stub call log until `name` appears; return its latest call."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            calls = self.stubs.calls_for(name)
            if calls:
                return calls[-1]
            await asyncio.sleep(0.5)
        seen = [c["tool"] for c in self.stubs.get_calls()]
        raise TimeoutError(f"tool {name!r} not called within {timeout}s (saw: {seen})")


@pytest.fixture
async def session(adapter_factory: Callable[[], FlowCatAdapter], stubs: StubsClient) -> Any:
    """Connected FlowCat session with the connect greeting already consumed."""
    adapter = adapter_factory()
    await adapter.connect()
    runner = TurnRunner(adapter, stubs)
    runner.greeting_pcm = await runner.await_bot_speech(timeout=GREETING_TIMEOUT_S)
    stubs.clear()  # greeting shouldn't touch tools, but start each test clean
    yield runner
    await adapter.close(graceful=True)
