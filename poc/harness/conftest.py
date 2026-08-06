"""pytest fixtures for the PoC harness.

Process lifecycle (stubs, shim, flowcat) is run_poc.sh's job, not ours:
fixtures only connect to already-running services.
"""

from __future__ import annotations

import asyncio
import contextlib
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

    async def send_fixture(self, name: str) -> dict[str, float]:
        """Push a fixture; return wire-time estimates for latency probes.

        The outbound track paces itself, so the fixture's first non-silent
        sample hits the wire at push time + queued backlog + lead silence.
        """
        pcm, rate = audio.load_wav(FIXTURES_DIR / name)
        assert rate == 16000, f"{name}: expected 16 kHz fixture, got {rate}"
        backlog_s = self.adapter.outbound_backlog_s
        push_ts = time.monotonic()
        await self.adapter.send_audio(pcm)
        lead_s = audio.first_audio_ts(pcm, rate) or 0.0
        dur_s = audio.duration_s(pcm, rate)
        speech_end_s = audio.last_audio_ts(pcm, rate) or dur_s
        return {
            "push_ts": push_ts,
            "backlog_s": backlog_s,
            "speech_onset_ts": push_ts + backlog_s + lead_s,
            "speech_end_ts": push_ts + backlog_s + speech_end_s,
            "end_ts": push_ts + backlog_s + dur_s,
        }

    async def await_bot_speech(self, timeout: float = 30.0) -> bytes:
        """Capture one bot utterance (turn boundary = trailing silence)."""
        return await audio.collect_speech(self.adapter.read_bot_audio, rate=16000, timeout=timeout)

    async def wait_for_tool(
        self,
        name: str,
        timeout: float = 60.0,
        match: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """Poll the stub call log until a call to `name` appears; return it.

        `match(args)` filters calls: whisper.cpp's ~4 s batch windows can
        split an utterance and fire a turn on a partial transcript, so the
        first call to the right tool may carry mangled args while the full
        window's call lands seconds later (CONTRACT.md known-fact 3).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            calls = self.stubs.calls_for(name)
            hits = [c for c in calls if match is None or match(c["args"])]
            if hits:
                return hits[-1]
            await asyncio.sleep(0.5)
        seen = [(c["tool"], c["args"]) for c in self.stubs.get_calls()]
        raise TimeoutError(f"no matching {name!r} call within {timeout}s (saw: {seen})")


@pytest.fixture
async def session_factory(
    adapter_factory: Callable[[], FlowCatAdapter], stubs: StubsClient
) -> Any:
    """Async factory for connected sessions (greeting consumed); closes all at teardown."""
    runners: list[TurnRunner] = []

    async def make() -> TurnRunner:
        adapter = adapter_factory()
        await adapter.connect()
        runner = TurnRunner(adapter, stubs)
        runner.greeting_pcm = await runner.await_bot_speech(timeout=GREETING_TIMEOUT_S)
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        with contextlib.suppress(Exception):
            await runner.adapter.close(graceful=True)


@pytest.fixture
async def session(session_factory: Any, stubs: StubsClient) -> Any:
    """Connected FlowCat session with the connect greeting already consumed."""
    runner = await session_factory()
    stubs.clear()  # greeting shouldn't touch tools, but start each test clean
    return runner
