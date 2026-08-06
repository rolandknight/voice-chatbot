"""SUT adapter interface + normalized Event (poc/CONTRACT.md).

The harness talks to any System Under Test exclusively through this
protocol; everything implementation-specific lives in the adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional, Protocol

EventKind = Literal["transcript-user", "transcript-bot", "state", "error"]


@dataclass
class Event:
    """Normalized SUT event. `raw` always keeps the original payload."""

    kind: EventKind
    raw: Any
    ts: float  # time.monotonic() at receipt
    text: Optional[str] = None


@dataclass
class Session:
    """Handle for one connected SUT session."""

    id: str
    meta: dict[str, Any] = field(default_factory=dict)


class SutAdapter(Protocol):
    """Black-box interface every SUT adapter implements."""

    async def connect(self) -> Session:
        """Negotiate transport, open the event stream."""
        ...

    async def send_audio(self, pcm: bytes) -> None:
        """Stream 16 kHz mono s16 PCM to the SUT."""
        ...

    def audio_out(self) -> AsyncIterator[bytes]:
        """Captured bot audio as 16 kHz mono s16 PCM chunks."""
        ...

    def events(self) -> AsyncIterator[Event]:
        """Normalized events (state/transcript/error)."""
        ...

    async def close(self, graceful: bool = True) -> None:
        """Graceful bye vs abrupt kill (for teardown tests)."""
        ...
