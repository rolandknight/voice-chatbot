"""FlowCat SUT adapter: WebRTC audio (aiortc) + events over a side WebSocket.

Wire details (paths + JSON field names) are FlowCat guesses pinned in
`FlowCatWire` — the Rust side confirms them and we adjust in one place.
Event normalization is deliberately permissive: unknown shapes are ignored,
`raw` is always preserved (poc/CONTRACT.md "Harness event normalization").
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from .adapter import Event, Session
from .audio import SPEECH_RMS_THRESHOLD, has_speech, resample


@dataclass
class FlowCatWire:
    """Configurable endpoint paths + JSON field names (CONTRACT.md guesses)."""

    offer_path: str = "/webrtc/offer"
    events_path_template: str = "/webrtc/events/{pc_id}"
    sdp_field: str = "sdp"
    type_field: str = "type"
    pc_id_field: str = "pc_id"


_TEXT_KEYS = ("text", "transcript", "content", "message", "utterance")
_ROLE_KEYS = ("role", "speaker", "source", "participant", "who", "from")
_BOT_WORDS = ("bot", "assistant", "agent", "tts")
_USER_WORDS = ("user", "human", "client", "stt")


def normalize_event(raw: Any, ts: Optional[float] = None) -> Optional[Event]:
    """Map a raw FlowCat event to a normalized Event; None -> ignore."""
    ts = time.monotonic() if ts is None else ts
    if not isinstance(raw, dict):
        return None
    etype = str(raw.get("type") or raw.get("event") or raw.get("kind") or "").lower()
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    text = next(
        (src[k] for src in (raw, data) for k in _TEXT_KEYS if isinstance(src.get(k), str)),
        None,
    )
    role = next(
        (src[k].lower() for src in (raw, data) for k in _ROLE_KEYS if isinstance(src.get(k), str)),
        "",
    )
    blob = f"{etype} {role}"
    if "error" in etype:
        return Event(kind="error", raw=raw, ts=ts, text=text)
    if text is not None and ("transcript" in etype or role):
        if any(w in blob for w in _BOT_WORDS):
            return Event(kind="transcript-bot", raw=raw, ts=ts, text=text)
        if any(w in blob for w in _USER_WORDS):
            return Event(kind="transcript-user", raw=raw, ts=ts, text=text)
    if "state" in etype or "state" in raw:
        return Event(kind="state", raw=raw, ts=ts, text=text)
    return None


class _OutboundPcmTrack(MediaStreamTrack):
    """Sendrecv audio track fed from a byte buffer; pads with silence."""

    kind = "audio"
    RATE = 48000
    SAMPLES = 960  # 20 ms

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()
        self._pts = 0
        self._start: Optional[float] = None

    def push(self, pcm48: bytes) -> None:
        self._buf.extend(pcm48)

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.monotonic()
        delay = self._start + self._pts / self.RATE - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        nbytes = self.SAMPLES * 2
        chunk = bytes(self._buf[:nbytes])
        del self._buf[: len(chunk)]
        chunk += b"\x00" * (nbytes - len(chunk))
        frame = AudioFrame(format="s16", layout="mono", samples=self.SAMPLES)
        frame.planes[0].update(chunk)
        frame.sample_rate = self.RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.RATE)
        self._pts += self.SAMPLES
        return frame


class FlowCatAdapter:
    """SutAdapter for the FlowCat PoC server (poc/CONTRACT.md, :6210)."""

    def __init__(self, base_url: str = "http://127.0.0.1:6210", wire: Optional[FlowCatWire] = None):
        self.base_url = base_url.rstrip("/")
        self.wire = wire or FlowCatWire()
        self.probes: dict[str, float] = {}  # connected / last_audio_sent / first_bot_*
        self.session: Optional[Session] = None
        self._pc: Optional[RTCPeerConnection] = None
        self._track = _OutboundPcmTrack()
        self._event_q: asyncio.Queue[Event] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._ws = None
        # Bot audio: continuously drained into a persistent buffer by
        # _pump_audio; readers advance a shared cursor. This makes
        # audio_out()/read_bot_audio repeatable — the stream only "ends"
        # when the track/connection truly ends.
        self._audio_buf = bytearray()
        self._audio_pos = 0
        self._audio_cond = asyncio.Condition()
        self._audio_ended = False

    async def connect(self) -> Session:
        pc = RTCPeerConnection()
        self._pc = pc
        pc.addTrack(self._track)
        pc.on("track", self._on_track)
        await pc.setLocalDescription(await pc.createOffer())  # gathers ICE
        payload = {
            self.wire.sdp_field: pc.localDescription.sdp,
            self.wire.type_field: pc.localDescription.type,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(self.base_url + self.wire.offer_path, json=payload)
            r.raise_for_status()
            answer = r.json()
        pc_id = next(
            (
                str(answer[k])
                for k in (self.wire.pc_id_field, "pc_id", "id", "pcId", "session_id", "peer_id")
                if answer.get(k) is not None
            ),
            "",
        )
        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=answer[self.wire.sdp_field],
                type=answer.get(self.wire.type_field, "answer"),
            )
        )
        if pc_id:
            self._tasks.append(asyncio.ensure_future(self._pump_events(pc_id)))
        self.probes["connected"] = time.monotonic()
        self.session = Session(id=pc_id, meta={"answer": answer})
        return self.session

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            self._tasks.append(asyncio.ensure_future(self._pump_audio(track)))

    async def _pump_audio(self, track: MediaStreamTrack) -> None:
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            while True:
                frame = await track.recv()
                for out in resampler.resample(frame):
                    pcm = out.to_ndarray().tobytes()
                    self.probes.setdefault("first_bot_frame", time.monotonic())
                    if "first_bot_speech" not in self.probes and has_speech(
                        pcm, 16000, SPEECH_RMS_THRESHOLD
                    ):
                        self.probes["first_bot_speech"] = time.monotonic()
                    async with self._audio_cond:
                        self._audio_buf.extend(pcm)
                        self._audio_cond.notify_all()
        except (MediaStreamError, asyncio.CancelledError):
            pass
        finally:
            self._audio_ended = True  # readers also poll, so a lost notify is safe
            with contextlib.suppress(BaseException):
                async with self._audio_cond:
                    self._audio_cond.notify_all()

    async def _pump_events(self, pc_id: str) -> None:
        ws_base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        url = ws_base + self.wire.events_path_template.format(pc_id=pc_id)
        with contextlib.suppress(asyncio.CancelledError):
            for attempt in range(5):
                try:
                    self._ws = await websockets.connect(url, open_timeout=5)
                    break
                except Exception:
                    await asyncio.sleep(0.5)
            else:
                return  # events unavailable; audio + stub log still carry the tests
            with contextlib.suppress(Exception):
                async for msg in self._ws:
                    try:
                        raw = json.loads(msg)
                    except (ValueError, TypeError):
                        continue
                    ev = normalize_event(raw)
                    if ev is not None:
                        self._event_q.put_nowait(ev)

    async def send_audio(self, pcm: bytes) -> None:
        """16 kHz mono s16 in; resampled to 48 kHz for the Opus track."""
        self._track.push(resample(pcm, 16000, self._track.RATE))
        self.probes["last_audio_sent"] = time.monotonic()

    async def read_bot_audio(self, timeout: float) -> Optional[bytes]:
        """New bot PCM past the shared cursor.

        Returns bytes when data is available, None on timeout, and b"" once
        the track/connection has truly ended and the buffer is drained.
        Repeatable: sequential readers share one cursor, so nothing is lost
        or duplicated between reads (e.g. greeting vs reply).
        """
        deadline = time.monotonic() + timeout
        while True:
            async with self._audio_cond:
                if self._audio_pos < len(self._audio_buf):
                    chunk = bytes(self._audio_buf[self._audio_pos :])
                    self._audio_pos = len(self._audio_buf)
                    return chunk
                if self._audio_ended:
                    return b""
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                with contextlib.suppress(asyncio.TimeoutError):
                    # Short slices double as a poll in case a notify is lost.
                    await asyncio.wait_for(self._audio_cond.wait(), min(remaining, 0.25))

    async def audio_out(self) -> AsyncIterator[bytes]:
        """Fresh iterator per call over the shared cursor (SutAdapter surface)."""
        while True:
            chunk = await self.read_bot_audio(timeout=3600.0)
            if chunk == b"":
                return
            if chunk:
                yield chunk

    def events(self) -> AsyncIterator[Event]:
        return self._drain(self._event_q)

    @staticmethod
    async def _drain(q: asyncio.Queue) -> AsyncIterator:
        while True:
            yield await q.get()

    async def close(self, graceful: bool = True) -> None:
        for t in self._tasks:
            t.cancel()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if graceful and self._pc is not None:
            with contextlib.suppress(Exception):
                await self._pc.close()
        # abrupt: leave the peer un-closed (no bye) — T7-style teardown probe
