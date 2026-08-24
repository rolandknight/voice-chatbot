"""aiortc glue: one RTCPeerConnection + RealtimeSession + PcmQueueTrack per call.

Knows about aiortc and the session; knows nothing about FastAPI or torch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from poc_tts_streaming.realtime.ids import new_id
from poc_tts_streaming.realtime.session import RealtimeSession
from poc_tts_streaming.track import PcmQueueTrack

logger = logging.getLogger(__name__)

EVENTS_CHANNEL = "oai-events"


@dataclass
class Call:
    id: str
    pc: RTCPeerConnection
    track: PcmQueueTrack
    session: RealtimeSession | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)


class CallRegistry:
    def __init__(self) -> None:
        self._calls: dict[str, Call] = {}

    def __len__(self) -> int:
        return len(self._calls)

    async def create(self, offer_sdp: str, build_session, *, session_patch: dict | None = None) -> tuple[str, str]:
        call = Call(id=new_id("call"), pc=RTCPeerConnection(), track=PcmQueueTrack())
        self._calls[call.id] = call
        pc = call.pc
        pc.addTrack(call.track)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            if channel.label != EVENTS_CHANNEL:
                logger.info("[%s] ignoring data channel %r", call.id, channel.label)
                return

            def send(event: dict) -> None:
                if channel.readyState == "open":
                    channel.send(json.dumps(event))

            call.session = build_session(send, call.track, session_patch)

            @channel.on("message")
            def on_message(message) -> None:
                if isinstance(message, str):
                    call.tasks.append(asyncio.ensure_future(call.session.handle(message)))

            call.tasks.append(asyncio.ensure_future(call.session.open()))

        @pc.on("track")
        def on_track(track) -> None:
            # A client that offers a mic gets it accepted and drained; this
            # server never listens. Draining keeps aiortc's receiver quiet.
            async def drain() -> None:
                try:
                    while True:
                        await track.recv()
                except MediaStreamError:
                    return
            call.tasks.append(asyncio.ensure_future(drain()))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            logger.info("[%s] connection state -> %s", call.id, pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.hangup(call.id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        await pc.setLocalDescription(await pc.createAnswer())
        return call.id, pc.localDescription.sdp

    async def hangup(self, call_id: str) -> bool:
        call = self._calls.pop(call_id, None)
        if call is None:
            return False
        if call.session is not None:
            await call.session.close()
        call.track.stop()
        for task in call.tasks:
            task.cancel()
        await call.pc.close()
        return True

    async def close_all(self) -> None:
        await asyncio.gather(*(self.hangup(cid) for cid in list(self._calls)), return_exceptions=True)
