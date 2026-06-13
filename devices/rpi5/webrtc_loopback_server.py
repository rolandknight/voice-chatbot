#!/usr/bin/env python3
"""Local WebRTC loopback server for Raspberry Pi audio testing.

The server accepts one or more WebRTC peers at POST /api/offer. Any inbound
audio track is relayed back to the same peer, so a Pi client can prove that
capture, WebRTC transport, and speaker playback all work before connecting to
the real voice-chatbot backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

log = logging.getLogger("rpi5.webrtc_loopback_server")
app = FastAPI()
relay = MediaRelay()
peers: set[RTCPeerConnection] = set()


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> PlainTextResponse:
    return PlainTextResponse(
        "Raspberry Pi WebRTC loopback server\n"
        "POST an SDP offer to /api/offer. Inbound audio is relayed back.\n"
    )


@app.post("/api/offer")
async def offer(request: Request) -> JSONResponse:
    payload = await request.json()
    if "sdp" not in payload or "type" not in payload:
        return JSONResponse({"error": "expected JSON body with sdp and type"}, 400)

    pc = RTCPeerConnection()
    peer_id = f"pc-{uuid4().hex[:8]}"
    peers.add(pc)
    log.info("[%s] peer created", peer_id)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        log.info("[%s] connection state: %s", peer_id, pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            peers.discard(pc)

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-untyped-def]
        log.info("[%s] inbound track: kind=%s id=%s", peer_id, track.kind, track.id)
        if track.kind == "audio":
            pc.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended() -> None:
            log.info("[%s] track ended: kind=%s", peer_id, track.kind)

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:  # type: ignore[no-untyped-def]
        log.info("[%s] datachannel: %s", peer_id, channel.label)

        @channel.on("message")
        def on_message(message) -> None:  # type: ignore[no-untyped-def]
            if isinstance(message, bytes):
                log.info("[%s] binary datachannel message: %d bytes", peer_id, len(message))
                return

            log.info("[%s] datachannel recv: %s", peer_id, message)
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                channel.send(json.dumps({"type": "error", "code": "bad_json"}))
                return

            channel.send(json.dumps({"type": "echo", "original": parsed}))
            if parsed.get("type") == "hello":
                channel.send(json.dumps({"type": "ready", "session_id": peer_id}))

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await asyncio.gather(*(pc.close() for pc in list(peers)), return_exceptions=True)
    peers.clear()


def _lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ip not in ips and ":" not in ip and not ip.startswith("127."):
                ips.append(ip)
    except socket.gaierror:
        pass
    return ips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"http://localhost:{args.port}")
    for ip in _lan_ips():
        print(f"http://{ip}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
