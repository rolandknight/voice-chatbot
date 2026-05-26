"""WebRTC smoke-test server.

Standalone FastAPI + aiortc app. No Pipecat, no STT/LLM/TTS. Its only job is
to prove the WebRTC transport works against a real browser before any
pipeline wiring lands:

  - Audio loopback: inbound audio track is relayed straight back out, so
    speaking into the mic plays through the browser speakers after one RTT.
  - DataChannel "control" echo: every message is logged and echoed back as
    {"type":"echo","original":<msg>}. A "hello" message also triggers a
    synthesized "ready" reply, matching the protocol the real backend will
    speak.

Run:
    pip install -r webrtc_smoke/requirements.txt
    python webrtc_smoke/server.py
    # open http://localhost:8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("webrtc_smoke")

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

app = FastAPI()
_relay = MediaRelay()
_peers: set[RTCPeerConnection] = set()


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/offer")
async def offer(request: Request) -> JSONResponse:
    payload = await request.json()
    if "sdp" not in payload or "type" not in payload:
        return JSONResponse({"error": "expected {sdp, type}"}, status_code=400)

    pc = RTCPeerConnection()
    pc_id = f"pc-{uuid4().hex[:8]}"
    _peers.add(pc)
    log.info("[%s] new peer connection", pc_id)

    @pc.on("connectionstatechange")
    async def on_state() -> None:
        log.info("[%s] connection state -> %s", pc_id, pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            _peers.discard(pc)

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-untyped-def]
        log.info("[%s] inbound track kind=%s id=%s", pc_id, track.kind, track.id)
        if track.kind == "audio":
            pc.addTrack(_relay.subscribe(track))

        @track.on("ended")
        async def on_ended() -> None:
            log.info("[%s] track ended kind=%s", pc_id, track.kind)

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:  # type: ignore[no-untyped-def]
        log.info("[%s] datachannel open label=%s", pc_id, channel.label)

        @channel.on("message")
        def on_message(message) -> None:  # type: ignore[no-untyped-def]
            if isinstance(message, bytes):
                log.info("[%s] dc binary %d bytes (ignored)", pc_id, len(message))
                return
            log.info("[%s] dc recv: %s", pc_id, message)
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                channel.send(json.dumps({"type": "error", "code": "bad_json"}))
                return

            channel.send(json.dumps({"type": "echo", "original": msg}))
            if msg.get("type") == "hello":
                channel.send(json.dumps({"type": "ready", "session_id": pc_id}))

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
    log.info("shutting down, closing %d peers", len(_peers))
    await asyncio.gather(*(pc.close() for pc in list(_peers)), return_exceptions=True)
    _peers.clear()


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


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


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("WEBRTC_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBRTC_PORT", "8080"))
    cert = os.environ.get("WEBRTC_SSL_CERT")
    key = os.environ.get("WEBRTC_SSL_KEY")
    scheme = "https" if cert and key else "http"

    print()
    print(f"  {scheme}://localhost:{port}")
    for ip in _lan_ips():
        print(f"  {scheme}://{ip}:{port}")
    if scheme == "http":
        print()
        print("  NOTE: browsers only grant mic access on http://localhost.")
        print("  For LAN clients use `make run-webrtc-smoke-lan` (HTTPS).")
    print()

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        app_dir=str(HERE),
        ssl_certfile=cert,
        ssl_keyfile=key,
    )
