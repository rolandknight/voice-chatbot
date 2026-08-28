"""Real WebRTC over loopback: an aiortc client peer in the test process
drives the app through httpx's ASGI transport and asserts the Realtime
event sequence plus audio on the remote track. No GPU: the engine's
synthesize_stream is replaced with a generator of tone chunks."""

import asyncio
import json
import time
from unittest.mock import MagicMock

import httpx
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from poc_tts_streaming.server import create_app


def fake_stream(text, voice, *, cancel=None, **knobs):
    for sentence in (s.strip() for s in text.split(".") if s.strip()):
        t = np.arange(24000 // 2) / 24000.0
        yield sentence + ".", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def make_app(tmp_path):
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    eng.synthesize_stream = fake_stream
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    return create_app(eng, {"generation": {}, "realtime": {"default_voice": "one-one.mp3"}},
                      voice_paths=[voices])


async def _scenario(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        token = (await http.post("/v1/realtime/client_secrets", json={})).json()["value"]
        # No STUN: real loopback only, so ICE never touches the network (see
        # webrtc.py) and stays sub-second instead of hitting aioice's retry
        # backoff against an unreachable public STUN server.
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        events: list[dict] = []
        got: asyncio.Queue = asyncio.Queue()
        frames: list = []

        channel = pc.createDataChannel("oai-events")
        channel.on("message", lambda m: (events.append(json.loads(m)), got.put_nowait(events[-1])))
        pc.addTransceiver("audio", direction="recvonly")

        @pc.on("track")
        def on_track(track):
            async def pull():
                while len(frames) < 60:
                    frames.append(await track.recv())
            asyncio.ensure_future(pull())

        await pc.setLocalDescription(await pc.createOffer())
        r = await http.post("/v1/realtime/calls", content=pc.localDescription.sdp,
                            headers={"content-type": "application/sdp", "authorization": f"Bearer {token}"})
        assert r.status_code == 201, r.text
        await pc.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))
        assert "typ srflx" not in pc.localDescription.sdp and "typ relay" not in pc.localDescription.sdp, \
            "client ICE must be host-only (no STUN/TURN)"
        assert "typ srflx" not in r.text and "typ relay" not in r.text, \
            "server ICE must be host-only (no STUN/TURN)"

        async def wait_for(type_, timeout=10):
            while True:
                ev = await asyncio.wait_for(got.get(), timeout)
                if ev["type"] == type_:
                    return ev

        await wait_for("conversation.created")
        channel.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Hello there. General Kenobi."}]}}))
        await wait_for("conversation.item.done")
        channel.send(json.dumps({"type": "response.create"}))
        await wait_for("output_audio_buffer.stopped")

        types = [e["type"] for e in events]
        assert types[:2] == ["session.created", "conversation.created"]
        i = types.index("response.created")
        assert types[i:] == [
            "response.created", "response.output_item.added", "response.content_part.added",
            "response.output_audio_transcript.delta", "output_audio_buffer.started",
            "response.output_audio_transcript.delta", "response.output_audio_transcript.done",
            "response.output_audio.done", "response.content_part.done", "response.output_item.done",
            "response.done", "output_audio_buffer.stopped",
        ]
        # audio really crossed the (loopback) wire
        for _ in range(200):
            if len(frames) >= 60:
                break
            await asyncio.sleep(0.05)
        assert len(frames) >= 60
        loud = [f for f in frames if np.abs(f.to_ndarray()).max() > 1000]
        assert loud, "expected non-silent decoded audio on the remote track"

        location = r.headers["location"]
        assert (await http.delete(location)).status_code == 200
        await pc.close()


def test_realtime_loopback_end_to_end(tmp_path):
    started = time.monotonic()
    asyncio.run(_scenario(make_app(tmp_path)))
    assert time.monotonic() - started < 30, "loopback scenario must finish well inside the wait_for budget"
