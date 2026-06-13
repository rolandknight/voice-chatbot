#!/usr/bin/env python3
"""Raspberry Pi 5 WebRTC voice client using an ALSA audio device.

The client captures microphone audio from a USB conference phone, sends it to
a WebRTC server via POST /api/offer signaling, and plays the remote WebRTC
audio track back to the same or another ALSA playback device.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from contextlib import suppress
from typing import Any

import aiohttp
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder

log = logging.getLogger("rpi5.webrtc_voice")


def _json_env(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    return json.loads(raw)


async def _wait_for_ice_gathering(pc: RTCPeerConnection, timeout: float) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        log.info("ICE gathering state: %s", pc.iceGatheringState)
        if pc.iceGatheringState == "complete":
            done.set()

    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(done.wait(), timeout=timeout)


def _build_player(device: str, sample_rate: int, channels: int) -> MediaPlayer:
    return MediaPlayer(
        device,
        format="alsa",
        options={
            "sample_rate": str(sample_rate),
            "channels": str(channels),
        },
    )


def _build_recorder(device: str) -> MediaRecorder:
    return MediaRecorder(device, format="alsa")


async def run(args: argparse.Namespace) -> None:
    ice_servers = [
        RTCIceServer(urls=urls)
        for urls in _json_env("WEBRTC_ICE_SERVERS", args.ice_server)
    ]
    pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
    recorder: MediaRecorder | None = None
    player: MediaPlayer | None = None
    stop_event = asyncio.Event()

    def stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop)

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:
        log.info("connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            stop_event.set()

    @pc.on("iceconnectionstatechange")
    def on_ice_connection_state_change() -> None:
        log.info("ICE connection state: %s", pc.iceConnectionState)

    @pc.on("track")
    async def on_track(track) -> None:  # type: ignore[no-untyped-def]
        nonlocal recorder
        log.info("remote track: kind=%s id=%s", track.kind, track.id)
        if track.kind != "audio":
            return

        recorder = _build_recorder(args.output_device)
        recorder.addTrack(track)
        await recorder.start()
        log.info("playing remote audio to ALSA device %r", args.output_device)

        @track.on("ended")
        async def on_ended() -> None:
            log.info("remote audio track ended")
            stop_event.set()

    dc = pc.createDataChannel("control")

    @dc.on("open")
    def on_datachannel_open() -> None:
        hello = {
            "type": "hello",
            "kind": "smart",
            "device_id": args.device_id,
            "device": "raspberry-pi-5",
            "audio_device": args.input_device,
            "capabilities": ["audio"],
        }
        dc.send(json.dumps(hello))
        log.info("control channel open; sent hello")

    @dc.on("message")
    def on_datachannel_message(message) -> None:  # type: ignore[no-untyped-def]
        log.info("control recv: %s", message)

    player = _build_player(args.input_device, args.sample_rate, args.channels)
    if player.audio is None:
        raise RuntimeError(f"ALSA input {args.input_device!r} did not expose an audio track")
    pc.addTrack(player.audio)
    log.info("capturing audio from ALSA device %r", args.input_device)

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await _wait_for_ice_gathering(pc, args.ice_gathering_timeout)

        payload = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(args.offer_url, json=payload) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"offer failed: HTTP {response.status}: {body}"
                    )
                answer = json.loads(body)

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        log.info("connected; press Ctrl+C to stop")
        await stop_event.wait()
    finally:
        if recorder is not None:
            await recorder.stop()
        if player is not None:
            player.audio.stop()
        await pc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offer-url",
        default=os.environ.get("WEBRTC_OFFER_URL", "http://127.0.0.1:8080/api/offer"),
        help="WebRTC signaling endpoint.",
    )
    parser.add_argument(
        "--input-device",
        default=os.environ.get("ALSA_INPUT_DEVICE", "default"),
        help="ALSA capture device, for example 'default' or 'plughw:CARD=Speaker,DEV=0'.",
    )
    parser.add_argument(
        "--output-device",
        default=os.environ.get("ALSA_OUTPUT_DEVICE", "default"),
        help="ALSA playback device, for example 'default' or 'plughw:CARD=Speaker,DEV=0'.",
    )
    parser.add_argument("--sample-rate", type=int, default=int(os.environ.get("AUDIO_SAMPLE_RATE", "48000")))
    parser.add_argument("--channels", type=int, default=int(os.environ.get("AUDIO_CHANNELS", "1")))
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "rpi5-jabra"))
    parser.add_argument(
        "--ice-server",
        action="append",
        default=[],
        help="ICE server URL. Repeatable. Defaults to no STUN for LAN/local testing.",
    )
    parser.add_argument("--ice-gathering-timeout", type=float, default=2.0)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
