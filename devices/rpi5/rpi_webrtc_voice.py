#!/usr/bin/env python3
"""Raspberry Pi 5 WebRTC voice client using an ALSA (Linux) or avfoundation
(macOS) audio device.

The client captures microphone audio from a USB conference phone, sends it to
a WebRTC server via POST /api/offer signaling, and plays the remote WebRTC
audio track back to the same or another playback device.

The audio backend is OS-aware so the exact same client can run on the target
Raspberry Pi (ALSA) and on a dev Mac (avfoundation capture + PortAudio
playback) — the "run the client here" workflow. See --audio-format.
"""

from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import logging
import os
import platform
import signal
import threading
import time
from collections import deque
from contextlib import suppress
from typing import Any

import aiohttp
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

log = logging.getLogger("rpi5.webrtc_voice")

# openWakeWord / capture rate for the --local-wake path. aiortc resamples to
# 48 kHz for Opus, and the server resamples back down for STT, so sending 16 kHz
# frames is fine and keeps one clean rate for the wake model.
WAKE_RATE = 16000
WAKE_CHUNK = 1280  # 80 ms @ 16 kHz


def _default_audio_format() -> str:
    """ffmpeg capture backend: alsa on Linux (the Pi), avfoundation on macOS."""
    return "avfoundation" if platform.system() == "Darwin" else "alsa"


def _sd_device(device: str | None) -> Any:
    """Map a device string to what sounddevice wants: None for the system
    default, an int for an index, or the name substring otherwise."""
    if not device or device == "default":
        return None
    try:
        return int(device)
    except (TypeError, ValueError):
        return device


async def _play_track_portaudio(
    track: Any,
    *,
    device: str,
    sample_rate: int,
    channels: int,
    stop_event: asyncio.Event,
) -> None:
    """Play a remote WebRTC audio track through PortAudio (sounddevice).

    Used when the ffmpeg backend has no audio *output* muxer — notably macOS,
    where avfoundation is capture-only. Frames are resampled to a fixed
    s16/rate/layout and written on a worker thread so the event loop keeps
    servicing the peer connection. numpy + sounddevice are imported lazily so
    the ALSA (Pi) path never needs them.
    """
    import sounddevice as sd  # noqa: WPS433 — optional, macOS/dev only
    from av.audio.resampler import AudioResampler

    layout = "mono" if channels == 1 else "stereo"
    resampler = AudioResampler(format="s16", layout=layout, rate=sample_rate)
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=_sd_device(device),
    )
    stream.start()
    log.info("playing remote audio via PortAudio device %r @ %d Hz", device, sample_rate)
    try:
        while not stop_event.is_set():
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            resampled = resampler.resample(frame)
            for r in resampled if isinstance(resampled, list) else [resampled]:
                arr = r.to_ndarray().reshape(-1, channels)
                await asyncio.to_thread(stream.write, arr)
    finally:
        with suppress(Exception):
            stream.stop()
            stream.close()
        stop_event.set()


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


def _build_player(device: str, sample_rate: int, channels: int, fmt: str) -> MediaPlayer:
    return MediaPlayer(
        device,
        format=fmt,
        options={
            "sample_rate": str(sample_rate),
            "channels": str(channels),
        },
    )


def _build_recorder(device: str, fmt: str) -> MediaRecorder:
    return MediaRecorder(device, format=fmt)


async def run(args: argparse.Namespace) -> None:
    ice_servers = [
        RTCIceServer(urls=urls)
        for urls in _json_env("WEBRTC_ICE_SERVERS", args.ice_server)
    ]
    pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
    recorder: MediaRecorder | None = None
    player: MediaPlayer | None = None
    play_task: asyncio.Task[None] | None = None
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
        nonlocal recorder, play_task
        log.info("remote track: kind=%s id=%s", track.kind, track.id)
        if track.kind != "audio":
            return

        if args.audio_format == "alsa":
            # ffmpeg's ALSA muxer doubles as an output device on Linux.
            recorder = _build_recorder(args.output_device, args.audio_format)
            recorder.addTrack(track)
            await recorder.start()
            log.info("playing remote audio to ALSA device %r", args.output_device)
        else:
            # avfoundation (macOS) is capture-only, so play via PortAudio.
            play_task = asyncio.ensure_future(
                _play_track_portaudio(
                    track,
                    device=args.output_device,
                    sample_rate=args.sample_rate,
                    channels=args.channels,
                    stop_event=stop_event,
                )
            )

            def _log_play_result(t: asyncio.Task[None]) -> None:
                # The play task runs detached; without this, a failure (missing
                # sounddevice, bad device, resampler error) would be swallowed
                # and just present as "no audio". Surface it.
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    log.error("remote audio playback failed: %r", exc)
                    stop_event.set()

            play_task.add_done_callback(_log_play_result)

        @track.on("ended")
        async def on_ended() -> None:
            log.info("remote audio track ended")
            stop_event.set()

    dc = pc.createDataChannel("control")

    @dc.on("open")
    def on_datachannel_open() -> None:
        capabilities = ["audio"]
        if args.mode == "wake":
            capabilities.append("wakeword")
        hello = {
            "type": "hello",
            "kind": "smart",
            "device_id": args.device_id,
            "device": "raspberry-pi-5",
            "audio_device": args.input_device,
            "capabilities": capabilities,
        }
        dc.send(json.dumps(hello))
        log.info("control channel open; sent hello")
        # Pin persona/backend for the session. Until on-device wake lands, this
        # is how you select a voice (e.g. --persona marvin) — otherwise the
        # server stays on its default persona in push mode.
        if args.persona:
            dc.send(json.dumps({"type": "persona", "name": args.persona}))
            log.info("requested persona %r", args.persona)
        if args.backend:
            dc.send(json.dumps({"type": "backend", "name": args.backend}))
            log.info("requested backend %r", args.backend)

    @dc.on("message")
    def on_datachannel_message(message) -> None:  # type: ignore[no-untyped-def]
        log.info("control recv: %s", message)

    player = _build_player(
        args.input_device, args.sample_rate, args.channels, args.audio_format
    )
    if player.audio is None:
        raise RuntimeError(
            f"{args.audio_format} input {args.input_device!r} did not expose an "
            "audio track (is the device connected / index correct?)"
        )
    pc.addTrack(player.audio)
    log.info(
        "capturing audio from %s device %r", args.audio_format, args.input_device
    )

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await _wait_for_ice_gathering(pc, args.ice_gathering_timeout)

        payload = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "mode": args.mode,
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
        # Graceful session end: tell the server we're leaving so it tears the
        # peer down immediately rather than waiting for its stale-session guard.
        with suppress(Exception):
            if dc.readyState == "open":
                dc.send(json.dumps({"type": "bye"}))
        if play_task is not None:
            play_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await play_task
        if recorder is not None:
            await recorder.stop()
        if player is not None:
            player.audio.stop()
        await pc.close()


# ───────────────────────── on-device wake lifecycle ──────────────────────────
#
# In --local-wake mode the device runs openWakeWord itself and stays
# *disconnected* until a wake word fires. One continuous PortAudio capture feeds
# both the detector and (once awake) the outbound WebRTC track, so a ~500 ms
# pre-roll can be replayed and the first phoneme isn't clipped. The server sees
# a normal push-mode client — no server-side wake. See docs/web-rtc.md.


class _MicTrack(MediaStreamTrack):
    """Outbound WebRTC audio track fed from the capture callback via a queue.

    A fresh track is created per wake session; `push` enqueues 16 kHz mono int16
    chunks (any size), `recv` hands aiortc exactly one 20 ms frame at a time.

    Why 20 ms: aiortc stamps every RTP packet from a single encoded frame with
    the *same* timestamp (rtcrtpsender.py). A larger frame is encoded into
    multiple Opus packets that then collide on one timestamp, and the receiver
    keeps only one — so anything bigger than one Opus frame (20 ms) silently
    drops audio. Pacing is natural: the queue only fills at real time.
    """

    kind = "audio"
    FRAME = WAKE_RATE * 20 // 1000  # 320 samples @ 16 kHz = 20 ms = one Opus packet

    def __init__(self, sample_rate: int = WAKE_RATE) -> None:
        super().__init__()
        import numpy as np

        self._np = np
        self._queue: asyncio.Queue = asyncio.Queue()
        self._buf = np.zeros(0, dtype=np.int16)
        self._sr = sample_rate
        self._pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)

    def push(self, samples) -> None:
        self._queue.put_nowait(samples)

    async def recv(self):
        import av  # aiortc dependency

        while len(self._buf) < self.FRAME:
            chunk = await self._queue.get()
            self._buf = self._np.concatenate([self._buf, chunk])
        out = self._buf[: self.FRAME]
        self._buf = self._buf[self.FRAME :]
        frame = av.AudioFrame.from_ndarray(out.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._sr
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self.FRAME
        return frame


class WakeClient:
    """Idle → (wake) → connect → stream → (silence) → close → re-arm loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = asyncio.Event()
        # Pre-roll ring buffer of recent 16 kHz chunks (list of np arrays).
        preroll_chunks = max(1, round(args.preroll_ms / 80.0))
        self._preroll: deque = deque(maxlen=preroll_chunks)
        # Active session state (None when idle/listening for wake).
        self._session: dict[str, Any] | None = None
        self._last_activity = 0.0
        # Server-reported bot speech state — keeps the session alive across a
        # whole reply even if the reply is longer than the silence timeout.
        self._bot_speaking = False
        # Earliest monotonic time a wake may re-fire after a session closes
        # (swallows the echo/settling right at teardown).
        self._rearm_at = 0.0
        # Playback buffer drained by the single duplex stream's output side.
        # One device stream for both directions avoids the input/output
        # contention two separate PortAudio streams cause on the Jabra.
        self._np: Any = None
        self._play_lock = threading.Lock()
        self._play_buf: Any = None  # np.int16 array, created in run()

    # ---- wake detector (lazy so import cost is paid once, on start) ----------
    def _build_detector(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from wake import LocalWakeDetector

        model_dir = self.args.wake_model_dir
        names = [n for n in self.args.wake_models.split(",") if n]
        paths = [os.path.join(model_dir, f"{n}.onnx") for n in names]
        for p in paths:
            if not os.path.exists(p):
                raise SystemExit(f"wake model not found: {p}")
        persona_for = dict(
            pair.split("=", 1) for pair in self.args.wake_persona_map.split(",") if "=" in pair
        )
        backend_for = dict(
            pair.split("=", 1) for pair in self.args.wake_backend_map.split(",") if "=" in pair
        )
        return LocalWakeDetector(
            model_paths=paths,
            persona_for_model=persona_for,
            backend_for_model=backend_for,
            threshold=self.args.threshold,
            cooldown_secs=self.args.cooldown,
        )

    async def run(self) -> None:
        import numpy as np
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        detector = self._build_detector()
        log.info("local wake armed: models=%s threshold=%.2f", detector.model_keys, self.args.threshold)

        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                self._loop.add_signal_handler(sig, self._stop.set)

        self._np = np
        self._play_buf = np.zeros(0, dtype=np.int16)
        audio_q: asyncio.Queue = asyncio.Queue()

        def _duplex_cb(indata, outdata, _frames, _t, status) -> None:  # PortAudio thread
            if status:
                log.debug("audio status: %s", status)
            # Capture -> event loop (never touch asyncio from this thread).
            self._loop.call_soon_threadsafe(audio_q.put_nowait, indata[:, 0].copy())
            # Playback <- whatever the remote-audio pump has queued.
            n = outdata.shape[0]
            with self._play_lock:
                buf = self._play_buf
                take = min(len(buf), n)
                if take:
                    outdata[:take, 0] = buf[:take]
                    self._play_buf = buf[take:]
                if take < n:
                    outdata[take:, 0] = 0

        dev = _sd_device(self.args.input_device)
        out_dev = _sd_device(self.args.output_device)
        # Log the actually-resolved devices so it's obvious whether we're on the
        # Jabra or fell back to the system default mic/speaker.
        try:
            in_name = sd.query_devices(dev, "input")["name"]
            out_name = sd.query_devices(out_dev, "output")["name"]
            log.info("audio in=%r  out=%r", in_name, out_name)
        except Exception as e:
            log.warning("could not resolve audio device %r: %r", self.args.input_device, e)
        stream = sd.Stream(
            samplerate=WAKE_RATE, channels=1, dtype="int16", blocksize=WAKE_CHUNK,
            device=(dev, out_dev), callback=_duplex_cb,
        )
        stream.start()
        log.info("listening for wake word on device %r (Ctrl+C to stop)", self.args.input_device)

        watchdog = asyncio.ensure_future(self._silence_watchdog())
        try:
            while not self._stop.is_set():
                try:
                    samples = await asyncio.wait_for(audio_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await self._on_chunk(samples, detector)
        finally:
            watchdog.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await watchdog
            with suppress(Exception):
                stream.stop()
                stream.close()
            with suppress(asyncio.CancelledError, Exception):
                await self._end_session("shutdown")

    async def _on_chunk(self, samples, detector) -> None:
        # Always feed the detector so its streaming feature buffers stay
        # continuous. If we stop feeding it during a session, the buffers freeze
        # on the wake word that opened the session and re-fire it the instant the
        # session ends (observed: immediate reconnect without anyone speaking).
        # We only *act* on a wake when idle and past the re-arm delay.
        ev = detector.process(samples)
        if self._session is None:
            self._preroll.append(samples)
            if ev is not None and time.monotonic() >= self._rearm_at:
                log.info("WAKE %r score=%.3f persona=%r", ev.model_key, ev.score, ev.persona)
                await self._start_session(ev)
        else:
            # In session: stream mic to the server. Session lifetime is driven by
            # the server's VAD/state messages (see _on_msg), so the bot's own
            # audio / echo can't hold it open.
            self._session["track"].push(samples)

    async def _start_session(self, ev) -> None:
        """Connect for a wake event, retrying with exponential backoff so a
        briefly-unreachable server doesn't waste the wake."""
        delay = self.args.reconnect_backoff
        for attempt in range(1, self.args.connect_retries + 1):
            if self._stop.is_set():
                return
            if await self._connect_once(ev):
                return  # session live
            if attempt < self.args.connect_retries:
                log.info(
                    "connect failed (attempt %d/%d); retrying in %.1fs",
                    attempt, self.args.connect_retries, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.args.reconnect_backoff_max)
        log.error("could not connect after %d attempts — back to listening", self.args.connect_retries)
        self._rearm_at = time.monotonic() + self.args.rearm_delay

    async def _connect_once(self, ev) -> bool:
        args = self.args
        ice = [RTCIceServer(urls=u) for u in _json_env("WEBRTC_ICE_SERVERS", args.ice_server)]
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice))
        track = _MicTrack(WAKE_RATE)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            log.info("connection state: %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await self._end_session("peer closed")

        @pc.on("track")
        async def _on_track(t) -> None:  # type: ignore[no-untyped-def]
            if t.kind != "audio":
                return
            # Remote TTS -> shared playback buffer, played by the duplex stream.
            pump = asyncio.ensure_future(self._pump_remote(t))
            if self._session is not None:
                self._session["pump"] = pump

        dc = pc.createDataChannel("control")

        @dc.on("open")
        def _on_open() -> None:
            dc.send(json.dumps({
                "type": "hello", "kind": "smart", "device_id": args.device_id,
                "device": "raspberry-pi-5", "capabilities": ["audio", "wakeword"],
            }))
            if ev.persona:
                dc.send(json.dumps({"type": "persona", "name": ev.persona}))
            if ev.backend:
                dc.send(json.dumps({"type": "backend", "name": ev.backend}))

        @dc.on("message")
        def _on_msg(message) -> None:  # type: ignore[no-untyped-def]
            # Server VAD/pipeline state drives the session keep-alive: real
            # user/bot activity resets the silence timer; sustained server
            # idle lets it lapse.
            try:
                data = json.loads(message)
            except Exception:
                return
            now = time.monotonic()
            mtype = data.get("type")
            if mtype == "state":
                st = data.get("state")
                if st == "speaking":
                    self._bot_speaking = True
                    self._last_activity = now
                elif st == "idle":
                    # Bot finished — start the silence countdown from here.
                    self._bot_speaking = False
                    self._last_activity = now
                elif st in ("listening", "thinking"):
                    self._last_activity = now
            elif mtype == "transcript":
                self._last_activity = now
            log.debug("control recv: %s", message)

        pc.addTrack(track)
        # Register the session BEFORE streaming so capture chunks flow in.
        self._session = {"pc": pc, "track": track, "dc": dc, "pump": None}
        self._bot_speaking = False
        self._last_activity = time.monotonic()
        # Replay pre-roll so the first phoneme after wake isn't clipped.
        for chunk in list(self._preroll):
            track.push(chunk)

        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await _wait_for_ice_gathering(pc, args.ice_gathering_timeout)
            payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "mode": "push"}
            async with aiohttp.ClientSession() as session:
                async with session.post(args.offer_url, json=payload) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"offer failed: HTTP {resp.status}: {body}")
                    answer = json.loads(body)
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            log.info("session open (persona=%r backend=%r)", ev.persona, ev.backend)
            return True
        except Exception as e:
            log.error("failed to open session: %r", e)
            # Tear down this attempt's peer without the full _end_session path
            # (no bye/rearm) so the retry loop stays in control.
            self._session = None
            with suppress(Exception):
                await pc.close()
            return False

    async def _pump_remote(self, track) -> None:
        """Read the remote TTS track, resample to 16 kHz mono, and append to the
        shared playback buffer that the duplex stream's output side drains."""
        from av.audio.resampler import AudioResampler

        resampler = AudioResampler(format="s16", layout="mono", rate=WAKE_RATE)
        max_len = WAKE_RATE * 2  # cap latency at ~2s if playback falls behind
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                for r in resampler.resample(frame):
                    arr = r.to_ndarray().reshape(-1).astype(self._np.int16)
                    with self._play_lock:
                        buf = self._np.concatenate([self._play_buf, arr])
                        if len(buf) > max_len:
                            buf = buf[-max_len:]
                        self._play_buf = buf
        except Exception:
            log.debug("remote pump ended", exc_info=True)

    async def _silence_watchdog(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            if self._session is None:
                continue
            # Never time out mid-response: keep alive for the whole bot reply
            # (server 'speaking'..'idle'). The audio track always carries frames
            # (silence between turns), so the playback buffer can't tell us this
            # — only the server's state messages can.
            if self._bot_speaking:
                self._last_activity = time.monotonic()
                continue
            if time.monotonic() - self._last_activity >= self.args.session_timeout:
                log.info("no speech for %.0fs — ending session", self.args.session_timeout)
                await self._end_session("silence")

    async def _end_session(self, reason: str) -> None:
        session = self._session
        if session is None:
            return
        self._session = None  # stop capture pushes immediately
        self._bot_speaking = False
        self._rearm_at = time.monotonic() + self.args.rearm_delay
        log.info("closing session (%s)", reason)
        pc, dc = session["pc"], session["dc"]
        with suppress(Exception):
            if dc.readyState == "open":
                dc.send(json.dumps({"type": "bye"}))
        pump = session.get("pump")
        if pump is not None:
            pump.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pump
        # Drop any buffered TTS so the next session doesn't play stale audio.
        with self._play_lock:
            if self._play_buf is not None:
                self._play_buf = self._np.zeros(0, dtype=self._np.int16)
        with suppress(Exception):
            await pc.close()


async def run_local_wake(args: argparse.Namespace) -> None:
    await WakeClient(args).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offer-url",
        default=os.environ.get("WEBRTC_OFFER_URL", "http://127.0.0.1:8080/api/offer"),
        help="WebRTC signaling endpoint.",
    )
    parser.add_argument(
        "--audio-format",
        default=os.environ.get("AUDIO_FORMAT", _default_audio_format()),
        choices=["alsa", "avfoundation"],
        help="ffmpeg capture backend. Default: alsa on Linux, avfoundation on "
        "macOS. avfoundation is capture-only, so playback goes through "
        "PortAudio (sounddevice) — used for the 'run the client here' dev loop.",
    )
    parser.add_argument(
        "--input-device",
        default=os.environ.get("ALSA_INPUT_DEVICE", "default"),
        help="Capture device. ALSA: 'default' or 'plughw:CARD=Speaker,DEV=0'. "
        "avfoundation: an audio index like ':0' (list with "
        "`ffmpeg -f avfoundation -list_devices true -i \"\"`). "
        "Defaults to ':0' when --audio-format=avfoundation.",
    )
    parser.add_argument(
        "--output-device",
        default=os.environ.get("ALSA_OUTPUT_DEVICE", "default"),
        help="Playback device. ALSA: 'default' or 'plughw:...'. avfoundation "
        "path: a PortAudio device index or name substring (e.g. 'Jabra'); "
        "'default' uses the system default output.",
    )
    parser.add_argument("--sample-rate", type=int, default=int(os.environ.get("AUDIO_SAMPLE_RATE", "48000")))
    parser.add_argument("--channels", type=int, default=int(os.environ.get("AUDIO_CHANNELS", "1")))
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "rpi5-jabra"))
    parser.add_argument(
        "--mode",
        default=os.environ.get("MODE", "push"),
        choices=["push", "wake"],
        help="Turn strategy the server should use. 'push' = VAD-only (device "
        "does its own wake). 'wake' = server-side wake detection + wake->persona "
        "switching (stopgap until on-device wake exists).",
    )
    parser.add_argument(
        "--persona",
        default=os.environ.get("PERSONA") or None,
        help="Pin a persona for the session (e.g. 'marvin'). Sent over the "
        "control channel right after hello.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("BACKEND") or None,
        help="Pin an LLM backend for the session (e.g. 'ollama' or 'claude').",
    )
    parser.add_argument(
        "--ice-server",
        action="append",
        default=[],
        help="ICE server URL. Repeatable. Defaults to no STUN for LAN/local testing.",
    )
    parser.add_argument("--ice-gathering-timeout", type=float, default=2.0)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))

    # --- on-device wake mode (--local-wake) ---
    wake = parser.add_argument_group("on-device wake (--local-wake)")
    wake.add_argument(
        "--local-wake",
        action="store_true",
        help="Run openWakeWord on-device and stay disconnected until it fires "
        "(the device owns wake + session end). Capture goes through PortAudio "
        "(sounddevice), so --input-device is a name substring / index (e.g. "
        "'Jabra'), NOT an ffmpeg device string.",
    )
    _wake_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "wakeword"
    )
    wake.add_argument("--wake-model-dir", default=os.path.normpath(_wake_model_dir))
    wake.add_argument("--wake-models", default=os.environ.get("WAKE_MODELS", "hey_babel,hey_marvin"))
    wake.add_argument(
        "--wake-persona-map",
        default=os.environ.get("WAKE_PERSONA_MAP", "hey_babel=babel,hey_marvin=marvin"),
        help="Comma-separated model=persona pairs applied when a model fires.",
    )
    wake.add_argument(
        "--wake-backend-map",
        default=os.environ.get("WAKE_BACKEND_MAP", ""),
        help="Comma-separated model=backend pairs (e.g. 'hey_claude=claude'). "
        "Sent alongside persona when that model fires; empty = server default.",
    )
    wake.add_argument("--threshold", type=float, default=float(os.environ.get("THRESHOLD", "0.5")))
    wake.add_argument("--cooldown", type=float, default=1.5)
    wake.add_argument(
        "--session-timeout", type=float, default=8.0,
        help="End the session after this many seconds without local speech.",
    )
    wake.add_argument("--preroll-ms", type=int, default=500, help="Audio replayed before the wake instant.")
    wake.add_argument(
        "--rearm-delay", type=float, default=1.5,
        help="Ignore wake fires for this long after a session closes (settling/echo).",
    )
    wake.add_argument(
        "--connect-retries", type=int, default=3,
        help="Attempts to reach the server per wake before giving up.",
    )
    wake.add_argument(
        "--reconnect-backoff", type=float, default=0.5,
        help="Initial delay between connect retries (doubles each attempt).",
    )
    wake.add_argument("--reconnect-backoff-max", type=float, default=4.0)
    wake.add_argument(
        "--speech-dbfs", type=float, default=-50.0,
        help="dBFS above which a chunk counts as speech (keeps the session alive).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # avfoundation has no "default" capture keyword — ':0' is the first audio
    # input device. Only override when the user didn't pass an explicit device.
    # (--local-wake uses PortAudio, where 'default' is valid, so skip it there.)
    if not args.local_wake and args.audio_format == "avfoundation" and args.input_device == "default":
        args.input_device = ":0"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_local_wake(args) if args.local_wake else run(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("stopped")


if __name__ == "__main__":
    main()
