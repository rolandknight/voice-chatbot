"""BBC live radio playback for the Babel voice agent.

Spawns an `mpv` subprocess pointed at a BBC HLS endpoint and the Jabra USB
speaker (selected via CoreAudio device name). Exposes start/stop/pause/resume
driven by mpv's JSON IPC socket.

mpv is GPL and the only required system dependency beyond what install_mac.sh
already pulls in. ffplay (already installed via ffmpeg) is kept as a degraded
fallback — it cannot target a specific output device on macOS, so it only
works when the Jabra is the system default output.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger


# BBC HLS endpoints. Each station has a distinct Akamai pool, so we list URLs
# explicitly rather than templating off the slug. If a station starts 404'ing,
# BBC has reshuffled pools — refresh from https://gist.github.com/bpsib/67089b959e4fa898af69fea59ad74bc3
# or https://github.com/groupmsl/BBCRadioStreams (community-maintained).
# Sports Extra is uk-live and will 403 outside the UK without a VPN.
_BASE_WW = "http://as-hls-ww-live.akamaized.net"
_BASE_UK = "http://as-hls-uk-live.akamaized.net"


def _ww(pool: str, slug: str) -> str:
    return (
        f"{_BASE_WW}/{pool}/live/ww/{slug}/{slug}.isml/"
        f"{slug}-audio%3d96000.norewind.m3u8"
    )


def _uk(pool: str, slug: str) -> str:
    return (
        f"{_BASE_UK}/{pool}/live/uk/{slug}/{slug}.isml/"
        f"{slug}-audio%3d96000.norewind.m3u8"
    )


@dataclass(frozen=True)
class Station:
    key: str
    display: str
    url: str
    # Longest-first ordering matters: "five sports extra" must beat "five live".
    aliases: tuple[str, ...]


STATIONS: tuple[Station, ...] = (
    Station(
        "radio_5_sports_extra", "BBC Radio 5 Sports Extra",
        _uk("pool_47700285", "bbc_radio_five_live_sports_extra"),
        ("5 sports extra", "five sports extra", "radio 5 sports extra"),
    ),
    Station(
        "radio_4_extra", "BBC Radio 4 Extra",
        _ww("pool_26173715", "bbc_radio_four_extra"),
        ("radio 4 extra", "radio four extra", "4 extra", "four extra"),
    ),
    Station(
        "radio_1xtra", "BBC Radio 1Xtra",
        _ww("pool_92079267", "bbc_1xtra"),
        ("1xtra", "one xtra", "1 xtra", "radio 1 xtra", "radio one xtra"),
    ),
    Station(
        "radio_5_live", "BBC Radio 5 Live",
        _ww("pool_89021708", "bbc_radio_five_live"),
        ("5 live", "five live", "radio 5 live", "radio five live"),
    ),
    Station(
        "radio_6_music", "BBC Radio 6 Music",
        _ww("pool_81827798", "bbc_6music"),
        ("6 music", "six music", "radio 6 music", "radio six music", "radio 6", "radio six"),
    ),
    Station(
        "radio_1", "BBC Radio 1",
        _ww("pool_01505109", "bbc_radio_one"),
        ("radio 1", "radio one", "radio won", "r1"),
    ),
    Station(
        "radio_2", "BBC Radio 2",
        _ww("pool_74208725", "bbc_radio_two"),
        ("radio 2", "radio two", "radio too", "radio to"),
    ),
    Station(
        "radio_3", "BBC Radio 3",
        _ww("pool_23461179", "bbc_radio_three"),
        ("radio 3", "radio three", "radio free"),
    ),
    Station(
        "radio_4", "BBC Radio 4",
        _ww("pool_55057080", "bbc_radio_fourfm"),
        ("radio 4", "radio four", "radio for"),
    ),
    Station(
        "asian_network", "BBC Asian Network",
        _ww("pool_22108647", "bbc_asian_network"),
        ("asian network",),
    ),
    Station(
        "world_service", "BBC World Service",
        _ww("pool_87948813", "bbc_world_service"),
        ("world service",),
    ),
)


_BY_KEY: dict[str, Station] = {s.key: s for s in STATIONS}


def build_alias_table(items, get_aliases):
    """Build a (alias, item) list sorted longest-first so longer aliases win.
    Shared with bbc_shows.py so on-demand show matching uses the same rules
    as station matching."""
    return sorted(
        ((alias, item) for item in items for alias in get_aliases(item)),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )


def match_alias(text, alias_table):
    """Return the item whose alias matches `text` (longest-first). Match is on
    a punctuation-stripped, single-spaced lower form so whisper's occasional
    comma/period quirks don't break the match."""
    normalised = re.sub(r"[^\w\s]", " ", text).lower()
    normalised = re.sub(r"\s+", " ", normalised).strip()
    for alias, item in alias_table:
        if re.search(rf"\b{re.escape(alias)}\b", normalised):
            return item
    return None


# Longest-first so "five sports extra" wins over "five live" when both appear.
_ALIAS_ORDER: list[tuple[str, Station]] = build_alias_table(
    STATIONS, lambda s: s.aliases
)


def match_station(text: str) -> Optional[Station]:
    """Return the longest matching station alias in `text`, case-insensitive."""
    return match_alias(text, _ALIAS_ORDER)


def _mpv_jabra_device() -> Optional[str]:
    """Run `mpv --audio-device=help`, return the coreaudio device string that
    contains 'jabra'. Returns None if mpv is missing or no match found."""
    mpv = shutil.which("mpv")
    if not mpv:
        return None
    try:
        out = subprocess.run(
            [mpv, "--audio-device=help"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"mpv --audio-device=help failed: {e}")
        return None
    for line in out.splitlines():
        # mpv prints lines like: "  'coreaudio/Jabra Speak2 40 UC' (Jabra Speak2 40 UC)"
        m = re.search(r"'(coreaudio[^']*)'", line)
        if m and "jabra" in line.lower():
            return m.group(1)
    return None


class RadioPlayer:
    """Owns a single mpv (or ffplay) subprocess and exposes lifecycle controls.

    Thread-safety: methods are called from the asyncio event loop in app.py,
    but they only spawn/signal subprocesses and write to a Unix socket — no
    awaits required, so plain blocking calls are fine and keep the code simple.
    """

    def __init__(self, ipc_path: str = "/tmp/babel-radio.sock"):
        self._ipc_path = ipc_path
        self._proc: Optional[subprocess.Popen] = None
        self._station: Optional[Station] = None
        self._paused = False
        self._device: Optional[str] = None
        self._engine: Optional[str] = None  # "mpv" or "ffplay"

    def _resolve_device(self) -> Optional[str]:
        if self._device is not None:
            return self._device
        self._device = _mpv_jabra_device()
        if self._device:
            logger.info(f"Radio output device: {self._device}")
        else:
            logger.warning(
                "Could not find a Jabra CoreAudio device via mpv. Radio will use "
                "the system default output."
            )
        return self._device

    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def current_station(self) -> Optional[Station]:
        return self._station if self.is_playing() else None

    def start(self, station: Station) -> None:
        if self.is_playing():
            self.stop()

        mpv = shutil.which("mpv")
        if mpv:
            self._start_mpv(mpv, station)
        else:
            ffplay = shutil.which("ffplay")
            if not ffplay:
                raise RuntimeError(
                    "Neither mpv nor ffplay is installed. Run `brew install mpv`."
                )
            logger.warning(
                "mpv not found; falling back to ffplay (cannot target Jabra "
                "specifically, will use system default output)."
            )
            self._start_ffplay(ffplay, station)

        self._station = station
        self._paused = False
        logger.info(f"Started radio: {station.display}")

    def _start_mpv(self, mpv: str, station: Station) -> None:
        # Best-effort cleanup of any stale socket from a prior run.
        try:
            os.unlink(self._ipc_path)
        except FileNotFoundError:
            pass
        cmd = [
            mpv,
            "--no-video",
            "--no-terminal",
            "--idle=no",
            f"--input-ipc-server={self._ipc_path}",
            "--cache=yes",
            "--demuxer-readahead-secs=4",
        ]
        device = self._resolve_device()
        if device:
            cmd.append(f"--audio-device={device}")
        cmd.append(station.url)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self._engine = "mpv"
        # Wait briefly for the IPC socket so the first pause() doesn't race.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if os.path.exists(self._ipc_path):
                return
            time.sleep(0.05)
        logger.warning(
            f"mpv IPC socket {self._ipc_path} did not appear within 2s; "
            "pause/resume may fail until it does."
        )

    def _start_ffplay(self, ffplay: str, station: Station) -> None:
        cmd = [
            ffplay,
            "-nodisp", "-autoexit", "-loglevel", "quiet",
            station.url,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self._engine = "ffplay"

    def stop(self) -> None:
        if not self.is_playing():
            self._proc = None
            self._station = None
            self._paused = False
            return
        assert self._proc is not None
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
        except (ProcessLookupError, OSError):
            pass
        logger.info(f"Stopped radio: {self._station.display if self._station else '?'}")
        self._proc = None
        self._station = None
        self._paused = False
        try:
            os.unlink(self._ipc_path)
        except FileNotFoundError:
            pass

    def pause(self) -> None:
        if not self.is_playing() or self._paused:
            return
        if self._engine == "mpv":
            if self._send_ipc({"command": ["set_property", "pause", True]}):
                self._paused = True
        else:
            # ffplay has no pause IPC; SIGSTOP the process to freeze it.
            try:
                self._proc.send_signal(signal.SIGSTOP)  # type: ignore[union-attr]
                self._paused = True
            except (ProcessLookupError, OSError) as e:
                logger.debug(f"ffplay SIGSTOP failed: {e}")

    def resume(self) -> None:
        if not self.is_playing() or not self._paused:
            return
        if self._engine == "mpv":
            if self._send_ipc({"command": ["set_property", "pause", False]}):
                self._paused = False
        else:
            try:
                self._proc.send_signal(signal.SIGCONT)  # type: ignore[union-attr]
                self._paused = False
            except (ProcessLookupError, OSError) as e:
                logger.debug(f"ffplay SIGCONT failed: {e}")

    def _send_ipc(self, payload: dict) -> bool:
        if not os.path.exists(self._ipc_path):
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(self._ipc_path)
                s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return True
        except (OSError, socket.timeout) as e:
            logger.debug(f"mpv IPC send failed: {e}")
            return False


def lookup(key: str) -> Optional[Station]:
    return _BY_KEY.get(key)
