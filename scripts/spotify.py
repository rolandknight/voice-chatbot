"""Spotify Connect playback for the Babel voice agent.

Audio flow:
    librespot --backend pipe (stdout PCM)  →  reader thread  →  registered sink

librespot advertises a Spotify Connect endpoint named "Babel" on the LAN;
the user binds it once from any Spotify client (phone, desktop), then we
control playback via the Web API through spotipy, always targeting that
device. librespot's `pipe` backend writes raw 44.1 kHz/s16le/stereo PCM to
stdout; a reader thread hands each chunk to whatever sink is registered via
`set_pcm_sink` — the active pipeline's SpotifyMediaInjector — which streams it
out that session's transport (WebRTC peer, or the server's Jabra in
--local-audio mode). Ducking is handled in the pipeline by the injector, not
here.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

# When run standalone (`python scripts/spotify.py …`), the project root isn't
# on sys.path, so `from config import …` would fail. app.py-driven imports
# add it already; this guards the standalone case.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get as get_config  # noqa: E402

from radio import _mpv_jabra_device


_SPOTIFY_SCOPE = (
    "user-modify-playback-state user-read-playback-state "
    "user-read-currently-playing playlist-read-private "
    "playlist-read-collaborative"
)

_CONFIG_DIR = Path.home() / ".config" / "babel"
_TOKEN_CACHE = _CONFIG_DIR / "spotify_token.json"
_DEVICE_CACHE = _CONFIG_DIR / "spotify_device.txt"

_LIBRESPOT_DEVICE_NAME = "Babel"
# librespot's pipe backend always emits 44.1kHz, 16-bit signed, stereo PCM.
# https://github.com/librespot-org/librespot/wiki/Audio-backends
_PCM_RATE = 44100


def _spotipy_module():
    """Lazy import so this file remains importable when spotipy isn't installed
    (Spotify can be disabled cleanly via env)."""
    import spotipy
    return spotipy


class _NoArtistTracksError(Exception):
    def __init__(self, artist_name: str):
        super().__init__(artist_name)
        self.artist_name = artist_name


def _is_device_not_found(exc: Exception) -> bool:
    """True if `exc` is a spotipy SpotifyException with HTTP 404 — that's
    Spotify's signal that the device_id we sent no longer exists (librespot
    restarted, the Connect session got reset, etc.)."""
    return getattr(exc, "http_status", None) == 404


class SpotifyPlayer:
    """Owns a librespot+mpv subprocess pair and a spotipy Web API client.

    Spotify Connect makes "Babel" appear in the user's Spotify clients; we
    control playback exclusively via the Web API targeting that device.
    Pause/resume for ducking goes through mpv IPC (zero-latency, no API
    call) — same trick as RadioPlayer.

    Thread-safety: methods are called from the asyncio event loop via
    asyncio.to_thread; they only spawn/signal subprocesses and make
    blocking HTTP via spotipy.
    """

    def __init__(
        self,
        ipc_path: str = "/tmp/babel-spotify.sock",
        device_name: str = _LIBRESPOT_DEVICE_NAME,
    ):
        self._ipc_path = ipc_path
        self._device_name = device_name
        self._librespot: Optional[subprocess.Popen] = None
        self._mpv: Optional[subprocess.Popen] = None
        self._paused = False  # legacy duck-pause state (mpv IPC path, unused)
        self._jabra: Optional[str] = None
        # librespot's raw PCM (44.1k/s16le/stereo) is read here and handed to a
        # registered sink — the active pipeline's Spotify injector — so music
        # plays out that session's transport (WebRTC peer or local Jabra)
        # instead of a local mpv → CoreAudio speaker.
        self._reader: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._pcm_sink: Optional[Callable[[bytes], None]] = None
        self._sink_lock = threading.Lock()
        self._sp = None  # spotipy.Spotify, lazy
        self._device_id: Optional[str] = self._load_cached_device_id()
        self._now_playing_cache: Optional[tuple[float, Optional[dict]]] = None
        # User-issued pause (voice command) — distinct from duck pause.
        self._user_paused = False

    # ---------- Audio sink ----------

    def set_pcm_sink(self, sink: Callable[[bytes], None]) -> None:
        """Register the consumer for librespot's raw PCM (44.1k/s16le/stereo).

        The active pipeline's Spotify injector calls this so music flows out its
        WebRTC peer (or the local Jabra) rather than a local speaker. Only one
        sink at a time — a new registration takes over."""
        with self._sink_lock:
            self._pcm_sink = sink

    def clear_pcm_sink(self, sink: Optional[Callable[[bytes], None]] = None) -> bool:
        """Clear the sink. If `sink` is given, only clear it if it's the current
        one (so a stale connection can't unhook a newer one). Returns True if a
        sink was actually cleared — the caller can then stop the audio sink."""
        with self._sink_lock:
            if sink is not None and self._pcm_sink is not sink:
                return False
            had = self._pcm_sink is not None
            self._pcm_sink = None
            return had

    def ensure_audio_sink(self) -> bool:
        """Start librespot (if not already running) and a reader thread that
        streams its PCM to the registered sink. Returns True on success.

        When this spawns a fresh sink the cached device id is invalidated —
        librespot generates a new id per session, so the next resolve_device()
        must re-query the API rather than trust the on-disk cache."""
        if self._sink_alive():
            return True
        # Fresh spawn coming up — drop any stale device id so resolve_device
        # re-queries.
        self._device_id = None

        librespot_bin = shutil.which("librespot")
        if librespot_bin is None:
            logger.error(
                "librespot not found. Install with `brew install librespot`."
            )
            return False

        # librespot 0.8.x: zeroconf discovery is on by default; there's only
        # --disable-discovery to opt out. Passing the (now-removed) flag
        # `--enable-discovery` makes librespot exit immediately.
        librespot_cmd = [
            librespot_bin,
            "--backend", "pipe",
            "--name", self._device_name,
            "--bitrate", "320",
            "--initial-volume", "80",
        ]
        try:
            self._librespot = subprocess.Popen(
                librespot_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except OSError as e:
            logger.error(f"Failed to spawn librespot: {e}")
            return False

        self._paused = False
        self._reader_stop.clear()
        self._reader = threading.Thread(
            target=self._pump_pcm, name="spotify-pcm", daemon=True
        )
        self._reader.start()
        logger.info(
            f"Spotify sink started: librespot[{self._librespot.pid}] -> pipeline injector"
        )
        return True

    def _pump_pcm(self) -> None:
        """Read librespot's stdout PCM and hand each chunk to the current sink.
        Runs on a daemon thread; exits when librespot's stdout closes."""
        stream = self._librespot.stdout if self._librespot else None
        if stream is None:
            return
        # 4096 bytes = 1024 stereo s16 frames ≈ 23 ms at 44.1 kHz — low latency.
        while not self._reader_stop.is_set():
            try:
                data = stream.read1(4096)
            except (ValueError, OSError):
                break
            if not data:
                break  # librespot exited / stdout closed
            with self._sink_lock:
                sink = self._pcm_sink
            if sink is not None:
                try:
                    sink(data)
                except Exception as e:
                    logger.debug(f"spotify pcm sink raised: {e!r}")

    def stop_audio_sink(self) -> None:
        self._reader_stop.set()
        self._terminate(self._librespot, "librespot")
        # Closing librespot's stdout unblocks the reader's read1().
        if self._librespot is not None and self._librespot.stdout is not None:
            try:
                self._librespot.stdout.close()
            except OSError:
                pass
        self._librespot = None
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        self._paused = False

    def _terminate(self, proc: Optional[subprocess.Popen], label: str) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        except (ProcessLookupError, OSError) as e:
            logger.debug(f"{label} terminate raised: {e}")

    def _sink_alive(self) -> bool:
        return self._librespot is not None and self._librespot.poll() is None

    def _jabra_device(self) -> Optional[str]:
        if self._jabra is None:
            self._jabra = _mpv_jabra_device()
            if self._jabra:
                logger.info(f"Spotify output device: {self._jabra}")
            else:
                logger.warning(
                    "Could not find a Jabra CoreAudio device via mpv. "
                    "Spotify will use the system default output."
                )
        return self._jabra

    # ---------- Spotipy client + device id ----------

    def _client(self):
        if self._sp is not None:
            return self._sp
        spotify_cfg = get_config().skills.spotify
        client_id = spotify_cfg.client_id.get_secret_value().strip()
        if not client_id:
            raise RuntimeError(
                "SPOTIPY_CLIENT_ID is not set in .env. Create a Spotify app at "
                "https://developer.spotify.com/dashboard."
            )
        redirect_uri = spotify_cfg.redirect_uri
        spotipy = _spotipy_module()
        from spotipy.oauth2 import SpotifyPKCE
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        auth = SpotifyPKCE(
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=_SPOTIFY_SCOPE,
            cache_handler=spotipy.cache_handler.CacheFileHandler(
                cache_path=str(_TOKEN_CACHE),
            ),
            open_browser=False,
        )
        # We never want a runtime handler to block on a browser prompt —
        # that's what --bootstrap is for. Inspect the cache directly so a
        # missing token surfaces as a clear error instead of triggering
        # SpotifyPKCE's interactive flow.
        if auth.cache_handler.get_cached_token() is None:
            raise RuntimeError(
                "Spotify isn't authorised yet. Run "
                "`python scripts/spotify.py --bootstrap`."
            )
        # retries=1 (default 3) + a short request timeout keeps the API
        # surface area responsive: a 429 burst produces one quick fail
        # instead of three retries' worth of stdout spam, and a hung
        # network never blocks shutdown for more than a few seconds.
        self._sp = spotipy.Spotify(
            auth_manager=auth, retries=1, status_retries=1, requests_timeout=4.0,
        )
        return self._sp

    def _load_cached_device_id(self) -> Optional[str]:
        try:
            return _DEVICE_CACHE.read_text().strip() or None
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.debug(f"Could not read cached spotify device: {e}")
            return None

    def _save_cached_device_id(self, device_id: str) -> None:
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _DEVICE_CACHE.write_text(device_id)
        except OSError as e:
            logger.debug(f"Could not write cached spotify device: {e}")

    def resolve_device(self, force_refresh: bool = False) -> Optional[str]:
        """Return the librespot device's Spotify Connect id, polling Spotify
        for up to ~5s if the device hasn't appeared yet. If the sink isn't
        running it gets started. Returns None if 'Babel' still isn't visible
        after polling (i.e. it's never been bound from a Spotify client)."""
        if self._device_id and not force_refresh:
            return self._device_id
        try:
            sp = self._client()
        except RuntimeError as e:
            logger.warning(f"Spotify client unavailable: {e}")
            return None

        if not self._sink_alive():
            self.ensure_audio_sink()

        # librespot can take a couple of seconds to register with Spotify
        # after spawn, so poll instead of one-shot. ~5s upper bound keeps
        # voice latency tolerable when the bind is genuinely missing.
        deadline = time.monotonic() + 5.0
        attempt = 0
        while True:
            attempt += 1
            try:
                devices = (sp.devices() or {}).get("devices", [])
            except Exception as e:
                logger.warning(f"sp.devices() failed: {e}")
                return None
            for d in devices:
                if d.get("name") == self._device_name:
                    new_id = d.get("id")
                    if new_id and new_id != self._device_id:
                        logger.info(
                            f"Spotify device id refreshed to {new_id} "
                            f"after {attempt} attempt(s)"
                        )
                    self._device_id = new_id
                    if self._device_id:
                        self._save_cached_device_id(self._device_id)
                    return self._device_id
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.4)

    # ---------- Playback ----------

    def is_playing(self) -> bool:
        return self._sink_alive() and not self._user_paused

    def now_playing(self) -> Optional[str]:
        info = self._current_playback()
        if not info or not info.get("item"):
            return None
        item = info["item"]
        track = item.get("name", "")
        artists = self._artists_str(item)
        if track and artists != "unknown":
            return f"{track} by {artists}"
        return track or None

    def _current_playback(self) -> Optional[dict]:
        now = time.monotonic()
        if self._now_playing_cache and now - self._now_playing_cache[0] < 2.0:
            return self._now_playing_cache[1]
        try:
            sp = self._client()
            info = sp.current_playback()
        except Exception as e:
            logger.debug(f"current_playback failed: {e}")
            return None
        self._now_playing_cache = (now, info)
        return info

    def search_and_play(
        self, query: str, kind: str = "any"
    ) -> tuple[bool, str]:
        """Returns (success, spoken_reply)."""
        query = (query or "").strip()
        if not query:
            return False, "Tell me what to play."

        if not self.ensure_audio_sink():
            return False, "I couldn't start the Spotify audio sink."

        device_id = self.resolve_device()
        if not device_id:
            return False, (
                "Spotify can't see the Babel device yet. Open Spotify on your "
                "phone, pick Connect, and choose Babel."
            )

        kind = (kind or "any").lower().strip()
        if kind not in {"track", "album", "artist", "any"}:
            kind = "any"

        try:
            sp = self._client()
        except RuntimeError as e:
            return False, str(e)

        search_type = (
            "track,album,artist" if kind == "any" else kind
        )
        try:
            # `market="from_token"` would tie results to the user's market
            # but requires the `user-read-private` scope, which we don't
            # request. Omitting it returns globally-available results,
            # which a Premium account can always play.
            results = sp.search(q=query, type=search_type, limit=5)
        except Exception as e:
            logger.warning(f"Spotify search failed: {e}")
            return False, "I couldn't reach Spotify search."

        pick_kind, item = self._pick_result(results, kind)
        if item is None:
            return False, f"I couldn't find {query} on Spotify."

        self._transfer_if_needed(sp, device_id)

        def _start(target_device_id: str) -> str:
            if pick_kind == "track":
                sp.start_playback(device_id=target_device_id, uris=[item["uri"]])
                return f"Playing {item['name']} by {self._artists_str(item)}."
            if pick_kind == "album":
                sp.start_playback(
                    device_id=target_device_id, context_uri=item["uri"]
                )
                return f"Playing {item['name']} by {self._artists_str(item)}."
            if pick_kind == "artist":
                top = (
                    sp.artist_top_tracks(item["id"]) or {}
                ).get("tracks") or []
                if not top:
                    raise _NoArtistTracksError(item["name"])
                sp.start_playback(
                    device_id=target_device_id,
                    uris=[t["uri"] for t in top[:10]],
                )
                return f"Playing top tracks from {item['name']}."
            raise ValueError(f"unexpected pick_kind: {pick_kind!r}")

        try:
            spoken = _start(device_id)
        except _NoArtistTracksError as e:
            return False, f"I couldn't find any tracks for {e.artist_name}."
        except Exception as e:
            if _is_device_not_found(e):
                # Cached/in-memory device id is stale. Re-resolve and retry once.
                logger.info("start_playback got 404; refreshing device id")
                device_id = self.resolve_device(force_refresh=True)
                if not device_id:
                    return False, (
                        "Spotify lost the Babel device. Open Spotify on "
                        "your phone and pick Babel from Connect."
                    )
                try:
                    spoken = _start(device_id)
                except _NoArtistTracksError as e2:
                    return False, f"I couldn't find any tracks for {e2.artist_name}."
                except Exception as e2:
                    logger.warning(f"start_playback retry failed: {e2}")
                    return False, "Spotify wouldn't start playback."
            else:
                logger.warning(f"start_playback failed: {e}")
                return False, "Spotify wouldn't start playback."

        self._user_paused = False
        self._now_playing_cache = None
        return True, spoken

    def _transfer_if_needed(self, sp, device_id: str) -> None:
        try:
            current = sp.current_playback()
        except Exception as e:
            logger.debug(f"current_playback (transfer check) failed: {e}")
            return
        if current and (current.get("device") or {}).get("id") != device_id:
            try:
                sp.transfer_playback(device_id=device_id, force_play=False)
            except Exception as e:
                logger.debug(f"transfer_playback failed: {e}")

    @staticmethod
    def _artists_str(item: dict) -> str:
        artists = item.get("artists") or []
        names = [a.get("name", "") for a in artists if a.get("name")]
        return ", ".join(names) if names else "unknown"

    def _pick_result(
        self, results: dict, kind: str
    ) -> tuple[str, Optional[dict]]:
        """For kind=any pick the bucket with the highest popularity, tie-break
        track > album > artist (users usually want one song)."""
        buckets = {
            "track": (results.get("tracks") or {}).get("items") or [],
            "album": (results.get("albums") or {}).get("items") or [],
            "artist": (results.get("artists") or {}).get("items") or [],
        }
        if kind in {"track", "album", "artist"}:
            items = buckets.get(kind) or []
            return kind, (items[0] if items else None)

        priority = {"track": 3, "album": 2, "artist": 1}
        best_kind: Optional[str] = None
        best_item: Optional[dict] = None
        best_score = -1
        for k, items in buckets.items():
            if not items:
                continue
            top = items[0]
            score = top.get("popularity", 0)
            if score > best_score or (
                score == best_score
                and priority[k] > priority.get(best_kind or "", 0)
            ):
                best_kind, best_item, best_score = k, top, score
        return best_kind or "track", best_item

    def play_playlist(self, name: str) -> tuple[bool, str]:
        name = (name or "").strip()
        if not name:
            return False, "Tell me which playlist to play."
        if not self.ensure_audio_sink():
            return False, "I couldn't start the Spotify audio sink."
        device_id = self.resolve_device()
        if not device_id:
            return False, (
                "Spotify can't see the Babel device yet. Open Spotify on your "
                "phone and pick Babel from Connect."
            )
        try:
            sp = self._client()
        except RuntimeError as e:
            return False, str(e)

        target = self._match_playlist(sp, name)
        if target is None:
            return False, f"I couldn't find a playlist called {name}."

        self._transfer_if_needed(sp, device_id)
        try:
            sp.start_playback(device_id=device_id, context_uri=target["uri"])
        except Exception as e:
            if _is_device_not_found(e):
                logger.info("start_playback (playlist) got 404; refreshing device id")
                device_id = self.resolve_device(force_refresh=True)
                if not device_id:
                    return False, (
                        "Spotify lost the Babel device. Open Spotify on "
                        "your phone and pick Babel from Connect."
                    )
                try:
                    sp.start_playback(device_id=device_id, context_uri=target["uri"])
                except Exception as e2:
                    logger.warning(f"start_playback (playlist) retry failed: {e2}")
                    return False, f"Spotify wouldn't start {target.get('name', name)}."
            else:
                logger.warning(f"start_playback (playlist) failed: {e}")
                return False, f"Spotify wouldn't start {target.get('name', name)}."

        self._user_paused = False
        self._now_playing_cache = None
        return True, f"Playing {target.get('name', name)}."

    @staticmethod
    def _normalise(text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", " ", text or "")
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def _match_playlist(self, sp, name: str) -> Optional[dict]:
        target_norm = self._normalise(name)
        target_tokens = set(target_norm.split())

        owned: list[dict] = []
        try:
            results = sp.current_user_playlists(limit=50)
            while results:
                owned.extend(results.get("items") or [])
                if not results.get("next"):
                    break
                results = sp.next(results)
        except Exception as e:
            logger.debug(f"current_user_playlists failed: {e}")

        for pl in owned:
            if self._normalise(pl.get("name", "")) == target_norm:
                return pl
        for pl in owned:
            if target_norm and target_norm in self._normalise(pl.get("name", "")):
                return pl
        best_pl: Optional[dict] = None
        best_score = 0.0
        for pl in owned:
            tokens = set(self._normalise(pl.get("name", "")).split())
            if not tokens or not target_tokens:
                continue
            score = len(tokens & target_tokens) / len(tokens | target_tokens)
            if score > best_score:
                best_pl, best_score = pl, score
        if best_pl is not None and best_score >= 0.4:
            return best_pl

        try:
            results = sp.search(q=name, type="playlist", limit=5)
            items = (results.get("playlists") or {}).get("items") or []
            if items:
                return items[0]
        except Exception as e:
            logger.debug(f"playlist search fallback failed: {e}")
        return None

    def pause(self) -> bool:
        try:
            sp = self._client()
            sp.pause_playback(device_id=self._device_id)
        except Exception as e:
            logger.debug(f"pause_playback failed: {e}")
            return False
        self._user_paused = True
        self._now_playing_cache = None
        return True

    def resume(self) -> bool:
        try:
            sp = self._client()
            sp.start_playback(device_id=self._device_id)
        except Exception as e:
            logger.debug(f"start_playback (resume) failed: {e}")
            return False
        self._user_paused = False
        self._now_playing_cache = None
        return True

    def skip_next(self) -> bool:
        try:
            sp = self._client()
            sp.next_track(device_id=self._device_id)
        except Exception as e:
            logger.debug(f"next_track failed: {e}")
            return False
        self._now_playing_cache = None
        return True

    def skip_previous(self) -> bool:
        try:
            sp = self._client()
            sp.previous_track(device_id=self._device_id)
        except Exception as e:
            logger.debug(f"previous_track failed: {e}")
            return False
        self._now_playing_cache = None
        return True

    # ---------- Ducking (mpv IPC) ----------

    def duck_pause(self) -> None:
        """No-op: ducking is now handled in the pipeline by the Spotify injector
        (it drops music frames while the bot speaks), so the shared player has
        nothing to pause. Kept so MediaDuckWatcher's capability dispatch still
        routes here instead of calling the API-level pause()."""

    def duck_resume(self) -> None:
        """No-op — see duck_pause."""

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

    # ---------- Lifecycle ----------

    def stop(self, api_pause: bool = True) -> None:
        """Tear down the audio sink. With api_pause=True (default), also
        send a best-effort `pause_playback` so the Spotify state on other
        clients reflects the stop — this is the right thing for the
        voice-command path. On bot shutdown (Ctrl+C) pass api_pause=False
        to skip the API call: killing mpv silences the speaker locally,
        and Spotify auto-pauses a few seconds after librespot disappears
        from Connect. Skipping it also avoids spotipy spamming stdout
        with retry messages when we're rate-limited at the worst moment."""
        if api_pause and self._sp is not None and self._device_id:
            try:
                self._sp.pause_playback(device_id=self._device_id)
            except Exception as e:
                logger.debug(f"stop pause_playback failed: {e}")
        self.stop_audio_sink()
        self._user_paused = False


def _bootstrap() -> int:
    spotipy = _spotipy_module()
    from spotipy.oauth2 import SpotifyPKCE
    spotify_cfg = get_config().skills.spotify
    client_id = spotify_cfg.client_id.get_secret_value().strip()
    if not client_id:
        print("ERROR: set SPOTIPY_CLIENT_ID in .env first.", file=sys.stderr)
        return 1
    redirect_uri = spotify_cfg.redirect_uri
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    auth = SpotifyPKCE(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=_SPOTIFY_SCOPE,
        cache_handler=spotipy.cache_handler.CacheFileHandler(
            cache_path=str(_TOKEN_CACHE),
        ),
        open_browser=True,
    )
    if auth.cache_handler.get_cached_token() is not None:
        sp = spotipy.Spotify(auth_manager=auth)
        me = sp.current_user()
        print(f"Already authorised as {me.get('display_name') or me.get('id')}.")
        return 0
    # No cached token — trigger the interactive flow. PKCE opens the
    # browser, captures the redirect, then caches the token to disk.
    token = auth.get_access_token()
    if not token:
        print("ERROR: bootstrap failed - no token cached.", file=sys.stderr)
        return 1
    sp = spotipy.Spotify(auth_manager=auth)
    me = sp.current_user()
    print(f"Authorised as {me.get('display_name') or me.get('id')}.")
    print(f"Token cached at {_TOKEN_CACHE}")
    return 0


def _start_sink_blocking() -> int:
    p = SpotifyPlayer()
    if not p.ensure_audio_sink():
        return 1
    print(
        f"Spotify sink running. Bind '{_LIBRESPOT_DEVICE_NAME}' from any "
        "Spotify client (Connect menu). Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(1)
            if not p._sink_alive():
                # Surface whichever process died first, with its stderr,
                # so the failure is debuggable instead of silent.
                for proc, label in (
                    (p._librespot, "librespot"),
                ):
                    if proc is None:
                        continue
                    rc = proc.poll()
                    if rc is None:
                        continue
                    err = b""
                    if proc.stderr is not None:
                        try:
                            err = proc.stderr.read() or b""
                        except OSError:
                            pass
                    print(f"{label} exited with code {rc}.")
                    if err:
                        print(f"--- {label} stderr ---")
                        print(err.decode("utf-8", errors="replace").rstrip())
                return 1
    except KeyboardInterrupt:
        print()
    finally:
        p.stop_audio_sink()
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spotify helper")
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Run the one-time OAuth flow.",
    )
    parser.add_argument(
        "--start-sink", action="store_true",
        help="Start the librespot+mpv sink and block. Use this when "
        "binding 'Babel' from a Spotify client.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        sys.exit(_bootstrap())
    if args.start_sink:
        sys.exit(_start_sink_blocking())
    parser.print_help()
    sys.exit(0)
