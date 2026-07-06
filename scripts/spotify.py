"""Spotify Connect playback for the Babel voice agent.

This module is **control-only**: it never touches audio. librespot runs on
the machine that owns the speaker — the Raspberry Pi client (ALSA → Jabra),
or the server itself in --local-audio mode — and advertises a Spotify Connect
endpoint named "Babel" on the LAN with a real audio backend, so music plays
at native 44.1 kHz stereo quality directly out that speaker.

The user binds "Babel" once from any Spotify client (phone, desktop); we then
drive playback entirely through the Web API via spotipy, always targeting that
device id. No PCM is piped through the voice pipeline (that path caused the
choppy/static playback — a 24 kHz-mono voice-tuned WebRTC channel is the wrong
transport for music). Because playback is decoupled from the voice session,
music keeps going even after the session idles out; the next wake reconnects
to issue pause/skip/stop.

See devices/rpi5/README.md for the librespot-on-Pi setup.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

# When run standalone (`python scripts/spotify.py …`), the project root isn't
# on sys.path, so `from config import …` would fail. app.py-driven imports
# add it already; this guards the standalone case.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get as get_config  # noqa: E402


_SPOTIFY_SCOPE = (
    "user-modify-playback-state user-read-playback-state "
    "user-read-currently-playing playlist-read-private "
    "playlist-read-collaborative"
)

_CONFIG_DIR = Path.home() / ".config" / "babel"
_TOKEN_CACHE = _CONFIG_DIR / "spotify_token.json"
_DEVICE_CACHE = _CONFIG_DIR / "spotify_device.txt"

_LIBRESPOT_DEVICE_NAME = "Babel"


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
    """Web API control for a librespot "Babel" Connect endpoint.

    librespot runs elsewhere (the Pi client, or the server in --local-audio
    mode) and plays audio natively out its own speaker; this class only drives
    playback via the Web API through spotipy, always targeting Babel's device
    id. It owns no subprocesses and no audio.

    Thread-safety: methods are called from the asyncio event loop via
    asyncio.to_thread; they only make blocking HTTP via spotipy.
    """

    def __init__(
        self,
        device_name: str = _LIBRESPOT_DEVICE_NAME,
    ):
        self._device_name = device_name
        self._sp = None  # spotipy.Spotify, lazy
        self._device_id: Optional[str] = self._load_cached_device_id()
        self._now_playing_cache: Optional[tuple[float, Optional[dict]]] = None
        # User-issued pause (voice command). `_playing` tracks whether we've
        # started playback this session (no local process to probe now that
        # librespot lives on the client), so is_playing() can gate cross-stop.
        self._user_paused = False
        self._playing = False

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
        """Return the "Babel" librespot device's Spotify Connect id, polling
        Spotify for up to ~5s if it hasn't appeared yet. Returns None if 'Babel'
        still isn't visible after polling — meaning librespot isn't running on
        the client, or the user hasn't bound it from a Spotify client yet."""
        if self._device_id and not force_refresh:
            return self._device_id
        try:
            sp = self._client()
        except RuntimeError as e:
            logger.warning(f"Spotify client unavailable: {e}")
            return None

        # librespot can take a couple of seconds to register with Spotify
        # after it starts, so poll instead of one-shot. ~5s upper bound keeps
        # voice latency tolerable when the device is genuinely missing.
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
        return self._playing and not self._user_paused

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

        device_id = self.resolve_device()
        if not device_id:
            return False, (
                "Spotify can't see the Babel device yet. Make sure librespot "
                "is running on the speaker, or pick Babel from the Connect menu."
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
        self._playing = True
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
        device_id = self.resolve_device()
        if not device_id:
            return False, (
                "Spotify can't see the Babel device yet. Make sure librespot "
                "is running on the speaker, or pick Babel from the Connect menu."
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
        self._playing = True
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
        self._playing = True
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

    # ---------- Ducking ----------

    def duck_pause(self) -> None:
        """No-op: ducking is disabled — Spotify plays natively on the client and
        keeps going while the bot speaks. Kept so MediaDuckWatcher's capability
        dispatch still routes here instead of calling the API-level pause()."""

    def duck_resume(self) -> None:
        """No-op — see duck_pause."""

    # ---------- Lifecycle ----------

    def stop(self, api_pause: bool = True) -> None:
        """Stop playback. With api_pause=True (default) send a best-effort
        `pause_playback` so playback actually halts on the client and the
        Spotify state on other clients reflects the stop — the right thing for
        the voice-command path. On bot shutdown (Ctrl+C) pass api_pause=False
        to skip the API call: it avoids spotipy spamming stdout with retry
        messages when we're rate-limited at the worst moment, and librespot on
        the client keeps whatever state it had (nothing local to tear down)."""
        if api_pause and self._sp is not None and self._device_id:
            try:
                self._sp.pause_playback(device_id=self._device_id)
            except Exception as e:
                logger.debug(f"stop pause_playback failed: {e}")
        self._playing = False
        self._user_paused = False


def _bootstrap(headless: bool = False) -> int:
    spotipy = _spotipy_module()
    from spotipy.oauth2 import SpotifyPKCE
    spotify_cfg = get_config().skills.spotify
    client_id = spotify_cfg.client_id.get_secret_value().strip()
    if not client_id:
        print("ERROR: set SPOTIPY_CLIENT_ID in .env first.", file=sys.stderr)
        return 1
    redirect_uri = spotify_cfg.redirect_uri
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # headless (e.g. SSH'd into the Pi with no browser): open_browser=False makes
    # spotipy PKCE print the auth URL and prompt for the redirected URL instead
    # of spinning up a local callback server (which the redirect would never
    # reach from a browser on another machine).
    auth = SpotifyPKCE(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=_SPOTIFY_SCOPE,
        cache_handler=spotipy.cache_handler.CacheFileHandler(
            cache_path=str(_TOKEN_CACHE),
        ),
        open_browser=not headless,
    )
    if auth.cache_handler.get_cached_token() is not None:
        sp = spotipy.Spotify(auth_manager=auth)
        me = sp.current_user()
        print(f"Already authorised as {me.get('display_name') or me.get('id')}.")
        return 0
    if headless:
        print(
            "Headless auth: open the URL below in a browser on ANY device,\n"
            "approve, then copy the FULL URL it redirects to (it'll show a\n"
            f"'can't connect' page at {redirect_uri} — that's fine) and paste\n"
            "it back at the prompt.\n"
        )
    # No cached token — trigger the interactive flow. PKCE captures the redirect
    # (local server if a browser is available, else the pasted URL), then caches
    # the token to disk.
    token = auth.get_access_token()
    if not token:
        print("ERROR: bootstrap failed - no token cached.", file=sys.stderr)
        return 1
    sp = spotipy.Spotify(auth_manager=auth)
    me = sp.current_user()
    print(f"Authorised as {me.get('display_name') or me.get('id')}.")
    print(f"Token cached at {_TOKEN_CACHE}")
    return 0


def _list_devices() -> int:
    """Print the Spotify Connect devices the account can see — handy for
    confirming the client's librespot "Babel" endpoint is visible."""
    p = SpotifyPlayer()
    try:
        sp = p._client()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    devices = (sp.devices() or {}).get("devices", [])
    if not devices:
        print(
            "No Spotify Connect devices visible. Start librespot on the client "
            f"(name it '{_LIBRESPOT_DEVICE_NAME}'), or open Spotify somewhere."
        )
        return 1
    for d in devices:
        active = " (active)" if d.get("is_active") else ""
        print(f"{d.get('name')!r} — id={d.get('id')} type={d.get('type')}{active}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spotify helper")
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Run the one-time OAuth flow.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="With --bootstrap: no local browser. Prints the auth URL and "
        "prompts for the redirected URL (paste it back). Use over SSH / on "
        "a headless Pi.",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List visible Spotify Connect devices (check the client's "
        f"'{_LIBRESPOT_DEVICE_NAME}' librespot endpoint is reachable).",
    )
    args = parser.parse_args()

    if args.bootstrap:
        sys.exit(_bootstrap(headless=args.headless))
    if args.list_devices:
        sys.exit(_list_devices())
    parser.print_help()
    sys.exit(0)
