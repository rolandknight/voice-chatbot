"""On-demand BBC Sounds show resolution for the babel voice agent.

Two-layer strategy:
  1. Curated RSS list — fast lookup of favourites (Archers omnibus, In Our Time,
     Desert Island Discs, Today programme, etc). Most BBC talk/drama shows
     publish a podcast feed at https://podcasts.files.bbci.co.uk/<pid>.rss.
  2. yt-dlp fallback — for anything not in the curated list (including music
     shows where licensing means there's no public RSS). Resolves a
     bbc.co.uk/sounds/play/<pid> URL into a playable stream URL that mpv can
     consume. Programme PID lookup uses BBC's public-but-undocumented
     experience search endpoint.

Both paths return a `ResolvedEpisode` whose `url` is something mpv can play
directly via the existing RadioPlayer.
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx
from loguru import logger

from radio import build_alias_table, match_alias

HTTP_TIMEOUT_SECS = 8.0


@dataclass(frozen=True)
class BbcShow:
    key: str
    display: str
    # When the show has a public RSS feed, the rss_url path is preferred — it's
    # fast (one HTTP GET) and gives us episode metadata for date/query filtering.
    # If 404'ing, refresh the PID from https://www.bbc.co.uk/programmes/<pid>
    # or fall back to programme_pid + yt-dlp.
    rss_url: Optional[str]
    # Programme PID for yt-dlp fallback (when RSS is missing or empty).
    programme_pid: Optional[str]
    # Longest-first ordering matters for match_alias.
    aliases: tuple[str, ...]


# Curated favourites. Each PID was verified against
# https://podcasts.files.bbci.co.uk/<pid>.rss returning 200 with the matching
# channel title. PIDs occasionally rotate at the BBC's end; if a feed starts
# 404'ing, search BBC Sounds for the show and pull the brand PID from the URL.
# Shows without a public RSS feed (Today programme, More or Less, etc.) are
# not listed — they fall through to the BBC search + yt-dlp path at request
# time, which is slower and BBC-extractor-dependent.
CURATED_SHOWS: tuple[BbcShow, ...] = (
    BbcShow(
        "archers_omnibus", "The Archers Omnibus",
        "https://podcasts.files.bbci.co.uk/b006qnkc.rss", "b006qnkc",
        ("the archers omnibus", "archers omnibus"),
    ),
    BbcShow(
        "archers", "The Archers",
        "https://podcasts.files.bbci.co.uk/b006qpgr.rss", "b006qpgr",
        ("the archers", "archers"),
    ),
    BbcShow(
        "in_our_time", "In Our Time",
        "https://podcasts.files.bbci.co.uk/b006qykl.rss", "b006qykl",
        ("in our time",),
    ),
    BbcShow(
        "desert_island_discs", "Desert Island Discs",
        "https://podcasts.files.bbci.co.uk/b006qnmr.rss", "b006qnmr",
        ("desert island discs",),
    ),
    BbcShow(
        "front_row", "Front Row",
        "https://podcasts.files.bbci.co.uk/b006qsq5.rss", "b006qsq5",
        ("front row",),
    ),
    BbcShow(
        "thinking_allowed", "Thinking Allowed",
        "https://podcasts.files.bbci.co.uk/b006qy05.rss", "b006qy05",
        ("thinking allowed",),
    ),
    BbcShow(
        "just_a_minute", "Just A Minute",
        "https://podcasts.files.bbci.co.uk/b006s5dp.rss", "b006s5dp",
        ("just a minute",),
    ),
    BbcShow(
        "friday_night_comedy", "Friday Night Comedy",
        "https://podcasts.files.bbci.co.uk/p02pc9pj.rss", "p02pc9pj",
        ("friday night comedy",),
    ),
)


_BY_KEY: dict[str, BbcShow] = {s.key: s for s in CURATED_SHOWS}
_ALIAS_TABLE = build_alias_table(CURATED_SHOWS, lambda s: s.aliases)


def lookup_show(key: str) -> Optional[BbcShow]:
    return _BY_KEY.get(key)


def match_show(text: str) -> Optional[BbcShow]:
    return match_alias(text, _ALIAS_TABLE)


@dataclass(frozen=True)
class ResolvedEpisode:
    url: str
    display: str


@dataclass(frozen=True)
class _RssItem:
    title: str
    description: str
    enclosure_url: Optional[str]
    pub_date: Optional[datetime]


_NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "media": "http://search.yahoo.com/mrss/",
}


def _parse_pub_date(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _parse_rss(xml_text: str) -> list[_RssItem]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"RSS parse failed: {e}")
        return []
    items: list[_RssItem] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        enclosure = item.find("enclosure")
        enclosure_url = enclosure.get("url") if enclosure is not None else None
        pub_date = _parse_pub_date(item.findtext("pubDate"))
        items.append(_RssItem(title, description, enclosure_url, pub_date))
    return items


async def _fetch_rss_items(rss_url: str) -> list[_RssItem]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        r = await client.get(rss_url, headers={"User-Agent": "babel-voice-bot/1.0"})
        r.raise_for_status()
        return _parse_rss(r.text)


def _pretty_pub_date(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.strftime("%A %-d %B")


def _pick_item(
    items: list[_RssItem],
    date: Optional[str],
    query: Optional[str],
) -> Optional[_RssItem]:
    if not items:
        return None
    if date:
        target = _parse_iso_date(date)
        if target is not None:
            for item in items:
                if item.pub_date and item.pub_date.date() == target:
                    return item
        return None
    if query:
        needle = query.lower()
        for item in items:
            haystack = f"{item.title} {item.description}".lower()
            if needle in haystack:
                return item
        return None
    return items[0]


def _parse_iso_date(text: str):
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _ytdlp_resolve_sync(url: str) -> tuple[str, str]:
    """Resolve a BBC Sounds URL to a playable audio stream URL.

    Runs in a worker thread (see _ytdlp_resolve). yt-dlp's BBC Sounds extractor
    handles the MPEG-DASH manifest and gives us a direct media URL that mpv
    can stream. For a programme/brand page yt-dlp returns the latest episode.
    """
    import yt_dlp  # imported lazily so the dep is only required at runtime

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": "bestaudio/best",
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    # Brand/series pages return a playlist; take its newest entry.
    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise RuntimeError("yt-dlp returned an empty playlist")
        info = entries[0]
        # Re-extract the leaf entry — playlist entries are often stubs.
        leaf_url = info.get("webpage_url") or info.get("url")
        if leaf_url:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(leaf_url, download=False)
    stream_url = info.get("url")
    if not stream_url:
        raise RuntimeError("yt-dlp returned no playable URL")
    title = info.get("title") or info.get("webpage_url") or url
    return stream_url, title


async def _ytdlp_resolve(url: str) -> ResolvedEpisode:
    stream_url, title = await asyncio.to_thread(_ytdlp_resolve_sync, url)
    return ResolvedEpisode(url=stream_url, display=title)


async def _bbc_sounds_search(query: str) -> Optional[str]:
    """Search BBC Sounds and return the play URL of the top episode hit.

    Endpoint is public-but-undocumented; treated as best-effort. We pull
    episode PIDs out of urn fields like 'urn:bbc:radio:episode:<pid>' so
    yt-dlp gets a playable URL it can actually extract. Brand/series PIDs
    are skipped because yt-dlp's BBC extractor needs an episode pid.
    """
    url = "https://rms.api.bbc.co.uk/v2/experience/inline/search"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
            r = await client.get(url, params={"q": query, "stations": "all"})
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning(f"BBC Sounds search failed: {e}")
        return None
    pid = _first_episode_pid(data)
    if not pid:
        return None
    return f"https://www.bbc.co.uk/sounds/play/{pid}"


_EPISODE_URN_RE = re.compile(r"^urn:bbc:radio:episode:([a-z][0-9a-z]{7})$")


def _first_episode_pid(node) -> Optional[str]:
    """Walk the JSON tree looking for the first urn:bbc:radio:episode:<pid>."""
    if isinstance(node, dict):
        urn = node.get("urn")
        if isinstance(urn, str):
            m = _EPISODE_URN_RE.match(urn)
            if m:
                return m.group(1)
        for value in node.values():
            found = _first_episode_pid(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _first_episode_pid(item)
            if found:
                return found
    return None


async def resolve_show_episode(
    show: str,
    date: Optional[str] = None,
    query: Optional[str] = None,
) -> Optional[ResolvedEpisode]:
    """Resolve a user-spoken show reference to a playable episode.

    Strategy:
      1. Look up `show` in the curated list. If found with rss_url, fetch
         and filter by date/query. RSS is fastest when it works.
      2. Anything else (curated but RSS unavailable, or not curated at all)
         falls through to a BBC Sounds search using the show name (plus
         query if given), then yt-dlp on the returned play URL.

    BBC's podcast feed PIDs occasionally rotate, so the search-then-yt-dlp
    path is the safety net that keeps things working when curated entries
    go stale.
    """
    show = (show or "").strip()
    if not show:
        return None

    curated = lookup_show(show) or match_show(show)

    if curated is not None and curated.rss_url:
        try:
            items = await _fetch_rss_items(curated.rss_url)
        except Exception as e:
            logger.warning(f"RSS fetch failed for {curated.display}: {e}")
            items = []
        picked = _pick_item(items, date=date, query=query)
        if picked and picked.enclosure_url:
            suffix = _pretty_pub_date(picked.pub_date)
            display = f"{curated.display}, {suffix}" if suffix else curated.display
            return ResolvedEpisode(url=picked.enclosure_url, display=display)
        logger.info(
            f"RSS empty/no-match for {curated.display}; falling back to search."
        )

    # Fallback: BBC Sounds search + yt-dlp. Use the curated display name when
    # we matched one (more accurate than the raw user text); otherwise the
    # user's phrasing. Layer the optional query on for keyword episode picks.
    search_terms = [curated.display if curated is not None else show]
    if query:
        search_terms.append(query)
    play_url = await _bbc_sounds_search(" ".join(search_terms))
    if not play_url:
        return None
    try:
        return await _ytdlp_resolve(play_url)
    except Exception as e:
        logger.warning(f"yt-dlp resolve failed for {play_url}: {e}")
        return None
