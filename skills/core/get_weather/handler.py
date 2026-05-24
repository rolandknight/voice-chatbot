from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass

import httpx
from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext

HTTP_TIMEOUT_SECS = 6.0
COREL_CLI_TIMEOUT_SECS = 5.0
IPGEO_TIMEOUT_SECS = 3.0

# WMO weather interpretation codes -> short spoken phrases.
WMO_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with rime",
    51: "drizzling lightly", 53: "drizzling", 55: "drizzling heavily",
    61: "raining lightly", 63: "raining", 65: "raining heavily",
    66: "freezing rain", 67: "freezing rain",
    71: "snowing lightly", 73: "snowing", 75: "snowing heavily",
    77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


@dataclass(frozen=True)
class _ResolvedLocation:
    lat: float
    lon: float
    place_name: str
    country_code: str | None = None


_LOCATION_CACHE: _ResolvedLocation | None = None


async def _resolve_corelocation() -> _ResolvedLocation | None:
    if sys.platform != "darwin" or shutil.which("CoreLocationCLI") is None:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "CoreLocationCLI", "-once", "-format",
            "%latitude\t%longitude\t%ISOcountryCode",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=COREL_CLI_TIMEOUT_SECS
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.debug("CoreLocationCLI timed out")
            return None
        if proc.returncode != 0:
            return None
        parts = stdout.decode().strip().split("\t")
        if len(parts) < 2:
            return None
        lat = float(parts[0])
        lon = float(parts[1])
        cc_raw = parts[2].strip().upper() if len(parts) >= 3 else ""
        country_code = cc_raw if len(cc_raw) == 2 and cc_raw.isalpha() else None
    except Exception as e:
        logger.debug(f"CoreLocation lookup failed: {e}")
        return None
    return _ResolvedLocation(
        lat=lat, lon=lon, place_name="your location", country_code=country_code
    )


async def _resolve_ip_geolocation() -> _ResolvedLocation | None:
    try:
        async with httpx.AsyncClient(timeout=IPGEO_TIMEOUT_SECS) as client:
            resp = await client.get("https://ipwho.is/")
            resp.raise_for_status()
            data = resp.json()
        if not data.get("success", False):
            logger.debug(f"IP geolocation returned error: {data.get('message')}")
            return None
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except Exception as e:
        logger.debug(f"IP geolocation failed: {e}")
        return None
    place_name = data.get("city") or data.get("region") or "your location"
    cc_raw = (data.get("country_code") or "").strip().upper()
    country_code = cc_raw if len(cc_raw) == 2 and cc_raw.isalpha() else None
    return _ResolvedLocation(
        lat=lat, lon=lon, place_name=place_name, country_code=country_code
    )


async def _resolve_current_location() -> _ResolvedLocation | None:
    global _LOCATION_CACHE
    if _LOCATION_CACHE is not None:
        return _LOCATION_CACHE
    resolved = await _resolve_corelocation()
    if resolved is None:
        resolved = await _resolve_ip_geolocation()
    if resolved is not None:
        _LOCATION_CACHE = resolved
    return resolved


def _use_imperial_units(country_code: str | None) -> bool:
    return country_code is None or country_code == "US"


async def _fetch_current_weather(
    client: httpx.AsyncClient, lat: float, lon: float, country_code: str | None
) -> dict:
    imperial = _use_imperial_units(country_code)
    forecast = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit" if imperial else "celsius",
            "wind_speed_unit": "mph" if imperial else "kmh",
        },
    )
    forecast.raise_for_status()
    return forecast.json().get("current", {})


def _format_weather_reply(
    current: dict, place_name: str, country_code: str | None
) -> str:
    temp = current.get("temperature_2m")
    code = current.get("weather_code")
    wind = current.get("wind_speed_10m")
    description = WMO_CODES.get(int(code) if code is not None else -1, "")
    imperial = _use_imperial_units(country_code)
    wind_unit = "miles per hour" if imperial else "kilometers per hour"
    wind_threshold = 15 if imperial else 24
    parts = []
    if description:
        parts.append(description.capitalize())
    if temp is not None:
        parts.append(f"{int(round(temp))} degrees")
    if wind is not None and wind >= wind_threshold:
        parts.append(f"wind around {int(round(wind))} {wind_unit}")
    body = ", ".join(parts) if parts else "conditions unavailable"
    return f"{body} in {place_name} right now."


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    location = (params.arguments.get("location") or "").strip()
    if not location:
        location = os.getenv("BABEL_DEFAULT_LOCATION", "").strip()

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
            if location:
                geo = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": location, "count": 1,
                        "language": "en", "format": "json",
                    },
                )
                geo.raise_for_status()
                results = geo.json().get("results") or []
                if not results:
                    await params.result_callback(
                        f"I couldn't find a place called {location}."
                    )
                    return
                place = results[0]
                lat = place["latitude"]
                lon = place["longitude"]
                place_name = place.get("name", location)
                cc_raw = (place.get("country_code") or "").strip().upper()
                country_code = (
                    cc_raw if len(cc_raw) == 2 and cc_raw.isalpha() else None
                )
            else:
                resolved = await _resolve_current_location()
                if resolved is None:
                    await params.result_callback(
                        "I couldn't figure out where you are. "
                        "Try asking again with a city name."
                    )
                    return
                lat = resolved.lat
                lon = resolved.lon
                place_name = resolved.place_name
                country_code = resolved.country_code

            current = await _fetch_current_weather(client, lat, lon, country_code)
    except Exception as e:
        logger.warning(f"Weather lookup failed: {e}")
        await params.result_callback("I couldn't reach the weather service right now.")
        return

    await params.result_callback(
        _format_weather_reply(current, place_name, country_code)
    )
