---
name: get_weather
description: >
  Get current weather conditions for a location. Use for any question about
  temperature, rain, snow, wind, or forecasts. If the user doesn't name a
  location, pass an empty string and the assistant will use the default or
  auto-detected current location.
category: core
always_available: true
parameters:
  location:
    type: string
    required: true
    description: >
      City name, optionally with region/country (e.g. 'San Francisco' or
      'Paris, France'). Empty string to use the default or auto-detected
      location.
triggers:
  - weather
  - temperature
  - how hot
  - how cold
  - degrees
  - rain
  - snowing
  - snow
  - forecast
  - sunny
  - cloudy
  - windy
---

# get_weather

Resolves the location via the Open-Meteo geocoding API (or CoreLocation / IP
fallback when no location is given), then fetches current conditions from
Open-Meteo's forecast API.
