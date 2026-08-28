---
name: stop_spotify
description: >
  Stop Spotify playback completely (not just pause). Use when the user says
  'stop Spotify', 'stop the music', 'kill Spotify', 'turn the music off'.
category: spotify
enabled_when: skills.spotify.enabled
requires: [spotify_player]
parameters: {}
triggers:
  - stop spotify
  - stop the music
  - kill spotify
  - turn the music off
  - turn off the music
---

# stop_spotify

Generic "stop whatever's playing" — also stops radio if it's playing, mirroring
stop_bbc_radio so the user always hears silence regardless of which stop_* the
LLM picked.
