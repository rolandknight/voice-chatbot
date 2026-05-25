---
name: pause_spotify
description: >
  Pause Spotify playback. Use when the user says 'pause', 'pause the music',
  'pause Spotify', 'hold on', 'wait' while music is playing. For stopping
  BBC radio use stop_bbc_radio.
category: spotify
enabled_when: skills.spotify.enabled
requires: [spotify_player]
parameters: {}
triggers:
  - pause
  - pause the music
  - hold on
  - wait
  - hang on
---

# pause_spotify

Pauses via the Spotify Web API.
