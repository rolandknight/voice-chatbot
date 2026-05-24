---
name: stop_bbc_radio
description: >
  Stop whatever BBC audio is currently playing — live radio or an on-demand
  show/podcast. Use this when the user says 'stop', 'stop the radio', 'stop
  the show', 'turn it off', 'silence', 'kill the radio', or anything similar
  while audio is playing.
category: radio
enabled_when: BABEL_RADIO_ENABLED
requires: [radio_player]
parameters: {}
triggers:
  - stop
  - stop the radio
  - stop the show
  - turn it off
  - silence
  - kill the radio
  - shut up
  - turn off the radio
---

# stop_bbc_radio

Generic "stop whatever's playing" — also stops Spotify if a track happens to
be playing, since the local model picks this tool for the bare word "stop"
regardless of which player owns the current audio.
