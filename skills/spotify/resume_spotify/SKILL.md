---
name: resume_spotify
description: >
  Resume previously-paused Spotify playback. Use when the user says 'resume',
  'unpause', 'keep playing', 'go on', 'continue the music'.
category: spotify
enabled_when: skills.spotify.enabled
requires: [spotify_player]
parameters: {}
triggers:
  - resume
  - unpause
  - keep playing
  - go on
  - continue
  - continue the music
---

# resume_spotify
