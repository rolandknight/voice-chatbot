---
name: skip_spotify
description: >
  Skip to the next or previous track on Spotify. Use for 'skip', 'next track',
  'next song', 'previous track', 'go back', 'last song again'.
category: spotify
enabled_when: skills.spotify.enabled
requires: [spotify_player]
parameters:
  direction:
    type: string
    enum: [next, previous]
    description: >
      'next' to advance to the next track (default), 'previous' to go back to
      the prior track.
triggers:
  - skip
  - next track
  - next song
  - previous track
  - previous song
  - go back
  - last song
  - back a song
---

# skip_spotify
