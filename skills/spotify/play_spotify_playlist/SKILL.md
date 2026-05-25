---
name: play_spotify_playlist
description: >
  Play a Spotify playlist by name. Use when the user names a playlist
  explicitly: 'play my Discover Weekly', 'play Release Radar', 'put on my
  workout playlist', 'play the Jazz Classics playlist'. For arbitrary
  songs/artists/albums use play_spotify.
category: spotify
enabled_when: skills.spotify.enabled
requires: [spotify_player]
parameters:
  name:
    type: string
    required: true
    description: >
      The playlist name the user named, verbatim. e.g. 'Discover Weekly',
      'Release Radar', 'workout', 'Jazz Classics'.
triggers:
  - playlist
  - discover weekly
  - release radar
  - daily mix
  - my playlist
  - liked songs
---

# play_spotify_playlist

Searches the user's saved playlists by name and starts playback. Cross-stops
radio if it's playing.
