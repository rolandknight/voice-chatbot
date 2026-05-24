---
name: play_spotify
description: >
  Play music from Spotify on the connected USB speaker. Use whenever the user
  asks to play a song, album, artist, or generally 'something by X' or 'a bit
  of Y' that isn't a BBC radio station or a named Spotify playlist. Examples:
  'play Radiohead', 'play Fake Plastic Trees', 'put on some jazz', 'play the
  new Taylor Swift album'. For playing a Spotify playlist by name (e.g.
  'play my Discover Weekly') use play_spotify_playlist instead.
category: spotify
enabled_when: BABEL_SPOTIFY_ENABLED
requires: [spotify_player]
parameters:
  query:
    type: string
    required: true
    description: >
      What the user wants to play, verbatim: a song title, artist, album, or
      descriptive phrase. e.g. 'Radiohead', 'Karma Police', 'OK Computer
      Radiohead', 'some Bach'.
  kind:
    type: string
    enum: [track, album, artist, any]
    description: >
      What to look for. 'track' for a specific song, 'album' for a named
      album, 'artist' to play an artist's top songs, 'any' (default) when
      unclear.
triggers:
  - play some
  - put on
  - play
  - song
  - album
  - artist
  - track
  - music
  - tune
  - jazz
  - rock
  - pop
  - classical
  - hip hop
  - rap
---

# play_spotify

Search-and-play via spotipy. Cross-stops radio if it's playing.
