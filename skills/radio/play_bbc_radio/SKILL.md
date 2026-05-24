---
name: play_bbc_radio
description: >
  Start streaming a live BBC radio station on the connected USB speaker. Use
  this whenever the user asks to play, put on, tune to, or switch to a BBC
  station (Radio 1, 1Xtra, Radio 2, Radio 3, Radio 4, Radio 4 Extra, Radio 5
  Live, Radio 5 Sports Extra, 6 Music, Asian Network, or the World Service).
category: radio
enabled_when: BABEL_RADIO_ENABLED
requires: [radio_player]
parameters:
  station:
    type: string
    required: true
    description: >
      The station the user named, verbatim. Examples: 'Radio 1', 'BBC Radio 4',
      'Radio 4 Extra', 'World Service', '6 Music', '5 Live', '1Xtra'.
triggers:
  - bbc
  - radio 1
  - radio 2
  - radio 3
  - radio 4
  - radio 5
  - 1xtra
  - 6 music
  - six music
  - world service
  - asian network
  - radio 4 extra
  - tune to
  - put on the radio
  - play the radio
  - play radio
---

# play_bbc_radio

Resolves the station via alias matching, cross-stops Spotify if it's playing,
then hands the station to RadioPlayer (mpv + ducking + USB output).
