---
name: play_bbc_show
description: >
  Play a specific BBC Sounds show, podcast, or programme episode on demand.
  Use for 'play the Archers omnibus', 'play yesterday's Today programme',
  'play the In Our Time about Spinoza'. For LIVE radio stations (Radio 1,
  Radio 4, 6 Music, World Service, etc.) use play_bbc_radio instead — this
  tool is only for on-demand programmes and podcast episodes.
category: shows
enabled_when: BABEL_SHOWS_ENABLED
requires: [radio_player]
parameters:
  show:
    type: string
    required: true
    description: >
      The show name the user named, verbatim. Examples: 'Archers omnibus',
      'In Our Time', 'Desert Island Discs', 'Today programme', 'Just A Minute'.
  date:
    type: string
    description: >
      Optional ISO date YYYY-MM-DD if the user asked for a specific day's
      episode. Resolve relative phrases like 'yesterday' or 'last Sunday'
      against today's date before passing. Omit for the latest episode.
  query:
    type: string
    description: >
      Optional keyword for finding an episode by topic, guest, or title
      (e.g. 'Spinoza', 'climate', 'Lynda Snell'). Omit for the latest episode.
triggers:
  - archers
  - in our time
  - desert island discs
  - today programme
  - just a minute
  - friday night comedy
  - omnibus
  - podcast
  - episode
  - bbc sounds
  - play the show
  - the programme
---

# play_bbc_show

Resolves the requested show via BBC Sounds, builds an ad-hoc Station, hands
it to RadioPlayer (same mpv + ducking pipeline as live radio).
