---
name: generate_sound_effect
description: >
  Generate and play a short (~5s) sound effect on the connected USB speaker
  using a foley synthesis model. Use whenever the user asks for a sound
  effect, foley, ambience, or noise — 'make a thunder sound', 'play a car
  engine', 'do a dog bark', 'sound effect of rain'. The description should
  be a vivid concrete noun phrase (what the listener would hear), not a
  request sentence.
category: sfx
requires: [sfx_backends, sfx_tracker]
parameters:
  description:
    type: string
    required: true
    description: >
      Concrete description of the sound, e.g. 'sportscar engine revving and
      driving away quickly', 'heavy rain on a tin roof', 'a dog barking
      twice'. Avoid request phrasing.
triggers:
  - sound effect
  - sound of
  - make a sound
  - play a sound
  - thunder
  - rain
  - dog bark
  - cat meow
  - explosion
  - foley
  - ambience
  - noise
  - fart
  - burp
  - laugh
  - cough
  - sneeze
  - snore
  - yawn
---

# generate_sound_effect

Generation runs in a background task; the ack TTS plays first, then mpv is
gated on the bot finishing its current speaking cycle (via sfx_tracker) so
the SFX doesn't stack on top of the ack.
