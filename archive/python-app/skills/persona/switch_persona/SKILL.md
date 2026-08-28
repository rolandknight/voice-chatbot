---
name: switch_persona
description: >
  Change the voice the assistant is speaking with by switching to a different
  declared persona. Use when the user asks to change voice, talk like a
  different character, or names a persona indirectly (e.g. 'use the butler
  voice', 'be the narrator now'). The persona argument must be one of the
  declared persona ids the user is told about; pass it lowercased.
category: persona
requires: [persona_switch_available]
parameters:
  persona:
    type: string
    required: true
    description: >
      The persona id to switch to. Must exactly match a declared persona id
      (e.g. 'babel', 'marvin').
triggers:
  - switch to
  - be the
  - talk like
  - use the
  - voice
  - persona
  - sound like
---

# switch_persona

Validates the requested id against the live PersonaConfig and applies the
swap via apply_skill_persona_switch — the handler never accepts an unknown
name silently, so the LLM can't make one up.
