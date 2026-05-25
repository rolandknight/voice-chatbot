---
name: set_timer
description: >
  Start a countdown timer. When it expires the assistant will speak the alert
  out loud. Use for any request like 'set a timer', 'remind me in N minutes',
  'wake me in N minutes'.
category: core
always_available: true
parameters:
  minutes:
    type: number
    required: true
    description: Duration of the timer in minutes. Can be fractional.
  label:
    type: string
    description: Optional short label spoken when the timer fires (e.g. 'tea', 'standup').
triggers:
  - timer
  - remind me
  - remind me in
  - wake me
  - wake me in
  - alarm in
  - set a timer
  - countdown
---

# set_timer

Sets a one-shot async timer. The timer fires by pushing a TTSSpeakFrame so the
spoken alert plays even if the user is mid-conversation.
