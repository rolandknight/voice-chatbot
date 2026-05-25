---
name: ask_claude
description: >
  Route the current conversation to Claude (the more capable cloud model)
  instead of the local Ollama model. Use this when the user explicitly asks
  to talk to Claude, asks Claude a question, or wants deeper reasoning,
  research, web search, or longer-form answers than the local model gives.
  The switch lasts for the active wake session only and resets automatically
  when the assistant goes back to sleep — it is NOT a permanent setting.
category: backends
requires: [backend_state]
parameters: {}
triggers:
  - ask claude
  - switch to claude
  - talk to claude
  - claude please
  - claude can you
  - hey claude
  - use claude
---

# ask_claude

Flips `backend_state["backend"]` to `claude` so the ParallelPipeline gate in
app.py routes the next LLMContextFrame into the Anthropic branch. The wake
timeout handler reverts it back to `ollama` on sleep, keeping the switch
scoped to the active session.

The handler returns a one-word confirmation so the LLM (which is still Ollama
at the moment the tool call is dispatched) doesn't try to answer the user's
real question — Claude will pick up the next user turn and answer it directly.
