---
name: web_search
description: >
  Search the web for current information. Use this for news, recent events,
  sports scores, prices, or any question whose answer may have changed
  recently or is outside the model's training data. Do NOT use it for things
  you already know or for simple chit-chat.
category: core
always_available: true
parameters:
  query:
    type: string
    required: true
    description: Natural-language search query.
triggers:
  - search
  - search for
  - look up
  - google
  - search the web
  - what's the news
  - whats the news
  - news
  - latest
  - current
  - score
  - scores
  - price
  - prices
---

# web_search

Provider chosen by BABEL_SEARCH_PROVIDER env (`duckduckgo` default, `brave`,
or `tavily` — Brave/Tavily need API keys).
