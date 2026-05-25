from __future__ import annotations

import httpx
from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from config import get as get_config
from skills._context import SkillContext

HTTP_TIMEOUT_SECS = 6.0


async def _search_duckduckgo(query: str) -> str:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        r = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": "babel-voice-bot/1.0"},
        )
        r.raise_for_status()
        data = r.json()
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        return abstract
    related = data.get("RelatedTopics") or []
    snippets: list[str] = []
    for item in related:
        if "Text" in item and item["Text"]:
            snippets.append(item["Text"])
        elif "Topics" in item:
            for sub in item["Topics"]:
                if "Text" in sub and sub["Text"]:
                    snippets.append(sub["Text"])
        if len(snippets) >= 3:
            break
    return " ".join(snippets[:3]).strip()


async def _search_brave(query: str) -> str:
    api_key = get_config().skills.web_search.brave_api_key.get_secret_value().strip()
    if not api_key:
        return "Brave search isn't configured. Add a BRAVE_API_KEY to .env to enable it."
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 3},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )
        r.raise_for_status()
        data = r.json()
    results = (data.get("web") or {}).get("results") or []
    snippets = []
    for item in results[:3]:
        desc = (item.get("description") or "").strip()
        if desc:
            snippets.append(desc)
    return " ".join(snippets).strip()


async def _search_tavily(query: str) -> str:
    api_key = get_config().skills.web_search.tavily_api_key.get_secret_value().strip()
    if not api_key:
        return "Tavily search isn't configured. Add a TAVILY_API_KEY to .env to enable it."
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 3,
                "include_answer": True,
                "search_depth": "basic",
            },
        )
        r.raise_for_status()
        data = r.json()
    answer = (data.get("answer") or "").strip()
    if answer:
        return answer
    snippets = []
    for item in data.get("results", [])[:3]:
        content = (item.get("content") or "").strip()
        if content:
            snippets.append(content)
    return " ".join(snippets).strip()


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    query = (params.arguments.get("query") or "").strip()
    if not query:
        await params.result_callback("I need a search query to look something up.")
        return

    provider = get_config().skills.web_search.provider
    try:
        if provider == "brave":
            text = await _search_brave(query)
        elif provider == "tavily":
            text = await _search_tavily(query)
        else:
            text = await _search_duckduckgo(query)
    except Exception as e:
        logger.warning(f"Web search ({provider}) failed: {e}")
        await params.result_callback("I couldn't reach the web right now.")
        return

    if not text:
        await params.result_callback(
            f"I searched for {query} but didn't get useful results."
        )
        return
    await params.result_callback(text)
