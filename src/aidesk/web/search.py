"""Web search with three interchangeable backends: DuckDuckGo (ddgs, no key), Tavily (key), fixture (tests)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

import httpx

from aidesk.config import REPO_ROOT, get_settings
from aidesk.llm.cache import ResponseCache, cache_key

FIXTURES = REPO_ROOT / "data" / "fixtures" / "web.json"
SEARCH_TTL = 6 * 3600


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict:
        return asdict(self)


class SearchError(RuntimeError):
    pass


def _ddgs(query: str, max_results: int) -> list[SearchResult]:
    from ddgs import DDGS

    try:
        rows = DDGS(timeout=15).text(query, max_results=max_results) or []
    except Exception as err:  # noqa: BLE001 - ddgs raises several exception types
        raise SearchError(f"DuckDuckGo search failed: {err}") from err
    return [SearchResult(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows if r.get("href")]


def _tavily(query: str, max_results: int) -> list[SearchResult]:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise SearchError("AIDESK_SEARCH=tavily needs TAVILY_API_KEY")
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        resp.raise_for_status()
    except httpx.HTTPError as err:
        raise SearchError(f"Tavily search failed: {err}") from err
    return [SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", "")[:400])
            for r in resp.json().get("results", []) if r.get("url")]


def load_fixtures() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _fixture(query: str, max_results: int) -> list[SearchResult]:
    data = load_fixtures()
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for i, r in enumerate(data["search"]):
        overlap = len(words & set(r.get("keywords", [])))
        scored.append((-overlap, i, r))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [SearchResult(r["title"], r["url"], r["snippet"]) for _, _, r in scored[:max_results]]


BACKENDS = {"ddgs": _ddgs, "tavily": _tavily, "fixture": _fixture}


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    query = " ".join((query or "").split())[:300]
    if not query:
        return []
    settings = get_settings()
    backend = settings.search_backend
    if backend not in BACKENDS:
        raise SearchError(f"unknown search backend {backend!r}; use ddgs, tavily or fixture")
    max_results = max(1, min(max_results, 8))
    cache = ResponseCache(settings.db_path, enabled=backend != "fixture")
    key = cache_key(f"search:{backend}", {"q": query.lower(), "n": max_results})
    hit = cache.get(key, max_age=SEARCH_TTL)
    if hit is not None:
        return [SearchResult(**r) for r in hit]
    results = BACKENDS[backend](query, max_results)
    cache.put(key, [r.as_dict() for r in results])
    return results
