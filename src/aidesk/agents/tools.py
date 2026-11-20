"""Tools the agents can call, in OpenAI function-calling format, plus a safe executor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from aidesk.guardrails.sanitize import sanitize_untrusted
from aidesk.web.fetch import BlockedURL, FetchError, fetch_page
from aidesk.web.search import SearchError, web_search

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs and short snippets. Use focused queries.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch a page returned by web_search and get its readable text and a source_id to cite.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "A URL from the web_search results"}},
            "required": ["url"],
        },
    },
}

WEB_TOOLS = [WEB_SEARCH_TOOL, FETCH_URL_TOOL]


@dataclass
class SourceRegistry:
    """Tracks what the agent has seen. fetch_url only accepts URLs that a search returned."""

    seen_urls: dict[str, str] = field(default_factory=dict)  # url -> title from search
    sources: dict[int, dict] = field(default_factory=dict)  # source_id -> page info
    removed_sentences: int = 0

    def id_for(self, url: str) -> int | None:
        for sid, src in self.sources.items():
            if src["url"] == url:
                return sid
        return None

    def to_state(self) -> dict:
        return {"seen_urls": self.seen_urls, "sources": {str(k): v for k, v in self.sources.items()},
                "removed_sentences": self.removed_sentences}

    @classmethod
    def from_state(cls, data: dict | None) -> SourceRegistry:
        data = data or {}
        return cls(dict(data.get("seen_urls", {})), {int(k): v for k, v in data.get("sources", {}).items()},
                   int(data.get("removed_sentences", 0)))


def run_tool(name: str, args: dict, registry: SourceRegistry, *, page_chars: int = 2500) -> dict:
    """Execute one tool call and return a JSON-serialisable result (errors are results, not exceptions)."""
    if name == "web_search":
        try:
            results = web_search(str(args.get("query", "")), max_results=5)
        except SearchError as err:
            return {"error": str(err)}
        for r in results:
            registry.seen_urls.setdefault(r.url, r.title)
        if not results:
            return {"results": [], "note": "no results; try a shorter query with different keywords"}
        return {"results": [r.as_dict() for r in results]}
    if name == "fetch_url":
        url = str(args.get("url", "")).strip()
        if url not in registry.seen_urls:
            return {"error": "fetch_url only accepts URLs returned by web_search in this conversation"}
        existing = registry.id_for(url)
        if existing:
            src = registry.sources[existing]
            return {"source_id": existing, "url": url, "title": src["title"], "text": src["text"]}
        try:
            page = fetch_page(url)
        except (BlockedURL, FetchError) as err:
            return {"error": f"could not fetch {url}: {err}"}
        text, removed = sanitize_untrusted(page.text[:page_chars])
        registry.removed_sentences += removed
        sid = len(registry.sources) + 1
        registry.sources[sid] = {"url": url, "title": page.title or registry.seen_urls.get(url, url), "text": text}
        return {"source_id": sid, "url": url, "title": registry.sources[sid]["title"], "text": text}
    return {"error": f"unknown tool {name}"}


def tool_message(call_id: str, result: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)}
