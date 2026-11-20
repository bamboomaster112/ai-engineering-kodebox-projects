# Lab 03: Ask-the-Web agent (ByteByteAI project 3)

**Goal:** a Perplexity-style agent that decides what to search, reads pages and answers with citations,
without letting web content take control.

## How it works (`src/aidesk/agents/web.py`)
A LangGraph loop: **agent** (main model with two tools) → **tools** → back to the agent, at most six tool
rounds, then **finalize**.
- `web_search` uses DuckDuckGo through the `ddgs` package (no key). Set `AIDESK_SEARCH=tavily` and
  `TAVILY_API_KEY` for Tavily, or `fixture` for offline tests.
- `fetch_url` (`src/aidesk/web/fetch.py`) only accepts URLs that a search in the same conversation
  returned, only public http(s) addresses on ports 80 and 443, and re-checks every redirect. Localhost,
  private ranges, link-local and cloud metadata addresses are refused (SSRF protection). Pages are capped
  at 2 MB and their readable text is extracted with trafilatura.
- Sentences in page text that try to instruct the assistant are removed before the model sees them.
- **finalize** keeps only citations that match fetched pages and builds the source list. An answer that
  tries to smuggle data out through a link is replaced.
- Progress streams to the browser as server-sent events.

## Try it
```bash
uv run aidesk web "What is the Model Context Protocol and who created it?"
```
Or the **Ask the Web** tab: each tool call appears as it happens.

## Verify
Offline: `tests/unit/test_web.py` (SSRF cases, fetch allow-list, sanitising, the full loop on fixtures).
Live: `tests/live/test_live_agents.py::test_ask_the_web_answers_with_fetched_citations`.

## Exercises
1. Add a `max_age` setting so questions about news always re-fetch pages.
2. Add a third tool, `search_kb`, so the agent can combine the help center with the web.
