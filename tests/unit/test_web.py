import pytest

from aidesk.agents import web
from aidesk.agents.tools import SourceRegistry, run_tool
from aidesk.web.fetch import BlockedURL, check_url
from aidesk.web.search import web_search


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8888/jupyter",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://user:pw@example.com/",
        "http://example.com:8080/",
    ],
)
def test_fetch_blocks_internal_and_odd_urls(url):
    with pytest.raises(BlockedURL):
        check_url(url)


def test_fixture_search_ranks_by_keywords():
    results = web_search("how does reciprocal rank fusion work")
    assert results[0].url.endswith("/rrf")


def test_fetch_tool_only_accepts_searched_urls():
    reg = SourceRegistry()
    out = run_tool("fetch_url", {"url": "https://notes.example.org/rrf"}, reg)
    assert "error" in out
    run_tool("web_search", {"query": "rank fusion"}, reg)
    out = run_tool("fetch_url", {"url": "https://notes.example.org/rrf"}, reg)
    assert out["source_id"] == 1 and "rank" in out["text"]


def test_fetched_pages_are_sanitised():
    reg = SourceRegistry()
    run_tool("web_search", {"query": "prompt injection risks"}, reg)
    out = run_tool("fetch_url", {"url": "https://notes.example.org/prompt-injection"}, reg)
    assert "Ignore all previous instructions" not in out["text"]
    assert reg.removed_sentences == 1


def test_ask_the_web_end_to_end_offline():
    out = web.ask("How does reciprocal rank fusion merge keyword and vector results?")
    assert out["event"] == "done"
    assert out["grounded"] and out["sources"][0]["url"].endswith("/rrf")
    assert "[1]" in out["answer"]
    steps = [t["step"] for t in out["trace"]]
    assert steps[:2] == ["agent", "tools"] and steps[-1] == "finalize"


def test_stream_emits_steps_then_done():
    events = list(web.stream("What is SQLite WAL mode?"))
    assert events[-1]["event"] == "done"
    assert any(e["event"] == "step" and e["node"] == "tools" for e in events)


def test_empty_question():
    assert list(web.stream("   "))[0]["event"] == "error"
