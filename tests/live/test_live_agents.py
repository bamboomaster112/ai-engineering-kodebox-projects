"""Live checks against KodeKey and the real web (run on the KodeBox):  uv run pytest -m live tests/live -s"""

from __future__ import annotations

import pytest

from aidesk.agents import research, web
from aidesk.playground import available_models, compare

pytestmark = pytest.mark.live


def test_playground_compares_three_models():
    out = compare("Name one benefit of continuous integration in one sentence.", available_models()[:3],
                  temperature=0.3, max_tokens=80)
    print("\nPLAYGROUND", [(r["model"], r["latency_ms"], r["error"]) for r in out["results"]])
    assert out["summary"]["errors"] == 0
    assert all(r["text"].strip() for r in out["results"])


def test_ask_the_web_answers_with_fetched_citations():
    out = web.ask("What is the Model Context Protocol and who created it?")
    print("\nWEB ANSWER", out.get("answer"), out.get("sources"))
    assert out["event"] == "done", out
    assert out["grounded"] and out["sources"]
    assert all(s["url"].startswith("http") for s in out["sources"])
    calls = [t for t in out["trace"] if t["step"] == "tools"]
    assert any(c["name"] == "fetch_url" and c.get("source_id") for t in calls for c in t["calls"])


def test_deep_research_produces_a_cited_report():
    research.reset_graph()
    out = research.run("What are the main techniques for evaluating retrieval-augmented generation systems?")
    print("\nREPORT\n", out.get("report", "")[:3000])
    assert out["event"] == "done", out
    assert out["citations_ok"]
    assert len(out["sources"]) >= 2
    steps = [t["step"] for t in out["trace"]]
    assert steps.count("research") >= 2 and "write" in steps
