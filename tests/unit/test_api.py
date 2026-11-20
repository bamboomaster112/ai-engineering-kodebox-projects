import json

import pytest
from fastapi.testclient import TestClient

from aidesk.agents import research, support
from aidesk.api import app as app_module


@pytest.fixture
def client():
    support.reset_graph()
    research.reset_graph()
    app_module._hits.clear()
    with TestClient(app_module.app) as c:
        yield c


def sse_events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_ui_and_health(client):
    page = client.get("/")
    assert page.status_code == 200 and "AI Desk" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    h = client.get("/api/health").json()
    assert h["provider"] == "fake" and h["kb"]["documents"] >= 13


def test_support_chat_and_tickets(client):
    out = client.post("/api/support/chat", json={"message": "How do I revoke an API token?"}).json()
    assert out["citations"] and not out["blocked"]
    t = client.post("/api/support/chat", json={"message": "I need a human to open a ticket", "thread_id": out["thread_id"]}).json()
    assert t["ticket"]
    assert client.get("/api/tickets").json()[0]["id"] == t["ticket"]["id"]


def test_input_validation(client):
    assert client.post("/api/support/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/support/chat", json={"message": "hi", "thread_id": "../etc"}).status_code == 422
    assert client.post("/api/research/resume", json={"run_id": "nope"}).status_code == 422


def test_playground(client):
    models = client.get("/api/playground/models").json()
    out = client.post("/api/playground/compare", json={"prompt": "hi", "models": models[:2]}).json()
    assert len(out["results"]) == 2
    bad = client.post("/api/playground/compare", json={"prompt": "hi", "models": ["not-allowed"]})
    assert bad.status_code == 400


def test_web_stream(client):
    resp = client.post("/api/web/ask", json={"question": "What is SQLite WAL mode?"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp)
    assert events[-1]["event"] == "done" and events[-1]["sources"]


def test_research_approval_flow(client):
    events = sse_events(client.post("/api/research/start", json={"question": "reciprocal rank fusion"}))
    assert events[-1]["event"] == "awaiting_approval"
    run_id = events[0]["run_id"]
    done = sse_events(client.post("/api/research/resume", json={"run_id": run_id, "plan": events[-1]["plan"]}))[-1]
    assert done["event"] == "done" and done["citations_ok"]


def test_rate_limit(client, monkeypatch):
    monkeypatch.setitem(app_module.LIMITS, "support", (2, 3600))
    for _ in range(2):
        assert client.post("/api/support/chat", json={"message": "What sizes of runners exist?"}).status_code == 200
    assert client.post("/api/support/chat", json={"message": "What sizes of runners exist?"}).status_code == 429


def test_usage(client):
    client.post("/api/support/chat", json={"message": "What sizes of runners exist?"})
    u = client.get("/api/usage").json()
    assert u["by_model_and_purpose"]
