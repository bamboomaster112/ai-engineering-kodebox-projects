"""End-to-end checks against the running stack on the KodeBox (after start-services.sh and publish-app.sh).

    uv run pytest -m e2e tests/e2e -s

Goes through nginx at AIDESK_E2E_URL (default http://127.0.0.1/app/) with the basic-auth login from
~/aidesk-credentials.txt, and talks to the MCP and A2A services on 127.0.0.1 directly.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from mcp import Client

from aidesk.a2a_client import ask_async, fetch_card

pytestmark = pytest.mark.e2e
BASE = os.environ.get("AIDESK_E2E_URL", "http://127.0.0.1/app/")
# Same questions as tests/live, so the response cache answers most model calls (saves tokens).
WEB_Q = "What is the Model Context Protocol and who created it?"
RESEARCH_Q = "What are the main techniques for evaluating retrieval-augmented generation systems?"


def creds() -> tuple[str, str]:
    text = (Path.home() / "aidesk-credentials.txt").read_text()
    fields = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    return fields["user"], fields["password"]


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=BASE, auth=creds(), timeout=600) as c:
        yield c


def sse(resp: httpx.Response) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_login_is_required():
    assert httpx.get(BASE, timeout=10).status_code == 401


def test_ui_and_health(http):
    assert "AI Desk" in http.get("").text
    h = http.get("api/health").json()
    print("\nHEALTH", h)
    assert h["provider"] == "kodekey" and h["kb"]["documents"] >= 13


def test_support_through_nginx(http):
    out = http.post("api/support/chat", json={"message": "My job failed with exit code 137. What does that mean and how do I fix it?"}).json()
    print("\nSUPPORT", out["answer"], out["citations"])
    assert out["citations"] and out["citations"][0]["doc_id"] == "troubleshooting-builds"


def test_web_agent_streams_through_nginx(http):
    events = sse(http.post("api/web/ask", json={"question": WEB_Q}))
    assert any(e["event"] == "step" for e in events)
    done = events[-1]
    print("\nWEB", done.get("answer"), done.get("sources"))
    assert done["event"] == "done" and done["sources"]


def test_research_with_plan_approval_through_nginx(http):
    start = sse(http.post("api/research/start", json={"question": "How do hybrid search systems combine BM25 and vector search?"}))
    assert start[-1]["event"] == "awaiting_approval", start
    plan = start[-1]["plan"][:2]
    done = sse(http.post("api/research/resume", json={"run_id": start[0]["run_id"], "plan": plan}))[-1]
    print("\nRESEARCH\n", done.get("report", "")[:2000])
    assert done["event"] == "done" and done["citations_ok"]


def test_mcp_server_over_streamable_http():
    async def go():
        async with Client("http://127.0.0.1:8001/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_kb", {"query": "rotate a personal access token", "k": 2})
        return {t.name for t in tools.tools}, result

    names, result = asyncio.run(go())
    assert "search_kb" in names and "create_ticket" in names
    payload = result.structured_content or json.loads(result.content[0].text)
    hits = payload.get("result", payload) if isinstance(payload, dict) else payload
    assert hits[0]["doc_id"] == "api-tokens"


def test_a2a_agent_card_and_task():
    card = asyncio.run(fetch_card("http://127.0.0.1:8002"))
    assert card.name == "AI Desk Deep Research"
    out = asyncio.run(ask_async("http://127.0.0.1:8002", RESEARCH_Q))
    print("\nA2A", out["state"], out["progress"][:6])
    assert out["state"] == "TASK_STATE_COMPLETED" and "## Sources" in out["report"]
