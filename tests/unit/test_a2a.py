import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from aidesk.a2a_client import ask_async
from aidesk.a2a_server import build_app


@pytest.fixture
def a2a_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(build_app(url), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield url
    server.should_exit = True
    thread.join(timeout=5)


def test_agent_card_is_published(a2a_url):
    card = httpx.get(f"{a2a_url}/.well-known/agent-card.json").json()
    assert card["name"] == "AI Desk Deep Research"
    iface = card["supportedInterfaces"][0]
    assert iface["protocolBinding"] == "JSONRPC" and iface["url"] == a2a_url
    assert card["skills"][0]["id"] == "deep_research"


@pytest.mark.parametrize("streaming", [True, False])
def test_research_over_a2a(a2a_url, streaming):
    result = asyncio.run(ask_async(a2a_url, "reciprocal rank fusion hybrid search", streaming=streaming))
    assert "COMPLETED" in result["state"], result
    assert "## Sources" in result["report"]
    if streaming:
        assert any(p.startswith("Plan:") for p in result["progress"])


def test_support_bot_delegates_research_over_a2a(a2a_url, monkeypatch):
    from aidesk.agents import support

    monkeypatch.setenv("AIDESK_A2A_URL", a2a_url)
    support.reset_graph()
    out = support.ask("Can you research and compare hybrid search with rank fusion for me?")
    assert out["intent"] == "research"
    assert out["trace"][-1]["step"] == "delegate" and out["trace"][-1]["ok"], out["trace"]
    assert "## Sources" in out["answer"]
