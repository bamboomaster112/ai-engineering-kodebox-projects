import json

from mcp import Client

from aidesk.mcp_server import mcp


def _payload(result):
    """Tool results come back as structured content (preferred) or JSON text."""
    if getattr(result, "structured_content", None):
        sc = result.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(result.content[0].text)


async def test_lists_the_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()
    names = {t.name for t in tools.tools}
    assert {"search_kb", "get_kb_article", "create_ticket", "list_tickets", "web_search", "fetch_url"} <= names


async def test_search_kb_over_mcp():
    async with Client(mcp) as client:
        result = await client.call_tool("search_kb", {"query": "exit code 137", "k": 2})
    hits = _payload(result)
    assert hits[0]["doc_id"] == "troubleshooting-builds"


async def test_ticket_round_trip_over_mcp():
    async with Client(mcp) as client:
        made = _payload(await client.call_tool("create_ticket", {"subject": "Runner stuck", "body": "mail me at bo@example.com"}))
        listed = _payload(await client.call_tool("list_tickets", {}))
    assert made["id"] == listed[0]["id"]


async def test_fetch_rule_holds_over_mcp():
    async with Client(mcp) as client:
        blocked = _payload(await client.call_tool("fetch_url", {"url": "http://169.254.169.254/latest/meta-data/"}))
    assert "error" in blocked
