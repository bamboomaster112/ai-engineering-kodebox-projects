# Lab 05: MCP tools and A2A agents

**Goal:** expose AI Desk's tools through the Model Context Protocol so any MCP client can use them, and
expose Deep Research as an agent that other agents can call through Agent2Agent (A2A).

## MCP server (`src/aidesk/mcp_server.py`)
Built with the MCP Python SDK 2.x (`MCPServer`, the class formerly called `FastMCP`), speaking the
2026-07-28 specification and serving older clients too. Tools: `search_kb`, `get_kb_article`,
`create_ticket`, `list_tickets`, `web_search`, `fetch_url` (with the same SSRF rules as the agents).

```bash
uv run aidesk mcp                       # stdio, for Claude Code and other local hosts
uv run aidesk mcp --http --port 8001    # stateless Streamable HTTP at http://127.0.0.1:8001/mcp
```
Use it from Claude Code on the box:
```bash
claude mcp add aidesk -- uv run --directory ~/ai-engineering-kodebox-projects aidesk mcp
claude -p "Use the aidesk search_kb tool to find how to rotate an API token, then summarise it."
```
Browse it with the MCP Inspector (pre-installed): point it at `http://127.0.0.1:8001/mcp`.

## A2A agent (`src/aidesk/a2a_server.py`, `src/aidesk/a2a_client.py`)
Built with `a2a-sdk` 1.x (A2A protocol 1.0). The agent card is at
`http://127.0.0.1:8002/.well-known/agent-card.json` and lists one skill, `deep_research`, over JSON-RPC.
A request runs the research graph with the plan auto-approved; each step streams back as a task status
update and the report arrives as a `text/markdown` artifact.

The support bot is an A2A client: when its classifier marks a message as a research request and
`AIDESK_A2A_URL` is set (the start script sets it), it hands the question to the research agent and
returns the report.

```bash
curl -s http://127.0.0.1:8002/.well-known/agent-card.json | python3 -m json.tool
uv run aidesk ask "Can you research how hybrid search compares with plain vector search?"
```

## Security choices
Both services listen on 127.0.0.1 only and are not published through nginx. Publishing them would need
authentication first (the MCP SDK supports OAuth; A2A cards can declare security schemes).

## Verify
Offline: `tests/unit/test_mcp.py` (in-process MCP client), `tests/unit/test_a2a.py` (real HTTP server,
streaming and non-streaming, plus the support-bot hand-off). End to end: `tests/e2e/test_stack.py`.
