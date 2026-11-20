"""Command line for AI Desk: run the services or try each agent from the terminal."""

from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aidesk", description="AI Desk on a KodeKloud AI Engineering KodeBox")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web API and UI on 127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    m = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    m.add_argument("--http", action="store_true", help="serve Streamable HTTP on 127.0.0.1 instead of stdio")
    m.add_argument("--port", type=int, default=8001)
    a = sub.add_parser("a2a", help="run the Deep Research A2A agent on 127.0.0.1")
    a.add_argument("--port", type=int, default=8002)
    i = sub.add_parser("ingest", help="(re)build the knowledge-base index")
    i.add_argument("--embed", choices=["auto", "on", "off"], default=None)
    q = sub.add_parser("ask", help="ask the support bot")
    q.add_argument("message")
    q.add_argument("--thread")
    w = sub.add_parser("web", help="ask the web agent")
    w.add_argument("question")
    r = sub.add_parser("research", help="run Deep Research with the plan auto-approved")
    r.add_argument("question")
    sub.add_parser("usage", help="token usage today")
    sub.add_parser("cache-clean", help="drop cached replies that were cut off or empty")
    t = sub.add_parser("traces", help="recent trace spans")
    t.add_argument("--last", type=int, default=20)
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("aidesk.api.app:app", host="127.0.0.1", port=args.port, log_level="info", proxy_headers=True,
                    forwarded_allow_ips="127.0.0.1")
    elif args.cmd == "mcp":
        from aidesk.mcp_server import main as mcp_main

        mcp_main(http=args.http, port=args.port)
    elif args.cmd == "a2a":
        from aidesk.a2a_server import main as a2a_main
        from aidesk.observability import setup_tracing

        setup_tracing("aidesk-a2a")
        a2a_main(port=args.port)
    elif args.cmd == "ingest":
        from aidesk.rag.store import ingest

        _print(ingest(embed=args.embed))
    elif args.cmd == "ask":
        from aidesk.agents.support import ask

        out = ask(args.message, args.thread)
        print(out["answer"])
        print(f"\n(thread {out['thread_id']}; citations: {[c['doc_id'] for c in out['citations']]})", file=sys.stderr)
    elif args.cmd == "web":
        from aidesk.agents.web import ask as web_ask

        out = web_ask(args.question)
        print(out.get("answer") or out.get("message"))
        for s in out.get("sources", []):
            print(f"[{s['n']}] {s['title']} - {s['url']}")
    elif args.cmd == "research":
        from aidesk.agents.research import run

        out = run(args.question)
        print(out.get("report") or out.get("message"))
    elif args.cmd == "usage":
        from aidesk.llm import get_gateway

        _print(get_gateway().ledger.summary())
    elif args.cmd == "cache-clean":
        from aidesk.config import get_settings
        from aidesk.llm.cache import purge_bad_entries

        print(f"removed {purge_bad_entries(get_settings().db_path)} cached replies")
    elif args.cmd == "traces":
        from aidesk.observability import recent_traces

        for span in recent_traces(args.last):
            attrs = span.get("attributes", {})
            print(f"{span['duration_ms']:>9} ms  {span['name']:<28} {attrs.get('llm.model', '')} "
                  f"{attrs.get('llm.prompt_tokens', '')}/{attrs.get('llm.completion_tokens', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
