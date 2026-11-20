"""The Deep Research agent as an A2A server (a2a-sdk 1.x, A2A protocol 1.0).

The agent card is published at /.well-known/agent-card.json and JSON-RPC is served at /.
Progress streams back as task status updates; the report arrives as a text/markdown artifact.
    aidesk a2a --port 8002
"""

from __future__ import annotations

import asyncio
import os
import threading

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from starlette.applications import Starlette

from aidesk.agents import research


def describe(event: dict) -> str:
    d = event.get("detail", {})
    step = d.get("step")
    if step == "plan":
        return "Plan: " + "; ".join(d.get("subquestions", []))
    if step == "research":
        return f"Researched '{d.get('subq', '')}' ({d.get('pages', 0)} pages)"
    if step == "collect":
        return f"Collected {d.get('sources', 0)} sources"
    if step == "approve":
        return "Plan approved"
    if step == "critic":
        return "Reviewing coverage" + (f": filling gaps {d.get('gaps')}" if d.get("gaps") else "")
    if step == "write":
        return "Writing the report"
    if step == "verify":
        return f"Checked citations ({d.get('cited', 0)} sources cited)"
    return str(step or event.get("event"))


class ResearchAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        question = (get_message_text(context.message) or "").strip()
        if not question:
            await updater.failed(new_text_message("Send a research question as text."))
            return
        await updater.update_status(state=TaskState.TASK_STATE_WORKING, message=new_text_message("Planning research"))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                for ev in research.start(question, auto_approve=True):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception as err:  # noqa: BLE001 - report any failure to the caller
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "message": str(err)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        final: dict | None = None
        while (ev := await queue.get()) is not None:
            if ev["event"] == "step":
                await updater.update_status(state=TaskState.TASK_STATE_WORKING, message=new_text_message(describe(ev)))
            elif ev["event"] in {"done", "error"}:
                final = ev

        if final and final["event"] == "done" and final.get("report"):
            await updater.add_artifact(parts=[new_text_part(text=final["report"], media_type="text/markdown")],
                                       name="research-report")
            note = "Report ready" + ("" if final.get("citations_ok") else " (citation check found problems)")
            await updater.complete(new_text_message(note))
        else:
            reason = (final or {}).get("message", "no report was produced")
            await updater.failed(new_text_message(f"Research failed: {reason}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancelling a research run is not supported")


def agent_card(public_url: str) -> AgentCard:
    skill = AgentSkill(
        id="deep_research",
        name="Deep research with citations",
        description="Plans sub-questions, researches them on the web in parallel and writes a cited Markdown report.",
        input_modes=["text/plain"],
        output_modes=["text/markdown"],
        tags=["research", "web", "citations"],
        examples=["Compare hybrid search approaches for RAG", "What are the main risks of AI agents using tools?"],
    )
    return AgentCard(
        name="AI Desk Deep Research",
        description="A research agent from AI Desk that answers broad questions with a cited report.",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/markdown"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=public_url, protocol_version="1.0")],
        skills=[skill],
    )


def build_app(public_url: str | None = None) -> Starlette:
    public_url = public_url or os.environ.get("AIDESK_A2A_PUBLIC_URL", "http://127.0.0.1:8002")
    card = agent_card(public_url)
    handler = DefaultRequestHandler(agent_executor=ResearchAgentExecutor(), task_store=InMemoryTaskStore(),
                                    agent_card=card)
    routes = []
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, "/"))
    return Starlette(routes=routes)


def main(port: int = 8002) -> None:
    import uvicorn

    uvicorn.run(build_app(os.environ.get("AIDESK_A2A_PUBLIC_URL", f"http://127.0.0.1:{port}")),
                host="127.0.0.1", port=port, log_level="info")
