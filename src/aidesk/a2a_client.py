"""Call a remote A2A agent: used by the support bot to hand research questions to the research agent."""

from __future__ import annotations

import asyncio

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState


async def fetch_card(base_url: str):
    async with httpx.AsyncClient(timeout=10) as http:
        return await A2ACardResolver(httpx_client=http, base_url=base_url).get_agent_card()


async def ask_async(base_url: str, text: str, *, streaming: bool = True, timeout: float = 600) -> dict:
    """Send one message; return the final report and the progress messages seen on the way."""
    card = await fetch_card(base_url)
    async with httpx.AsyncClient(timeout=timeout) as http:
        client = await create_client(agent=card, client_config=ClientConfig(streaming=streaming, httpx_client=http))
        progress: list[str] = []
        report = ""
        state = ""
        try:
            request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))
            async for chunk in client.send_message(request):
                if chunk.HasField("status_update"):
                    state = TaskState.Name(chunk.status_update.status.state)
                    msg = get_stream_response_text(chunk)
                    if msg:
                        progress.append(msg)
                elif chunk.HasField("artifact_update"):
                    report = get_artifact_text(chunk.artifact_update.artifact) or report
                elif chunk.HasField("task"):
                    state = TaskState.Name(chunk.task.status.state)
                    for artifact in chunk.task.artifacts:
                        report = get_artifact_text(artifact) or report
        finally:
            await client.close()
    return {"agent": card.name, "state": state, "report": report, "progress": progress}


def ask_research_agent(base_url: str, question: str) -> str:
    """Blocking helper for synchronous code (the support bot's LangGraph node)."""
    result = asyncio.run(ask_async(base_url, question))
    if not result["report"]:
        raise RuntimeError(f"research agent finished without a report (state {result['state']})")
    return result["report"]
