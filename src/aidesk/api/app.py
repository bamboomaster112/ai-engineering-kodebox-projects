"""AI Desk HTTP API and web UI (FastAPI). Binds to 127.0.0.1; nginx publishes it at /app/ with basic auth."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from pydantic import BaseModel, Field

from aidesk import __version__
from aidesk.agents import research, support, web
from aidesk.config import get_settings
from aidesk.llm import get_gateway
from aidesk.observability import recent_traces, setup_tracing
from aidesk.playground import PlaygroundError, available_models, compare
from aidesk.rag.store import ensure_index, index_meta
from aidesk.tickets import list_tickets

STATIC = Path(__file__).parent / "static"
tracer = trace.get_tracer("aidesk.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_tracing("aidesk-api")
    ensure_index()
    yield


app = FastAPI(title="AI Desk", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json",
              lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---- safety: per-client rate limits and security headers -------------------------------

LIMITS = {"playground": (30, 3600), "support": (120, 3600), "web": (30, 3600), "research": (8, 3600)}
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_hits_lock = Lock()


def client_id(request: Request) -> str:
    # nginx sets X-Real-IP; only trust it when the request came through the local proxy.
    peer = request.client.host if request.client else "unknown"
    if peer in {"127.0.0.1", "::1"}:
        return request.headers.get("x-real-ip", peer)
    return peer


def rate_limit(request: Request, bucket: str) -> None:
    limit, window = LIMITS[bucket]
    key = (client_id(request), bucket)
    now = time.time()
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, f"rate limit: {limit} {bucket} requests per hour")
        q.append(now)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if not request.url.path.startswith("/api/docs"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
    return response


_DONE = object()


def in_one_thread(events: Iterator[dict]) -> Iterator[dict]:
    """Run a whole event generator in one worker thread.

    Starlette advances sync generators on pooled threads with a fresh context each time, which would
    lose the run's token-budget scope between steps. One dedicated thread keeps it intact.
    """
    q: queue.Queue = queue.Queue(maxsize=256)

    def worker() -> None:
        try:
            for ev in events:
                q.put(ev)
        except Exception as err:  # noqa: BLE001 - report to the client instead of dropping the stream
            q.put({"event": "error", "message": f"internal error: {type(err).__name__}"})
        finally:
            q.put(_DONE)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not _DONE:
        yield item


def sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        for ev in in_one_thread(events):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- pages and status ------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "version": __version__, "provider": s.provider, "roles": s.models,
            "search": s.search_backend, "kb": index_meta()}


@app.get("/api/usage")
def usage() -> dict:
    return get_gateway().ledger.summary()


@app.get("/api/traces")
def traces(limit: int = 30) -> list[dict]:
    return recent_traces(max(1, min(limit, 200)))


@app.get("/api/tickets")
def tickets(status: str = "open") -> list[dict]:
    return list_tickets(status)


# ---- project 1: playground ----------------------------------------------------------------


class CompareIn(BaseModel):
    prompt: str = Field(max_length=4000)
    models: list[str] = Field(min_length=1, max_length=6)
    system: str | None = Field(default=None, max_length=2000)
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int = 300
    runs: int = 1


@app.get("/api/playground/models")
def playground_models() -> list[str]:
    return available_models()


@app.post("/api/playground/compare")
def playground_compare(body: CompareIn, request: Request) -> dict:
    rate_limit(request, "playground")
    with tracer.start_as_current_span("api.playground.compare"):
        try:
            return compare(body.prompt, body.models, system=body.system, temperature=body.temperature,
                           top_p=body.top_p, max_tokens=body.max_tokens, runs=body.runs)
        except PlaygroundError as err:
            raise HTTPException(400, str(err)) from err


# ---- project 2: support bot ---------------------------------------------------------------


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


@app.post("/api/support/chat")
def support_chat(body: ChatIn, request: Request) -> dict:
    rate_limit(request, "support")
    with tracer.start_as_current_span("api.support.chat"):
        return support.ask(body.message, body.thread_id)


# ---- project 3: ask the web ---------------------------------------------------------------


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=600)


@app.post("/api/web/ask")
def web_ask(body: QuestionIn, request: Request) -> StreamingResponse:
    rate_limit(request, "web")
    return sse(web.stream(body.question))


# ---- project 4: deep research -------------------------------------------------------------


class ResearchStartIn(QuestionIn):
    auto_approve: bool = False


class ResearchResumeIn(BaseModel):
    run_id: str = Field(pattern=r"^research-[a-f0-9]{12}$")
    plan: list[str] | None = Field(default=None, max_length=4)
    approved: bool = True


@app.post("/api/research/start")
def research_start(body: ResearchStartIn, request: Request) -> StreamingResponse:
    rate_limit(request, "research")
    return sse(research.start(body.question, auto_approve=body.auto_approve))


@app.post("/api/research/resume")
def research_resume(body: ResearchResumeIn) -> StreamingResponse:
    plan = [p[:300] for p in body.plan] if body.plan else None
    return sse(research.resume(body.run_id, plan=plan, approved=body.approved))


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # pragma: no cover - last resort
    return JSONResponse({"detail": "internal error"}, status_code=500)
