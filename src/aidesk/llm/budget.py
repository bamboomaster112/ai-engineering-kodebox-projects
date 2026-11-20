"""Token ledger and caps. Every model call is recorded; caps are checked before a call is sent."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from aidesk.db import connect
from aidesk.llm.types import BudgetExceeded, Usage

_current_run: ContextVar[tuple[str, int] | None] = ContextVar("aidesk_run", default=None)


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@contextmanager
def run_scope(cap: int | None = None, run_id: str | None = None):
    """Group calls under one run id so a single request cannot overspend. Yields the run id."""
    rid = run_id or uuid.uuid4().hex[:12]
    token = _current_run.set((rid, cap or 0))
    try:
        yield rid
    finally:
        try:
            _current_run.reset(token)
        except ValueError:  # exited from another context (for example a generator resumed elsewhere)
            _current_run.set(None)


def in_this_context(fn):
    """Wrap fn so it runs with the caller's run scope when handed to a thread pool."""
    import contextvars

    ctx = contextvars.copy_context()
    return lambda *a, **k: ctx.copy().run(fn, *a, **k)


def current_run() -> tuple[str | None, int]:
    value = _current_run.get()
    return (value[0], value[1]) if value else (None, 0)


class Ledger:
    def __init__(self, db_path: Path, run_cap: int, daily_cap: int):
        self.db_path = db_path
        self.run_cap = run_cap
        self.daily_cap = daily_cap

    def spent_today(self) -> int:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM usage "
                "WHERE day = ? AND cached = 0 AND provider != 'fake'",
                (today(),),
            ).fetchone()
        return int(row[0])

    def spent_in_run(self, run_id: str) -> int:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM usage "
                "WHERE run_id = ? AND cached = 0",
                (run_id,),
            ).fetchone()
        return int(row[0])

    def check(self, estimate: int, provider: str, run_id: str | None, run_cap: int) -> None:
        cap = run_cap or self.run_cap
        if run_id and cap and self.spent_in_run(run_id) + estimate > cap:
            raise BudgetExceeded(f"this run would pass its {cap:,}-token cap")
        if provider != "fake" and self.daily_cap and self.spent_today() + estimate > self.daily_cap:
            raise BudgetExceeded(
                f"today's {self.daily_cap:,}-token cap is reached; raise AIDESK_DAILY_TOKEN_CAP to continue"
            )

    def record(
        self,
        *,
        model: str,
        purpose: str,
        usage: Usage,
        cached: bool,
        latency_ms: int,
        provider: str,
        run_id: str | None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO usage (ts, day, run_id, model, purpose, prompt_tokens, completion_tokens, "
                "cached, latency_ms, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    today(),
                    run_id,
                    model,
                    purpose,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    int(cached),
                    latency_ms,
                    provider,
                ),
            )

    def summary(self, day: str | None = None) -> dict:
        day = day or today()
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT model, purpose, COUNT(*) AS calls, SUM(cached) AS cached_calls, "
                "SUM(CASE WHEN cached = 0 THEN prompt_tokens + completion_tokens ELSE 0 END) AS tokens "
                "FROM usage WHERE day = ? GROUP BY model, purpose ORDER BY tokens DESC",
                (day,),
            ).fetchall()
        items = [dict(r) for r in rows]
        return {
            "day": day,
            "tokens": sum(int(r["tokens"] or 0) for r in items),
            "daily_cap": self.daily_cap,
            "by_model_and_purpose": items,
        }
