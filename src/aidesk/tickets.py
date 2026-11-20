"""A tiny ticket desk in SQLite: the support bot's hand-off to humans."""

from __future__ import annotations

import time

from aidesk.config import get_settings
from aidesk.db import connect
from aidesk.guardrails.patterns import redact_pii


def create_ticket(subject: str, body: str, *, customer: str | None = None, thread_id: str | None = None,
                  priority: str = "normal") -> dict:
    subject, _ = redact_pii(subject.strip()[:120] or "Support request")
    body, _ = redact_pii(body.strip()[:4000])
    if priority not in {"low", "normal", "high", "urgent"}:
        priority = "normal"
    with connect(get_settings().db_path) as conn:
        cur = conn.execute(
            "INSERT INTO tickets (created, thread_id, subject, body, customer, priority) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), thread_id, subject, body, customer, priority),
        )
        ticket_id = cur.lastrowid
    return {"id": ticket_id, "subject": subject, "priority": priority, "status": "open"}


def list_tickets(status: str = "open", limit: int = 20) -> list[dict]:
    with connect(get_settings().db_path) as conn:
        rows = conn.execute(
            "SELECT id, created, thread_id, subject, body, customer, priority, status FROM tickets "
            "WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, max(1, min(limit, 100))),
        ).fetchall()
    return [dict(r) for r in rows]
