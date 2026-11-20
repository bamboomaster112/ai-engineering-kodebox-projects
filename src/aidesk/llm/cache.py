"""Response cache: identical requests (same model, messages and settings) return the stored answer."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from aidesk.db import connect


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return namespace + ":" + hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, db_path: Path, enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled

    def get(self, key: str, max_age: float | None = None) -> Any | None:
        if not self.enabled:
            return None
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT response, created FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if not row or (max_age is not None and time.time() - row[1] > max_age):
            return None
        return json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response, created) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )


def purge_bad_entries(db_path: Path) -> int:
    """Delete cached chat replies that were cut off or empty (saved by older versions). Returns how many."""
    removed = 0
    with connect(db_path) as conn:
        for key, raw in conn.execute("SELECT key, response FROM llm_cache WHERE key LIKE 'chat:%'").fetchall():
            value = json.loads(raw)
            message = value.get("message", {}) if isinstance(value, dict) else {}
            empty = not ((message.get("content") or "").strip() or message.get("tool_calls"))
            if isinstance(value, dict) and (value.get("finish_reason") == "length" or empty):
                conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                removed += 1
    return removed
