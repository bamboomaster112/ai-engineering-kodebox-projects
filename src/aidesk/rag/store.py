"""The knowledge-base index: SQLite FTS5 (BM25) for keywords, plus optional vectors for meaning."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aidesk.config import get_settings
from aidesk.db import connect
from aidesk.llm import LLMError, get_gateway
from aidesk.rag.chunking import Chunk, chunk_document, load_documents

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "and", "or", "in", "on", "for", "with", "my",
    "i", "do", "does", "how", "what", "can", "it", "this", "that", "we", "you", "your", "our", "me", "at",
    "by", "from", "as", "if", "about", "there", "which", "when", "why", "will", "would", "should", "could",
}


@dataclass
class Hit:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "doc_id": self.doc_id, "title": self.title, "section": self.section,
                "text": self.text, "scores": self.scores}


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: quoted terms joined with OR (no user syntax gets through)."""
    terms = []
    for word in _WORD.findall(text.lower()):
        word = word.strip(".-_")
        if len(word) < 2 or word in STOPWORDS:
            continue
        if word not in terms:
            terms.append(word)
    return " OR ".join(f'"{t}"' for t in terms[:24])


def ingest(kb_dir: Path | None = None, *, embed: str | None = None) -> dict:
    """(Re)build the index from markdown files. Returns a summary of what was indexed."""
    settings = get_settings()
    kb_dir = kb_dir or settings.kb_dir
    mode = (embed or settings.embed_mode).lower()
    docs = load_documents(kb_dir)
    chunks: list[Chunk] = [c for d in docs for c in chunk_document(d)]

    vectors: np.ndarray | None = None
    dense_note = "off"
    if mode != "off" and chunks:
        try:
            vectors = get_gateway().embed([f"{c.title}. {c.section}. {c.text}" for c in chunks], purpose="ingest")
            dense_note = "on"
        except LLMError as err:
            if mode == "on":
                raise
            dense_note = f"unavailable ({str(err)[:120]})"

    with connect(settings.db_path) as conn:
        conn.execute("DELETE FROM kb_chunks")
        conn.execute("DELETE FROM kb_fts")
        for i, c in enumerate(chunks):
            blob = vectors[i].astype(np.float32).tobytes() if vectors is not None else None
            conn.execute(
                "INSERT INTO kb_chunks (id, doc_id, title, section, text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (c.id, c.doc_id, c.title, c.section, c.text, blob),
            )
            conn.execute("INSERT INTO kb_fts (id, title, section, text) VALUES (?, ?, ?, ?)",
                         (c.id, c.title, c.section, c.text))
        meta = {
            "documents": len(docs),
            "chunks": len(chunks),
            "dense": vectors is not None,
            "dense_note": dense_note,
            "embed_model": settings.model_for("embed") if vectors is not None else None,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('index', ?)", (json.dumps(meta),))
    return meta


def index_meta() -> dict:
    with connect(get_settings().db_path) as conn:
        row = conn.execute("SELECT value FROM kb_meta WHERE key = 'index'").fetchone()
    return json.loads(row[0]) if row else {"documents": 0, "chunks": 0, "dense": False}


def ensure_index() -> dict:
    meta = index_meta()
    if not meta.get("chunks"):
        meta = ingest()
    return meta


def _row_to_hit(row, **scores) -> Hit:
    return Hit(id=row["id"], doc_id=row["doc_id"], title=row["title"], section=row["section"],
               text=row["text"], scores=dict(scores))


def search_lexical(query: str, k: int = 12) -> list[Hit]:
    q = fts_query(query)
    if not q:
        return []
    with connect(get_settings().db_path) as conn:
        rows = conn.execute(
            "SELECT c.id, c.doc_id, c.title, c.section, c.text, bm25(kb_fts, 0.0, 3.0, 2.0, 1.0) AS score "
            "FROM kb_fts JOIN kb_chunks c ON c.id = kb_fts.id WHERE kb_fts MATCH ? ORDER BY score LIMIT ?",
            (q, k),
        ).fetchall()
    return [_row_to_hit(r, bm25=float(-r["score"])) for r in rows]


def search_dense(query: str, k: int = 12) -> list[Hit]:
    with connect(get_settings().db_path) as conn:
        rows = conn.execute(
            "SELECT id, doc_id, title, section, text, embedding FROM kb_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    qvec = get_gateway().embed([query], purpose="query_embed")[0]
    sims = matrix @ qvec
    order = np.argsort(-sims)[:k]
    return [_row_to_hit(rows[i], cosine=float(sims[i])) for i in order]


def get_article(doc_id: str) -> dict | None:
    with connect(get_settings().db_path) as conn:
        rows = conn.execute(
            "SELECT title, section, text FROM kb_chunks WHERE doc_id = ? ORDER BY rowid", (doc_id,)
        ).fetchall()
    if not rows:
        return None
    sections = []
    for r in rows:
        if not sections or sections[-1]["section"] != r["section"]:
            sections.append({"section": r["section"], "text": r["text"]})
        else:
            sections[-1]["text"] += " " + r["text"]
    return {"doc_id": doc_id, "title": rows[0]["title"], "sections": sections}
