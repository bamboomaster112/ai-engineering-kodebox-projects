"""Retrieval pipeline: query expansion, keyword and vector search, rank fusion, LLM reranking.

When the key has no embeddings model (a standard KodeKey key, at the time of writing), the vector
step is skipped and recall comes from searching the original question plus LLM-written rewrites.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from aidesk.guardrails.sanitize import sanitize_untrusted
from aidesk.llm import LLMError, get_gateway
from aidesk.rag.store import Hit, ensure_index, search_dense, search_lexical

RRF_K = 60


class Expansions(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=4)


class Ranking(BaseModel):
    ranking: list[str]


@dataclass
class Retrieval:
    query: str
    hits: list[Hit]
    queries: list[str]
    dense: bool
    reranked: bool
    removed_lines: int = 0
    notes: list[str] = field(default_factory=list)


def expand_query(question: str) -> list[str]:
    gw = get_gateway()
    try:
        out = gw.structured(
            [
                {
                    "role": "user",
                    "content": "Write up to 3 short search queries (keywords, synonyms, product terms) that would find "
                    "help-center articles answering this customer question. Do not answer it.\n"
                    f"<question>{html.escape(question)}</question>",
                }
            ],
            Expansions,
            role="fast",
            purpose="expand",
            max_tokens=800,
        )
        return [q for q in out.queries if q.strip()][:3]
    except LLMError:
        return []


def rrf(rankings: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Reciprocal rank fusion: robust, needs no score calibration between keyword and vector search."""
    fused: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            if hit.id in fused:
                fused[hit.id].scores.update(hit.scores)
            else:
                fused[hit.id] = Hit(hit.id, hit.doc_id, hit.title, hit.section, hit.text, dict(hit.scores))
    for hid, score in scores.items():
        fused[hid].scores["rrf"] = round(score, 5)
    return sorted(fused.values(), key=lambda h: -scores[h.id])


def rerank(question: str, hits: list[Hit], top_k: int) -> tuple[list[Hit], bool]:
    """Ask the fast model to order candidates by how well they answer the question (listwise)."""
    if len(hits) <= 1:
        return hits[:top_k], False
    candidates = "\n".join(
        f'<candidate id="{h.id}">{html.escape(h.title)} / {html.escape(h.section)}: {html.escape(h.text[:600])}</candidate>'
        for h in hits
    )
    try:
        out = get_gateway().structured(
            [
                {
                    "role": "user",
                    "content": "Rank the candidate passages by how directly they help answer the query. "
                    "Return every useful candidate id, best first, and leave out irrelevant ones.\n"
                    f"<query>{html.escape(question)}</query>\n{candidates}",
                }
            ],
            Ranking,
            role="fast",
            purpose="rerank",
            max_tokens=1000,
        )
    except LLMError:
        return hits[:top_k], False
    by_id = {h.id: h for h in hits}
    ordered = [by_id[i] for i in out.ranking if i in by_id]
    if not ordered:
        return hits[:top_k], False
    for pos, h in enumerate(ordered):
        h.scores["rerank_pos"] = pos + 1
    return ordered[:top_k], True


def retrieve(question: str, k: int = 4, *, expand: bool = True, use_rerank: bool = True, pool: int = 10) -> Retrieval:
    meta = ensure_index()
    dense = bool(meta.get("dense"))
    queries = [question] + (expand_query(question) if expand else [])
    rankings = [search_lexical(q, k=12) for q in queries]
    if dense:
        rankings.append(search_dense(question, k=12))
    candidates = rrf([r for r in rankings if r])[:pool]
    hits, reranked = rerank(question, candidates, k) if use_rerank else (candidates[:k], False)

    removed = 0
    for h in hits:
        h.text, n = sanitize_untrusted(h.text)
        removed += n
    notes = [] if dense else ["vector search off: the embeddings model is not available, using keyword search"]
    return Retrieval(query=question, hits=hits, queries=queries, dense=dense, reranked=reranked,
                     removed_lines=removed, notes=notes)
