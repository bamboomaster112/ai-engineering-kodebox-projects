from aidesk.config import get_settings
from aidesk.guardrails.sanitize import sanitize_untrusted
from aidesk.rag.chunking import chunk_document, load_documents
from aidesk.rag.retrieve import retrieve, rrf
from aidesk.rag.store import Hit, fts_query, get_article, index_meta, ingest, search_lexical


def test_documents_and_chunks_load():
    docs = load_documents(get_settings().kb_dir)
    assert len(docs) >= 13
    assert all(d.doc_id and d.title for d in docs)
    chunks = [c for d in docs for c in chunk_document(d)]
    assert len({c.id for c in chunks}) == len(chunks)


def test_fts_query_is_safe():
    q = fts_query('How do I "rotate" my API token? OR DROP TABLE; NEAR(x)')
    assert q.startswith('"rotate"') and '"api"' in q and '"token"' in q
    assert "DROP" not in q  # lower-cased and quoted, so it is just a word
    assert fts_query("the and of") == ""


def test_ingest_builds_index_with_fake_vectors():
    meta = ingest()
    assert meta["documents"] >= 13 and meta["chunks"] > meta["documents"]
    assert meta["dense"] is True  # the fake provider has embeddings
    assert index_meta()["chunks"] == meta["chunks"]


def test_ingest_falls_back_to_keywords_when_embeddings_fail(monkeypatch):
    from aidesk.llm import LLMError
    from aidesk.llm.gateway import Gateway

    def boom(self, *a, **k):
        raise LLMError("403 not available for this key")

    monkeypatch.setattr(Gateway, "embed", boom)
    meta = ingest()
    assert meta["dense"] is False and "unavailable" in meta["dense_note"]


def test_lexical_search_finds_the_right_article():
    ingest(embed="off")
    hits = search_lexical("exit code 137 out of memory")
    assert hits and hits[0].doc_id == "troubleshooting-builds"


def test_rrf_merges_and_scores():
    a = Hit("x#1", "x", "X", "s", "t", {"bm25": 1.0})
    b = Hit("y#1", "y", "Y", "s", "t", {"bm25": 0.5})
    fused = rrf([[a, b], [b]])
    assert [h.id for h in fused] == ["y#1", "x#1"]


def test_retrieve_end_to_end_offline():
    ingest(embed="off")
    r = retrieve("How many build minutes are in the Team plan?", k=3)
    assert r.hits and r.hits[0].doc_id == "billing-plans"
    assert r.dense is False and r.notes
    assert len(r.queries) >= 1


def test_poisoned_article_is_sanitised_before_use():
    ingest(embed="off")
    r = retrieve("community tip cache key lock file fallback key", k=4)
    poisoned = [h for h in r.hits if h.doc_id == "community-cache-tips"]
    assert poisoned, "the planted article should be retrieved for this query"
    assert "brightwave-help.example" not in poisoned[0].text
    assert r.removed_lines >= 1


def test_sanitize_keeps_clean_text():
    text = "Use a cache key. Keep caches small."
    assert sanitize_untrusted(text) == (text, 0)


def test_get_article():
    ingest(embed="off")
    art = get_article("api-tokens")
    assert art and art["title"].startswith("Personal access tokens")
    assert any(s["section"] == "Rotating a token" for s in art["sections"])
    assert get_article("nope") is None


def test_ingest_falls_back_when_the_budget_is_spent(monkeypatch):
    from aidesk.llm import BudgetExceeded
    from aidesk.llm.gateway import Gateway

    def spent(self, *a, **k):
        raise BudgetExceeded("today's cap is reached")

    monkeypatch.setattr(Gateway, "embed", spent)
    meta = ingest()
    assert meta["dense"] is False and meta["chunks"] > 0
