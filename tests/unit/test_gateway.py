import pytest
from pydantic import BaseModel

from aidesk.config import get_settings
from aidesk.llm import BudgetExceeded, Gateway, LLMError, extract_json
from aidesk.llm.budget import run_scope


class Pair(BaseModel):
    a: int
    b: str


class ScriptedProvider:
    """Returns canned replies in order, so tests can drive the gateway precisely."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, request, purpose):
        self.calls += 1
        text = self.replies.pop(0)
        return {
            "model": request["model"],
            "message": {"role": "assistant", "content": text},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def stream(self, request, purpose):
        yield {"delta": "x"}

    def embed(self, texts, model, dimensions):
        return {"vectors": [[1.0] + [0.0] * (dimensions - 1) for _ in texts], "prompt_tokens": 1}


def test_fake_chat_records_usage():
    gw = Gateway()
    r = gw.chat([{"role": "user", "content": "hello there"}], purpose="probe")
    assert r.text == "pong"
    assert r.usage.total > 0
    summary = gw.ledger.summary()
    assert summary["by_model_and_purpose"][0]["purpose"] == "probe"


def test_cache_returns_same_answer_without_a_second_call():
    provider = ScriptedProvider(["first", "second"])
    gw = Gateway(provider=provider)
    msgs = [{"role": "user", "content": "same question"}]
    a = gw.chat(msgs, purpose="t")
    b = gw.chat(msgs, purpose="t")
    assert (a.text, b.text) == ("first", "first")
    assert b.cached and provider.calls == 1


def test_cache_can_be_bypassed():
    provider = ScriptedProvider(["first", "second"])
    gw = Gateway(provider=provider)
    msgs = [{"role": "user", "content": "q"}]
    gw.chat(msgs, purpose="t", cache=False)
    assert gw.chat(msgs, purpose="t", cache=False).text == "second"


def test_run_cap_stops_a_runaway_run():
    provider = ScriptedProvider(["ok"] * 50)
    gw = Gateway(provider=provider)
    with run_scope(cap=40), pytest.raises(BudgetExceeded):
        for i in range(10):
            gw.chat([{"role": "user", "content": f"call {i}"}], purpose="t", cache=False)
    assert provider.calls < 10


def test_structured_repairs_once():
    provider = ScriptedProvider(["not json at all", '{"a": 3, "b": "x"}'])
    gw = Gateway(provider=provider)
    out = gw.structured([{"role": "user", "content": "give a pair"}], Pair, purpose="t")
    assert out == Pair(a=3, b="x")


def test_structured_gives_up_with_a_clear_error():
    gw = Gateway(provider=ScriptedProvider(["nope", "still nope"]))
    with pytest.raises(LLMError, match="Pair"):
        gw.structured([{"role": "user", "content": "x"}], Pair, purpose="t")


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 2} hope it helps') == {"a": 2}
    assert extract_json("[1, 2]") == [1, 2]


def test_embeddings_are_normalised_and_cached():
    gw = Gateway()
    v1 = gw.embed(["rotate an api token", "rotate the api token"])
    assert v1.shape == (2, get_settings().embed_dimensions)
    assert abs(float(v1[0] @ v1[0]) - 1.0) < 1e-5
    before = gw.ledger.summary()["by_model_and_purpose"]
    gw.embed(["rotate an api token"])
    assert gw.ledger.summary()["by_model_and_purpose"] == before


def test_stream_records_usage():
    gw = Gateway()
    text = "".join(gw.stream([{"role": "user", "content": "say something"}], purpose="probe"))
    assert text == "pong"
    assert gw.ledger.summary()["tokens"] >= 0


class TruncatingProvider(ScriptedProvider):
    """First reply is cut off before any text (hidden thinking used the budget), then a real one."""

    def complete(self, request, purpose):
        self.calls += 1
        self.seen_max_tokens = getattr(self, "seen_max_tokens", []) + [request["max_tokens"]]
        if self.calls == 1:
            return {"model": request["model"], "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length", "usage": {"prompt_tokens": 10, "completion_tokens": request["max_tokens"]}}
        return {"model": request["model"], "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def test_truncated_empty_reply_is_retried_with_a_bigger_budget():
    provider = TruncatingProvider([])
    gw = Gateway(provider=provider)
    r = gw.chat([{"role": "user", "content": "grade this"}], max_tokens=200, purpose="t")
    assert r.text == "done" and provider.seen_max_tokens == [200, 600]
    rows = gw.ledger.summary()["by_model_and_purpose"]
    assert rows[0]["calls"] == 2  # both attempts are booked


def test_cut_off_replies_are_not_cached(tmp_path):
    class AlwaysCut(ScriptedProvider):
        def complete(self, request, purpose):
            self.calls += 1
            return {"model": request["model"], "message": {"role": "assistant", "content": "partial"},
                    "finish_reason": "length", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    provider = AlwaysCut([])
    gw = Gateway(provider=provider)
    gw.chat([{"role": "user", "content": "x"}], purpose="t")
    gw.chat([{"role": "user", "content": "x"}], purpose="t")
    assert provider.calls == 2


def test_purge_bad_entries():
    from aidesk.llm.cache import cache_key, purge_bad_entries

    gw = Gateway()
    gw.cache.put(cache_key("chat:x", {"a": 1}), {"message": {"content": ""}, "finish_reason": "length"})
    gw.cache.put(cache_key("chat:x", {"a": 2}), {"message": {"content": "fine"}, "finish_reason": "stop"})
    assert purge_bad_entries(gw.settings.db_path) == 1
