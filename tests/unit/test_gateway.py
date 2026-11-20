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
