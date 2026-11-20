"""A deterministic stand-in for KodeKey.

It lets the whole app (API, agents, MCP, A2A) run in CI and on a laptop without spending tokens.
It reads the same XML-tagged prompts the real models get and answers by rules, keyed on the
call's purpose. It is a test double, not a model: answers are extractive and simple.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any

import numpy as np

from aidesk.guardrails.patterns import injection_hits

_TAG = r"<{tag}(?P<attrs>\s[^>]*)?>(?P<body>.*?)</{tag}>"
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
OFF_TOPIC_WORDS = {"weather", "recipe", "poem", "horoscope", "football", "lottery", "song"}


def _tags(text: str, tag: str) -> list[tuple[dict[str, str], str]]:
    out = []
    for m in re.finditer(_TAG.format(tag=tag), text, flags=re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group("attrs") or ""))
        out.append((attrs, m.group("body").strip()))
    return out


def _first(text: str, tag: str) -> str:
    found = _tags(text, tag)
    return found[0][1] if found else ""


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    parts = _SENTENCE_END.split(text, maxsplit=1)
    sentence = parts[0] if parts else text
    return sentence[:280]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _content_words(text: str) -> set[str]:
    from aidesk.rag.store import STOPWORDS

    return {w.rstrip("s") for w in _words(text) if w not in STOPWORDS and len(w) > 2}


def _all_text(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(chunks)


def _last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


class FakeProvider:
    name = "fake"

    def complete(self, request: dict[str, Any], purpose: str) -> dict[str, Any]:
        messages = request.get("messages", [])
        handler = getattr(self, f"_p_{purpose}", None)
        if request.get("tools") and purpose in {"agent", "research_agent"}:
            message = self._agent(messages)
        elif handler:
            message = {"role": "assistant", "content": handler(messages, request)}
        else:
            message = {"role": "assistant", "content": self._default(messages, request)}
        prompt_tokens = max(1, len(json.dumps(messages)) // 4)
        completion_tokens = max(1, len(json.dumps(message)) // 4)
        return {
            "model": request.get("model", "fake"),
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }

    def stream(self, request: dict[str, Any], purpose: str) -> Iterator[dict[str, Any]]:
        result = self.complete(request, purpose)
        text = result["message"].get("content") or ""
        for i, word in enumerate(text.split(" ")):
            yield {"delta": word if i == 0 else " " + word}
        yield {"usage": result["usage"]}

    def embed(self, texts: list[str], model: str, dimensions: int | None) -> dict[str, Any]:
        dims = dimensions or 256
        vectors = [hashed_embedding(t, dims).tolist() for t in texts]
        return {"vectors": vectors, "prompt_tokens": sum(len(t) // 4 for t in texts)}

    # ---- purposes -------------------------------------------------------------------------

    def _default(self, messages, request) -> str:
        user = " ".join(_last_user(messages).split()[:12])
        return f"(offline fake model {request.get('model', '')}) You asked: {user}"

    def _p_probe(self, messages, request) -> str:
        return "pong"

    def _p_guard(self, messages, request) -> str:
        text = _first(_all_text(messages), "user_input") or _last_user(messages)
        hits = injection_hits(text)
        off_topic = bool(_words(text) & OFF_TOPIC_WORDS)
        words = _words(text)
        intent = "support"
        if words & {"human", "person", "agent", "ticket", "escalate"}:
            intent = "human"
        elif words & {"research", "compare", "comparison", "market", "landscape"}:
            intent = "research"
        return json.dumps(
            {
                "injection": bool(hits),
                "off_topic": off_topic,
                "intent": intent,
                "reason": ("matched: " + ", ".join(hits)) if hits else ("off topic" if off_topic else "ok"),
            }
        )

    def _p_expand(self, messages, request) -> str:
        question = _first(_all_text(messages), "question") or _last_user(messages)
        keywords = [w for w in _WORD.findall(question.lower()) if len(w) > 3][:6]
        return json.dumps({"queries": [" ".join(keywords)] if keywords else []})

    def _p_condense(self, messages, request) -> str:
        return _first(_all_text(messages), "follow_up") or _last_user(messages)

    def _p_rerank(self, messages, request) -> str:
        text = _all_text(messages)
        query = _content_words(_first(text, "query"))
        scored = []
        for pos, (attrs, body) in enumerate(_tags(text, "candidate")):
            overlap = len(query & _content_words(body))
            scored.append((-overlap, pos, attrs.get("id", "")))
        scored.sort()
        return json.dumps({"ranking": [cid for _, _, cid in scored]})

    def _p_answer(self, messages, request) -> str:
        sources = _tags(_all_text(messages), "source")
        if not sources:
            return "I could not find this in the knowledge base, so I can't answer it reliably."
        parts = [f"{_first_sentence(body)} [{attrs.get('id', '?')}]" for attrs, body in sources[:2]]
        return " ".join(parts)

    def _p_research_notes(self, messages, request) -> str:
        sources = _tags(_all_text(messages), "source")
        if not sources:
            return "- No usable sources were found for this sub-question."
        return "\n".join(f"- {_first_sentence(body)} [{attrs.get('id', '?')}]" for attrs, body in sources[:3])

    def _p_plan(self, messages, request) -> str:
        question = _first(_all_text(messages), "question") or _last_user(messages)
        q = question.rstrip("?. ")
        return json.dumps(
            {
                "subquestions": [
                    f"What is {q} and how does it work?",
                    f"What are the main trade-offs and risks of {q}?",
                ]
            }
        )

    def _p_critic(self, messages, request) -> str:
        return json.dumps({"sufficient": True, "gaps": []})

    def _p_write(self, messages, request) -> str:
        text = _all_text(messages)
        question = _first(text, "question")
        notes = [body for _, body in _tags(text, "notes")]
        lines = [f"# {question}", "", "## Summary", ""]
        lines += [n for n in notes if n] or ["No findings were collected."]
        return "\n".join(lines)

    def _p_judge(self, messages, request) -> str:
        return json.dumps({"faithfulness": 1.0, "relevancy": 1.0, "reason": "offline fake judge"})

    # ---- tool-using agent -------------------------------------------------------------------

    def _agent(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        last = messages[-1] if messages else {}
        tool_names = {}
        for m in messages:
            for tc in m.get("tool_calls") or []:
                tool_names[tc["id"]] = tc["function"]["name"]

        def call(name: str, args: dict) -> dict:
            n = sum(1 for m in messages if m.get("role") == "tool") + 1
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                ],
            }

        if last.get("role") == "user":
            return call("web_search", {"query": last.get("content", "")[:200]})
        if last.get("role") == "tool":
            name = tool_names.get(last.get("tool_call_id", ""), "")
            try:
                payload = json.loads(last.get("content") or "null")
            except json.JSONDecodeError:
                payload = None
            if name == "web_search":
                results = payload.get("results", []) if isinstance(payload, dict) else []
                if results:
                    return call("fetch_url", {"url": results[0]["url"]})
                return {"role": "assistant", "content": "I could not find any web results for that."}
            if name == "fetch_url" and isinstance(payload, dict):
                sid = payload.get("source_id", 1)
                sentence = _first_sentence(payload.get("text", "")) or "The page had no readable text."
                return {"role": "assistant", "content": f"{sentence} [{sid}]"}
        return {"role": "assistant", "content": "I could not complete the research."}


def hashed_embedding(text: str, dims: int) -> np.ndarray:
    """Feature-hashing embedding: deterministic, lexical, good enough for offline tests."""
    vec = np.zeros(dims, dtype=np.float32)
    for word in _WORD.findall(text.lower()):
        h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
        vec[h % dims] += 1.0 if (h >> 63) & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec
