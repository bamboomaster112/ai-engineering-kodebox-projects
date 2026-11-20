"""The LLM gateway: every model call in AI Desk goes through here.

It picks the model for a role, applies the token caps, reuses cached answers, records usage,
and hides the difference between KodeKey and the offline fake.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from typing import Any, TypeVar

import numpy as np
from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from aidesk.config import Settings, get_settings
from aidesk.llm.budget import Ledger, current_run
from aidesk.llm.cache import ResponseCache, cache_key
from aidesk.llm.fake import FakeProvider
from aidesk.llm.providers import OpenAICompatProvider, Provider
from aidesk.llm.types import ChatResult, LLMError, ToolCall, Usage

T = TypeVar("T", bound=BaseModel)
tracer = trace.get_tracer("aidesk.llm")


def _estimate_tokens(payload: Any) -> int:
    return max(1, len(json.dumps(payload, default=str)) // 4)


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}
        calls.append(ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=args))
    return calls


MAX_RETRY_TOKENS = 8000


def _truncated_empty(response: dict[str, Any]) -> bool:
    message = response.get("message", {})
    return response.get("finish_reason") == "length" and not (message.get("content") or "").strip() \
        and not message.get("tool_calls")


def _cacheable(response: dict[str, Any]) -> bool:
    """Never cache a reply that was cut off or came back empty, or a re-run would replay the failure."""
    message = response.get("message", {})
    if response.get("finish_reason") == "length":
        return False
    return bool((message.get("content") or "").strip() or message.get("tool_calls"))


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model reply (models sometimes add prose or fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in the model reply")


class Gateway:
    def __init__(self, settings: Settings | None = None, provider: Provider | None = None):
        self.settings = settings or get_settings()
        if provider is not None:
            self.provider = provider
        elif self.settings.provider == "fake":
            self.provider = FakeProvider()
        else:
            self.provider = OpenAICompatProvider(self.settings.base_url, self.settings.api_key)
        self.cache = ResponseCache(self.settings.db_path, self.settings.cache_enabled)
        self.ledger = Ledger(self.settings.db_path, self.settings.run_token_cap, self.settings.daily_token_cap)

    @property
    def provider_name(self) -> str:
        return getattr(self.provider, "name", "custom")

    def model_for(self, role: str) -> str:
        return self.settings.model_for(role)

    # ---- chat -----------------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str = "main",
        model: str | None = None,
        temperature: float | None = 0.2,
        top_p: float | None = None,
        max_tokens: int = 800,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        purpose: str = "chat",
        cache: bool | None = None,
        run_id: str | None = None,
    ) -> ChatResult:
        model = model or self.model_for(role)
        request: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if tools:
            request["tools"] = tools
            if tool_choice is not None:
                request["tool_choice"] = tool_choice
        if response_format:
            request["response_format"] = response_format

        scope_run, scope_cap = current_run()
        run_id = run_id or scope_run
        use_cache = self.cache.enabled if cache is None else cache
        key = cache_key(f"chat:{self.provider_name}", {**request, "purpose": purpose})

        with tracer.start_as_current_span(f"llm.{purpose}") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.purpose", purpose)
            started = time.perf_counter()
            hit = self.cache.get(key) if use_cache else None
            if hit is not None:
                response, cached = hit, True
                usage = Usage(**response.get("usage", {}))
            else:
                self.ledger.check(_estimate_tokens(request), self.provider_name, run_id, scope_cap)
                response, cached = self.provider.complete(request, purpose), False
                usage = Usage(**response.get("usage", {}))
                if _truncated_empty(response) and request["max_tokens"] < MAX_RETRY_TOKENS:
                    # Some models think before answering, and hidden thinking counts against max_tokens.
                    # A reply cut off before any text gets one retry with a larger budget.
                    self._record(model, purpose, usage, False, started, run_id)
                    request = {**request, "max_tokens": min(request["max_tokens"] * 3, MAX_RETRY_TOKENS)}
                    span.set_attribute("llm.retried_after_truncation", True)
                    response = self.provider.complete(request, purpose)
                    usage = Usage(**response.get("usage", {}))
                if use_cache and _cacheable(response):
                    self.cache.put(key, response)
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._record(model, purpose, usage, cached, started, run_id)
            span.set_attribute("llm.cached", cached)
            span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
            span.set_attribute("llm.completion_tokens", usage.completion_tokens)

        message = response.get("message", {})
        return ChatResult(
            text=message.get("content") or "",
            model=response.get("model", model),
            usage=usage,
            latency_ms=latency_ms,
            cached=cached,
            tool_calls=_parse_tool_calls(message),
            finish_reason=response.get("finish_reason"),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str = "main",
        model: str | None = None,
        temperature: float | None = 0.2,
        max_tokens: int = 800,
        purpose: str = "chat",
    ) -> Iterator[str]:
        """Yield text as it arrives. Streams are never cached; usage is recorded at the end."""
        model = model or self.model_for(role)
        request = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            request["temperature"] = temperature
        run_id, cap = current_run()
        self.ledger.check(_estimate_tokens(request), self.provider_name, run_id, cap)
        started = time.perf_counter()
        usage = Usage()
        produced = []
        for event in self.provider.stream(request, purpose):
            if "delta" in event:
                produced.append(event["delta"])
                yield event["delta"]
            if "usage" in event:
                usage = Usage(**event["usage"])
        if not usage.total:
            usage = Usage(_estimate_tokens(messages), _estimate_tokens("".join(produced)))
        self.ledger.record(
            model=model,
            purpose=purpose,
            usage=usage,
            cached=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
            provider=self.provider_name,
            run_id=run_id,
        )

    def _record(self, model: str, purpose: str, usage: Usage, cached: bool, started: float, run_id: str | None) -> None:
        self.ledger.record(
            model=model,
            purpose=purpose,
            usage=usage,
            cached=cached,
            latency_ms=int((time.perf_counter() - started) * 1000),
            provider=self.provider_name,
            run_id=run_id,
        )

    # ---- structured output -----------------------------------------------------------------

    def structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        role: str = "fast",
        purpose: str = "structured",
        max_tokens: int = 1500,
        run_id: str | None = None,
    ) -> T:
        """Ask for JSON matching a Pydantic model, validate it, and repair once if needed.

        Prompt-based JSON works on every model behind KodeKey, unlike provider-specific
        response_format options, so this is the portable default.
        """
        schema_json = json.dumps(schema.model_json_schema())
        instruction = {
            "role": "system",
            "content": "Reply with one JSON object only, with no prose and no code fences. "
            f"It must validate against this JSON schema:\n{schema_json}",
        }
        convo = [instruction, *messages]
        last_error = ""
        for _attempt in range(2):
            result = self.chat(convo, role=role, temperature=0, max_tokens=max_tokens, purpose=purpose, run_id=run_id)
            try:
                return schema.model_validate(extract_json(result.text))
            except (ValueError, ValidationError) as err:
                last_error = str(err)[:400]
                convo = [
                    *convo,
                    {"role": "assistant", "content": result.text},
                    {"role": "user", "content": f"That was not valid ({last_error}). Reply with the JSON object only."},
                ]
        raise LLMError(f"model did not return valid {schema.__name__} JSON: {last_error}")

    # ---- embeddings ------------------------------------------------------------------------

    def embed(self, texts: list[str], *, purpose: str = "embed", batch_size: int = 64) -> np.ndarray:
        """Embed texts (cached per text), returning L2-normalised float32 vectors."""
        model = self.model_for("embed")
        dims = self.settings.embed_dimensions
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        keys = [cache_key(f"embed:{self.provider_name}", {"m": model, "d": dims, "t": t}) for t in texts]
        for i, key in enumerate(keys):
            hit = self.cache.get(key)
            if hit is not None:
                vectors[i] = hit
            else:
                missing.append(i)
        for start in range(0, len(missing), batch_size):
            idx = missing[start : start + batch_size]
            batch = [texts[i] for i in idx]
            run_id, cap = current_run()
            self.ledger.check(_estimate_tokens(batch), self.provider_name, run_id, cap)
            t0 = time.perf_counter()
            out = self.provider.embed(batch, model, dims)
            self.ledger.record(
                model=model,
                purpose=purpose,
                usage=Usage(prompt_tokens=int(out.get("prompt_tokens", 0))),
                cached=False,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                provider=self.provider_name,
                run_id=run_id,
            )
            for i, vec in zip(idx, out["vectors"], strict=True):
                vectors[i] = vec
                self.cache.put(keys[i], vec)
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


_gateway: Gateway | None = None
_lock = threading.Lock()


def get_gateway() -> Gateway:
    global _gateway
    with _lock:
        if _gateway is None:
            _gateway = Gateway()
        return _gateway


def reset_gateway() -> None:
    global _gateway
    with _lock:
        _gateway = None
