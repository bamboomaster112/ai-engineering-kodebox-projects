"""Model providers. KodeKey speaks the OpenAI API, so one client covers every model it serves."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Protocol

import openai

from aidesk.llm.types import LLMError

log = logging.getLogger(__name__)


class Provider(Protocol):
    name: str

    def complete(self, request: dict[str, Any], purpose: str) -> dict[str, Any]: ...

    def stream(self, request: dict[str, Any], purpose: str) -> Iterator[dict[str, Any]]: ...

    def embed(self, texts: list[str], model: str, dimensions: int | None) -> dict[str, Any]: ...


class OpenAICompatProvider:
    """Talks to KodeKey (or any OpenAI-compatible gateway) with the official OpenAI SDK."""

    name = "kodekey"

    def __init__(self, base_url: str, api_key: str, timeout: float = 90.0):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is empty: put your KodeKey key in ~/.env, or set AIDESK_PROVIDER=fake")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        # Some models reject max_tokens or sampling settings; remember what each model needs.
        self._quirks: dict[str, set[str]] = {}

    def _adapt(self, request: dict[str, Any]) -> dict[str, Any]:
        req = dict(request)
        quirks = self._quirks.get(req.get("model", ""), set())
        if "max_completion_tokens" in quirks and "max_tokens" in req:
            req["max_completion_tokens"] = req.pop("max_tokens")
        if "no_sampling" in quirks:
            req.pop("temperature", None)
            req.pop("top_p", None)
        if "no_top_p" in quirks:
            req.pop("top_p", None)
        return req

    def _learn(self, model: str, err: Exception) -> bool:
        """Record a quirk from a 400 error. Returns True when a retry may now succeed."""
        text = str(err).lower()
        quirks = self._quirks.setdefault(model, set())
        before = set(quirks)
        if "max_completion_tokens" in text:
            quirks.add("max_completion_tokens")
        if "temperature" in text and ("unsupported" in text or "only" in text or "does not support" in text):
            quirks.add("no_sampling")
        if "top_p" in text and "temperature" in text and "cannot both" in text:
            quirks.add("no_top_p")
        return quirks != before

    def complete(self, request: dict[str, Any], purpose: str) -> dict[str, Any]:
        model = request.get("model", "")
        for _attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**self._adapt(request))
                break
            except openai.BadRequestError as err:
                if self._learn(model, err):
                    continue
                raise LLMError(f"{model} rejected the request: {_short(err)}") from err
            except openai.APIStatusError as err:
                raise LLMError(f"{model} returned HTTP {err.status_code}: {_short(err)}") from err
            except openai.APIError as err:
                raise LLMError(f"{model} call failed: {_short(err)}") from err
        else:  # pragma: no cover - three quirks in a row
            raise LLMError(f"{model} kept rejecting the request")

        if not resp.choices:
            raise LLMError(f"{model} returned no choices")
        choice = resp.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        usage = resp.usage.model_dump() if resp.usage else {}
        return {
            "model": resp.model or model,
            "message": message,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
        }

    def stream(self, request: dict[str, Any], purpose: str) -> Iterator[dict[str, Any]]:
        model = request.get("model", "")
        req = self._adapt({**request, "stream": True, "stream_options": {"include_usage": True}})
        try:
            stream = self.client.chat.completions.create(**req)
            for chunk in stream:
                if chunk.usage:
                    yield {
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens or 0,
                            "completion_tokens": chunk.usage.completion_tokens or 0,
                        }
                    }
                for choice in chunk.choices or []:
                    delta = choice.delta.content if choice.delta else None
                    if delta:
                        yield {"delta": delta}
        except openai.APIError as err:
            raise LLMError(f"{model} stream failed: {_short(err)}") from err

    def embed(self, texts: list[str], model: str, dimensions: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "input": texts}
        if dimensions:
            kwargs["dimensions"] = dimensions
        try:
            resp = self.client.embeddings.create(**kwargs)
        except openai.APIError as err:
            raise LLMError(f"embedding call failed: {_short(err)}") from err
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        tokens = resp.usage.prompt_tokens if resp.usage else 0
        return {"vectors": vectors, "prompt_tokens": tokens}


def _short(err: Exception, limit: int = 300) -> str:
    text = " ".join(str(err).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
