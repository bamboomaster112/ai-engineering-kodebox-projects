from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResult:
    text: str
    model: str
    usage: Usage
    latency_ms: int
    cached: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def assistant_message(self) -> dict[str, Any]:
        """The message to append to the conversation, in OpenAI chat format."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push a run or the day past its token cap."""


class LLMError(RuntimeError):
    """A model call failed after retries. The message is safe to show to a user."""


class ContentBlocked(LLMError):
    """The gateway's own content screening (KodeKey uses Model Armor) refused the request."""
