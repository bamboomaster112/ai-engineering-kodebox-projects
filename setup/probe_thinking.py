"""Does KodeKey turn on extended thinking for Claude models, and can a request turn it off?

Thinking tokens count against max_tokens, so a short budget can leave no room for the visible reply.
Spends roughly 2,000 tokens. Run on the KodeBox:  uv run python setup/probe_thinking.py
"""

from __future__ import annotations

import json

import openai

from aidesk.config import get_settings

VARIANTS = {
    "default": {},
    "thinking_disabled": {"extra_body": {"thinking": {"type": "disabled"}}},
    "reasoning_effort_none": {"extra_body": {"reasoning_effort": "none"}},
    "reasoning_effort_low": {"extra_body": {"reasoning_effort": "low"}},
}


HARD_PROMPT = (
    "Grade this support answer. Question: 'Can I move my organization from the US region to the EU region?' "
    "Source: 'The region cannot be changed after the organization is created. To move regions, create a new "
    "organization and migrate your projects.' Answer: 'Yes, you can switch regions any time in settings.' "
    'Reply with JSON only: {"faithfulness": number 0-1, "relevancy": number 0-1, "reason": "short"}'
)


def main() -> None:
    import sys

    prompt = HARD_PROMPT if "--hard" in sys.argv else "In one short sentence: what does CI stand for?"
    s = get_settings()
    client = openai.OpenAI(base_url=s.base_url, api_key=s.api_key, timeout=60, max_retries=1)
    results = {}
    roles = ("fast", "reasoning") if "--hard" in sys.argv else ("fast", "main", "reasoning")
    for role in roles:
        model = s.model_for(role)
        for name, extra in VARIANTS.items():
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    **extra,
                )
                msg = r.choices[0].message.model_dump(exclude_none=True)
                thinking = msg.get("reasoning_content") or ""
                row = {"ok": bool(msg.get("content")), "completion_tokens": r.usage.completion_tokens,
                       "thinking_chars": len(thinking), "finish": r.choices[0].finish_reason,
                       "reply": (msg.get("content") or "")[:60]}
            except openai.APIError as err:
                row = {"ok": False, "error": " ".join(str(err).split())[:140]}
            results[f"{role}:{name}"] = row
            print(f"{role:<9} {model:<20} {name:<22} {json.dumps(row)}")


if __name__ == "__main__":
    main()
