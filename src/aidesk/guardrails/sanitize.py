"""Treat retrieved text (knowledge base chunks, web pages) as data: remove instruction-like sentences."""

from __future__ import annotations

import re

from aidesk.guardrails.patterns import injection_hits

_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
PLACEHOLDER = "[removed: instruction-like text]"


def sanitize_untrusted(text: str) -> tuple[str, int]:
    """Drop sentences that try to instruct the assistant. Returns (clean text, sentences removed)."""
    if not text or not injection_hits(text):
        return text, 0
    kept, removed = [], 0
    for sentence in _SENTENCES.split(text):
        if sentence and injection_hits(sentence):
            removed += 1
            if not kept or kept[-1] != PLACEHOLDER:
                kept.append(PLACEHOLDER)
        elif sentence:
            kept.append(sentence)
    return " ".join(kept), removed
