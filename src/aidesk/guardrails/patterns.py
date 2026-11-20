"""Fast, offline checks used by the guardrails (and by the offline fake model)."""

from __future__ import annotations

import re

# Phrases that try to override the assistant's instructions. Kept deliberately narrow: an LLM
# classifier handles the subtle cases, these catch the obvious ones for free and without a call.
INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[\w\s,]{0,40}\b(instructions?|rules|guidelines|prompts?|guardrails)\b",
        re.I,
    ),
    "reveal_system_prompt": re.compile(
        r"\b(reveal|show|print|repeat|output|leak|dump)\b[\w\s,]{0,30}\b(system|hidden|initial|original)\s+(prompt|instructions|message)",
        re.I,
    ),
    "role_hijack": re.compile(
        r"\b(you are now|from now on you are|act as|pretend (to be|you are))\b[\w\s,]{0,40}\b(unrestricted|unfiltered|jailbroken|developer mode|DAN|no rules)\b",
        re.I,
    ),
    "jailbreak_terms": re.compile(r"\b(jailbreak|do anything now|developer mode enabled)\b", re.I),
    "fake_system_turn": re.compile(r"(^|\n)\s*(system|assistant)\s*:\s|<\|im_start\|>|\[/?INST\]", re.I),
    "exfiltration": re.compile(
        r"(!\[[^\]]*\]\(https?://)|\b(send|post|upload|forward)\b[\w\s,]{0,40}\b(to|at)\s+https?://", re.I
    ),
    "encoded_payload": re.compile(r"[A-Za-z0-9+/]{240,}={0,2}"),
}

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "secret": re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[bap]-[A-Za-z0-9-]{10,})\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)"),
}


def injection_hits(text: str) -> list[str]:
    return [name for name, pat in INJECTION_PATTERNS.items() if pat.search(text or "")]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask emails, phone numbers, card numbers and API tokens. Returns (text, kinds found)."""
    found: list[str] = []

    def mask(kind: str):
        def _sub(m: re.Match[str]) -> str:
            raw = m.group(0)
            if kind == "card":
                digits = re.sub(r"\D", "", raw)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return raw
            if kind == "phone" and len(re.sub(r"\D", "", raw)) < 9:
                return raw
            found.append(kind)
            return f"[{kind.upper()} REDACTED]"

        return _sub

    out = text or ""
    for kind in ("secret", "email", "card", "phone"):
        out = PII_PATTERNS[kind].sub(mask(kind), out)
    return out, sorted(set(found))
