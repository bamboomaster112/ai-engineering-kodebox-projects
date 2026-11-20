"""Project 1: the LLM playground. Send one prompt to several models and compare the replies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from aidesk.config import get_settings
from aidesk.llm import BudgetExceeded, LLMError, get_gateway
from aidesk.llm.budget import in_this_context, run_scope

MAX_PROMPT_CHARS = 4000


@dataclass
class RunResult:
    model: str
    run: int
    text: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    error: str | None = None


class PlaygroundError(ValueError):
    pass


def available_models() -> list[str]:
    return list(get_settings().playground_models)


def _one(model: str, run: int, messages, temperature, top_p, max_tokens) -> RunResult:
    try:
        r = get_gateway().chat(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            purpose="playground",
            # Reusing a cached reply hides sampling differences, so only cache deterministic calls.
            cache=temperature == 0 and run == 1,
        )
        return RunResult(model, run, r.text, r.latency_ms, r.usage.prompt_tokens, r.usage.completion_tokens, r.cached)
    except (LLMError, BudgetExceeded) as err:
        return RunResult(model, run, error=str(err)[:300])


def compare(
    prompt: str,
    models: list[str],
    *,
    system: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int = 300,
    runs: int = 1,
    token_cap: int = 12_000,
) -> dict:
    prompt = (prompt or "").strip()
    if not prompt:
        raise PlaygroundError("prompt is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PlaygroundError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
    allowed = set(available_models())
    unknown = [m for m in models if m not in allowed]
    if not models or unknown:
        raise PlaygroundError(f"choose models from the playground list; not allowed: {unknown or 'none chosen'}")
    if not 0 <= temperature <= 2 or (top_p is not None and not 0 < top_p <= 1):
        raise PlaygroundError("temperature must be 0 to 2 and top_p 0 to 1")
    runs = max(1, min(int(runs), 3))
    max_tokens = max(16, min(int(max_tokens), 1000))

    messages = ([{"role": "system", "content": system.strip()}] if system and system.strip() else [])
    messages.append({"role": "user", "content": prompt})
    jobs = [(m, r) for m in models[:6] for r in range(1, runs + 1)]
    with run_scope(cap=token_cap) as run_id, ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
        futures = [pool.submit(in_this_context(_one), m, r, messages, temperature, top_p, max_tokens)
                   for m, r in jobs]
        results = [f.result() for f in futures]

    ok = [r for r in results if not r.error]
    return {
        "run_id": run_id,
        "settings": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "runs": runs},
        "results": [asdict(r) for r in results],
        "summary": {
            "fastest": min(ok, key=lambda r: r.latency_ms).model if ok else None,
            "tokens": sum(r.prompt_tokens + r.completion_tokens for r in ok if not r.cached),
            "errors": sum(1 for r in results if r.error),
        },
    }
