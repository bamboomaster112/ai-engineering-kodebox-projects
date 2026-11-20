"""Phase 0 probe: which KodeKey models can this key call, and which features work?

Spends roughly 2,000 tokens. Writes reports/kodekey-probe.json and prints a summary table.
Run on the KodeBox:  uv run python setup/probe_kodekey.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from aidesk.config import REPO_ROOT, get_settings
from aidesk.llm import Gateway, LLMError
from aidesk.llm.budget import run_scope

ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up the shipping status of an order by its number.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order number"}},
            "required": ["order_id"],
        },
    },
}


class Capital(BaseModel):
    country: str
    capital: str


def ping(gw: Gateway, model: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = gw.chat(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            model=model,
            temperature=0,
            max_tokens=64,
            purpose="probe",
            cache=False,
        )
        return {
            "ok": "pong" in r.text.lower(),
            "reply": r.text.strip()[:40],
            "latency_ms": r.latency_ms,
            "tokens": r.usage.total,
            "served_by": r.model,
        }
    except LLMError as err:
        return {"ok": False, "error": str(err)[:200], "latency_ms": int((time.perf_counter() - t0) * 1000)}


def tool_call(gw: Gateway, model: str) -> dict:
    try:
        r = gw.chat(
            [{"role": "user", "content": "What is the status of order 4471? Use the tool."}],
            model=model,
            temperature=0,
            max_tokens=200,
            tools=[ORDER_TOOL],
            purpose="probe",
            cache=False,
        )
        names = [tc.name for tc in r.tool_calls]
        args = [tc.arguments for tc in r.tool_calls]
        return {"ok": names == ["get_order_status"] and "4471" in json.dumps(args), "calls": names, "args": args}
    except LLMError as err:
        return {"ok": False, "error": str(err)[:200]}


def structured(gw: Gateway, model_role: str) -> dict:
    try:
        out = gw.structured(
            [{"role": "user", "content": "Give the capital of Japan."}], Capital, role=model_role, purpose="probe"
        )
        return {"ok": out.capital.lower() == "tokyo", "value": out.model_dump()}
    except LLMError as err:
        return {"ok": False, "error": str(err)[:200]}


def native_json_schema(gw: Gateway, model: str) -> dict:
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "capital", "schema": Capital.model_json_schema(), "strict": True},
    }
    try:
        r = gw.chat(
            [{"role": "user", "content": "Give the capital of France as JSON."}],
            model=model,
            temperature=0,
            max_tokens=100,
            response_format=fmt,
            purpose="probe",
            cache=False,
        )
        Capital.model_validate_json(r.text)
        return {"ok": True}
    except Exception as err:  # noqa: BLE001 - informational probe
        return {"ok": False, "error": str(err)[:160]}


def streaming(gw: Gateway, model: str) -> dict:
    try:
        chunks = list(
            gw.stream(
                [{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
                model=model,
                max_tokens=40,
                purpose="probe",
            )
        )
        text = "".join(chunks)
        return {"ok": "5" in text and len(chunks) >= 1, "chunks": len(chunks), "text": text.strip()[:40]}
    except LLMError as err:
        return {"ok": False, "error": str(err)[:200]}


def embeddings(gw: Gateway) -> dict:
    try:
        vecs = gw.embed(
            ["How do I rotate an API token?", "Steps to regenerate an access token", "Best pizza in Naples"],
            purpose="probe",
        )
        close = float(vecs[0] @ vecs[1])
        far = float(vecs[0] @ vecs[2])
        return {"ok": vecs.shape[1] == gw.settings.embed_dimensions and close > far, "dims": int(vecs.shape[1]),
                "similar": round(close, 3), "unrelated": round(far, 3)}
    except LLMError as err:
        return {"ok": False, "error": str(err)[:200]}


def main() -> int:
    settings = get_settings()
    if settings.provider != "kodekey":
        print("OPENAI_API_KEY is not set (or AIDESK_PROVIDER=fake); the probe needs your KodeKey key.")
        return 2
    gw = Gateway()
    report: dict = {"base_url": settings.base_url, "roles": settings.models, "chat": {}, "features": {}}
    with run_scope(cap=20_000):
        for model in settings.probe_models:
            report["chat"][model] = ping(gw, model)
            status = "ok " if report["chat"][model]["ok"] else "FAIL"
            detail = report["chat"][model].get("error") or report["chat"][model].get("reply")
            print(f"  [{status}] {model:<32} {report['chat'][model].get('latency_ms', 0):>6} ms  {detail}")

        for role in ("fast", "main", "reasoning"):
            model = settings.model_for(role)
            report["features"][role] = {
                "model": model,
                "tool_calling": tool_call(gw, model),
                "streaming": streaming(gw, model),
                "native_json_schema": native_json_schema(gw, model),
            }
        report["features"]["structured_fast"] = structured(gw, "fast")
        report["features"]["embeddings"] = embeddings(gw)

    report["tokens_spent"] = gw.ledger.summary()["tokens"]
    out = REPO_ROOT / "reports" / "kodekey-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\nFeatures by role:")
    for role in ("fast", "main", "reasoning"):
        f = report["features"][role]
        marks = {k: ("ok" if v.get("ok") else "FAIL") for k, v in f.items() if isinstance(v, dict)}
        print(f"  {role:<10} {f['model']:<28} {marks}")
    print(f"  structured output (fast): {'ok' if report['features']['structured_fast'].get('ok') else 'FAIL'}")
    e = report["features"]["embeddings"]
    print(f"  embeddings: {'ok' if e.get('ok') else 'FAIL'} {e}")
    print(f"\nTokens spent today so far: {report['tokens_spent']:,}. Report: {out.relative_to(REPO_ROOT)}")

    roles_ok = all(report["chat"].get(settings.model_for(r), {}).get("ok", True) for r in ("fast", "main", "reasoning"))
    core_ok = roles_ok and report["features"]["embeddings"].get("ok") and report["features"]["structured_fast"].get("ok")
    print("PROBE RESULT:", "PASS" if core_ok else "CHECK ROLES")
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
