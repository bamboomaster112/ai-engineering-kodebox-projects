# Lab 01: LLM Playground (ByteByteAI project 1)

**Goal:** see how models differ on the same prompt, and how sampling settings change their output.

## How it works
- `src/aidesk/playground.py` sends one prompt to up to six models in parallel threads and collects the text,
  latency and token counts of each run.
- Only models listed under `[playground]` in `config/models.toml` are accepted, so the UI cannot be used
  to call arbitrary (or blocked) models.
- Every call goes through the **LLM gateway** (`src/aidesk/llm/gateway.py`), which records tokens in a
  SQLite ledger, enforces a per-run and a daily token cap, and caches deterministic replies.
- At temperature 0 a repeated prompt is served from the cache; above 0 caching is off so you can see sampling.

## Try it
UI: open `http://<server-ip>/app/`, choose **Playground**, tick three models, set **Runs per model** to 3 and
temperature to 1.0, and compare how much the replies vary. Then set temperature to 0 and run twice: the
second run shows `cached` and costs no tokens.

API:
```bash
curl -s -u "desk:<password>" http://<server-ip>/app/api/playground/compare \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain CI in one sentence","models":["claude-haiku-5-5","gpt-5.4-mini"],"temperature":0.7}'
```

## Verify
`tests/unit/test_playground.py` (offline) and `tests/live/test_live_agents.py::test_playground_compares_three_models`.

## Exercises
1. Add a "same seed" comparison: run one model three times at temperature 0 with caching turned off
   (`cache=False`) and check whether the replies are identical.
2. Show tokens per second: completion tokens divided by latency, per model.
