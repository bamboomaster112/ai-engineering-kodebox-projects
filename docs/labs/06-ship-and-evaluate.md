# Lab 06: Ship it, measure it, keep it safe (the capstone)

**Goal:** run the whole application as a service, prove it works end to end, and gate changes on quality
and security checks.

## Run it as a service
```bash
./setup/start-services.sh     # tmux session "aidesk": api :8000, mcp :8001, a2a :8002, all on 127.0.0.1
./setup/publish-app.sh        # nginx /app/ with basic auth; password saved in ~/aidesk-credentials.txt
```
Why tmux instead of systemd: the KodeBox runs a platform security monitor, and services that run as your user
in tmux change nothing outside your home folder except the two nginx files that publish `/app/`. Re-run the
start script after the server restarts (auto-shutdown keeps your files).

## Three kinds of tests
| Suite | Command | Needs | Tokens |
|---|---|---|---|
| Offline unit and integration | `uv run pytest` | nothing (fake model, fixture web) | 0 |
| Live agents | `uv run pytest -m live tests/live -s` | KodeKey key, internet | about 30k cold, little when cached |
| Support evals (quality gate) | `uv run pytest -m live tests/evals -s` | KodeKey key | about 60k cold, little when cached |
| End to end | `uv run pytest -m e2e tests/e2e -s` | running services and nginx | mostly cached |

The offline suite exercises the real code paths (LangGraph graphs, SQLite, FastAPI, MCP in-process,
A2A over real HTTP) with a deterministic fake model that reads the same XML-tagged prompts.

## The quality gate
`tests/evals/test_evals.py` asks 14 golden questions (12 answerable, 2 not in the help center) and sends 11
guardrail probes (direct and subtle attacks, an off-topic request, two benign look-alikes and an indirect
injection through a planted article). It measures:

| Metric | How | Threshold |
|---|---|---|
| Retrieval hit rate | the expected article is among the 4 passages given to the model | 0.9 |
| Expected facts | the answer contains the key fact (for example "10,000" build minutes) | 0.8 |
| Cites the right article | a citation points at the expected article | 0.9 |
| Faithfulness, relevancy | LLM judge (claude-haiku-5-5), 0 to 1 | 0.8 each |
| Abstention | out-of-scope questions get "not found" and no citations | all |
| Guardrails | each probe blocked or allowed as expected; the planted instruction never reaches the answer | all |

Results are written to `reports/live/support-evals.json`.

## Security checks
- `ruff` with the bandit-derived `S` rules, `bandit -ll`, and `pip-audit` on the locked dependency set.
- CI (`.github/workflows/ci.yml`) runs all of them plus the offline tests on every push, with actions pinned
  by commit SHA and read-only permissions.
- Secrets never enter the repo: the key lives in `~/.env` (mode 600), `.gitignore` excludes `.env`,
  credentials files and the runtime folder.

## Cost control
`uv run aidesk usage` shows today's tokens by model and purpose. Every run has a token cap (support 12k,
web 30k, research 45k by default) and there is a daily cap (`AIDESK_DAILY_TOKEN_CAP`, 150k by default).
Identical calls are served from the cache, which is why a second `run-all.sh` costs a fraction of the first.

## Exercises
1. Add a golden question that currently fails, fix the retrieval or prompt, and watch the gate go green.
2. Point `OTEL_EXPORTER_OTLP_ENDPOINT` at a local Phoenix and compare the span timings of two runs.
3. Raise the support bot's run cap to 0 (unlimited) and measure the real cost of one turn per model role.
