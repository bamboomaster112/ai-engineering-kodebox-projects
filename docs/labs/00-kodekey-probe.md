# Lab 00: Find out what your KodeKey key can do

**Goal:** before writing any AI code, measure which models your key can call and which API features work,
so the design rests on facts instead of the model list.

## Why this matters
`GET /v1/models` on KodeKey listed 61 models, but listing is not permission. The probe makes a tiny request
(a few tokens) to each candidate and tests tool calling, streaming, JSON output and embeddings on the models
AI Desk uses for its three roles.

## Run it
```bash
cd ~/ai-engineering-kodebox-projects
./setup/bootstrap.sh
uv run python setup/probe_kodekey.py        # about 2,000 tokens
```
It prints a table and writes `reports/kodekey-probe.json`.

## What we found (20 November 2026, standard KodeKey key)
| Model | Result |
|---|---|
| claude-haiku-4-5-20251001, claude-haiku-5-5 | ok (about 650 ms) |
| claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-2 | ok (1.1 to 1.6 s) |
| gpt-5.4-mini, google/gemini-3.5-flash, qwen/qwen3.8-flash, deepseek/deepseek-v4-flash | ok |
| gpt-5-mini, gpt-4.1, kodekloud/claude-haiku, kodekloud/claude-sonnet | HTTP 403 for this key |
| text-embedding-3-large | HTTP 403 for this key |

| Feature (fast, main, reasoning roles) | Result |
|---|---|
| Tool calling | ok on all three |
| Streaming | ok on all three |
| Native `response_format: json_schema` | not supported through the gateway |
| Prompted JSON validated with Pydantic | ok |

## Decisions this drove
- **Roles** in `config/models.toml`: fast and judge = claude-haiku-5-5, main = claude-sonnet-5-5,
  reasoning = claude-opus-5-5.
- **Structured output** is prompt-based JSON validated by Pydantic with one automatic repair turn
  (`Gateway.structured`), because native JSON schema mode is not available through KodeKey.
- **Retrieval without embeddings**: with no embeddings model, the RAG pipeline uses keyword search (SQLite
  FTS5 BM25) plus LLM query expansion and LLM reranking. The vector path switches on automatically if the
  key ever gains an embeddings model (`[embeddings] mode = "auto"`).

## A second probe: hidden thinking
`uv run python setup/probe_thinking.py --hard` showed that the Claude 5.5 models on KodeKey can think before
answering, that the thinking text is not returned, and that request options such as
`thinking: {type: disabled}` or `reasoning_effort` make no visible difference through the gateway. Because
thinking counts against `max_tokens`, AI Desk uses generous ceilings (you pay only for tokens used) and the
gateway retries a reply that was cut off before any text.

## Exercise
Add two more models to `[probe] chat` and compare their latency. Then point the `fast` role at the quickest
one with `AIDESK_MODEL_FAST=...` and re-run the support evals to see whether quality holds.
