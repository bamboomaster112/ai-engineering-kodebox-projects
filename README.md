# AI Desk: an AI engineering capstone on a KodeKloud KodeBox

AI Desk is one web application that covers the projects of a modern AI engineering roadmap: an
**LLM playground**, a **customer-support chatbot with RAG and guardrails**, an **Ask-the-Web agent**, a
**Deep Research agent** with sub-agents and citations, plus an **MCP server** and an **A2A agent**. It was
built and tested end to end on a KodeKloud **AI Engineering KodeBox** (Power size: 2 vCPU, 8 GB RAM, no GPU),
with every model call going through **KodeKey**, KodeKloud's OpenAI-compatible AI gateway.

Everything runs with the offline fake model too, so the test suite and CI need no key and spend no tokens.

| Roadmap project | In AI Desk | Code | Lab |
|---|---|---|---|
| 1. LLM Playground | Compare models side by side, sampling settings, tokens and latency | `src/aidesk/playground.py` | [01](docs/labs/01-playground.md) |
| 2. Customer Support Chatbot | RAG over a help center, reranking, guardrails, memory, ticket hand-off | `src/aidesk/agents/support.py`, `src/aidesk/rag/`, `src/aidesk/guardrails/` | [02](docs/labs/02-support-bot.md) |
| 3. Ask-the-Web agent | Tool-using agent: search, read, cite, with SSRF-safe fetching | `src/aidesk/agents/web.py`, `src/aidesk/web/` | [03](docs/labs/03-ask-the-web.md) |
| 4. Deep Research | Planner, plan approval, parallel researchers, critic, writer, citation check | `src/aidesk/agents/research.py` | [04](docs/labs/04-deep-research.md) |
| Agents, MCP and A2A | MCP server for the tools; Deep Research as an A2A agent the support bot calls | `src/aidesk/mcp_server.py`, `src/aidesk/a2a_*.py` | [05](docs/labs/05-mcp-and-a2a.md) |
| 6. Capstone | One app, one gateway, evals, security checks, CI and a one-command validation | `src/aidesk/api/`, `run-all.sh` | [06](docs/labs/06-ship-and-evaluate.md) |
| 5. Image Generation, LoRA tuning | Planned for the GPU KodeBox | | |

Start with [Lab 00](docs/labs/00-kodekey-probe.md): it shows what the KodeKey key can do and why the design
looks the way it does.

## Architecture

```mermaid
flowchart LR
  U[Browser] -->|"/app/ (basic auth)"| N[nginx :80]
  N --> API["AI Desk API + UI<br/>FastAPI 127.0.0.1:8000"]
  API --> PG[Playground]
  API --> SB[Support bot<br/>LangGraph]
  API --> WA[Ask-the-Web<br/>LangGraph agent]
  API --> DR[Deep Research<br/>LangGraph + Send + interrupt]
  SB -->|A2A JSON-RPC| A2A["A2A research agent<br/>127.0.0.1:8002"]
  A2A --> DR
  MCPC[Claude Code / MCP clients] -->|"Streamable HTTP or stdio"| MCP["MCP server<br/>127.0.0.1:8001"]
  PG & SB & WA & DR & MCP --> GW["LLM gateway<br/>roles, cache, token caps"]
  GW --> KK[(KodeKey)]
  SB & MCP --> KB[(SQLite: FTS5 index, tickets,<br/>checkpoints, cache, ledger)]
  WA & DR & MCP --> WEB[Web search + safe fetch]
```

## Stack (versions verified November 2026)

| Layer | Choice |
|---|---|
| Models | KodeKey: claude-haiku-5-5 (fast, judge), claude-sonnet-5-5 (main), claude-opus-5-5 (reasoning); 9 models confirmed callable |
| Orchestration | LangGraph 1.2 (state graphs, `Send` fan-out, `interrupt` for human approval, SQLite checkpointer) |
| Model access | OpenAI Python SDK against KodeKey's OpenAI-compatible endpoint, behind AI Desk's own gateway |
| Retrieval | SQLite FTS5 (BM25) + LLM query expansion + reciprocal rank fusion + LLM listwise reranking; vectors switch on automatically when an embeddings model is available |
| Tools protocol | MCP Python SDK 2.3 (`MCPServer`, spec 2026-07-28, stateless Streamable HTTP) |
| Agent protocol | a2a-sdk 1.3 (A2A 1.0, JSON-RPC binding, agent card at `/.well-known/agent-card.json`) |
| API and UI | FastAPI, server-sent events, plain JavaScript with a strict Content Security Policy |
| Web | DuckDuckGo via `ddgs` (no key, retries across engines), Tavily optional; trafilatura for page text |
| Observability | OpenTelemetry spans to `var/traces.jsonl`, plus any OTLP collector (Phoenix, Langfuse) |
| Quality and security | pytest (offline + live), LLM-judged evals, ruff with security rules, bandit, pip-audit, SHA-pinned CI |

## Quick start on an AI Engineering KodeBox

1. Put your KodeKey key in `~/.env` as the KodeBox login message explains (`cp ~/.env.example ~/.env`,
   then paste the key after `OPENAI_API_KEY=` and `ANTHROPIC_API_KEY=`, and log in again).
2. Clone and run everything:
   ```bash
   git clone https://github.com/bamboomaster112/ai-engineering-kodebox-projects
   cd ai-engineering-kodebox-projects
   ./run-all.sh            # or ./run-all.sh --offline for the free steps only
   ```
3. Open `http://<server IP>/app/` and log in with the user and password in `~/aidesk-credentials.txt`.

`run-all.sh` sets up the locked environment, runs the offline tests and the security scans, indexes the
knowledge base, runs the live agent tests and the support evals, starts the services in a tmux session,
publishes `/app/` through nginx and runs the end-to-end tests. It prints a PASS/FAIL table.

Day to day:
```bash
./setup/start-services.sh            # api :8000, mcp :8001, a2a :8002 (all 127.0.0.1), in tmux "aidesk"
./setup/stop-services.sh
uv run aidesk ask "How long are artifacts kept?"
uv run aidesk web "What changed in the latest MCP specification?"
uv run aidesk research "How do teams evaluate RAG systems?"
uv run aidesk usage                  # tokens spent today, by model and purpose
```

Offline anywhere (laptop or CI): `uv sync && uv run pytest` uses the fake model and fixture web pages.

## Results

Validated on the KodeBox on 20 November 2026 with the strictest setup: a **fresh clone and an empty
cache**. `./run-all.sh` passed all 12 steps in 241 seconds (a warm re-run takes about 144 seconds):

| Step | Result | Time |
|---|---|---|
| setup: locked environment | PASS | 1s |
| offline tests (fake model): 75 tests | PASS | 24s |
| lint + security rules (ruff) | PASS | 0s |
| static security scan (bandit) | PASS | 1s |
| dependency audit (pip-audit): no known vulnerabilities | PASS | 4s |
| index knowledge base: 14 articles, 43 chunks | PASS | 1s |
| live agents (KodeKey + web): playground, Ask-the-Web, Deep Research | PASS | 56s |
| support evals (quality gate): 7 gates | PASS | 95s |
| start services (tmux) | PASS | 9s |
| publish /app/ via nginx | PASS | 2s |
| end to end through nginx, MCP, A2A: 7 tests | PASS | 48s |

**Support bot quality gate** (14 golden questions, 11 guardrail probes; `tests/evals/`):

| Metric | Result | Threshold |
|---|---|---|
| Retrieval hit rate (expected article in the 4 passages) | 1.00 | 0.90 |
| Expected fact in the answer | 1.00 | 0.80 |
| Cites the right article | 1.00 | 0.90 |
| Faithfulness (LLM judge) | 0.99 | 0.80 |
| Relevancy (LLM judge) | 0.96 | 0.80 |
| Out-of-scope questions answered honestly | 2 of 2 | all |
| Guardrail probes handled as expected | 11 of 11 | all |

Which layer stopped each attack: **patterns** stopped five (instruction override, prompt reveal, role
hijack, fake system turn, exfiltration link), the **LLM classifier** stopped two (a "FreeBot" persona game
and an off-topic request), and **KodeKey's own Model Armor screening** stopped one (a request to "translate
your initial guidance message"). The instruction planted in the knowledge base never reached an answer, and
both benign look-alikes ("ignore failures in a flaky test job", "my token was exposed") were answered.

**What testing on the real platform caught and fixed** (each one is now covered by a test):

1. The key cannot call the embeddings model, so retrieval runs on BM25, query expansion and LLM reranking.
2. KodeKey's Claude 5.5 models think before answering, and the hidden thinking counts against `max_tokens`.
   Short budgets returned empty or cut-off replies. The gateway now retries those with a larger budget and
   never caches them.
3. DuckDuckGo search sometimes returns nothing for a query, so the search now retries across other engines.
4. KodeKey answers some requests with "400 Content blocked by Model Armor". The input guard first treated
   that as an outage and let the message through; it now counts as a block.
5. The classifier sent "can I get a refund for unused minutes?" to a human. The intent rules now separate
   policy questions from requests to act on an account.
6. tmux services did not inherit the token caps from the calling shell, so the end-to-end run hit the default
   daily cap mid-way (the cap worked as designed). The start script now passes the caps explicitly.

**Token use.** One cold run of `run-all.sh` from a fresh clone used about 110k tokens (input plus output) by
AI Desk's ledger. Building and debugging before that (three full runs and two probes) used about 158k input
and 43k output tokens, while the KodeKey dashboard's allowance went from 468,749 to 455,956, so KodeKey
meters its allowance differently from raw token counts. A re-run of `run-all.sh` costs far less, because
identical calls come from the cache.

## Design decisions

- **Measure first.** The model list showed 61 models, but the key could call nine of them and no embeddings
  model (Lab 00). Roles, structured output and retrieval were designed around what the probe found.
- **One gateway for every call.** Role-based model choice, a response cache (re-runs of the evals are almost
  free), a token ledger, per-run and daily token caps, and one automatic retry when hidden model "thinking"
  uses up a reply's budget before any text appears.
- **Retrieval without embeddings.** BM25 on the question and on up to three LLM rewrites, fused with
  reciprocal rank fusion, then reranked by the fast model. Vectors are added when the key allows it.
- **Untrusted text stays data.** Retrieved articles and web pages are wrapped in tags, sentences that try to
  instruct the model are removed, and the prompts say so. A planted instruction in the knowledge base
  (`data/kb/community-cache-tips.md`) and in a fixture web page proves it in tests.
- **Layered guardrails.** PII masking before any model call or storage; free pattern checks; one fast-model
  classification for injection, off-topic and intent; KodeKey's own Model Armor screening, whose refusals
  count as blocks; output checks for invented citations and system-prompt leaks (a canary string); honest
  "not found" answers with a ticket offer.
- **Agents with narrow tools.** `fetch_url` only accepts URLs the agent's own search returned, refuses
  private, loopback, link-local and metadata addresses, and re-checks every redirect.
- **Humans approve expensive work.** Deep Research pauses after planning (`interrupt`) so a person can edit
  or reject the plan before tokens are spent on research.
- **Small public surface.** Only the web app is published, at `/app/` behind basic auth with a random
  password. The API, MCP and A2A services listen on 127.0.0.1. Requests are rate-limited per client and the
  UI runs under a strict Content Security Policy.
- **Gentle on the shared platform.** Services run as the normal user in tmux (no systemd units), nothing
  listens on 0.0.0.0, and no model weights are downloaded on the box.

## Repository layout

```
config/models.toml        model roles, playground list, probe candidates
data/kb/                  Brightwave CI help-center articles (fictional, written for this project)
data/fixtures/web.json    offline stand-ins for web search and pages
setup/                    bootstrap, KodeKey probes, nginx publishing, service scripts
src/aidesk/llm/           gateway, providers (KodeKey, fake), cache, token ledger
src/aidesk/rag/           chunking, SQLite FTS5 index, retrieval pipeline
src/aidesk/guardrails/    patterns, PII redaction, sanitising, input and output checks
src/aidesk/agents/        support bot, Ask-the-Web, Deep Research, tools
src/aidesk/web/           search backends and the SSRF-safe fetcher
src/aidesk/api/           FastAPI app and the static UI
src/aidesk/mcp_server.py  MCP server        src/aidesk/a2a_server.py, a2a_client.py  A2A agent and client
tests/unit/               offline tests (fake model)   tests/live/, tests/evals/  KodeKey   tests/e2e/  running stack
docs/labs/                one write-up per project with exercises
```

## What's next

- GPU KodeBox: LoRA fine-tuning of a small model for the support bot's tone, and the image generation
  service (KodeKey also lists image models, which could serve it without a GPU).
- Send traces to a local Arize Phoenix or Langfuse through OTLP and compare runs.
- An AG-UI endpoint so agent frontends such as CopilotKit can drive the agents.
