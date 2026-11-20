# Lab 04: Deep Research with sub-agents and citations (ByteByteAI project 4)

**Goal:** turn a broad question into a short, cited report, with a person approving the plan first.

## The graph (`src/aidesk/agents/research.py`)
```
plan ─► approve (interrupt) ─► research × N in parallel (Send) ─► next_round ─► collect ─► critic
                                     ▲                                                     │
                                     └──────────── gaps (one extra round at most) ◄────────┘
critic (enough) ─► write ─► verify ─► report
```
- **plan**: the reasoning model (claude-opus-5-5) proposes two to four sub-questions.
- **approve**: `interrupt()` pauses the run and the API returns the plan. The UI lets you edit it;
  `Command(resume=...)` continues with your version. The A2A agent and tests auto-approve.
- **research**: one researcher per sub-question, in parallel through LangGraph `Send`. Each searches, reads
  two pages through the same safe fetcher, and writes cited notes with the fast model.
- **collect**: gives every distinct URL one global number and rewrites each researcher's local citations.
- **critic**: the main model decides whether the notes cover the question; if not, up to two gap questions
  get one more research round.
- **write**: the reasoning model writes the report from the notes only.
- **verify**: removes citations that do not match a collected source and appends the source list.
- Every model call in a run, including the parallel branches, is booked to the run's id, and the run
  stops if it would pass its token cap (45,000 by default).

## Try it
UI: **Deep Research** tab → **Make a plan** → edit the lines → **Approve and run**.
CLI (plan auto-approved): `uv run aidesk research "How do teams evaluate RAG systems in production?"`

## Verify
Offline: `tests/unit/test_research.py` (end to end, approval with edits, rejection, invented citations).
Live: `tests/live/test_live_agents.py::test_deep_research_produces_a_cited_report`.

## Exercises
1. Let the critic choose which earlier sub-question to deepen instead of adding new ones.
2. Store finished reports in SQLite and list them in the UI.
