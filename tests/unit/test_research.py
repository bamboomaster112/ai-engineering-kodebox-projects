from aidesk.agents import research


def setup_function():
    research.reset_graph()


def test_research_runs_end_to_end_with_citations():
    out = research.run("reciprocal rank fusion for hybrid search")
    assert out["event"] == "done", out
    assert out["citations_ok"], out["report"]
    assert "## Sources" in out["report"]
    assert out["sources"] and all(s["url"].startswith("https://notes.example.org/") for s in out["sources"])
    steps = [t["step"] for t in out["trace"]]
    assert steps[0] == "plan" and steps[-1] == "verify"
    # every model call, including those inside parallel researcher branches, is booked to this run
    from aidesk.config import get_settings
    from aidesk.db import connect

    with connect(get_settings().db_path) as conn:
        rows = conn.execute("SELECT run_id, purpose FROM usage WHERE purpose != 'probe'").fetchall()
    assert rows and all(r["run_id"] == out["run_id"] for r in rows), [tuple(r) for r in rows]
    assert steps.count("research") == len(out["plan"])


def test_plan_waits_for_approval_then_resumes_with_edits():
    events = list(research.start("prompt injection risks for AI agents"))
    assert events[0]["event"] == "started"
    assert events[-1]["event"] == "awaiting_approval"
    run_id = events[0]["run_id"]
    plan = events[-1]["plan"]
    assert 1 <= len(plan) <= research.MAX_SUBQUESTIONS

    edited = ["What is indirect prompt injection?"]
    resumed = list(research.resume(run_id, plan=edited))
    done = resumed[-1]
    assert done["event"] == "done" and done["plan"] == edited
    assert sum(1 for e in resumed if e["event"] == "step" and e["node"] == "research") == 1


def test_rejected_plan_ends_without_research():
    events = list(research.start("a2a agent protocols"))
    run_id = events[0]["run_id"]
    resumed = list(research.resume(run_id, approved=False))
    assert resumed[-1]["event"] == "done"
    assert resumed[-1]["report"] == ""


def test_resume_unknown_run_is_an_error():
    assert list(research.resume("research-nope"))[-1]["event"] == "error"


def test_verify_strips_invented_citations():
    state = {"sources": [{"n": 1, "url": "https://a.example", "title": "A"}], "report": "Fact [1]. Made up [7]."}
    out = research.verify_node(state)
    assert "[7]" not in out["report"] and out["citations_ok"] is False
