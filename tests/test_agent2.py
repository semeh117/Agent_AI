"""Offline Agent 2 regression tests. Live demo: python -m examples.run_agent2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

import next_chapter.agents.agent2 as agent2_workflow
import next_chapter.agents.agent2_interview as agent2_interview
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo
from tests.support.agent2_fakes import _fake_cv, _fake_ranked_result, _FakeAgentLLM, _inject_workflow_fakes, _ManualMonkeyPatch
from next_chapter.storage.agent2_checkpointer import Agent2SqliteSaver


def test_agent2_langgraph_has_explicit_nodes():
    nodes = set(agent2_workflow._get_agent2_graph().nodes)
    assert {
        "load_cv",
        "build_query",
        "match_jobs",
        "persist_recommendations",
        "generate_cover_letter",
        "choose_delivery",
        "deliver",
        "finalize",
    }.issubset(nodes)


def test_agent2_langgraph_completes_with_supplied_channel(monkeypatch):
    _inject_workflow_fakes(monkeypatch)
    result = agent2_workflow.run_agent2_full_auto(
        _fake_cv(),
        delivery_channel="telegram",
        interactive_delivery=False,
    )

    assert result["status"] == "completed"
    assert result["delivery"]["channel"] == "telegram"
    assert result["delivery"]["messages_sent"] == 1
    assert result["ranked_jobs"][0]["final_score"] == 90.0
    assert result["completed_steps"] == [
        "cv_parsed",
        "query_built",
        "jobs_ranked",
        "recommendations_saved",
        "cover_letter_generated",
        "delivery_selected",
        "results_delivered",
        "workflow_finalized",
    ]


def test_agent2_langgraph_pauses_and_resumes_for_streamlit(monkeypatch):
    _inject_workflow_fakes(monkeypatch)
    started = agent2_workflow.run_agent2_full_auto(
        _fake_cv(),
        interactive_delivery=False,
    )

    assert started["status"] == "awaiting_delivery"
    assert started["interrupt"]["type"] == "delivery_choice"

    resumed = agent2_workflow.resume_agent2_workflow(
        started["workflow_id"],
        "gmail",
    )
    assert resumed["status"] == "completed"
    assert resumed["delivery"] == {
        "channel": "gmail",
        "status": "completed",
        "draft_id": "draft-test-1",
    }


def test_agent2_langgraph_resumes_after_graph_restart(monkeypatch):
    """A new graph object can resume a pause persisted by an older one."""

    _inject_workflow_fakes(monkeypatch)
    workflow_id = "persistent-workflow-test"
    config = {"configurable": {"thread_id": workflow_id}}
    initial_state = {
        "workflow_id": workflow_id,
        "cv_info": _fake_cv().model_dump(),
        "location": "",
        "results_count": 1,
        "use_cache": True,
        "delivery_channel": "",
        "completed_steps": [],
        "warnings": [],
        "error": None,
        "status": "started",
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent2.sqlite3"
        first_graph = agent2_workflow._build_agent2_graph(
            Agent2SqliteSaver(database_path)
        )
        first_graph.invoke(initial_state, config=config)
        assert any(
            getattr(task, "interrupts", ())
            for task in first_graph.get_state(config).tasks
        )

        # Rebuild both saver and graph to simulate a Streamlit/app restart.
        second_graph = agent2_workflow._build_agent2_graph(
            Agent2SqliteSaver(database_path)
        )
        resumed = second_graph.invoke(Command(resume="telegram"), config=config)
        assert resumed["status"] == "completed"
        assert resumed["delivery"]["channel"] == "telegram"


# ---------------------------------------------------------------------------
# Agent 2 on-demand interview graph (src/next_chapter/agents/agent2_interview.py)
# ---------------------------------------------------------------------------


class _FakeStructuredInterview:
    """Deterministic stand-in for Groq's structured interview output."""

    def invoke(self, _prompt):
        def question(number: int, category: str) -> dict:
            return {
                "question": f"{category} interview question {number}?",
                "competency": f"{category} competency {number}",
                "why_asked": "This checks relevant role knowledge.",
                "job_connection": "The posting requires reliable production delivery.",
                "answer_strategy": "Explain the approach and connect it to CV evidence.",
                "sample_answer": (
                    "I would begin with the Python project stated in my CV and "
                    "explain my decisions honestly."
                ),
                "cv_evidence": ["Python"],
                "learning_plan": (
                    "Build a small practice project and request feedback."
                    if category == "Gap"
                    else ""
                ),
            }

        return {
            "role_summary": (
                "This AI Engineer role focuses on grounded production systems "
                "and reliable Python delivery."
            ),
            "role_family": "Machine Learning Engineering",
            "primary_focus": "Reliable production AI systems",
            "key_competencies": ["Python", "model evaluation", "reliability"],
            "candidate_strengths": ["Python"],
            "primary_gaps": ["Kubernetes"],
            "technical_questions": [question(i, "Technical") for i in range(1, 6)],
            "gap_questions": [question(i, "Gap") for i in range(1, 3)],
            "behavioral_questions": [question(i, "Behavioral") for i in range(1, 4)],
            "questions_to_ask": [
                "How do you evaluate model quality?",
                "How is the AI team organized?",
                "What does success look like in ninety days?",
                "How are production incidents handled?",
            ],
            "preparation_checklist": [
                "Review the job description.",
                "Prepare two project examples.",
                "Practice concise technical explanations.",
                "Review the missing skills honestly.",
                "Prepare questions for the interviewer.",
            ],
        }


class _FakeInterviewLLM:
    calls = 0

    def with_structured_output(self, _schema, **kwargs):
        assert kwargs == {"method": "json_schema", "strict": True}
        _FakeInterviewLLM.calls += 1
        return _FakeStructuredInterview()


def _save_interview_fixture_application(database_path: Path) -> str:
    from next_chapter.services.application_tracker import save_application

    candidate = {
        **_fake_cv().model_dump(),
        "skill_evidence": {"Python": "Production AI project"},
    }
    job = {
        **_fake_ranked_result()["ranked_jobs"][0],
        "skills_detail": {
            "matching": [{"job_skill": "Python", "matched_via": "Python"}],
            "missing": ["Kubernetes"],
        },
    }
    return save_application(candidate, job, database_path=database_path).application_id


def test_agent2_interview_graph_has_explicit_nodes():
    with tempfile.TemporaryDirectory() as temporary_directory:
        graph = agent2_interview._build_agent2_interview_graph(
            Agent2SqliteSaver(Path(temporary_directory) / "agent2.sqlite3")
        )
    assert {
        "load_application",
        "validate_application",
        "generate_interview_content",
        "render_and_persist_pdf",
        "finalize",
    }.issubset(set(graph.nodes))


def test_agent2_interview_graph_completes_and_returns_pdf():
    from next_chapter.services.interview_preparation import list_interview_preparations

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        database_path = root / "agent2.sqlite3"
        application_id = _save_interview_fixture_application(database_path)
        _FakeInterviewLLM.calls = 0

        result = agent2_interview.run_agent2_interview_preparation(
            application_id,
            workflow_id="interview-test-1",
            database_path=database_path,
            output_directory=root / "pdf",
            llm=_FakeInterviewLLM(),
        )

        assert result["status"] == "completed"
        assert result["error"] is None
        assert result["workflow_id"] == "interview-test-1"
        # Records every node in order.
        assert result["completed_steps"] == [
            "application_loaded",
            "application_validated",
            "interview_content_generated",
            "pdf_rendered_and_persisted",
            "workflow_finalized",
        ]
        # Returns an absolute PDF path that really exists.
        pdf_path = Path(result["pdf_path"])
        assert pdf_path.is_absolute() and pdf_path.exists()
        assert pdf_path.stat().st_size > 1_000
        assert result["preparation_id"]
        assert result["provider"] and result["model"]
        assert "Interview preparation generated successfully." in result["output"]
        assert "AI Engineer @ Example AI" in result["output"]
        # Exactly one generation call, no CV re-parsing needed.
        assert _FakeInterviewLLM.calls == 1
        assert len(result["interview_content"]["technical_questions"]) == 5
        # Persisted through the shared service, so Agent 3 / Streamlit see it.
        stored = list_interview_preparations(application_id, database_path=database_path)
        assert [record.preparation_id for record in stored] == [result["preparation_id"]]
        assert stored[0].pdf_path == result["pdf_path"]


def test_agent2_interview_graph_handles_missing_application():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        _FakeInterviewLLM.calls = 0
        result = agent2_interview.run_agent2_interview_preparation(
            "does-not-exist",
            database_path=root / "agent2.sqlite3",
            output_directory=root / "pdf",
            llm=_FakeInterviewLLM(),
        )

        assert result["status"] == "failed"
        assert result["error"].startswith("load_application failed:")
        assert "was not found" in result["error"]
        assert result["completed_steps"] == [
            "load_application_failed",
            "workflow_finalized",
        ]
        assert "Interview preparation failed." in result["output"]
        assert "pdf_path" not in result
        # No LLM tokens are spent on a missing application.
        assert _FakeInterviewLLM.calls == 0


def test_agent2_interview_graph_rejects_incomplete_application():
    from next_chapter.services.application_tracker import save_application

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        database_path = root / "agent2.sqlite3"
        # No description -> validation must stop before generation.
        job = {**_fake_ranked_result()["ranked_jobs"][0], "description": ""}
        application_id = save_application(
            _fake_cv().model_dump(), job, database_path=database_path
        ).application_id
        _FakeInterviewLLM.calls = 0

        result = agent2_interview.run_agent2_interview_preparation(
            application_id,
            database_path=database_path,
            output_directory=root / "pdf",
            llm=_FakeInterviewLLM(),
        )

        assert result["status"] == "failed"
        assert result["error"].startswith("validate_application failed:")
        assert "job description is missing" in result["error"]
        assert result["completed_steps"] == [
            "application_loaded",
            "validate_application_failed",
            "workflow_finalized",
        ]
        assert _FakeInterviewLLM.calls == 0


def _run_offline_tests() -> int:
    patcher = _ManualMonkeyPatch()
    try:
        test_agent2_langgraph_has_explicit_nodes()
        test_agent2_langgraph_completes_with_supplied_channel(patcher)
        test_agent2_langgraph_pauses_and_resumes_for_streamlit(patcher)
        test_agent2_langgraph_resumes_after_graph_restart(patcher)
    finally:
        patcher.undo()
    print("Agent 2 LangGraph offline tests: PASS (4/4)")
    test_agent2_interview_graph_has_explicit_nodes()
    test_agent2_interview_graph_completes_and_returns_pdf()
    test_agent2_interview_graph_handles_missing_application()
    test_agent2_interview_graph_rejects_incomplete_application()
    print("Agent 2 interview LangGraph offline tests: PASS (4/4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_offline_tests())
