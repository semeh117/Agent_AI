"""Agent 2 graph tests plus the interactive end-to-end demo.

Pytest runs only the injected offline workflow checks. Running this file
directly starts the real CV -> LinkedIn -> cover letter -> delivery workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

import agent.agent2 as agent2_workflow
import agent.agent2_interview as agent2_interview
from core.agent2_cv_parser import Agent2CVInfo
from storage.agent2_checkpointer import Agent2SqliteSaver


def _fake_cv() -> Agent2CVInfo:
    return Agent2CVInfo(
        full_name="Test Candidate",
        skills=["Python", "LangGraph", "RAG"],
        job_titles=["AI Engineer"],
        experience_years=4.0,
        education=["B.S. Computer Science"],
        highest_education_level="Bachelor",
        mail="candidate@example.com",
    )


def _fake_ranked_result(**_kwargs):
    job = {
        "job_title": "AI Engineer",
        "company": "Example AI",
        "url": "https://example.com/job/1",
        "description": "Build production AI systems with Python and LangGraph.",
        "skills_score": 80.0,
        "experience_score": 100.0,
        "education_score": 100.0,
        "final_score": 90.0,
        "score_percent": 90.0,
        "skills_detail": {"matching": [], "missing": []},
        "inconclusive": False,
    }
    return {
        "ranked_jobs": [job],
        "skipped_count": 0,
        "skipped_jobs": [],
    }


class _FakeStructuredQuery:
    def invoke(self, _prompt):
        return {"query": "AI Engineer Python LangGraph RAG"}


class _FakeAgentLLM:
    def with_structured_output(self, _schema):
        return _FakeStructuredQuery()


def _inject_workflow_fakes(monkeypatch):
    import pipeline.cover_letter as cover_letter_pipeline
    import pipeline.send_results_email as email_pipeline
    import pipeline.send_results_telegram as telegram_pipeline
    import services.application_tracker as tracker_service

    monkeypatch.setattr(
        agent2_workflow,
        "get_agent_llm",
        lambda **_kwargs: _FakeAgentLLM(),
    )
    monkeypatch.setattr(agent2_workflow, "match_linkedin_jobs", _fake_ranked_result)
    monkeypatch.setattr(
        cover_letter_pipeline,
        "generate_cover_letter",
        lambda *_args, **_kwargs: "A grounded test cover letter.",
    )
    monkeypatch.setattr(
        email_pipeline,
        "create_results_draft",
        lambda *_args, **_kwargs: {"id": "draft-test-1"},
    )
    monkeypatch.setattr(
        telegram_pipeline,
        "create_results_telegram",
        lambda *_args, **_kwargs: {"messages": [{"ok": True}]},
    )
    monkeypatch.setattr(
        tracker_service,
        "save_application",
        lambda _candidate, job, **_kwargs: SimpleNamespace(
            application_id="application-test-1",
            candidate_id="candidate-test-1",
            job_id="job-test-1",
            url=job["url"],
        ),
    )


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
# Agent 2 on-demand interview graph (agent/agent2_interview.py)
# ---------------------------------------------------------------------------


class _FakeStructuredInterview:
    """Deterministic stand-in for Groq's structured interview output."""

    def invoke(self, _prompt):
        def question(number: int, category: str) -> dict:
            return {
                "question": f"{category} interview question {number}?",
                "why_asked": "This checks relevant role knowledge.",
                "answer_strategy": "Explain the approach and connect it to CV evidence.",
                "sample_answer": (
                    "I would begin with the Python project stated in my CV and "
                    "explain my decisions honestly."
                ),
                "cv_evidence": ["Python"],
            }

        return {
            "role_summary": (
                "This AI Engineer role focuses on grounded production systems "
                "and reliable Python delivery."
            ),
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
    from services.application_tracker import save_application

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
    from services.interview_preparation import list_interview_preparations

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
    from services.application_tracker import save_application

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


class _ManualMonkeyPatch:
    """Tiny pytest-free patch helper used only by ``--offline``.

    Like pytest's fixture, it restores every patched attribute in ``undo`` so
    fakes injected for the search graph (for example the tracker's
    ``save_application``) never leak into the interview-graph tests.
    """

    def __init__(self) -> None:
        self._originals: list[tuple[object, str, object]] = []

    def setattr(self, target, name, value):
        self._originals.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        while self._originals:
            target, name, original = self._originals.pop()
            setattr(target, name, original)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Agent 2's LangGraph workflow.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run injected graph tests without APIs, Chrome, or delivery.",
    )
    parser.add_argument(
        "--interview",
        metavar="APPLICATION_ID",
        help="Live: run the on-demand interview LangGraph for one saved application.",
    )
    args = parser.parse_args()
    if args.offline:
        return _run_offline_tests()
    if args.interview:
        result = agent2_interview.run_agent2_interview_preparation(args.interview)
        print(f"Workflow ID: {result['workflow_id']}")
        print(f"Completed nodes: {' -> '.join(result.get('completed_steps', []))}\n")
        print(result["output"])
        return 0 if result.get("status") == "completed" else 1

    cv_folder = PROJECT_ROOT / "cv"
    pdf_files = sorted(cv_folder.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No CV PDF files were found in '{cv_folder}'.")

    print("Available CVs:")
    for index, path in enumerate(pdf_files, start=1):
        print(f"  {index}. {path.name}")

    choice = input("\nWhich CV? (enter number): ").strip()
    selected_index = int(choice) - 1
    if selected_index < 0 or selected_index >= len(pdf_files):
        raise ValueError("The selected CV number is out of range.")

    selected_path = pdf_files[selected_index]
    location = input("LinkedIn location (leave empty for any location): ").strip()

    print("\n=== FULL LINKEDIN AGENT 2 LANGGRAPH WORKFLOW ===")
    result = agent2_workflow.run_agent2_full_auto_from_pdf(
        str(selected_path),
        results_count=3,
        location=location,
    )
    cv_info = result["cv_info"]
    print(
        f"\nCandidate: {cv_info.full_name}, "
        f"{len(cv_info.skills)} skills, "
        f"{cv_info.experience_years} years of experience, "
        f"education={cv_info.highest_education_level or 'not found'}"
    )
    print(f"CV extraction: {result['cv_extraction_backend']}")
    print(f"Workflow ID: {result['workflow_id']}")
    print(f"Completed nodes: {' -> '.join(result.get('completed_steps', []))}\n")
    for warning in result.get("cv_extraction_warnings", []):
        print(f"CV extraction warning: {warning}")
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
