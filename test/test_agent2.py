"""Agent 2 graph tests plus the interactive end-to-end demo.

Pytest runs only the injected offline workflow checks. Running this file
directly starts the real CV -> LinkedIn -> cover letter -> delivery workflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import agent.agent2 as agent2_workflow
from core.agent2_cv_parser import Agent2CVInfo


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

    monkeypatch.setattr(agent2_workflow, "get_agent_llm", lambda **_kwargs: _FakeAgentLLM())
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


def test_agent2_langgraph_has_explicit_nodes():
    nodes = set(agent2_workflow._get_agent2_graph().nodes)
    assert {
        "load_cv",
        "build_query",
        "match_jobs",
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


class _ManualMonkeyPatch:
    """Tiny pytest-free patch helper used only by ``--offline``."""

    @staticmethod
    def setattr(target, name, value):
        setattr(target, name, value)


def _run_offline_tests() -> int:
    patcher = _ManualMonkeyPatch()
    test_agent2_langgraph_has_explicit_nodes()
    test_agent2_langgraph_completes_with_supplied_channel(patcher)
    test_agent2_langgraph_pauses_and_resumes_for_streamlit(patcher)
    print("Agent 2 LangGraph offline tests: PASS (3/3)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Agent 2's LangGraph workflow.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run injected graph tests without APIs, Chrome, or delivery.",
    )
    args = parser.parse_args()
    if args.offline:
        return _run_offline_tests()

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
