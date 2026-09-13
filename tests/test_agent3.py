"""Offline tests for Agent 3's classic ReAct orchestration."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake import FakeListLLM
import pytest

import next_chapter.agents.agent3 as agent3
import next_chapter.agents.tools.cover_letter as cover_letter_tool
import next_chapter.agents.tools.agent3_react_tools as react_tools
import next_chapter.delivery.telegram as telegram_delivery
from next_chapter.agents.react_output_parser import RequiredToolsVerifyingParser
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo


def _cv() -> Agent2CVInfo:
    return Agent2CVInfo(
        full_name="Test Candidate",
        skills=["Python", "LangGraph", "RAG"],
        job_titles=["AI Engineer"],
        experience_years=4.0,
        education=["B.S. Computer Science"],
        highest_education_level="Bachelor",
        mail="candidate@example.com",
    )


def _ranked_jobs() -> list[dict[str, Any]]:
    return [
        {
            "job_title": "AI Engineer",
            "company": "Example AI",
            "url": "https://example.com/job/1",
            "description": "Build AI systems with Python.",
            "skills_score": 80.0,
            "experience_score": 100.0,
            "education_score": 100.0,
            "final_score": 90.0,
            "score_percent": 90.0,
            "skills_detail": {
                "matching": [{"job_skill": "Python", "matched_via": "Python"}],
                "missing": ["Docker", "AWS"],
            },
            "inconclusive": False,
        },
        {
            "job_title": "ML Engineer",
            "company": "Example ML",
            "url": "https://example.com/job/2",
            "description": "Deploy ML services.",
            "skills_score": 60.0,
            "experience_score": 100.0,
            "education_score": 100.0,
            "final_score": 80.0,
            "score_percent": 80.0,
            "skills_detail": {
                "matching": [{"job_skill": "Python", "matched_via": "Python"}],
                "missing": ["Docker"],
            },
            "inconclusive": False,
        },
    ]


def _responses() -> list[str]:
    return [
        (
            "I need relevant LinkedIn postings.\n"
            "Action: search_linkedin_jobs\n"
            "Action Input: AI Engineer Python LangGraph"
        ),
        (
            "The titles are relevant, so I should evaluate them.\n"
            "Action: evaluate_linkedin_results\n"
            "Action Input: none"
        ),
        (
            "I need evidence about recurring skill gaps.\n"
            "Action: analyze_skill_gaps\n"
            "Action Input: none"
        ),
        (
            "The first ranked job is the valid top match.\n"
            "Action: write_cover_letter_for_job\n"
            "Action Input: https://example.com/job/1"
        ),
        (
            "I now have the observed results.\n"
            "Final Answer: Two jobs were ranked and the top job received a cover letter."
        ),
    ]


def _install_fakes(monkeypatch):
    calls = {"search": 0, "match": 0, "cover": 0, "telegram": 0}

    def fake_search(**_kwargs):
        calls["search"] += 1
        return [
            {
                "title": "AI Engineer",
                "company": "Example AI",
                "url": "https://example.com/job/1",
                "description": "Build AI systems with Python.",
            },
            {
                "title": "ML Engineer",
                "company": "Example ML",
                "url": "https://example.com/job/2",
                "description": "Deploy ML services.",
            },
        ]

    def fake_match(**_kwargs):
        calls["match"] += 1
        return {
            "ranked_jobs": _ranked_jobs(),
            "skipped_count": 0,
            "skipped_jobs": [],
        }

    def fake_persist(_cv_info, jobs):
        return {
            "candidate_id": "candidate-1",
            "tracked_applications": [
                {"application_id": f"application-{index}", "url": job["url"]}
                for index, job in enumerate(jobs, start=1)
            ],
        }

    def fake_cover(*_args, **_kwargs):
        calls["cover"] += 1
        return "Grounded cover letter."

    def fake_telegram(*_args, **_kwargs):
        calls["telegram"] += 1
        return {"messages": [{"ok": True}]}

    import next_chapter.search.linkedin as linkedin_search

    monkeypatch.setattr(linkedin_search, "search_jobs", fake_search)
    monkeypatch.setattr(react_tools, "match_linkedin_jobs", fake_match)
    monkeypatch.setattr(react_tools, "persist_ranked_jobs", fake_persist)
    monkeypatch.setattr(cover_letter_tool, "generate_cover_letter", fake_cover)
    monkeypatch.setattr(
        telegram_delivery,
        "create_results_telegram",
        fake_telegram,
    )
    return calls


def test_agent3_uses_classic_react_prompt_and_tools(monkeypatch):
    monkeypatch.setattr(
        agent3,
        "get_agent3_llm",
        lambda **_kwargs: FakeListLLM(responses=_responses()),
    )
    context = react_tools.Agent3RunContext(_cv())
    executor = agent3.build_agent3_executor(context, verbose=False)

    assert agent3.build_agent3_prompt().input_variables == [
        "agent_scratchpad",
        "input",
        "tool_names",
        "tools",
    ]
    assert tuple(tool.name for tool in executor.tools) == agent3.AGENT3_TOOL_NAMES
    assert "Thought:" in agent3.AGENT3_REACT_PROMPT
    assert "Action:" in agent3.AGENT3_REACT_PROMPT
    assert "Observation:" in agent3.AGENT3_REACT_PROMPT
    assert "Action: one tool from" not in agent3.AGENT3_REACT_PROMPT


def test_agent3_corrects_observed_linkedin_tool_misspelling():
    parser = RequiredToolsVerifyingParser(
        required_tools=set(),
        only_if_any=set(),
        tool_aliases={"search_linkinned_jobs": "search_linkedin_jobs"},
    )

    action = parser.parse(
        "I need relevant jobs.\n"
        "Action: search_linkinned_jobs\n"
        "Action Input: Data Scientist Python XGBoost"
    )

    assert action.tool == "search_linkedin_jobs"
    assert action.tool_input == "Data Scientist Python XGBoost"
    assert "Action: search_linkedin_jobs" in action.log
    assert "search_linkinned_jobs" not in action.log


def test_agent3_runs_real_action_observation_sequence(monkeypatch):
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(
        agent3,
        "get_agent3_llm",
        lambda **_kwargs: FakeListLLM(responses=_responses()),
    )

    result = agent3.run_agent3_full_auto(
        _cv(), delivery_channel="telegram", verbose=False
    )

    assert [step["tool"] for step in result["react_trace"]] == list(
        agent3.AGENT3_TOOL_NAMES
    )
    assert result["react_validation"]["status"] == "passed"
    assert result["status"] == "completed"
    assert result["delivery"]["channel"] == "telegram"
    assert result["delivery"]["status"] == "completed"
    assert result["ranked_jobs"][0]["url"] == "https://example.com/job/1"
    assert result["skill_gap_analysis"]["recurring_missing_skills"] == [
        {"skill": "Docker", "jobs": 2}
    ]
    assert result["cover_letter"] == "Grounded cover letter."
    assert calls == {"search": 1, "match": 1, "cover": 1, "telegram": 1}


def test_agent3_does_not_repair_a_skipped_react_action(monkeypatch):
    calls = _install_fakes(monkeypatch)
    responses = [
        _responses()[0],
        _responses()[1],
        "I am finished.\nFinal Answer: Jobs were ranked.",
        "I am still finished.\nFinal Answer: Jobs were ranked.",
        "Final Answer: I could not complete every required action.",
    ]
    monkeypatch.setattr(
        agent3,
        "get_agent3_llm",
        lambda **_kwargs: FakeListLLM(responses=responses),
    )

    result = agent3.run_agent3_full_auto(
        _cv(), delivery_channel="telegram", verbose=False
    )

    assert result["react_validation"]["status"] == "failed"
    assert result["status"] == "incomplete"
    assert result["delivery"]["status"] == "skipped"
    assert calls["cover"] == 0
    assert calls["telegram"] == 0


def test_agent3_streamlit_delivery_uses_reviewed_result_without_rerun(monkeypatch):
    import next_chapter.delivery.gmail as gmail_delivery

    calls = []
    monkeypatch.setattr(
        gmail_delivery,
        "create_results_draft",
        lambda cv, jobs, letter, to_email: calls.append(
            (cv, jobs, letter, to_email)
        ) or {"id": "draft-agent3"},
    )
    result = {
        "workflow_type": "agent3",
        "workflow_id": "agent3-ui-test",
        "status": "awaiting_delivery",
        "delivery": {"status": "awaiting_choice"},
        "cv_info": _cv(),
        "ranked_jobs": _ranked_jobs(),
        "cover_letter": "Original letter",
    }

    delivered = agent3.deliver_agent3_result(
        result,
        "gmail",
        cover_letter="Reviewed letter",
    )

    assert delivered["status"] == "completed"
    assert delivered["cover_letter"] == "Reviewed letter"
    assert delivered["delivery"]["observation"].endswith("draft-agent3).")
    assert calls[0][2:] == ("Reviewed letter", "candidate@example.com")
    with pytest.raises(ValueError, match="already been delivered"):
        agent3.deliver_agent3_result(delivered, "gmail")
