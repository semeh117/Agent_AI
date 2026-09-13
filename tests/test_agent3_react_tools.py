"""Unit tests for Agent 3's run-scoped ReAct tools."""

from __future__ import annotations

import json

from next_chapter.agents.tools.agent3_react_tools import (
    Agent3RunContext,
    build_agent3_react_tools,
)
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo


def _cv() -> Agent2CVInfo:
    return Agent2CVInfo(
        full_name="Tool Test",
        skills=["Python"],
        job_titles=["Backend Engineer"],
        experience_years=3.0,
        education=["BSc"],
        highest_education_level="Bachelor",
    )


def _by_name(context: Agent3RunContext):
    return {tool.name: tool for tool in build_agent3_react_tools(context)}


def test_evaluation_and_gap_tools_require_prior_state():
    tools = _by_name(Agent3RunContext(_cv()))

    evaluation = json.loads(tools["evaluate_linkedin_results"].invoke("none"))
    gaps = json.loads(tools["analyze_skill_gaps"].invoke("none"))

    assert "search_linkedin_jobs first" in evaluation["error"]
    assert "evaluate_linkedin_results first" in gaps["error"]


def test_search_tool_enforces_two_search_limit(monkeypatch):
    import next_chapter.search.linkedin as linkedin_search

    monkeypatch.setattr(
        linkedin_search,
        "search_jobs",
        lambda **kwargs: [
            {
                "title": "Backend Engineer",
                "company": "Example",
                "url": f"https://example.com/{kwargs['query'].replace(' ', '-')}",
                "description": "Python backend services.",
            }
        ],
    )
    context = Agent3RunContext(_cv(), max_searches=2)
    search = _by_name(context)["search_linkedin_jobs"]

    first = json.loads(search.invoke("Backend Engineer Python"))
    second = json.loads(search.invoke('{"query": "Python API Engineer"}'))
    third = json.loads(search.invoke("Software Engineer"))

    assert first["search_number"] == 1
    assert second["search_number"] == 2
    assert second["query"] == "Python API Engineer"
    assert "maximum of 2" in third["error"]
    assert context.search_count == 2


def test_skill_gap_tool_aggregates_recurring_skills():
    context = Agent3RunContext(_cv())
    context.match_result = {
        "ranked_jobs": [
            {
                "skills_detail": {
                    "matching": [{"job_skill": "Python"}],
                    "missing": ["Docker", "AWS"],
                }
            },
            {
                "skills_detail": {
                    "matching": [{"job_skill": "Python"}],
                    "missing": ["Docker"],
                }
            },
        ]
    }
    gaps = _by_name(context)["analyze_skill_gaps"]

    result = json.loads(gaps.invoke("none"))

    assert result["jobs_analyzed"] == 2
    assert result["recurring_missing_skills"] == [
        {"skill": "Docker", "jobs": 2}
    ]
    assert result["strongest_existing_skills"] == [
        {"skill": "Python", "jobs": 2}
    ]
