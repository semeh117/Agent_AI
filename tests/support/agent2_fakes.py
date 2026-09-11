"""Deterministic services used by workflow and Streamlit regression tests."""
from types import SimpleNamespace
import next_chapter.agents.agent2 as agent2_workflow
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo


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
    import next_chapter.services.cover_letter as cover_letter_pipeline
    import next_chapter.delivery.gmail as email_pipeline
    import next_chapter.delivery.telegram as telegram_pipeline
    import next_chapter.services.application_tracker as tracker_service

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
