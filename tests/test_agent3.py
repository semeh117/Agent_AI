"""Offline Agent 3 regression tests. Live demo: python -m examples.run_agent3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import next_chapter.agents.agent3 as agent3_workflow
import next_chapter.agents.tools.cover_letter as cover_letter_tool
import next_chapter.agents.tools.gmail as gmail_tool
import next_chapter.agents.tools.interview_preparation as interview_tool
import next_chapter.agents.tools.linkedin_match_tool as linkedin_tool
import next_chapter.agents.tools.telegram_tool as telegram_tool
from next_chapter.agents.agent3 import run_agent3_full_auto_from_pdf
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo


# ---------------------------------------------------------------------------
# Offline fakes
# ---------------------------------------------------------------------------


class _ScriptedToolCallingModel(BaseChatModel):
    """Chat model that replays scripted tool calls, then a final answer.

    ``create_tool_calling_agent`` only needs ``bind_tools`` and ``invoke``. Each
    scripted step is a list of ``(tool_name, args)`` pairs; an empty list means
    "answer without tools", which ends the AgentExecutor loop.
    """

    script: list[list[tuple[str, dict[str, Any]]]]
    final_answer: str = "Scripted final answer."
    seen_tool_names: list[str] = []
    step: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools, **_kwargs):
        self.seen_tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        calls = self.script[self.step] if self.step < len(self.script) else []
        self.step += 1
        if calls:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call-{self.step}-{index}"}
                    for index, (name, args) in enumerate(calls)
                ],
            )
        else:
            message = AIMessage(content=self.final_answer)
        return ChatResult(generations=[ChatGeneration(message=message)])


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


def _fake_ranked_jobs() -> list[dict[str, Any]]:
    def job(index: int, score: float) -> dict[str, Any]:
        return {
            "job_title": f"AI Engineer {index}",
            "company": f"Example AI {index}",
            "url": f"https://example.com/job/{index}",
            "description": "Build production AI systems with Python and LangGraph.",
            "skills_score": score - 10,
            "experience_score": 100.0,
            "education_score": 100.0,
            "final_score": score,
            "score_percent": score,
            "skills_detail": {
                "matching": [{"job_skill": "Python", "matched_via": "Python"}],
                "missing": ["Kubernetes"],
            },
            "inconclusive": False,
        }

    return [job(1, 90.0), job(2, 75.0), job(3, 60.0)]


def _fake_match_linkedin_jobs(**_kwargs) -> dict[str, Any]:
    return {
        "ranked_jobs": _fake_ranked_jobs(),
        "skipped_count": 0,
        "skipped_jobs": [],
    }


class _Patcher:
    """Minimal monkeypatch replacement that restores attributes afterwards."""

    def __init__(self) -> None:
        self._originals: list[tuple[object, str, object]] = []

    def setattr(self, target, name, value):
        self._originals.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        while self._originals:
            target, name, original = self._originals.pop()
            setattr(target, name, original)


class _OfflineAgent3Environment:
    """Temporary database plus fakes for every external dependency."""

    def __init__(self, model: _ScriptedToolCallingModel) -> None:
        self.model = model
        self.patcher = _Patcher()
        self.calls: dict[str, int] = {
            "match": 0,
            "cover_letter": 0,
            "gmail": 0,
            "telegram": 0,
            "interview": 0,
        }

    def __enter__(self):
        import next_chapter.services.interview_preparation as interview_service

        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self._previous_db = os.environ.get("AGENT2_DATABASE_PATH")
        os.environ["AGENT2_DATABASE_PATH"] = str(self.root / "agent2.sqlite3")

        def fake_match(**kwargs):
            self.calls["match"] += 1
            return _fake_match_linkedin_jobs(**kwargs)

        def fake_cover_letter(*_args, **_kwargs):
            self.calls["cover_letter"] += 1
            return "A grounded test cover letter."

        def fake_gmail(*_args, **_kwargs):
            self.calls["gmail"] += 1
            return {"id": "draft-test-1"}

        def fake_telegram(*_args, **_kwargs):
            self.calls["telegram"] += 1
            return {"messages": [{"ok": True}]}

        def fake_interview_content(_application, _profile, *, llm=None):
            self.calls["interview"] += 1
            return interview_service.InterviewPreparationContent.model_validate(
                _fake_interview_payload()
            )

        # The shared tools bind these names at import time, so the fakes are
        # installed on the tool modules themselves (not on the pipelines).
        self.patcher.setattr(agent3_workflow, "get_agent_llm", lambda **_k: self.model)
        self.patcher.setattr(linkedin_tool, "match_linkedin_jobs", fake_match)
        self.patcher.setattr(cover_letter_tool, "generate_cover_letter", fake_cover_letter)
        self.patcher.setattr(gmail_tool, "create_results_draft", fake_gmail)
        self.patcher.setattr(telegram_tool, "create_results_telegram", fake_telegram)
        self.patcher.setattr(
            interview_service, "generate_interview_content", fake_interview_content
        )
        self.patcher.setattr(
            interview_service,
            "DEFAULT_INTERVIEW_PDF_DIRECTORY",
            self.root / "pdf",
        )
        return self

    def __exit__(self, *_exc):
        self.patcher.undo()
        if self._previous_db is None:
            os.environ.pop("AGENT2_DATABASE_PATH", None)
        else:
            os.environ["AGENT2_DATABASE_PATH"] = self._previous_db
        self._directory.cleanup()
        return False


def _fake_interview_payload() -> dict[str, Any]:
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


def _full_auto_script() -> list[list[tuple[str, dict[str, Any]]]]:
    """The scripted model performs the five-tool workflow, then answers."""

    return [
        [("match_linkedin_jobs_for_agent", {"query": "AI Engineer Python LangGraph"})],
        [("write_cover_letter", {"url": "https://example.com/job/1"})],
        [("send_results_telegram", {"_unused_input": ""})],
        [],
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent3_exposes_interview_tool():
    names = [tool.name for tool in agent3_workflow.TOOLS]
    assert names == [
        "match_linkedin_jobs_for_agent",
        "write_cover_letter",
        "ask_user_delivery_channel",
        "send_results_draft",
        "send_results_telegram",
        "generate_interview_preparation_pdf",
    ]
    assert "explicitly asks" in agent3_workflow.generate_interview_preparation_pdf.description
    assert "user-triggered only" in agent3_workflow.AGENT3_SYSTEM_PROMPT


def test_agent3_full_auto_persists_ranking_without_interview():
    from next_chapter.services.application_tracker import list_applications

    model = _ScriptedToolCallingModel(script=_full_auto_script())
    with _OfflineAgent3Environment(model) as env:
        result = agent3_workflow.run_agent3_full_auto(
            _fake_cv(),
            results_count=3,
            delivery_channel="telegram",
        )

        # Existing behaviour intact: ranking, cover letter, delivery.
        assert model.seen_tool_names == [tool.name for tool in agent3_workflow.TOOLS]
        assert [job["final_score"] for job in result["ranked_jobs"]] == [90.0, 75.0, 60.0]
        assert result["cover_letter"] == "A grounded test cover letter."
        assert result["cover_letter_job"]["url"] == "https://example.com/job/1"
        assert result["delivery"] == {
            "channel": "telegram",
            "status": "completed",
            "observation": result["delivery"]["observation"],
        }
        assert env.calls == {
            "match": 1,
            "cover_letter": 1,
            "gmail": 0,
            "telegram": 1,
            "interview": 0,
        }
        called_tools = [action.tool for action, _ in result["intermediate_steps"]]
        assert "generate_interview_preparation_pdf" not in called_tools
        assert interview_tool.get_last_interview_preparation() is None

        # Ranking persisted deterministically in the shared Agent 2 tracker,
        # in ranking order, with URLs, scores and match details.
        tracked = result["tracked_applications"]
        assert [row["url"] for row in tracked] == [
            "https://example.com/job/1",
            "https://example.com/job/2",
            "https://example.com/job/3",
        ]
        assert result["candidate_id"]
        assert "persistence_warning" not in result
        stored = list_applications(result["candidate_id"])
        assert len(stored) == 3
        by_url = {record.url: record for record in stored}
        assert by_url["https://example.com/job/1"].final_score == 90.0
        assert by_url["https://example.com/job/1"].status == "discovered"
        assert by_url["https://example.com/job/1"].match_details["missing"] == ["Kubernetes"]
        assert by_url["https://example.com/job/1"].candidate_email == "candidate@example.com"

        # The observation the model saw stays compact and unchanged in order.
        match_observation = json.loads(result["intermediate_steps"][0][1])
        assert [job["final_score"] for job in match_observation["ranked_jobs"]] == [90.0, 75.0, 60.0]
        assert all("description" not in job for job in match_observation["ranked_jobs"])
        assert len(match_observation["tracked_applications"]) == 3

        # A second run deduplicates instead of inserting again.
        model.step = 0
        agent3_workflow.run_agent3_full_auto(
            _fake_cv(), results_count=3, delivery_channel="telegram"
        )
        assert len(list_applications(result["candidate_id"])) == 3


def test_agent3_persistence_failure_keeps_ranking():
    import next_chapter.services.application_tracker as tracker_service

    model = _ScriptedToolCallingModel(script=_full_auto_script())
    with _OfflineAgent3Environment(model) as env:

        def broken_save(*_args, **_kwargs):
            raise RuntimeError("disk unavailable")

        env.patcher.setattr(tracker_service, "save_application", broken_save)
        result = agent3_workflow.run_agent3_full_auto(
            _fake_cv(), results_count=3, delivery_channel="telegram"
        )
        assert [job["final_score"] for job in result["ranked_jobs"]] == [90.0, 75.0, 60.0]
        assert result["tracked_applications"] == []
        assert "disk unavailable" in result["persistence_warning"]
        assert result["delivery"]["status"] == "completed"


def test_agent3_interview_tool_calls_shared_service():
    from next_chapter.services.application_tracker import save_application
    from next_chapter.services.interview_preparation import list_interview_preparations

    model = _ScriptedToolCallingModel(script=[])
    with _OfflineAgent3Environment(model) as env:
        application = save_application(_fake_cv(), _fake_ranked_jobs()[0])

        observation = json.loads(
            agent3_workflow.generate_interview_preparation_pdf.func(
                application.application_id
            )
        )
        assert env.calls["interview"] == 1
        assert observation["job_title"] == "AI Engineer 1"
        assert observation["company"] == "Example AI 1"
        assert observation["provider"] and observation["model"]
        assert Path(observation["pdf_path"]).exists()
        stored = list_interview_preparations(application.application_id)
        assert [record.preparation_id for record in stored] == [observation["preparation_id"]]
        assert interview_tool.get_last_interview_preparation() == observation

        missing = json.loads(
            agent3_workflow.generate_interview_preparation_pdf.func("does-not-exist")
        )
        assert "was not found" in missing["error"]
        assert "pdf_path" not in missing
        assert env.calls["interview"] == 1


def test_agent3_interview_entry_point_uses_react_executor():
    from next_chapter.services.application_tracker import save_application

    with _OfflineAgent3Environment(_ScriptedToolCallingModel(script=[])) as env:
        application = save_application(_fake_cv(), _fake_ranked_jobs()[0])
        model = _ScriptedToolCallingModel(
            script=[
                [("generate_interview_preparation_pdf", {"application_id": application.application_id})],
                [],
            ],
            final_answer="Interview PDF ready.",
        )
        env.patcher.setattr(agent3_workflow, "get_agent_llm", lambda **_k: model)

        result = agent3_workflow.run_agent3_interview_preparation(
            application.application_id, verbose=False
        )
        called_tools = [action.tool for action, _ in result["intermediate_steps"]]
        assert called_tools == ["generate_interview_preparation_pdf"]
        assert env.calls["match"] == 0 and env.calls["cover_letter"] == 0
        assert env.calls["interview"] == 1
        assert result["interview_preparation"]["application_id"] == application.application_id
        assert Path(result["interview_preparation"]["pdf_path"]).exists()
        assert result["output"] == "Interview PDF ready."

        # Model skips the tool: the guard still produces an honest result.
        lazy_model = _ScriptedToolCallingModel(script=[[]], final_answer="Done.")
        env.patcher.setattr(agent3_workflow, "get_agent_llm", lambda **_k: lazy_model)
        guarded = agent3_workflow.run_agent3_interview_preparation(
            application.application_id, verbose=False
        )
        assert env.calls["interview"] == 2
        assert guarded["interview_preparation"]["pdf_path"]
        assert "deterministic guard" in guarded["output"]

        # Missing application: no PDF is claimed.
        env.patcher.setattr(agent3_workflow, "get_agent_llm", lambda **_k: _ScriptedToolCallingModel(script=[[]]))
        failed = agent3_workflow.run_agent3_interview_preparation("does-not-exist", verbose=False)
        assert failed["interview_preparation"] is None
        assert "was not found" in failed["interview_error"]
        assert env.calls["interview"] == 2


def _run_offline_tests() -> int:
    test_agent3_exposes_interview_tool()
    test_agent3_full_auto_persists_ranking_without_interview()
    test_agent3_persistence_failure_keeps_ranking()
    test_agent3_interview_tool_calls_shared_service()
    test_agent3_interview_entry_point_uses_react_executor()
    print("Agent 3 tool-calling offline tests: PASS (5/5)")
    return 0


# ---------------------------------------------------------------------------
# Interactive live demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(_run_offline_tests())
