"""Offline schema tests for Agent 2's dedicated SQLite database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent

from next_chapter.storage.agent2_database import (  # noqa: E402
    agent2_connection,
    initialize_agent2_database,
)
from next_chapter.services.application_tracker import (  # noqa: E402
    add_application_note,
    get_application_status_history,
    get_tracker_summary,
    get_tracked_job_urls,
    list_applications,
    list_candidates,
    save_application,
    update_application_status,
)
from next_chapter.services.interview_preparation import (  # noqa: E402
    build_interview_prompt,
    generate_interview_content,
    generate_interview_preparation,
    list_interview_preparations,
)
from pypdf import PdfReader  # noqa: E402


EXPECTED_TABLES = {
    "schema_migrations",
    "candidates",
    "candidate_profiles",
    "jobs",
    "applications",
    "application_status_history",
    "application_match_details",
    "interview_preparations",
    "graph_checkpoints",
    "graph_writes",
    "graph_blobs",
}


def test_interview_prompt_adapts_to_applied_role_family() -> None:
    profile = {"skills": ["Python", "SQL"], "job_titles": ["Data Analyst"]}
    common = {
        "company": "Example",
        "match_details": {"matching": [], "missing": []},
        "final_score": 80.0,
    }
    data_engineering = SimpleNamespace(
        **common,
        job_title="Data Engineer",
        description="Build ETL pipelines, maintain dbt models, and monitor quality.",
    )
    data_science = SimpleNamespace(
        **common,
        job_title="Data Scientist",
        description="Design experiments and validate predictive models.",
    )

    engineering_prompt = build_interview_prompt(data_engineering, profile)
    science_prompt = build_interview_prompt(data_science, profile)

    assert '"role_family": "Data Engineering"' in engineering_prompt
    assert "batch or streaming pipeline design" in engineering_prompt
    assert '"role_family": "Data Science"' in science_prompt
    assert "prevention of data leakage" in science_prompt


def test_agent2_database_schema() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent2.sqlite3"
        first_path = initialize_agent2_database(database_path)
        second_path = initialize_agent2_database(database_path)
        assert first_path == second_path == database_path.resolve()

        with agent2_connection(database_path) as connection:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert EXPECTED_TABLES.issubset(tables)
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            version = connection.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0]
            from next_chapter.storage.agent2_database import SCHEMA_VERSION
            assert version == SCHEMA_VERSION


def test_agent2_database_constraints() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent2.sqlite3"
        initialize_agent2_database(database_path)

        with agent2_connection(database_path) as connection:
            connection.execute(
                "INSERT INTO candidates(id, full_name, email) VALUES (?, ?, ?)",
                ("candidate-1", "Test Candidate", "candidate@example.com"),
            )
            connection.execute(
                """INSERT INTO jobs(id, external_url, title, company)
                   VALUES (?, ?, ?, ?)""",
                ("job-1", "https://example.com/job/1", "AI Engineer", "Example"),
            )

        try:
            with agent2_connection(database_path) as connection:
                connection.execute(
                    """INSERT INTO applications
                       (id, candidate_id, job_id, status)
                       VALUES (?, ?, ?, ?)""",
                    ("application-1", "candidate-1", "job-1", "not-a-status"),
                )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Invalid application status was accepted.")


def test_agent2_application_tracker() -> None:
    candidate = {
        "full_name": "Test Candidate",
        "mail": "candidate@example.com",
        "skills": ["Python", "LangGraph", "RAG"],
        "job_titles": ["AI Engineer"],
        "experience_years": 4.0,
        "education": ["B.S. Computer Science"],
        "highest_education_level": "Bachelor",
    }
    job = {
        "job_title": "AI Engineer",
        "company": "Example AI",
        "url": "https://example.com/jobs/1",
        "description": "Build grounded AI systems.",
        "skills_score": 80.0,
        "experience_score": 90.0,
        "education_score": 100.0,
        "final_score": 87.0,
        "skills_detail": {
            "matching": [{"job_skill": "Python", "matched_via": "Python"}],
            "missing": ["Kubernetes"],
        },
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent2.sqlite3"
        saved = save_application(
            candidate,
            job,
            database_path=database_path,
        )
        duplicate = save_application(
            candidate,
            job,
            database_path=database_path,
        )
        assert duplicate.application_id == saved.application_id
        assert duplicate.status == "saved"
        assert duplicate.match_details["missing"] == ["Kubernetes"]
        assert get_tracked_job_urls(
            candidate,
            database_path=database_path,
        ) == {job["url"]}

        second_job = {
            **job,
            "job_title": "Senior ML Engineer",
            "company": "Second Company",
            "url": "https://example.com/jobs/2",
            "skills_score": 90.0,
            "experience_score": 100.0,
            "education_score": 100.0,
            "final_score": 95.0,
        }
        discovered = save_application(
            candidate,
            second_job,
            status="discovered",
            cover_letter="A tailored cover letter.",
            candidate_id=saved.candidate_id,
            database_path=database_path,
        )

        applications = list_applications(
            saved.candidate_id,
            database_path=database_path,
        )
        assert {record.application_id for record in applications} == {
            saved.application_id,
            discovered.application_id,
        }
        assert [record.application_id for record in list_applications(
            saved.candidate_id,
            status="discovered",
            minimum_score=90,
            database_path=database_path,
        )] == [discovered.application_id]
        assert len(list_applications(
            search="second company",
            limit=1,
            database_path=database_path,
        )) == 1

        candidates = list_candidates(database_path=database_path)
        assert len(candidates) == 1
        assert candidates[0].application_count == 2

        summary = get_tracker_summary(
            saved.candidate_id,
            database_path=database_path,
        )
        assert summary.total_applications == 2
        assert summary.status_counts["saved"] == 1
        assert summary.status_counts["discovered"] == 1
        assert summary.average_final_score == 91.0
        assert summary.highest_final_score == 95.0
        assert summary.applications_with_cover_letter == 1

        applied = update_application_status(
            saved.application_id,
            "applied",
            database_path=database_path,
        )
        assert applied.status == "applied"
        assert applied.applied_at is not None

        noted = add_application_note(
            saved.application_id,
            "Application submitted through the company portal.",
            database_path=database_path,
        )
        assert "company portal" in (noted.notes or "")

        with agent2_connection(database_path) as connection:
            history = connection.execute(
                """SELECT previous_status, new_status
                   FROM application_status_history
                   WHERE application_id = ? ORDER BY id""",
                (saved.application_id,),
            ).fetchall()
        assert [tuple(row) for row in history] == [
            (None, "saved"),
            ("saved", "applied"),
        ]
        timeline = get_application_status_history(
            saved.application_id,
            database_path=database_path,
        )
        assert [event.new_status for event in timeline] == ["saved", "applied"]


class _FakeStructuredInterview:
    def invoke(self, _prompt):
        def question(number: int, category: str) -> dict:
            return {
                "question": f"{category} interview question {number}?",
                "competency": f"{category} competency {number}",
                "why_asked": "This checks relevant role knowledge.",
                "job_connection": "The posting requires reliable production delivery.",
                "answer_strategy": "Explain the approach and connect it to CV evidence.",
                "sample_answer": "I would begin with the Python project stated in my CV and explain my decisions honestly.",
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
            "technical_questions": [
                question(index, "Technical") for index in range(1, 6)
            ],
            "gap_questions": [
                question(index, "Gap") for index in range(1, 3)
            ],
            "behavioral_questions": [
                question(index, "Behavioral") for index in range(1, 4)
            ],
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
    def with_structured_output(self, _schema, **kwargs):
        assert kwargs == {"method": "json_schema", "strict": True}
        return _FakeStructuredInterview()


class _FakeGroqSchemaError(Exception):
    def __init__(self, failed_generation: dict | str):
        super().__init__("Generated JSON does not match the expected schema.")
        self.body = {
            "error": {
                "code": "json_validate_failed",
                "message": "Failed to validate JSON.",
                "failed_generation": (
                    failed_generation
                    if isinstance(failed_generation, str)
                    else json.dumps(failed_generation)
                ),
            }
        }


class _FakeIncompleteStructuredInterview(_FakeStructuredInterview):
    def invoke(self, prompt):
        payload = super().invoke(prompt)
        payload.pop("questions_to_ask")
        payload.pop("preparation_checklist")
        raise _FakeGroqSchemaError(payload)


class _FakeIncompleteInterviewLLM:
    def with_structured_output(self, _schema, **kwargs):
        assert kwargs == {"method": "json_schema", "strict": True}
        return _FakeIncompleteStructuredInterview()


class _FakeRetryInterviewLLM:
    def __init__(self, *, always_fail: bool = False):
        self.always_fail = always_fail
        self.calls = 0

    def with_structured_output(self, _schema, **kwargs):
        assert kwargs == {"method": "json_schema", "strict": True}
        return self

    def invoke(self, prompt):
        self.calls += 1
        if self.always_fail or self.calls == 1:
            raise _FakeGroqSchemaError("")
        return _FakeStructuredInterview().invoke(prompt)


def test_agent2_interview_preparation_pdf() -> None:
    candidate = {
        "full_name": "Test Candidate",
        "mail": "candidate@example.com",
        "skills": ["Python", "LangGraph", "RAG"],
        "skill_evidence": {"Python": "Production AI project"},
        "job_titles": ["AI Engineer"],
        "experience_years": 4.0,
        "education": ["B.S. Computer Science"],
        "highest_education_level": "Bachelor",
    }
    job = {
        "job_title": "AI Engineer",
        "company": "Example AI",
        "url": "https://example.com/jobs/interview",
        "description": "Build reliable AI systems with Python and LangGraph.",
        "skills_score": 80.0,
        "experience_score": 100.0,
        "education_score": 100.0,
        "final_score": 90.0,
        "skills_detail": {
            "matching": [{"job_skill": "Python", "matched_via": "Python"}],
            "missing": ["Kubernetes"],
        },
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        database_path = root / "agent2.sqlite3"
        application = save_application(
            candidate,
            job,
            database_path=database_path,
        )
        preparation = generate_interview_preparation(
            application.application_id,
            llm=_FakeInterviewLLM(),
            output_directory=root / "pdf",
            database_path=database_path,
        )
        pdf_path = Path(preparation.pdf_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1_000
        reader = PdfReader(pdf_path)
        assert len(reader.pages) >= 2
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        assert "Interview Preparation Guide" in extracted
        assert "1. Role Snapshot" in extracted
        assert "Machine Learning Engineering" in extracted
        assert "2. Technical Interview Questions" in extracted
        assert "What this tests" in extracted
        assert "Job connection" in extracted
        assert list_interview_preparations(
            application.application_id,
            database_path=database_path,
        )[0].preparation_id == preparation.preparation_id


def test_agent2_interview_recovers_missing_trailing_lists() -> None:
    candidate = {
        "full_name": "Recovery Candidate",
        "skills": ["Python", "RAG"],
        "experience_years": 3.0,
        "highest_education_level": "Bachelor",
    }
    job = {
        "job_title": "AI Engineer",
        "company": "Recovery AI",
        "url": "https://example.com/jobs/recovery",
        "description": "Build reliable AI systems.",
        "skills_score": 70.0,
        "experience_score": 80.0,
        "education_score": 100.0,
        "final_score": 79.0,
        "skills_detail": {"matching": [], "missing": ["Kubernetes"]},
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        database_path = root / "agent2.sqlite3"
        application = save_application(candidate, job, database_path=database_path)
        preparation = generate_interview_preparation(
            application.application_id,
            llm=_FakeIncompleteInterviewLLM(),
            output_directory=root / "pdf",
            database_path=database_path,
        )
        assert len(preparation.content.questions_to_ask) == 4
        assert len(preparation.content.preparation_checklist) == 5
        assert Path(preparation.pdf_path).exists()


def test_agent2_interview_retries_empty_schema_failure() -> None:
    profile = {
        "full_name": "Retry Candidate",
        "skills": ["Python", "RAG"],
        "experience_years": 3.0,
        "highest_education_level": "Bachelor",
    }
    job = {
        "job_title": "AI Engineer",
        "company": "Retry AI",
        "url": "https://example.com/jobs/retry",
        "description": "Build reliable AI systems.",
        "final_score": 79.0,
        "skills_detail": {"matching": [], "missing": ["Kubernetes"]},
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent2.sqlite3"
        application = save_application(profile, job, database_path=database_path)
        llm = _FakeRetryInterviewLLM()
        content = generate_interview_content(application, profile, llm=llm)
        assert llm.calls == 2
        assert len(content.technical_questions) == 5


def test_agent2_interview_falls_back_after_repeated_schema_failure() -> None:
    profile = {
        "full_name": "Fallback Candidate",
        "skills": ["Python", "RAG"],
        "skill_evidence": {"Python": "Python project listed in the CV."},
        "experience_years": 3.0,
        "highest_education_level": "Bachelor",
    }
    job = {
        "job_title": "AI Engineer",
        "company": "Fallback AI",
        "url": "https://example.com/jobs/fallback",
        "description": "Build reliable AI systems with Python and Kubernetes.",
        "final_score": 79.0,
        "skills_detail": {
            "matching": [{"job_skill": "Python", "matched_via": "Python"}],
            "missing": ["Kubernetes"],
        },
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent2.sqlite3"
        application = save_application(profile, job, database_path=database_path)
        llm = _FakeRetryInterviewLLM(always_fail=True)
        content = generate_interview_content(application, profile, llm=llm)
        assert llm.calls == 2
        assert len(content.technical_questions) == 5
        assert len(content.gap_questions) == 2
        assert len(content.behavioral_questions) == 3
        assert "Kubernetes" in content.gap_questions[0].question


def main() -> int:
    test_interview_prompt_adapts_to_applied_role_family()
    test_agent2_database_schema()
    test_agent2_database_constraints()
    test_agent2_application_tracker()
    test_agent2_interview_preparation_pdf()
    test_agent2_interview_recovers_missing_trailing_lists()
    test_agent2_interview_retries_empty_schema_failure()
    test_agent2_interview_falls_back_after_repeated_schema_failure()
    print("Agent 2 SQLite and interview preparation tests: PASS (8/8)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
