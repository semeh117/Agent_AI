"""Offline schema tests for Agent 2's dedicated SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from storage.agent2_database import (  # noqa: E402
    agent2_connection,
    initialize_agent2_database,
)
from services.application_tracker import (  # noqa: E402
    add_application_note,
    get_application_status_history,
    get_tracker_summary,
    get_tracked_job_urls,
    list_applications,
    list_candidates,
    save_application,
    update_application_status,
)


EXPECTED_TABLES = {
    "schema_migrations",
    "candidates",
    "jobs",
    "applications",
    "application_status_history",
    "application_match_details",
    "interview_preparations",
    "graph_checkpoints",
    "graph_writes",
    "graph_blobs",
}


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
            assert version == 3


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


def _show_live_tracker() -> int:
    summary = get_tracker_summary()
    print("AGENT 2 APPLICATION TRACKER")
    print(f"Total applications: {summary.total_applications}")
    print(f"Average score: {summary.average_final_score}")
    print(f"Highest score: {summary.highest_final_score}")
    print(f"Cover letters: {summary.applications_with_cover_letter}")
    active_counts = {
        status: count
        for status, count in summary.status_counts.items()
        if count
    }
    print(f"Statuses: {active_counts}")

    for candidate in list_candidates():
        print(
            f"\n{candidate.full_name or 'Unknown candidate'} "
            f"({candidate.application_count} application(s))"
        )
        for application in list_applications(candidate.candidate_id):
            print(
                f"  [{application.status}] {application.job_title} @ "
                f"{application.company} - {application.final_score}%"
            )
            print(f"    {application.url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test or inspect Agent 2's application tracker."
    )
    parser.add_argument(
        "--show-live",
        action="store_true",
        help="Read and display the real Agent 2 tracker database.",
    )
    args = parser.parse_args()
    if args.show_live:
        return _show_live_tracker()

    test_agent2_database_schema()
    test_agent2_database_constraints()
    test_agent2_application_tracker()
    print("Agent 2 SQLite database tests: PASS (3/3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
