"""Deterministic Agent 2 application-tracking service.

This module owns application business rules. It accepts structured candidate
and ranked-job data, delegates persistence to Agent 2's SQLite layer, and
returns stable Pydantic records suitable for LangGraph or Streamlit.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel

from storage.agent2_database import (
    AGENT2_APPLICATION_STATUSES,
    agent2_connection,
    initialize_agent2_database,
)


class ApplicationRecord(BaseModel):
    application_id: str
    candidate_id: str
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    job_id: str
    source: str
    job_title: str
    company: str
    url: str
    description: Optional[str] = None
    skills_score: Optional[float] = None
    experience_score: Optional[float] = None
    education_score: Optional[float] = None
    final_score: Optional[float] = None
    status: str
    notes: Optional[str] = None
    cover_letter: Optional[str] = None
    match_details: dict[str, Any]
    created_at: str
    updated_at: str
    applied_at: Optional[str] = None


class CandidateRecord(BaseModel):
    candidate_id: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    application_count: int
    latest_activity_at: Optional[str] = None


class ApplicationStatusEvent(BaseModel):
    history_id: int
    application_id: str
    previous_status: Optional[str] = None
    new_status: str
    changed_at: str


class TrackerSummary(BaseModel):
    candidate_id: Optional[str] = None
    total_applications: int
    status_counts: dict[str, int]
    average_final_score: Optional[float] = None
    highest_final_score: Optional[float] = None
    applications_with_cover_letter: int
    latest_activity_at: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_optional(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _validated_status(status: str) -> str:
    normalized = str(status or "").strip().casefold()
    if normalized not in AGENT2_APPLICATION_STATUSES:
        allowed = ", ".join(AGENT2_APPLICATION_STATUSES)
        raise ValueError(f"Invalid application status '{status}'. Use: {allowed}.")
    return normalized


def _score(job: Any, name: str) -> Optional[float]:
    value = _field(job, name)
    if value is None:
        return None
    score = float(value)
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"{name} must be between 0 and 100.")
    return score


def _candidate_identity(
    connection,
    candidate: Any,
    candidate_id: Optional[str],
) -> str:
    full_name = _normalized_optional(_field(candidate, "full_name"))
    email = _normalized_optional(_field(candidate, "mail"))
    if email is None:
        email = _normalized_optional(_field(candidate, "email"))
    if email is not None:
        email = email.casefold()

    resolved_id = _normalized_optional(candidate_id)
    if resolved_id is None and email is not None:
        row = connection.execute(
            "SELECT id FROM candidates WHERE lower(email) = ?",
            (email,),
        ).fetchone()
        if row is not None:
            resolved_id = row["id"]
    if resolved_id is None:
        resolved_id = str(uuid4())

    now = _utc_now()
    connection.execute(
        """INSERT INTO candidates(id, full_name, email, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               full_name = COALESCE(excluded.full_name, candidates.full_name),
               email = COALESCE(excluded.email, candidates.email),
               updated_at = excluded.updated_at""",
        (resolved_id, full_name, email, now, now),
    )
    return resolved_id


def _job_identity(connection, job: Any) -> str:
    url = _normalized_optional(_field(job, "url"))
    title = _normalized_optional(
        _field(job, "job_title", _field(job, "title"))
    )
    company = _normalized_optional(_field(job, "company"))
    if url is None:
        raise ValueError("A job URL is required to track an application.")
    if title is None or company is None:
        raise ValueError("A job title and company are required.")

    existing = connection.execute(
        "SELECT id FROM jobs WHERE external_url = ?",
        (url,),
    ).fetchone()
    job_id = existing["id"] if existing is not None else str(uuid4())
    now = _utc_now()
    connection.execute(
        """INSERT INTO jobs
           (id, source, external_url, title, company, description, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(external_url) DO UPDATE SET
               title = excluded.title,
               company = excluded.company,
               description = COALESCE(excluded.description, jobs.description),
               updated_at = excluded.updated_at""",
        (
            job_id,
            _normalized_optional(_field(job, "source")) or "linkedin",
            url,
            title,
            company,
            _normalized_optional(_field(job, "description")),
            now,
            now,
        ),
    )
    return job_id


_APPLICATION_SELECT = """
SELECT
    a.id AS application_id,
    a.candidate_id,
    c.full_name AS candidate_name,
    c.email AS candidate_email,
    a.job_id,
    j.source,
    j.title AS job_title,
    j.company,
    j.external_url AS url,
    j.description,
    a.skills_score,
    a.experience_score,
    a.education_score,
    a.final_score,
    a.status,
    a.notes,
    a.cover_letter,
    COALESCE(d.details_json, '{}') AS match_details_json,
    a.created_at,
    a.updated_at,
    a.applied_at
FROM applications AS a
JOIN candidates AS c ON c.id = a.candidate_id
JOIN jobs AS j ON j.id = a.job_id
LEFT JOIN application_match_details AS d ON d.application_id = a.id
"""


def _record(row) -> ApplicationRecord:
    data = dict(row)
    raw_details = data.pop("match_details_json", "{}")
    try:
        details = json.loads(raw_details or "{}")
    except (TypeError, json.JSONDecodeError):
        details = {}
    data["match_details"] = details if isinstance(details, dict) else {}
    return ApplicationRecord.model_validate(data)


def save_application(
    candidate: Any,
    job: Any,
    *,
    status: str = "saved",
    notes: Optional[str] = None,
    cover_letter: Optional[str] = None,
    candidate_id: Optional[str] = None,
    database_path: Optional[str | Path] = None,
) -> ApplicationRecord:
    """Save one candidate/job pair once and refresh its current match data."""

    normalized_status = _validated_status(status)
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        resolved_candidate_id = _candidate_identity(
            connection,
            candidate,
            candidate_id,
        )
        job_id = _job_identity(connection, job)
        existing = connection.execute(
            """SELECT id FROM applications
               WHERE candidate_id = ? AND job_id = ?""",
            (resolved_candidate_id, job_id),
        ).fetchone()
        now = _utc_now()
        if existing is None:
            application_id = str(uuid4())
            applied_at = now if normalized_status == "applied" else None
            connection.execute(
                """INSERT INTO applications
                   (id, candidate_id, job_id, skills_score, experience_score,
                    education_score, final_score, status, notes, cover_letter,
                    created_at, updated_at, applied_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    application_id,
                    resolved_candidate_id,
                    job_id,
                    _score(job, "skills_score"),
                    _score(job, "experience_score"),
                    _score(job, "education_score"),
                    _score(job, "final_score"),
                    normalized_status,
                    _normalized_optional(notes),
                    _normalized_optional(cover_letter),
                    now,
                    now,
                    applied_at,
                ),
            )
            connection.execute(
                """INSERT INTO application_status_history
                   (application_id, previous_status, new_status, changed_at)
                   VALUES (?, NULL, ?, ?)""",
                (application_id, normalized_status, now),
            )
        else:
            application_id = existing["id"]
            connection.execute(
                """UPDATE applications SET
                       skills_score = ?, experience_score = ?,
                       education_score = ?, final_score = ?,
                       notes = COALESCE(?, notes),
                       cover_letter = COALESCE(?, cover_letter),
                       updated_at = ?
                   WHERE id = ?""",
                (
                    _score(job, "skills_score"),
                    _score(job, "experience_score"),
                    _score(job, "education_score"),
                    _score(job, "final_score"),
                    _normalized_optional(notes),
                    _normalized_optional(cover_letter),
                    now,
                    application_id,
                ),
            )

        details = _field(job, "skills_detail", {})
        details_json = json.dumps(
            details if isinstance(details, Mapping) else {},
            ensure_ascii=False,
        )
        connection.execute(
            """INSERT INTO application_match_details(application_id, details_json)
               VALUES (?, ?)
               ON CONFLICT(application_id) DO UPDATE SET
                   details_json = excluded.details_json""",
            (application_id, details_json),
        )
        row = connection.execute(
            _APPLICATION_SELECT + " WHERE a.id = ?",
            (application_id,),
        ).fetchone()
    return _record(row)


def get_application(
    application_id: str,
    *,
    database_path: Optional[str | Path] = None,
) -> ApplicationRecord:
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        row = connection.execute(
            _APPLICATION_SELECT + " WHERE a.id = ?",
            (str(application_id).strip(),),
        ).fetchone()
    if row is None:
        raise LookupError(f"Application '{application_id}' was not found.")
    return _record(row)


def list_applications(
    candidate_id: Optional[str] = None,
    *,
    status: Optional[str] = None,
    search: Optional[str] = None,
    minimum_score: Optional[float] = None,
    limit: Optional[int] = None,
    database_path: Optional[str | Path] = None,
) -> list[ApplicationRecord]:
    """List tracker records using UI-friendly deterministic filters."""

    initialize_agent2_database(database_path)
    clauses: list[str] = []
    parameters: list[Any] = []
    if candidate_id is not None:
        resolved_candidate_id = str(candidate_id).strip()
        if not resolved_candidate_id:
            raise ValueError("candidate_id cannot be empty.")
        clauses.append("a.candidate_id = ?")
        parameters.append(resolved_candidate_id)
    if status is not None:
        clauses.append("a.status = ?")
        parameters.append(_validated_status(status))
    search_text = _normalized_optional(search)
    if search_text is not None:
        clauses.append("(lower(j.title) LIKE ? OR lower(j.company) LIKE ?)")
        pattern = f"%{search_text.casefold()}%"
        parameters.extend([pattern, pattern])
    if minimum_score is not None:
        resolved_score = float(minimum_score)
        if not 0.0 <= resolved_score <= 100.0:
            raise ValueError("minimum_score must be between 0 and 100.")
        clauses.append("a.final_score >= ?")
        parameters.append(resolved_score)
    if limit is not None:
        resolved_limit = int(limit)
        if resolved_limit <= 0:
            raise ValueError("limit must be greater than zero.")
        resolved_limit = min(resolved_limit, 1000)
    else:
        resolved_limit = None

    query = _APPLICATION_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY a.updated_at DESC, a.created_at DESC"
    if resolved_limit is not None:
        query += " LIMIT ?"
        parameters.append(resolved_limit)
    with agent2_connection(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_record(row) for row in rows]


def list_candidates(
    *,
    database_path: Optional[str | Path] = None,
) -> list[CandidateRecord]:
    """Return candidates for the Streamlit candidate selector."""

    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        rows = connection.execute(
            """SELECT
                   c.id AS candidate_id,
                   c.full_name,
                   c.email,
                   COUNT(a.id) AS application_count,
                   MAX(a.updated_at) AS latest_activity_at
               FROM candidates AS c
               LEFT JOIN applications AS a ON a.candidate_id = c.id
               GROUP BY c.id, c.full_name, c.email
               ORDER BY latest_activity_at DESC, c.full_name"""
        ).fetchall()
    return [CandidateRecord.model_validate(dict(row)) for row in rows]


def get_tracker_summary(
    candidate_id: Optional[str] = None,
    *,
    database_path: Optional[str | Path] = None,
) -> TrackerSummary:
    """Calculate dashboard totals without involving an LLM."""

    initialize_agent2_database(database_path)
    clause = ""
    parameters: list[Any] = []
    resolved_candidate_id: Optional[str] = None
    if candidate_id is not None:
        resolved_candidate_id = str(candidate_id).strip()
        if not resolved_candidate_id:
            raise ValueError("candidate_id cannot be empty.")
        clause = " WHERE candidate_id = ?"
        parameters.append(resolved_candidate_id)

    with agent2_connection(database_path) as connection:
        aggregate = connection.execute(
            """SELECT
                   COUNT(*) AS total_applications,
                   AVG(final_score) AS average_final_score,
                   MAX(final_score) AS highest_final_score,
                   SUM(CASE WHEN cover_letter IS NOT NULL
                                 AND trim(cover_letter) <> ''
                            THEN 1 ELSE 0 END) AS with_cover_letter,
                   MAX(updated_at) AS latest_activity_at
               FROM applications"""
            + clause,
            parameters,
        ).fetchone()
        status_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM applications"
            + clause
            + " GROUP BY status",
            parameters,
        ).fetchall()

    status_counts = {status: 0 for status in AGENT2_APPLICATION_STATUSES}
    status_counts.update({row["status"]: int(row["count"]) for row in status_rows})
    average = aggregate["average_final_score"]
    highest = aggregate["highest_final_score"]
    return TrackerSummary(
        candidate_id=resolved_candidate_id,
        total_applications=int(aggregate["total_applications"] or 0),
        status_counts=status_counts,
        average_final_score=round(float(average), 1) if average is not None else None,
        highest_final_score=round(float(highest), 1) if highest is not None else None,
        applications_with_cover_letter=int(aggregate["with_cover_letter"] or 0),
        latest_activity_at=aggregate["latest_activity_at"],
    )


def get_application_status_history(
    application_id: str,
    *,
    database_path: Optional[str | Path] = None,
) -> list[ApplicationStatusEvent]:
    """Return the auditable status timeline for one application."""

    resolved_id = str(application_id or "").strip()
    if not resolved_id:
        raise ValueError("application_id is required.")
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM applications WHERE id = ?",
            (resolved_id,),
        ).fetchone()
        if exists is None:
            raise LookupError(f"Application '{application_id}' was not found.")
        rows = connection.execute(
            """SELECT
                   id AS history_id,
                   application_id,
                   previous_status,
                   new_status,
                   changed_at
               FROM application_status_history
               WHERE application_id = ?
               ORDER BY id""",
            (resolved_id,),
        ).fetchall()
    return [ApplicationStatusEvent.model_validate(dict(row)) for row in rows]


def get_tracked_job_urls(
    candidate: Any,
    *,
    database_path: Optional[str | Path] = None,
) -> set[str]:
    """Return LinkedIn URLs already saved for the same candidate."""

    email = _normalized_optional(_field(candidate, "mail"))
    if email is None:
        email = _normalized_optional(_field(candidate, "email"))
    full_name = _normalized_optional(_field(candidate, "full_name"))
    if email is None and full_name is None:
        return set()

    initialize_agent2_database(database_path)
    if email is not None:
        identity_clause = "lower(c.email) = ?"
        identity_value = email.casefold()
    else:
        identity_clause = "lower(c.full_name) = ?"
        identity_value = full_name.casefold()

    with agent2_connection(database_path) as connection:
        rows = connection.execute(
            f"""SELECT j.external_url
                FROM applications AS a
                JOIN candidates AS c ON c.id = a.candidate_id
                JOIN jobs AS j ON j.id = a.job_id
                WHERE {identity_clause}""",
            (identity_value,),
        ).fetchall()
    return {str(row["external_url"]).split("?", 1)[0].rstrip("/") for row in rows}


def update_application_status(
    application_id: str,
    status: str,
    *,
    database_path: Optional[str | Path] = None,
) -> ApplicationRecord:
    normalized_status = _validated_status(status)
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        current = connection.execute(
            "SELECT status, applied_at FROM applications WHERE id = ?",
            (str(application_id).strip(),),
        ).fetchone()
        if current is None:
            raise LookupError(f"Application '{application_id}' was not found.")
        if current["status"] != normalized_status:
            now = _utc_now()
            applied_at = current["applied_at"]
            if normalized_status == "applied" and applied_at is None:
                applied_at = now
            connection.execute(
                """UPDATE applications
                   SET status = ?, applied_at = ?, updated_at = ?
                   WHERE id = ?""",
                (normalized_status, applied_at, now, application_id),
            )
            connection.execute(
                """INSERT INTO application_status_history
                   (application_id, previous_status, new_status, changed_at)
                   VALUES (?, ?, ?, ?)""",
                (application_id, current["status"], normalized_status, now),
            )
        row = connection.execute(
            _APPLICATION_SELECT + " WHERE a.id = ?",
            (application_id,),
        ).fetchone()
    return _record(row)


def add_application_note(
    application_id: str,
    note: str,
    *,
    database_path: Optional[str | Path] = None,
) -> ApplicationRecord:
    clean_note = str(note or "").strip()
    if not clean_note:
        raise ValueError("Application note cannot be empty.")
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        current = connection.execute(
            "SELECT notes FROM applications WHERE id = ?",
            (str(application_id).strip(),),
        ).fetchone()
        if current is None:
            raise LookupError(f"Application '{application_id}' was not found.")
        timestamped_note = f"[{_utc_now()}] {clean_note}"
        notes = (
            f"{current['notes']}\n{timestamped_note}"
            if current["notes"]
            else timestamped_note
        )
        connection.execute(
            "UPDATE applications SET notes = ?, updated_at = ? WHERE id = ?",
            (notes, _utc_now(), application_id),
        )
        row = connection.execute(
            _APPLICATION_SELECT + " WHERE a.id = ?",
            (application_id,),
        ).fetchone()
    return _record(row)
