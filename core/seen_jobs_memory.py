"""
seen_jobs_memory.py
--------------
Per-candidate memory of jobs already shown in a previous run. Keyed by
(candidate_email, job_url) — URL is Himalayas' stable identifier for a
specific posting, more reliable than title+company which could
theoretically collide across different postings.

Used to: (1) filter out already-shown jobs from a fresh search, so a
repeat run surfaces something new instead of the same results, and
(2) let search_real_jobs decide when to page deeper or fall back to a
category-based search, if everything the API returns has already been
seen before.

Scoped per-candidate (by email) — unlike skill_memory.py, which is
deliberately GLOBAL. A job being already-shown to candidate A must not
hide it from candidate B; identity here is the point, not the skill
relationship.
"""

import sqlite3
from pathlib import Path
from typing import List, Set

DB_PATH = Path("cache/seen_jobs_memory.db")


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            candidate_email TEXT NOT NULL,
            job_url TEXT NOT NULL,
            job_title TEXT,
            company TEXT,
            score_percent REAL,
            matching_skills TEXT,
            missing_skills TEXT,
            seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (candidate_email, job_url)
        )
    """)
    return conn


def get_seen_urls(candidate_email: str) -> Set[str]:
    """All job URLs already shown to this candidate, across every past run."""
    if not candidate_email:
        return set()
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT job_url FROM seen_jobs WHERE candidate_email = ?",
            (candidate_email.strip().lower(),),
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _job_key(job_title: str, company: str) -> str:
    """Stable key for a posting that has no URL — small LLMs often drop the
    'url' field when they retype a job for evaluation. Without this, a scored
    job with no recorded URL would never be remembered and would be scored
    AGAIN on the next run."""
    return f"missing|{job_title.strip()}|{company.strip()}"


def record_seen(candidate_email: str, job_url: str, job_title: str, company: str,
                 score_percent: float, matching_skills: list, missing_skills: list) -> None:
    """Mark a job as scored/seen for this candidate. Keyed by URL when we have
    one, else by `missing|title|company` so no scored job is ever silently
    dropped — see search.job_search.search_real_jobs, which filters on BOTH
    forms so a job remembered this way is still skipped on the next search."""
    if not candidate_email or not job_title:
        return  # nothing to key this record by — skip silently rather than error
    if not job_url:
        job_url = _job_key(job_title, company)
    if not job_url:
        return
    import json
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO seen_jobs
               (candidate_email, job_url, job_title, company, score_percent, matching_skills, missing_skills)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(candidate_email, job_url) DO UPDATE SET
                   score_percent = excluded.score_percent,
                   matching_skills = excluded.matching_skills,
                   missing_skills = excluded.missing_skills,
                   seen_at = CURRENT_TIMESTAMP""",
            (candidate_email.strip().lower(), job_url, job_title, company,
             score_percent, json.dumps(matching_skills), json.dumps(missing_skills)),
        )
        conn.commit()
    finally:
        conn.close()


def clear_seen(candidate_email: str = None) -> int:
    """
    Deletes seen-job records. Pass a candidate_email to wipe just that
    candidate's history, or omit to wipe EVERYONE's — useful for starting
    a clean dev/test slate rather than manually deleting the DB file.
    Returns the number of rows deleted.
    """
    conn = _get_connection()
    try:
        if candidate_email:
            cursor = conn.execute(
                "DELETE FROM seen_jobs WHERE candidate_email = ?",
                (candidate_email.strip().lower(),),
            )
        else:
            cursor = conn.execute("DELETE FROM seen_jobs")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()