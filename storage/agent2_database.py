"""SQLite foundation dedicated to Agent 2 application data.

Agent 1's ``cache/seen_jobs_memory.db`` remains an independent legacy store.
This module owns Agent 2's database path, connections, pragmas, and schema;
agents and UI code must access it through services instead of writing SQL.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Generator, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT2_DATABASE_PATH = PROJECT_ROOT / "runtime" / "agent2.sqlite3"
SCHEMA_VERSION = 3

AGENT2_APPLICATION_STATUSES = (
    "discovered",
    "saved",
    "applied",
    "interview",
    "offer",
    "rejected",
    "withdrawn",
)

_STATUS_SQL = ", ".join(f"'{status}'" for status in AGENT2_APPLICATION_STATUSES)

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    full_name TEXT,
    email TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_email
ON candidates(lower(email))
WHERE email IS NOT NULL AND trim(email) <> '';

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'linkedin',
    external_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS applications (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    skills_score REAL CHECK (skills_score BETWEEN 0 AND 100),
    experience_score REAL CHECK (experience_score BETWEEN 0 AND 100),
    education_score REAL CHECK (education_score BETWEEN 0 AND 100),
    final_score REAL CHECK (final_score BETWEEN 0 AND 100),
    status TEXT NOT NULL DEFAULT 'discovered'
        CHECK (status IN ({_STATUS_SQL})),
    notes TEXT,
    cover_letter TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    applied_at TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    UNIQUE (candidate_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_applications_candidate_status
ON applications(candidate_id, status);

CREATE TABLE IF NOT EXISTS application_status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL,
    previous_status TEXT,
    new_status TEXT NOT NULL CHECK (new_status IN ({_STATUS_SQL})),
    changed_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (application_id) REFERENCES applications(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_status_history_application
ON application_status_history(application_id, changed_at);

CREATE TABLE IF NOT EXISTS application_match_details (
    application_id TEXT PRIMARY KEY,
    details_json TEXT NOT NULL DEFAULT '{{}}',
    FOREIGN KEY (application_id) REFERENCES applications(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interview_preparations (
    id TEXT PRIMARY KEY,
    application_id TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (application_id) REFERENCES applications(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_interview_preparations_application
ON interview_preparations(application_id, created_at);

-- LangGraph persistence is deliberately isolated from application-tracker
-- records by the graph_ prefix. Values are serialized by LangGraph's safe
-- msgpack serializer; pickle fallback is not enabled.
CREATE TABLE IF NOT EXISTS graph_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    checkpoint_type TEXT NOT NULL,
    checkpoint BLOB NOT NULL,
    metadata_type TEXT NOT NULL,
    metadata BLOB NOT NULL,
    parent_checkpoint_id TEXT,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_checkpoints_latest
ON graph_checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC);

CREATE TABLE IF NOT EXISTS graph_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    write_index INTEGER NOT NULL,
    channel TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value BLOB NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (
        thread_id,
        checkpoint_ns,
        checkpoint_id,
        task_id,
        write_index
    )
);

CREATE TABLE IF NOT EXISTS graph_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
"""


def get_agent2_database_path(
    database_path: Optional[str | Path] = None,
) -> Path:
    """Resolve Agent 2's database independently of the process working directory."""

    configured = database_path or os.getenv("AGENT2_DATABASE_PATH")
    if configured is None or not str(configured).strip():
        return DEFAULT_AGENT2_DATABASE_PATH

    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@contextmanager
def agent2_connection(
    database_path: Optional[str | Path] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Open one short, transactional Agent 2 SQLite connection."""

    path = get_agent2_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_agent2_database(
    database_path: Optional[str | Path] = None,
) -> Path:
    """Create or validate Agent 2's versioned schema and return its path."""

    path = get_agent2_database_path(database_path)
    with agent2_connection(path) as connection:
        connection.executescript(_SCHEMA_SQL)
        existing = connection.execute(
            "SELECT MAX(version) AS version FROM schema_migrations"
        ).fetchone()["version"]
        if existing is not None and int(existing) > SCHEMA_VERSION:
            raise RuntimeError(
                "Agent 2 database schema is newer than this application "
                f"({existing} > {SCHEMA_VERSION})."
            )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
    return path
