"""Agent-facing LinkedIn matching tool for Agent 2.

    Agent 2 decides what query and location to use. This tool runs the
deterministic production workflow:

    LinkedIn scrape -> job parsing -> skills cosine similarity
    -> experience/education scoring -> weighted ranking
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from pipeline.linkedin_cosine_pipeline import match_linkedin_jobs


_current_cv_info: Any = None
_current_location = ""
_current_results_count = 3
_last_match_result: dict[str, Any] | None = None
_last_tracked_applications: list[dict[str, Any]] = []


def sync_linkedin_results_for_shared_tools(result: dict[str, Any]) -> None:
    """Expose Agent 2 rankings in the shape expected by shared delivery tools.

    The cover-letter, Gmail, and Telegram tools predate Agent 2 and read their
    ranked jobs from ``job_evaluator``. Keeping this adapter here lets Agent 2
    reuse those tools without coupling them to LinkedIn or cosine matching.
    """

    import agent.tools.job_evaluator as job_evaluator

    job_evaluator._all_evaluations = [
        {
            **job,
            "score_percent": job.get("score_percent", job.get("final_score", 0.0)),
            "matching_skills": job.get("skills_detail", {}).get("matching", []),
            "missing_skills": job.get("skills_detail", {}).get("missing", []),
        }
        for job in result.get("ranked_jobs", [])
    ]


def set_candidate_profile(
    cv_info: Any,
    location: str = "",
    results_count: int = 3,
) -> None:
    """Set the candidate and run options used by subsequent tool calls."""

    global _current_cv_info, _current_location, _current_results_count
    global _last_match_result, _last_tracked_applications
    _current_cv_info = cv_info
    _current_location = (location or "").strip()
    _current_results_count = max(1, min(int(results_count), 20))
    _last_match_result = None
    _last_tracked_applications = []


def get_last_match_result() -> dict[str, Any] | None:
    """Return the full latest result, including stored job descriptions."""

    return _last_match_result


def get_last_tracked_applications() -> list[dict[str, Any]]:
    """Return the tracker rows saved for the latest ranking (may be empty)."""

    return list(_last_tracked_applications)


def persist_ranked_jobs(cv_info: Any, ranked_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Save ranked jobs in the shared Agent 2 tracker, deterministically.

    This is the same ``save_application`` call Agent 2's
    ``persist_recommendations`` node makes, so both architectures write to one
    database with identical deduplication. Failures are reported as a warning
    and never discard the valid ranking.
    """

    tracked: list[dict[str, Any]] = []
    if not ranked_jobs:
        return {"tracked_applications": tracked, "candidate_id": ""}
    try:
        from services.application_tracker import save_application

        candidate_id: str | None = None
        for job in ranked_jobs:
            record = save_application(
                cv_info,
                job,
                status="discovered",
                candidate_id=candidate_id,
            )
            candidate_id = record.candidate_id
            tracked.append(
                {
                    "application_id": record.application_id,
                    "job_id": record.job_id,
                    "url": record.url,
                }
            )
        return {"tracked_applications": tracked, "candidate_id": candidate_id or ""}
    except Exception as exc:
        return {
            "tracked_applications": tracked,
            "candidate_id": "",
            "persistence_warning": (
                "Ranked jobs could not be saved to the application tracker "
                f"({type(exc).__name__}: {exc})."
            ),
        }


@tool
def match_linkedin_jobs_for_agent(
    query: str,
) -> str:
    """Search LinkedIn and rank jobs against the loaded candidate profile.

    Use a concise search query based on the candidate's strongest target role
    and most relevant skills. Pass only the plain query string. The location
    and requested result count were configured when the candidate profile was
    loaded, so they must not be included in the tool input.

    The returned JSON contains ``ranked_jobs``. Every ranked job includes:
    ``skills_score`` (required-skill coverage after exact/ESCO/cosine matching),
    ``experience_score``,
    ``education_score``, and the weighted ``final_score`` calculated as:

        skills_score * 0.5
        + experience_score * 0.3
        + education_score * 0.2

    Jobs are already sorted by ``final_score`` descending. Do not recalculate
    or reorder the scores in the agent response. The ranked jobs are also
    saved automatically to the application tracker; ``tracked_applications``
    lists each saved ``application_id`` with its ``url`` and
    ``persistence_warning`` appears only if saving failed. Do not call any
    other tool to save them. ESCO normalization counts are
    dataset-coverage statistics, never candidate-job match counts. A 100 score
    with a ``No ... requirement stated`` note means no penalty was applied; it
    is not evidence that an unstated requirement was positively matched.
    """

    if _current_cv_info is None:
        return json.dumps(
            {"error": "No candidate profile has been loaded."},
            ensure_ascii=False,
        )

    resolved_query = (query or "").strip()
    resolved_location = _current_location

    if not resolved_query:
        return json.dumps(
            {"error": "LinkedIn search query cannot be empty."},
            ensure_ascii=False,
        )

    global _last_match_result, _last_tracked_applications

    try:
        result = match_linkedin_jobs(
            cv_info=_current_cv_info,
            query=resolved_query,
            location=resolved_location,
            max_jobs=_current_results_count,
            use_cache=True,
        )
    except Exception as exc:
        return json.dumps(
            {
                "error": f"LinkedIn matching failed: {exc}",
                "query": resolved_query,
                "location": resolved_location,
            },
            indent=2,
            ensure_ascii=False,
        )

    # Persist the ranking in the shared tracker without changing its order.
    # This is deterministic Python, not a separate LLM-controlled tool.
    persistence = persist_ranked_jobs(_current_cv_info, result.get("ranked_jobs", []))
    result = {
        **result,
        "tracked_applications": persistence["tracked_applications"],
        "candidate_id": persistence["candidate_id"],
    }
    if persistence.get("persistence_warning"):
        result["persistence_warning"] = persistence["persistence_warning"]
    _last_tracked_applications = persistence["tracked_applications"]

    _last_match_result = result
    sync_linkedin_results_for_shared_tools(result)

    # Full descriptions are retained in module state for the cover-letter
    # stage, but excluded from the LLM observation to save context tokens.
    compact_result = {
        **result,
        "ranked_jobs": [
            {key: value for key, value in job.items() if key != "description"}
            for job in result["ranked_jobs"]
        ],
    }
    return json.dumps(compact_result, indent=2, ensure_ascii=False)
