"""Agent-facing LinkedIn matching tool for Agent 2.

The future Agent 2 decides what query and location to use. This tool runs the
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


def set_candidate_profile(
    cv_info: Any,
    location: str = "",
    results_count: int = 3,
) -> None:
    """Set the candidate and run options used by subsequent tool calls."""

    global _current_cv_info, _current_location, _current_results_count
    global _last_match_result
    _current_cv_info = cv_info
    _current_location = (location or "").strip()
    _current_results_count = max(1, min(int(results_count), 20))
    _last_match_result = None


def get_last_match_result() -> dict[str, Any] | None:
    """Return the full latest result, including stored job descriptions."""

    return _last_match_result


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
    ``skills_score`` (cosine similarity), ``experience_score``,
    ``education_score``, and the weighted ``final_score`` calculated as:

        skills_score * 0.5
        + experience_score * 0.3
        + education_score * 0.2

    Jobs are already sorted by ``final_score`` descending. Do not recalculate
    or reorder the scores in the agent response.
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

    global _last_match_result

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

    _last_match_result = result

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
