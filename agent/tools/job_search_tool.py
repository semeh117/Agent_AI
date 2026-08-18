"""
job_search_tool.py
--------------
Agent-facing wrapper around search/job_search.py's _search_jobs_core().
Same underlying paging/fallback/seen-filtering logic as search_real_jobs,
but the Observation returned here includes a natural-language note when
jobs were filtered out because this candidate already evaluated them in
a previous run.

Why a separate tool instead of changing search_real_jobs itself:
search_real_jobs's output is consumed by json.loads() directly in
job_matching_pipeline.py and dev/capture_job_fixture.py — it must stay
a strict JSON string of a list, no extra text. The agent, by contrast,
just reads the Observation as a string, so it's free to include prose
framing the model can actually reason from in its next Thought (e.g.
"3 jobs were already seen — I should refine my query" instead of
silently getting a shorter list with no explanation).

Same module pattern as search/job_search_fixture.py and dev/cv_fixture.py
in this project: a separate, purpose-built function sitting alongside
the "real" one rather than overloading a single function with two
different output contracts.
"""

import json
import os
from langchain_core.tools import tool
from search.job_search import _search_jobs_core

# Description length shown in the Observation. Runs are kept small
# (results_count=3), so the agent sees the FULL text of each posting —
# it needs as much of the description as possible to judge relevance and
# ground the cover letter. The cap is only a safety net for a
# pathological oversized posting: with a multi-step ReAct loop every
# Observation is re-sent on the next call, so a truly enormous description
# would still blow through small-model context windows. The FULL stored
# text in job_search._last_search_results is never affected by this cap.
DESCRIPTION_MAX_CHARS = int(os.getenv("JOB_DESCRIPTION_MAX_CHARS", "2500"))
DESCRIPTION_SUMMARY_CHARS = int(os.getenv("JOB_DESCRIPTION_SUMMARY_CHARS", "20000"))


def _shorten(description: str, limit: int = DESCRIPTION_MAX_CHARS) -> str:
    if len(description) <= limit:
        return description
    return description[:limit].rstrip() + " … [truncated]"


@tool
def search_jobs_for_agent(query: str, results_count: int = 3) -> str:
    """
    Searches for real, currently active remote job postings matching the
    query. Same underlying search as search_real_jobs (paging, seen-job
    filtering, category-based fallback), but the Observation starts with
    a short status note before the JSON list.

    Input:
        query: search terms, e.g. "full stack developer"
        results_count: how many fresh postings to return (default 3)

    Output: a short status line, then a JSON list of job postings (same
    fields as search_real_jobs: title, company, employment_type,
    seniority, description, categories, url, salary).

    If the status note says jobs were filtered out as already-seen, or
    that fewer postings came back than requested, do not just proceed
    with a short list — refine the query and call this tool again before
    moving on to evaluation.
    """
    result = _search_jobs_core(query, results_count)

    # Shallow-copy each job BEFORE shortening: _search_jobs_core already
    # stored these SAME dicts (by reference) in job_search._last_search_results,
    # and evaluate_job_match + the cover-letter step fetch the FULL posting
    # back from there. Mutating them in place would overwrite the full
    # description with the summary and silently degrade scoring.
    jobs = [dict(job) for job in result["jobs"]]
    for job in jobs:
        job["description"] = _shorten(
            job.get("description", ""), DESCRIPTION_SUMMARY_CHARS
        )
    filtered = result["filtered_seen_count"]
    returned = result["returned_count"]
    requested = result["requested_count"]

    if filtered > 0:
        note = (
            f"Note: {filtered} job posting(s) matching this query were filtered "
            f"out because you already evaluated them for this candidate in a "
            f"previous run. Returning {returned} fresh posting(s) instead."
        )
    else:
        note = f"Returning {returned} fresh posting(s)."

    if returned < requested:
        note += (
            f" This is fewer than the {requested} requested — consider trying "
            f"a different or more specific search query (e.g. different skill "
            f"keywords or job title) to surface other relevant postings before "
            f"evaluating what you have."
        )

    return f"{note}\n\n{json.dumps(jobs, indent=2, ensure_ascii=False)}"