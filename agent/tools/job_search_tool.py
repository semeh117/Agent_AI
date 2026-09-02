"""
job_search_tool.py
--------------
Agent-facing wrapper around search/job_search.py's _search_jobs_core().
Same underlying paging/fallback/seen-filtering logic as search_real_jobs,
but the Observation returned here includes a natural-language note when
jobs were filtered out because this candidate already evaluated them in
a previous run.

Why a separate tool instead of changing search_real_jobs itself:
search_real_jobs keeps a strict JSON-list output contract. The agent, by
contrast, reads the Observation as a string, so it's free to include prose
framing the model can actually reason from in its next Thought (e.g.
"3 jobs were already seen — I should refine my query" instead of
silently getting a shorter list with no explanation).

This remains a separate, purpose-built wrapper rather than overloading
one function with two different output contracts.
"""

import json
from langchain_core.tools import tool
from search.job_search import _search_jobs_core

# Only metadata needed for the next ReAct decision is returned to the agent.
# Full descriptions remain in search.job_search._last_search_results, where
# evaluate_job_match and write_cover_letter retrieve them by URL. Keeping the
# descriptions out of the scratchpad prevents each subsequent LLM call from
# resending tens of thousands of tokens.
AGENT_VISIBLE_JOB_FIELDS = (
    "title",
    "company",
    "employment_type",
    "seniority",
    "url",
    "salary_min",
    "salary_max",
)


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

    Output: a short status line, then a compact JSON list containing title,
    company, employment type, seniority, URL, and available salary. Full job
    descriptions are retained internally and recovered by URL during matching
    and cover-letter generation; they are intentionally excluded here to keep
    the agent context within provider token limits.

    If the status note says jobs were filtered out as already-seen, or
    that fewer postings came back than requested, do not just proceed
    with a short list — refine the query and call this tool again before
    moving on to evaluation.
    """
    result = _search_jobs_core(query, results_count)

    # Build new compact dictionaries; never mutate _search_jobs_core's jobs,
    # because those same objects hold the full descriptions used downstream.
    jobs = [
        {field: job.get(field) for field in AGENT_VISIBLE_JOB_FIELDS}
        for job in result["jobs"]
    ]
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
