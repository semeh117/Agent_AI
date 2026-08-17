"""
job_search.py
--------------
Real job search via Himalayas' public API. Free, no API key required.
Himalayas returns FULL job descriptions (not truncated snippets), which
is essential for reliable skill extraction, and uses proper server-side
keyword search — unlike RemoteOK, which was dropped after an AND/OR
matching bug, and Adzuna, which was dropped for truncated descriptions.
"""

import html
import re
import json
import requests
from langchain_core.tools import tool

HIMALAYAS_SEARCH_URL = "https://himalayas.app/jobs/api/search"
_candidate_email = None
_last_search_results = []   # every job returned so far THIS run — lets
                            # job_evaluator.py recover a dropped URL by
                            # matching title+company against what we returned
_session_returned_urls = set()  # URLs surfaced earlier THIS run — a second
                                # search_real_jobs call must skip them too,
                                # even if the agent hasn't evaluated them yet
def set_candidate_email(email: str) -> None:
    """Called once, before the agent runs — see job_evaluator.py's
    set_candidate_profile(), which calls this automatically so no other
    file needs to change. Resets per-run search memory so a fresh run
    (or a different candidate) starts with an empty session."""
    global _candidate_email, _last_search_results, _session_returned_urls
    _candidate_email = email
    _last_search_results = []
    _session_returned_urls = set()


def _is_seen(job: dict, seen_urls: set) -> bool:
    """True if this candidate already saw this posting. Matches by URL when
    the posting has one, OR by the `missing|title|company` key used when a
    previous evaluation was recorded without a URL (see record_seen)."""
    if job.get("url") and job["url"] in seen_urls:
        return True
    return f"missing|{job.get('title', '').strip()}|{job.get('company', '').strip()}" in seen_urls


def _record_returned(jobs: list) -> None:
    """Remember the jobs we just surfaced, so (1) job_evaluator can recover
    URLs from ANY of this run's searches, and (2) the next search_real_jobs
    call filters them out even if nothing has been scored/recorded yet."""
    global _last_search_results, _session_returned_urls
    for job in jobs:
        if job not in _last_search_results:
            _last_search_results.append(job)
        if job.get("url"):
            _session_returned_urls.add(job["url"])


def _clean_description(raw_html: str) -> str:
    """Strips HTML tags from the description for clean LLM input."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def _fetch_page(query: str, limit: int, page: int) -> list:
    """One raw call to Himalayas, cleaned into this project's job shape.
    No filtering applied here — that happens in the caller."""
    params = {"q": query, "limit": limit, "page": page}
    response = requests.get(HIMALAYAS_SEARCH_URL, params=params, timeout=30)
    if response.status_code != 200:
        return []

    data = response.json()
    raw_jobs = data.get("jobs", [])

    jobs = []
    for job in raw_jobs:
        jobs.append({
            "title": job.get("title", ""),
            "company": job.get("companyName", ""),
            "employment_type": job.get("employmentType", ""),
            "seniority": job.get("seniority", []),
            "description": _clean_description(job.get("description", "")),
            "categories": job.get("categories", []),
            "url": job.get("applicationLink", ""),
            "salary_min": job.get("minSalary"),
            "salary_max": job.get("maxSalary"),
        })
    return jobs



@tool
def search_real_jobs(query: str, results_count: int = 3) -> str:
    """
    Searches Himalayas for real, currently active remote job postings
    using their built-in keyword search (not client-side filtering).

    Input:
        query: search terms, e.g. "full stack developer"
        results_count: how many postings to return (default 5, max 20)

    Output: JSON string — list of job postings, each with:
        title, company, employment_type, seniority, full cleaned
        description, categories, url, salary (if available).
    """
    query = query.strip()
    if query.startswith("{") and query.endswith("}"):
        try:
            parsed = json.loads(query)
            if isinstance(parsed, dict) and "query" in parsed:
                query = parsed["query"]
        except json.JSONDecodeError:
            pass
    results_count = min(results_count, 20)
    from core.seen_jobs_memory import get_seen_urls

    # Previously-scored jobs (this candidate's memory) PLUS anything already
    # surfaced earlier in THIS run — the second guarantees a repeat
    # search_real_jobs call never re-serves the same postings, even before
    # the agent has evaluated/recorded them.
    seen_urls = get_seen_urls(_candidate_email) if _candidate_email else set()
    seen_urls = seen_urls | set(_session_returned_urls)

    fresh_jobs = []
    all_seen_this_search = []

    MAX_PAGE_ATTEMPTS = 3

    for page in range(1, MAX_PAGE_ATTEMPTS + 1):
        page_jobs = _fetch_page(query, limit=min(results_count * 2, 10), page=page)
        if not page_jobs:
            break  # himalays has no more results for this query

        for job in page_jobs:
            all_seen_this_search.append(job)
            # A scored posting is remembered by URL when we had one, or by a
            # `missing|title|company` key when we didn't (see record_seen).
            # Skip it if EITHER form says this candidate already saw it.
            if not _is_seen(job, seen_urls):
                if job["url"] not in {j["url"] for j in fresh_jobs}:  # de-dupe within this search too
                    fresh_jobs.append(job)

        if len(fresh_jobs) >= results_count:
            _record_returned(fresh_jobs[:results_count])
            print(f"Loaded {len(fresh_jobs)} fresh jobs for '{query}' "
                  f"(page {page}, {len(seen_urls)} previously seen filtered out)")
            return json.dumps(fresh_jobs[:results_count], indent=2, ensure_ascii=False)

        print(f"  Page {page}: only {len(fresh_jobs)} fresh job(s) so far "
              f"(candidate has already seen {len(seen_urls)} jobs total) — trying next page...")

        # Paging ran dry for this query — fall back to a category-based search,
        # trying each DISTINCT category seen in what the API returned, until we
        # hit the requested count or run out of categories.
        if len(fresh_jobs) < results_count and all_seen_this_search:
            categories_tried = set()
            for job in all_seen_this_search:
                for category in job.get("categories", []):
                    if category in categories_tried:
                        continue
                    categories_tried.add(category)

                    fallback_query = category.replace("-", " ")
                    print(f"  All pages exhausted for '{query}' — falling back to "
                          f"category-based search: '{fallback_query}'")

                    fallback_jobs = _fetch_page(fallback_query, limit=max(results_count * 2, 10), page=1)
                    for job2 in fallback_jobs:
                        if (not _is_seen(job2, seen_urls)
                                and job2["url"] not in {j["url"] for j in fresh_jobs}):
                            fresh_jobs.append(job2)

                    if len(fresh_jobs) >= results_count:
                        break
            if len(fresh_jobs) >= results_count:
                break

    if not fresh_jobs:
        print(f"  No new jobs found for '{query}' after paging and category fallback — "
              f"candidate may have already seen everything currently available.")

    print(f"Loaded {len(fresh_jobs)} fresh job(s) total for '{query}' "
          f"({len(seen_urls)} previously seen filtered out)")
    _record_returned(fresh_jobs[:results_count])
    return json.dumps(fresh_jobs[:results_count], indent=2, ensure_ascii=False)