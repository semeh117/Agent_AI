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
_last_search_results = []
def set_candidate_email(email: str) -> None:
    """Called once, before the agent runs — see job_evaluator.py's
    set_candidate_profile(), which calls this automatically so no other
    file needs to change."""
    global _candidate_email
    _candidate_email = email


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
    results_count=min(results_count,20)
    from core.seen_jobs_memory import get_seen_urls

    seen_urls = get_seen_urls(_candidate_email) if _candidate_email else set()

    fresh_jobs=[]
    all_seen_this_search = []

    MAX_PAGE_ATTEMPTS = 3

    for page in range (1  ,MAX_PAGE_ATTEMPTS+1):
        page_jobs =_fetch_page(query ,limit=min(results_count *2, 10), page =page)
        if not page_jobs:
            break # himalays has no more results for this query 

        for job in page_jobs:
            all_seen_this_search.append(job)
            if job["url"] and job["url"] not in seen_urls:
                if job["url"] not in {j["url"] for j in fresh_jobs}:#de-dupe within this search too
                    fresh_jobs.append(job)


        if len(fresh_jobs) >= results_count:
            print(f"Loaded {len(fresh_jobs)} fresh jobs for '{query}' "
                  f"(page {page}, {len(seen_urls)} previously seen filtered out)")
            return json.dumps(fresh_jobs[:results_count], indent=2, ensure_ascii=False)

        print(f"  Page {page}: only {len(fresh_jobs)} fresh job(s) so far "
              f"(candidate has already seen {len(seen_urls)} jobs total) — trying next page...")

        if len(fresh_jobs)< results_count and all_seen_this_search:
            categories_tried =set()
            for job in all_seen_this_search:
                for category in job.get("categories",[]):
                    if category in categories_tried:
                        continue
                    categories_tried.add(category)

                    fallback_query = category.replace("-", " ")
                print(f"  All pages exhausted for '{query}' — falling back to "
                      f"category-based search: '{fallback_query}'")

                fallback_jobs = _fetch_page(fallback_query, limit=max(results_count * 2, 10), page=1)
                for job2 in fallback_jobs:
                    if (job2["url"] and job2["url"] not in seen_urls
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
    global _last_search_results
    _last_search_results = fresh_jobs[:results_count]
    return json.dumps(fresh_jobs[:results_count], indent=2, ensure_ascii=False)