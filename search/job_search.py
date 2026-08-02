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


def _clean_description(raw_html: str) -> str:
    """Strips HTML tags from the description for clean LLM input."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


@tool
def search_real_jobs(query: str, results_count: int = 5) -> str:
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
    params = {
        "q": query,
        "limit": min(results_count, 20),
        "page": 1,
    }

    response = requests.get(HIMALAYAS_SEARCH_URL, params=params, timeout=30)
    if response.status_code != 200:
        return f'{{"error": "Himalayas API returned status {response.status_code}"}}'

    data = response.json()

    raw_jobs = data.get("jobs", [])[:results_count]

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

    print(f"Loaded {len(jobs)} jobs from Himalayas API for query '{query}'")

    return json.dumps(jobs, indent=2, ensure_ascii=False)