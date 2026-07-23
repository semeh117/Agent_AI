"""
job_search.py
--------------
Real job search via the Adzuna API — no scraping, legitimate REST API,
returns real live job postings (title, company, description, url).
"""

import os
import requests
from langchain_core.tools import tool

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "gb")
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"


@tool
def search_real_jobs(query: str, results_count: int = 5) -> str:
    """
    Searches for real, currently active job postings using the Adzuna API.

    Input:
        query: search terms, e.g. "Data Scientist Machine Learning"
        results_count: how many postings to return (default 5, max 20)

    Output: JSON string — a list of job postings, each with:
        title, company, location, description, url, salary (if available)
    """
    url = f"{ADZUNA_BASE_URL}/{ADZUNA_COUNTRY}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": min(results_count, 20),
        "what": query,
        "content-type": "application/json",
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        return f'{{"error": "Adzuna API returned status {response.status_code}: {response.text[:200]}"}}'

    data = response.json()
    jobs = []
    for job in data.get("results", []):
        jobs.append({
            "title": job.get("title", ""),
            "company": job.get("company", {}).get("display_name", "Unknown"),
            "location": job.get("location", {}).get("display_name", "Unknown"),
            "description": job.get("description", ""),
            "url": job.get("redirect_url", ""),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
        })

    import json
    return json.dumps(jobs, indent=2)