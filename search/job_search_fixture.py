"""
job_search_fixture.py
--------------
Deterministic, offline stand-in for search_real_jobs() — reads a frozen
JSON snapshot from disk instead of calling the live Himalayas API.

This is a SEPARATE function from search_real_jobs (job_search.py), on
purpose: the live tool is what the agent and the fixed pipeline actually
use in production, and neither of those files is touched by this one.
Use this only when you want to test prompt/logic changes (e.g. agent.py's
ReAct prompt, matcher.py's scoring) against a job pool that's guaranteed
NOT to change between two runs — isolating "did my code change do
something different" from "did the live job pool just change."

Not a LangChain @tool — this is a plain Python function meant to be
called directly from a dev/test script, not handed to an agent.

Usage:
    from job_search_fixture import search_jobs_from_fixture
    jobs_json = search_jobs_from_fixture(results_count=3)
"""

import json
from pathlib import Path

DEFAULT_FIXTURE_PATH = "fixtures/job_search_fixture.json"


def search_jobs_from_fixture(results_count: int = 5, fixture_path: str = DEFAULT_FIXTURE_PATH) -> str:
    """
    Returns the same JSON-string-of-a-list shape as search_real_jobs(),
    but sourced from a frozen fixture file instead of a live API call.

    Input:
        results_count: how many postings to return from the fixture
            (capped to however many the fixture actually contains).
        fixture_path: path to a fixture JSON file produced by
            capture_job_fixture.py. Defaults to the standard location.

    Output: JSON string — list of job postings, same shape as
        search_real_jobs's output. Returns an error-object JSON string
        (matching search_real_jobs's own error format) if the fixture
        file doesn't exist yet.
    """
    path = Path(fixture_path)
    if not path.exists():
        return (
            f'{{"error": "Fixture not found at {fixture_path}. '
            f'Run capture_job_fixture.py first to create one."}}'
        )

    fixture = json.loads(path.read_text(encoding="utf-8"))
    jobs = fixture.get("jobs", [])[:results_count]

    print(f"Loaded {len(jobs)} jobs from fixture '{fixture_path}' "
          f"(captured query: '{fixture.get('captured_query', 'unknown')}')")

    return json.dumps(jobs, indent=2, ensure_ascii=False)