"""
capture_job_fixture.py
--------------
Run this manually, occasionally, to (re)freeze a real Himalayas response
to disk as a test fixture. This is NOT part of the pipeline or the agent
— it's a standalone dev utility.

Why this exists:
Himalayas is a live API — the same query can return different postings
on different days (postings get added/filled/removed), and the agent's
own search-query text can vary slightly run to run even at temperature=0.
That makes it impossible to tell "did my prompt change improve the
result" apart from "did the live job pool just change underneath me."

Usage:
    python capture_job_fixture.py "AI engineer python langchain"
    python capture_job_fixture.py "AI engineer python langchain" --out fixtures/ai_engineer.json
"""

import argparse
import json
from pathlib import Path
from search.job_search import search_real_jobs

DEFAULT_FIXTURE_PATH = "fixtures/job_search_fixture.json"


def main():
    parser = argparse.ArgumentParser(description="Capture a live Himalayas search result as a fixture.")
    parser.add_argument("query", help="Search query to send to Himalayas, e.g. 'AI engineer python langchain'")
    parser.add_argument("--results-count", type=int, default=5, help="How many postings to capture (default 5)")
    parser.add_argument("--out", default=DEFAULT_FIXTURE_PATH, help=f"Output path (default {DEFAULT_FIXTURE_PATH})")
    args = parser.parse_args()

    print(f"Calling live Himalayas API for query: '{args.query}'...")
    jobs_json = search_real_jobs.invoke({"query": args.query, "results_count": args.results_count})
    jobs = json.loads(jobs_json)

    if isinstance(jobs, dict) and "error" in jobs:
        print(f"  [ERROR] Capture failed: {jobs['error']}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixture = {
        "captured_query": args.query,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    out_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  -> Captured {len(jobs)} jobs to {out_path}")
    print("  This fixture is now frozen — reruns of search_jobs_from_fixture() "
          "will always return this exact same set until you recapture it.")


if __name__ == "__main__":
    main()