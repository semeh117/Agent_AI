"""Scrape real LinkedIn jobs and capture raw descriptions + parser outputs.

Usage:
    python -m dev.capture_linkedin_job_parser_fixture "AI Engineer" --count 3
    python -m dev.capture_linkedin_job_parser_fixture "AI Engineer" --location Germany
    python -m dev.capture_linkedin_job_parser_fixture --reparse-existing

Parsing is fresh by default so the fixture measures the currently configured
parser model. Pass ``--use-cache`` only when a repeatable cached capture is
preferred over evaluating the model again.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.agent2_parser import JOB_CACHE_VERSION, extract_job_requirements_agent2
from search.job_scraper import search_jobs


DEFAULT_OUTPUT = Path("fixtures/linkedin_job_parser_fixture.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture LinkedIn postings beside their parsed requirements."
    )
    parser.add_argument("query", nargs="?", help="LinkedIn job search query.")
    parser.add_argument("--location", default="")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reparse-existing",
        action="store_true",
        help="Reparse LinkedIn jobs already stored in --out without scraping.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached job parsing. Omit to test the parser model fresh.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than zero.")

    if args.reparse_existing:
        if not args.out.is_file():
            raise FileNotFoundError(f"Existing job fixture not found: {args.out}")
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        scraped_jobs = [
            entry["linkedin_job"]
            for entry in existing.get("jobs", [])
            if entry.get("linkedin_job")
        ]
        query = str(existing.get("query") or "")
        location = str(existing.get("location") or "")
        requested_count = int(existing.get("requested_count") or len(scraped_jobs))
    else:
        if not args.query:
            parser.error("query is required unless --reparse-existing is used.")
        query = args.query
        location = args.location
        requested_count = args.count
        scraped_jobs = search_jobs(
            query=query,
            location=location,
            max_jobs=args.count,
        )
    captured_jobs = []
    for job in scraped_jobs:
        title = str(job.get("title") or "").strip()
        description = str(job.get("description") or "").strip()
        entry = {
            "linkedin_job": {
                "title": title,
                "company": str(job.get("company") or ""),
                "url": str(job.get("url") or ""),
                "description": description,
            },
            "parsed_requirements": None,
            "parser_error": None,
        }
        if not title or not description:
            entry["parser_error"] = "Scraper returned a missing title or description."
        else:
            try:
                parsed = extract_job_requirements_agent2(
                    job_title=title,
                    job_description=description,
                    use_cache=args.use_cache,
                )
                entry["parsed_requirements"] = parsed.model_dump()
            except Exception as exc:
                entry["parser_error"] = str(exc)
        captured_jobs.append(entry)

    fixture = {
        "fixture_type": "linkedin_job_parser",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "LinkedIn public jobs scraper",
        "query": query,
        "location": location,
        "requested_count": requested_count,
        "scraped_count": len(scraped_jobs),
        "parser_provider": os.getenv("PARSER_PROVIDER", "openrouter"),
        "parser_model": os.getenv("PARSER_MODEL", "qwen/qwen-2.5-7b-instruct"),
        "parser_version": JOB_CACHE_VERSION,
        "used_extraction_cache": args.use_cache,
        "jobs": captured_jobs,
    }
    parsed_count = sum(
        entry["parsed_requirements"] is not None for entry in captured_jobs
    )
    if args.reparse_existing and scraped_jobs and parsed_count == 0:
        raise RuntimeError(
            "Every parser call failed; the existing LinkedIn fixture was left "
            "unchanged. Check provider connectivity and retry."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Captured LinkedIn parser fixture: {args.out}")
    source_action = "Loaded" if args.reparse_existing else "Scraped"
    print(
        f"{source_action} {len(scraped_jobs)} job(s); "
        f"parsed {parsed_count} successfully."
    )
    for index, entry in enumerate(captured_jobs, start=1):
        parsed = entry["parsed_requirements"]
        status = (
            f"{len(parsed['required_skills'])} skills"
            if parsed
            else f"ERROR: {entry['parser_error']}"
        )
        print(f"  {index}. {entry['linkedin_job']['title']} — {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
