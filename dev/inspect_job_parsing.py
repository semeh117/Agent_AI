"""Interactive LinkedIn job parsing review for the Agent 2 hybrid pipeline.

Asks the user for a search query (and optional location), scrapes a few real
LinkedIn job postings, then shows each posting's region slicing and the
structured parsing result field by field.

Usage:
    python dev/inspect_job_parsing.py
    python dev/inspect_job_parsing.py --query "machine learning engineer"
    python dev/inspect_job_parsing.py --query "ml engineer" --location "Tunisia" --count 2
    python dev/inspect_job_parsing.py --no-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent2_job_parser import _find_job_section_headings, _job_regions  # noqa: E402
from core.agent2_parser import extract_job_requirements_agent2  # noqa: E402


DESCRIPTION_PREVIEW_CHARS = 900


def _choose_query() -> tuple[str, str]:
    while True:
        query = input("LinkedIn search query (e.g. 'machine learning engineer'): ").strip()
        if not query:
            print("A query is required.")
            continue
        location = input("Location (blank = anywhere): ").strip()
        return query, location


def _print_headings(description: str) -> None:
    headings = _find_job_section_headings(description)
    print("\nDetected section headings (in order):")
    if not headings:
        print("  - none -> the fallback requirement-sentence filter will be used.")
        return
    for heading in headings:
        print(
            f"  - {heading.label!r}  ->  {heading.kind}"
            f"   (chars {heading.start}..{heading.end})"
        )


def _print_regions(description: str) -> None:
    regions = _job_regions(description)
    print("\nRegions fed to the parser:")
    for label in ("responsibility", "required", "preferred", "ignored"):
        text = getattr(regions, label)
        print(f"\n  [{label}] ({len(text)} chars)"
              + ("" if label != "ignored" else "  -- never sent to the LLM"))
        for line in (text or "(empty)").splitlines()[:12]:
            print(f"    {line}")


def _print_evidence(
    skills: list[str], evidence: dict[str, str], label: str
) -> None:
    print(f"\n  {label} ({len(skills)}):")
    if not skills:
        print("    - none")
        return
    for skill in skills:
        excerpt = evidence.get(skill, "")
        print(f"    - {skill!r}  (evidence: ...{excerpt[:80]}...)")


def _print_parse(job: dict[str, Any], use_cache: bool) -> None:
    title = str(job.get("title") or "").strip()
    description = str(job.get("description") or "").strip()

    print(f"\n{'=' * 72}")
    print(f"JOB: {title}  @ {job.get('company') or 'unknown company'}")
    print(f"URL: {job.get('url') or 'unknown'}")
    print(
        f"Type: {job.get('employment_type') or '?'}"
        f"  |  Seniority: {', '.join(job.get('seniority') or []) or '?'}"
        f"  |  Salary: {job.get('salary_min')}-{job.get('salary_max')}"
    )
    print("=" * 72)

    if not description:
        print("No description scraped - nothing to parse.")
        return

    print(f"\nRaw description ({len(description)} chars, first "
          f"{DESCRIPTION_PREVIEW_CHARS} shown):")
    print("-" * 72)
    print(description[:DESCRIPTION_PREVIEW_CHARS])
    print("-" * 72)

    _print_headings(description)
    _print_regions(description)

    try:
        requirements = extract_job_requirements_agent2(
            job_title=title,
            job_description=description,
            use_cache=use_cache,
        )
    except Exception as exc:
        print(f"\nParsing refused: {type(exc).__name__}: {exc}")
        return

    print(f"\nParsed job title: {requirements.job_title or 'not returned'}")
    print(f"Seniority: {requirements.seniority_level or 'not returned'}")
    print(
        f"Required experience: "
        f"{requirements.required_experience_years or 'not stated'} years"
    )
    print(
        f"Required education: "
        f"{requirements.required_education_level or 'not stated'}"
    )
    print(
        f"Preferred education: "
        f"{requirements.preferred_education_level or 'not stated'}"
    )

    _print_evidence(
        requirements.required_skills,
        requirements.required_skill_evidence,
        "Required skills",
    )
    _print_evidence(
        requirements.preferred_skills,
        requirements.preferred_skill_evidence,
        "Preferred skills",
    )

    print(
        f"\n  Required alternative groups "
        f"({len(requirements.required_skill_groups)}):  "
        f"scoring counts each as 'satisfy at least one'"
    )
    if not requirements.required_skill_groups:
        print("    - none")
    for group in requirements.required_skill_groups:
        print(f"    - {group}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape a few live LinkedIn jobs and review their parsing."
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Search query; omit to be prompted interactively.",
    )
    parser.add_argument(
        "--location",
        default="",
        help="Optional location filter.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="How many jobs to scrape (default 2).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh parser LLM calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    use_cache = not args.no_cache

    query = (args.query or "").strip()
    location = (args.location or "").strip()
    if not query:
        query, location = _choose_query()
    if args.count <= 0:
        raise ValueError("--count must be greater than zero.")

    from search.job_scraper import search_jobs

    print(f"Searching LinkedIn for {query!r}"
          + (f" in {location!r}" if location else " anywhere"))
    print("Starting real browser scraping - this can take a minute.\n")
    jobs = search_jobs(
        query=query,
        location=location,
        max_jobs=args.count,
    )
    print(f"Scraped {len(jobs)} job(s).")

    if not jobs:
        print("No jobs were scraped; nothing to parse.")
        return 1

    for job in jobs:
        _print_parse(job, use_cache=use_cache)
        print()

    print("Done. Reruns reuse the parser cache for identical descriptions; "
          "use --no-cache to force fresh LLM calls.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())