"""
test_job_parser.py
------------------
Tests the Agent 2 job parser against real jobs scraped from LinkedIn.

Flow:
    LinkedIn scraper
        ↓
    Job title + description
        ↓
    agent2_parser.py
        ↓
    Extracted skills / seniority / experience / education
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from search.job_scraper import search_jobs
from core.agent2_parser import extract_job_requirements_agent2


def main() -> int:
    print("=" * 80)
    print("LINKEDIN → AGENT 2 JOB PARSER TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Scrape a small number of LinkedIn jobs
    # ---------------------------------------------------------

    print("\n[1] Scraping LinkedIn jobs...\n")

    jobs = search_jobs(
        query="Machine Learning Engineer",
        location="Germany",
        max_jobs=3,
    )

    print(f"Scraped {len(jobs)} jobs.")

    if not jobs:
        print("No jobs found.")
        return

    # ---------------------------------------------------------
    # 2. Parse every scraped job
    # ---------------------------------------------------------

    parsed_count = 0

    for index, job in enumerate(jobs, start=1):

        print("\n" + "=" * 80)
        print(f"JOB {index}")
        print("=" * 80)

        print(f"Title:   {job.get('title', '')}")
        print(f"Company: {job.get('company', '')}")
        print(f"URL:     {job.get('url', '')}")

        description = job.get("description", "")

        print(
            f"\nDescription: {len(description)} characters, "
            f"{len(description.splitlines())} preserved line(s)"
        )

        if not description:
            print("WARNING: Job has no description.")
            continue

        # -----------------------------------------------------
        # 3. Send the LinkedIn job to Agent 2's dedicated parser
        # -----------------------------------------------------

        print("\nExtracting requirements...")

        try:
            result = extract_job_requirements_agent2(
                job_title=job.get("title", ""),
                job_description=description,
                use_cache=False,
            )

        except Exception as e:
            print(f"ERROR while parsing job: {e}")
            continue

        parsed_count += 1

        # -----------------------------------------------------
        # 4. Display extracted information
        # -----------------------------------------------------

        print("\n--- EXTRACTED RESULT ---")

        print(f"Job title:              {result.job_title}")
        print(f"Seniority:              {result.seniority_level}")
        print(f"Experience:             {result.required_experience_years}")
        print(f"Education:              {result.required_education_level}")

        print("\nRequired skills:")

        if result.required_skills:
            for skill in result.required_skills:
                print(f"  - {skill}")
        else:
            print("  No skills extracted.")

        print("\nPreferred skills (not included in required-skill cosine scoring):")

        if result.preferred_skills:
            for skill in result.preferred_skills:
                print(f"  - {skill}")
        else:
            print("  No preferred skills extracted.")

        print("\nAlternative requirement groups:")
        if result.required_skill_groups:
            for group in result.required_skill_groups:
                print(f"  - {' OR '.join(group)}")
        else:
            print("  No alternative groups extracted.")

    print("\n" + "=" * 80)
    print("TEST FINISHED")
    print("=" * 80)
    print(f"Successfully parsed {parsed_count}/{len(jobs)} scraped jobs.")
    return 0 if parsed_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
