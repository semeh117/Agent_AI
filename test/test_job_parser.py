"""
test_job_parser.py
------------------
Tests the job_parser against real jobs scraped from LinkedIn.

Flow:
    LinkedIn scraper
        ↓
    Job title + description
        ↓
    job_parser.py
        ↓
    Extracted skills / seniority / experience / education
"""

from search.job_scraper import search_jobs
from core.job_parser import extract_job_requirements


def main():
    print("=" * 80)
    print("LINKEDIN → JOB PARSER TEST")
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

    for index, job in enumerate(jobs, start=1):

        print("\n" + "=" * 80)
        print(f"JOB {index}")
        print("=" * 80)

        print(f"Title:   {job.get('title', '')}")
        print(f"Company: {job.get('company', '')}")
        print(f"URL:     {job.get('url', '')}")

        description = job.get("description", "")

        print(f"\nDescription length: {len(description)} characters")

        if not description:
            print("WARNING: Job has no description.")
            continue

        # -----------------------------------------------------
        # 3. Send the LinkedIn job to job_parser.py
        # -----------------------------------------------------

        print("\nExtracting requirements...")

        try:
            result = extract_job_requirements(
                job_title=job.get("title", ""),
                job_description=description,
                use_cache=False,
            )

        except Exception as e:
            print(f"ERROR while parsing job: {e}")
            continue

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

    print("\n" + "=" * 80)
    print("TEST FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()