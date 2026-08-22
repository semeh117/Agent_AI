"""Live end-to-end test for the LinkedIn cosine-matching pipeline.

Flow:
    CV PDF
      -> CV text extraction
      -> structured CV parsing
      -> LinkedIn query construction
      -> LinkedIn scraping
      -> structured parsing of every scraped job
      -> embedding generation
      -> weighted skills/experience/education ranking

This is intentionally a manual integration test: it opens Chrome, accesses
LinkedIn, calls the configured LLM, and may download the embedding model on
its first run. It is not executed automatically by the unit test suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_linkedin_query(cv_info: Any) -> str:
    """Build a compact first-pass LinkedIn query from the parsed CV."""

    titles = [title.strip() for title in cv_info.job_titles if title.strip()]
    skills = [skill.strip() for skill in cv_info.skills if skill.strip()]

    terms: list[str] = []
    if titles:
        terms.append(titles[0])
    terms.extend(skills[:3])

    if not terms:
        raise ValueError("The parsed CV contains no job titles or skills.")

    return " ".join(terms)


def print_cv_profile(cv_info: Any) -> None:
    print("\nPARSED CV")
    print("-" * 80)
    print(f"Candidate:  {cv_info.full_name or 'Not extracted'}")
    print(f"Roles:      {', '.join(cv_info.job_titles) or 'None'}")
    print(f"Experience: {cv_info.experience_years} years")
    print(f"Education:  {cv_info.highest_education_level or 'Not extracted'}")
    print(f"Skills ({len(cv_info.skills)}):")
    for skill in cv_info.skills:
        print(f"  - {skill}")


def print_job_requirements(index: int, job: dict, parsed: Any) -> None:
    print(f"\nPARSED JOB {index}")
    print("-" * 80)
    print(f"Title:       {job.get('title') or 'Not scraped'}")
    print(f"Company:     {job.get('company') or 'Not scraped'}")
    print(f"URL:         {job.get('url') or 'Not scraped'}")
    print(f"Description: {len(job.get('description', ''))} characters")
    print(f"Seniority:   {parsed.seniority_level or 'Not extracted'}")
    print(
        "Experience:  "
        f"{parsed.required_experience_years} years"
        if parsed.required_experience_years is not None
        else "Experience:  Not specified"
    )
    print(f"Education:   {parsed.required_education_level or 'Not specified'}")
    print(f"Skills ({len(parsed.required_skills)}):")
    if parsed.required_skills:
        for skill in parsed.required_skills:
            print(f"  - {skill}")
    else:
        print("  - None extracted")


def print_ranking(ranked_jobs: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("FINAL WEIGHTED JOB-MATCH RANKING")
    print("=" * 80)

    for position, result in enumerate(ranked_jobs, start=1):
        status = "INCONCLUSIVE" if result["inconclusive"] else "SCORED"
        print(
            f"\n{position}. {result['job_title']} @ "
            f"{result['company']} - {result['final_score']:.1f}% [{status}]"
        )
        print(
            "   Components: "
            f"skills={result['skills_score']:.1f}% x 0.5, "
            f"experience={result['experience_score']:.1f}% x 0.3, "
            f"education={result['education_score']:.1f}% x 0.2"
        )
        print(f"   URL: {result['url']}")
        print(
            "   Required skills: "
            + (", ".join(result["required_skills"]) or "None extracted")
        )
        print(
            "   Experience: candidate="
            f"{result['candidate_experience_years']}, "
            f"required={result['required_experience_years']}"
        )
        print(
            "   Education: candidate="
            f"{result['candidate_education']}, "
            f"required={result['required_education']}"
        )


def run_pipeline(
    cv_path: Path,
    location: str,
    max_jobs: int,
    query: str | None = None,
    use_cache: bool = False,
) -> list[dict]:
    # Imports are intentionally local: merely importing this manual test must
    # not launch browser/LLM dependencies during ordinary test discovery.
    from core.cosine_matcher import rank_jobs_by_cosine
    from core.cv_parser import extract_cv_info, extract_text_from_pdf
    from core.job_parser import extract_job_requirements
    from search.job_scraper import search_jobs

    print("=" * 80)
    print("CV -> LINKEDIN -> JOB PARSER -> COSINE MATCHING")
    print("=" * 80)

    if not cv_path.is_file():
        raise FileNotFoundError(f"CV file does not exist: {cv_path}")

    print(f"\n[1/5] Reading CV: {cv_path}")
    cv_text = extract_text_from_pdf(cv_path)
    print(f"Extracted {len(cv_text)} characters from the PDF.")

    print("\n[2/5] Parsing CV with the configured LLM...")
    cv_info = extract_cv_info(cv_text, use_cache=use_cache)
    print_cv_profile(cv_info)

    linkedin_query = (query or "").strip() or build_linkedin_query(cv_info)
    print("\n[3/5] Scraping LinkedIn jobs...")
    print(f"Query:    {linkedin_query}")
    print(f"Location: {location or 'Any location'}")
    print(f"Maximum:  {max_jobs}")

    jobs = search_jobs(
        query=linkedin_query,
        location=location,
        max_jobs=max_jobs,
    )
    print(f"Scraped {len(jobs)} job(s).")

    if not jobs:
        raise RuntimeError("LinkedIn returned no usable jobs.")

    print("\n[4/5] Parsing scraped job descriptions...")
    parsed_jobs: list[tuple[dict, Any]] = []

    for index, job in enumerate(jobs, start=1):
        description = (job.get("description") or "").strip()
        title = (job.get("title") or "").strip()

        if not title or not description:
            print(
                f"\nSkipping job {index}: missing "
                f"{'title' if not title else 'description'}."
            )
            continue

        try:
            parsed = extract_job_requirements(
                job_title=title,
                job_description=description,
                use_cache=use_cache,
            )
        except Exception as exc:
            print(f"\nSkipping job {index}: parsing failed: {exc}")
            continue

        parsed_jobs.append((job, parsed))
        print_job_requirements(index, job, parsed)

    if not parsed_jobs:
        raise RuntimeError("None of the scraped jobs could be parsed.")

    print("\n[5/5] Calculating weighted job-match scores...")
    ranked_jobs = rank_jobs_by_cosine(cv_info, parsed_jobs)
    print_ranking(ranked_jobs)

    if len(ranked_jobs) != len(parsed_jobs):
        raise AssertionError("Every parsed job must appear in the final ranking.")

    conclusive = [job for job in ranked_jobs if not job["inconclusive"]]
    conclusive_scores = [job["final_match"] for job in conclusive]
    if conclusive_scores != sorted(conclusive_scores, reverse=True):
        raise AssertionError("Scored jobs are not ordered by descending similarity.")

    first_inconclusive = next(
        (index for index, job in enumerate(ranked_jobs) if job["inconclusive"]),
        len(ranked_jobs),
    )
    if any(not job["inconclusive"] for job in ranked_jobs[first_inconclusive:]):
        raise AssertionError("Inconclusive jobs must appear after scored jobs.")

    print("\n" + "=" * 80)
    print("FULL PIPELINE TEST PASSED")
    print("=" * 80)
    return ranked_jobs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live CV-to-LinkedIn cosine matching test."
    )
    parser.add_argument(
        "--cv",
        type=Path,
        default=PROJECT_ROOT / "cv" / "Semah_Mechi_.pdf",
        help="Path to the candidate CV PDF.",
    )
    parser.add_argument(
        "--location",
        default="Germany",
        help="LinkedIn job-search location; pass an empty string for any location.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=3,
        help="Maximum number of LinkedIn jobs to scrape.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional explicit query; otherwise it is built from the parsed CV.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached CV/job extractions instead of forcing fresh LLM parsing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.max_jobs <= 0:
        raise ValueError("--max-jobs must be greater than zero.")

    run_pipeline(
        cv_path=args.cv.resolve(),
        location=args.location.strip(),
        max_jobs=args.max_jobs,
        query=args.query,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    main()
