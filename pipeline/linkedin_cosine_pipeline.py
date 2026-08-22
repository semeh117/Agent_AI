"""Reusable LinkedIn parsing and cosine-ranking pipeline for Agent 2.

The pipeline receives an already-parsed candidate profile. Query generation,
scraping, job parsing, and ranking are coordinated here so the future agent
only has to decide when to call this workflow and how to present its result.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from core.cosine_matcher import rank_jobs_by_cosine


def build_linkedin_query(cv_info: Any, max_skills: int = 3) -> str:
    """Build a compact LinkedIn query from the candidate's parsed CV."""

    if max_skills < 0:
        raise ValueError("max_skills cannot be negative.")

    titles = [
        str(title).strip()
        for title in (getattr(cv_info, "job_titles", []) or [])
        if str(title).strip()
    ]
    skills = [
        str(skill).strip()
        for skill in (getattr(cv_info, "skills", []) or [])
        if str(skill).strip()
    ]

    terms: list[str] = []
    if titles:
        terms.append(titles[0])
    terms.extend(skills[:max_skills])

    if not terms:
        raise ValueError("The candidate has no job titles or skills for a query.")

    return " ".join(terms)


def match_linkedin_jobs(
    cv_info: Any,
    query: Optional[str] = None,
    location: str = "",
    max_jobs: int = 3,
    use_cache: bool = True,
    *,
    search_fn: Optional[Callable[..., list[dict[str, Any]]]] = None,
    parser_fn: Optional[Callable[..., Any]] = None,
    embeddings: Optional[Any] = None,
) -> dict[str, Any]:
    """Scrape, parse, and rank LinkedIn jobs for one candidate.

    ``search_fn``, ``parser_fn``, and ``embeddings`` are injectable so this
    orchestration can be tested without Chrome, an LLM, or a downloaded
    embedding model. In production, omitting them selects the real project
    implementations.
    """

    if max_jobs <= 0:
        raise ValueError("max_jobs must be greater than zero.")

    resolved_query = (query or "").strip() or build_linkedin_query(cv_info)
    resolved_location = (location or "").strip()

    if search_fn is None:
        # Lazy imports keep unit tests independent from Selenium and the LLM
        # provider when fake implementations are injected.
        from search.job_scraper import search_jobs

        search_fn = search_jobs

    if parser_fn is None:
        from core.job_parser import extract_job_requirements

        parser_fn = extract_job_requirements

    scraped_jobs = search_fn(
        query=resolved_query,
        location=resolved_location,
        max_jobs=max_jobs,
    )

    parsed_jobs: list[tuple[dict[str, Any], Any]] = []
    skipped_jobs: list[dict[str, str]] = []

    for job in scraped_jobs:
        title = str(job.get("title") or "").strip()
        description = str(job.get("description") or "").strip()

        if not title or not description:
            missing = "title" if not title else "description"
            skipped_jobs.append(
                {
                    "title": title,
                    "company": str(job.get("company") or ""),
                    "url": str(job.get("url") or ""),
                    "reason": f"Missing {missing}.",
                }
            )
            continue

        try:
            requirements = parser_fn(
                job_title=title,
                job_description=description,
                use_cache=use_cache,
            )
        except Exception as exc:
            skipped_jobs.append(
                {
                    "title": title,
                    "company": str(job.get("company") or ""),
                    "url": str(job.get("url") or ""),
                    "reason": f"Requirement parsing failed: {exc}",
                }
            )
            continue

        parsed_jobs.append((job, requirements))

    ranked_jobs = (
        rank_jobs_by_cosine(cv_info, parsed_jobs, embeddings=embeddings)
        if parsed_jobs
        else []
    )

    return {
        "query": resolved_query,
        "location": resolved_location,
        "requested_count": max_jobs,
        "scraped_count": len(scraped_jobs),
        "parsed_count": len(parsed_jobs),
        "skipped_count": len(skipped_jobs),
        "skipped_jobs": skipped_jobs,
        "ranked_jobs": ranked_jobs,
    }
