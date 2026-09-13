"""Run-scoped tools used by Agent 3's classic ReAct executor.

The tools deliberately expose separate search, evaluation, skill-gap, and
cover-letter actions. Heavy work remains in the existing deterministic
services; the ReAct model only decides which action to take next after reading
each compact observation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from next_chapter.pipelines.linkedin_matching import match_linkedin_jobs


def sync_linkedin_results_for_shared_tools(result: dict[str, Any]) -> None:
    """Expose Agent 3 rankings in the shape used by cover/delivery adapters."""

    import next_chapter.agents.tools.job_evaluator as job_evaluator

    job_evaluator._all_evaluations = [
        {
            **job,
            "score_percent": job.get("score_percent", job.get("final_score", 0.0)),
            "matching_skills": job.get("skills_detail", {}).get("matching", []),
            "missing_skills": job.get("skills_detail", {}).get("missing", []),
        }
        for job in result.get("ranked_jobs", [])
    ]


def persist_ranked_jobs(
    cv_info: Any,
    ranked_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save Agent 3 rankings through the shared application tracker."""

    tracked: list[dict[str, Any]] = []
    if not ranked_jobs:
        return {"tracked_applications": tracked, "candidate_id": ""}
    try:
        from next_chapter.services.application_tracker import save_application

        candidate_id: str | None = None
        for job in ranked_jobs:
            record = save_application(
                cv_info,
                job,
                status="discovered",
                candidate_id=candidate_id,
            )
            candidate_id = record.candidate_id
            tracked.append(
                {
                    "application_id": record.application_id,
                    "job_id": record.job_id,
                    "url": record.url,
                }
            )
        return {
            "tracked_applications": tracked,
            "candidate_id": candidate_id or "",
        }
    except Exception as exc:
        return {
            "tracked_applications": tracked,
            "candidate_id": "",
            "persistence_warning": (
                "Ranked jobs could not be saved to the application tracker "
                f"({type(exc).__name__}: {exc})."
            ),
        }


@dataclass
class Agent3RunContext:
    """Mutable state owned by one Agent 3 executor invocation."""

    cv_info: Any
    location: str = ""
    results_count: int = 3
    search_pool_size: int | None = None
    max_searches: int = 2
    search_count: int = 0
    query: str = ""
    scraped_jobs: list[dict[str, Any]] = field(default_factory=list)
    match_result: dict[str, Any] | None = None
    skill_gap_analysis: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.location = str(self.location or "").strip()
        self.results_count = max(1, min(int(self.results_count), 20))
        if self.search_pool_size is None:
            self.search_pool_size = self.results_count
        self.search_pool_size = max(
            self.results_count,
            min(int(self.search_pool_size), 20),
        )
        self.max_searches = max(1, int(self.max_searches))


def _clean_text_input(value: str, key: str | None = None) -> str:
    """Accept plain ReAct input while tolerating a small JSON object."""

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if key and isinstance(payload, dict):
                return str(payload.get(key) or "").strip()
    return text.strip('"').strip("'").strip()


def _compact_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": str(job.get("title") or ""),
            "company": str(job.get("company") or ""),
            "url": str(job.get("url") or ""),
        }
        for job in jobs
    ]


def _skill_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("job_skill")
            or value.get("skill")
            or value.get("required_skill")
            or ""
        ).strip()
    return str(value or "").strip()


def build_agent3_react_tools(context: Agent3RunContext) -> list[BaseTool]:
    """Create four single-input tools bound to one Agent 3 run."""

    def search_linkedin_jobs(query: str) -> str:
        """Scrape LinkedIn for a concise job query and return compact postings."""

        resolved_query = _clean_text_input(query, "query")
        if not resolved_query:
            return json.dumps({"error": "LinkedIn search query cannot be empty."})
        if context.search_count >= context.max_searches:
            return json.dumps(
                {
                    "error": (
                        f"The maximum of {context.max_searches} LinkedIn searches "
                        "for this run has already been reached."
                    )
                }
            )

        try:
            from next_chapter.search.linkedin import search_jobs
            from next_chapter.services.application_tracker import get_tracked_job_urls

            excluded_urls = get_tracked_job_urls(context.cv_info)
            jobs = search_jobs(
                query=resolved_query,
                location=context.location,
                max_jobs=context.search_pool_size,
                posted_within_hours=30 * 24,
                exclude_urls=excluded_urls,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "error": f"LinkedIn scraping failed: {exc}",
                    "query": resolved_query,
                    "location": context.location,
                },
                ensure_ascii=False,
            )

        context.search_count += 1
        context.query = resolved_query
        context.scraped_jobs = list(jobs)
        context.match_result = None
        context.skill_gap_analysis = None
        return json.dumps(
            {
                "query": resolved_query,
                "location": context.location,
                "search_number": context.search_count,
                "searches_remaining": context.max_searches - context.search_count,
                "scraped_count": len(context.scraped_jobs),
                "jobs": _compact_jobs(context.scraped_jobs),
            },
            indent=2,
            ensure_ascii=False,
        )

    def evaluate_linkedin_results(_unused_input: str = "") -> str:
        """Parse, score, rank, and save the jobs from the latest search."""

        if not context.query:
            return json.dumps(
                {"error": "No LinkedIn search exists. Call search_linkedin_jobs first."}
            )
        if not context.scraped_jobs:
            return json.dumps(
                {"error": "The latest LinkedIn search returned no jobs to evaluate."}
            )

        stored_jobs = list(context.scraped_jobs)

        def use_stored_jobs(**_kwargs) -> list[dict[str, Any]]:
            return list(stored_jobs)

        try:
            result = match_linkedin_jobs(
                cv_info=context.cv_info,
                query=context.query,
                location=context.location,
                max_jobs=context.results_count,
                use_cache=True,
                search_fn=use_stored_jobs,
                posted_within_hours=30 * 24,
                search_pool_size=max(context.results_count, len(stored_jobs)),
            )
        except Exception as exc:
            return json.dumps(
                {
                    "error": f"LinkedIn evaluation failed: {exc}",
                    "query": context.query,
                },
                ensure_ascii=False,
            )

        persistence = persist_ranked_jobs(
            context.cv_info, result.get("ranked_jobs", [])
        )
        result = {
            **result,
            "tracked_applications": persistence["tracked_applications"],
            "candidate_id": persistence["candidate_id"],
        }
        if persistence.get("persistence_warning"):
            result["persistence_warning"] = persistence["persistence_warning"]
        context.match_result = result
        sync_linkedin_results_for_shared_tools(result)

        compact_result = {
            **result,
            "ranked_jobs": [
                {key: value for key, value in job.items() if key != "description"}
                for job in result.get("ranked_jobs", [])
            ],
        }
        return json.dumps(compact_result, indent=2, ensure_ascii=False)

    def analyze_skill_gaps(_unused_input: str = "") -> str:
        """Summarize recurring matched and missing skills in ranked jobs."""

        ranked_jobs = (context.match_result or {}).get("ranked_jobs", [])
        if not ranked_jobs:
            return json.dumps(
                {
                    "error": (
                        "No ranked jobs exist. Call evaluate_linkedin_results first."
                    )
                }
            )

        missing: Counter[str] = Counter()
        matching: Counter[str] = Counter()
        display_names: dict[str, str] = {}
        for job in ranked_jobs:
            details = job.get("skills_detail", {}) or {}
            for category, counter in (
                (details.get("missing", []), missing),
                (details.get("matching", []), matching),
            ):
                for raw_skill in category:
                    name = _skill_name(raw_skill)
                    normalized = name.casefold()
                    if not normalized:
                        continue
                    display_names.setdefault(normalized, name)
                    counter[normalized] += 1

        def ordered(counter: Counter[str]) -> list[dict[str, Any]]:
            return [
                {"skill": display_names[key], "jobs": count}
                for key, count in sorted(
                    counter.items(), key=lambda item: (-item[1], item[0])
                )
            ]

        recurring = [item for item in ordered(missing) if item["jobs"] >= 2]
        strongest = ordered(matching)
        recommendation = (
            "Prioritize " + ", ".join(item["skill"] for item in recurring[:3])
            if recurring
            else "No missing skill recurred across multiple ranked jobs."
        )
        analysis = {
            "jobs_analyzed": len(ranked_jobs),
            "recurring_missing_skills": recurring,
            "all_missing_skills": ordered(missing),
            "strongest_existing_skills": strongest,
            "recommendation": recommendation,
        }
        context.skill_gap_analysis = analysis
        return json.dumps(analysis, indent=2, ensure_ascii=False)

    def write_cover_letter_for_job(job_url: str) -> str:
        """Write one cover letter for a ranked job, identified by its URL."""

        resolved_url = _clean_text_input(job_url, "url")
        ranked_jobs = (context.match_result or {}).get("ranked_jobs", [])
        if not ranked_jobs:
            return "Error: No ranked jobs exist. Call evaluate_linkedin_results first."
        selected = next(
            (job for job in ranked_jobs if str(job.get("url") or "") == resolved_url),
            None,
        )
        if selected is None:
            return "Error: The supplied URL is not present in the current ranked jobs."

        from next_chapter.agents.tools.cover_letter import write_cover_letter

        return str(write_cover_letter.func(url=resolved_url))

    return [
        StructuredTool.from_function(
            func=search_linkedin_jobs,
            name="search_linkedin_jobs",
            description=(
                "Scrape LinkedIn using one concise plain-text job query. Returns "
                "compact titles, companies, and URLs. At most two searches are "
                "allowed, so refine only when the first result list is clearly poor."
            ),
        ),
        StructuredTool.from_function(
            func=evaluate_linkedin_results,
            name="evaluate_linkedin_results",
            description=(
                "Parse, score, rank, and save the jobs from the latest LinkedIn "
                "search. Pass an empty string. Scores are deterministic and must "
                "not be recalculated or reordered."
            ),
        ),
        StructuredTool.from_function(
            func=analyze_skill_gaps,
            name="analyze_skill_gaps",
            description=(
                "Analyze recurring matching and missing skills in the current "
                "ranked jobs. Pass an empty string."
            ),
        ),
        StructuredTool.from_function(
            func=write_cover_letter_for_job,
            name="write_cover_letter_for_job",
            description=(
                "Write a cover letter for one job from the current ranking. Pass "
                "only that job's URL as plain text. Use the first ranked job."
            ),
        ),
    ]
