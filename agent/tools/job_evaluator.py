"""
Agent-facing tool: evaluates ONE job posting against the candidate's
already-extracted profile. Wraps job_parser.py (requirement extraction)
+ matcher.py (scoring) into a single tool call the agent can invoke
per job it finds.

Also records this job as SEEN for this candidate (seen_jobs_memory.py)
after scoring it — so a future run's search_real_jobs call for the
SAME candidate automatically filters this job out and surfaces
something new instead.

_all_evaluations accumulates every evaluate_job_match result for the
CURRENT run — used by agent1.py to build the full ranked job list for
the results email/message, not just the single top match.
"""

import json
from langchain_core.tools import tool
from core.job_parser import extract_job_requirements
from core.matcher import calculate_compatibility
from core.seen_jobs_memory import record_seen
from search.job_search import set_candidate_email
import search.job_search as job_search

_current_cv_info = None
_all_evaluations = []  # every evaluate_job_match result THIS run — reset
                        # per run inside set_candidate_profile()


def set_candidate_profile(cv_info):
    global _current_cv_info, _all_evaluations
    _current_cv_info = cv_info
    _all_evaluations = []  # reset — otherwise a second run in the same
    # process would keep appending onto the previous run's jobs
    set_candidate_email(cv_info.mail if cv_info else None)


def get_ranked_evaluations() -> list:
    """
    Deterministic final ranking for the current run, safe to use from
    anywhere (email builder, agent completion, agent1):

      - de-duplicates: the LLM sometimes calls evaluate_job_match twice for
        the same posting (a re-typed job with the same URL), which would
        otherwise double-count it in the ranked list. Keyed by URL when
        present, else by title|company.
      - orders by the prompt's rule: any evaluation with extractable
        requirements (inconclusive: false) ranks ABOVE all inconclusive
        ones, regardless of raw score_percent — an inconclusive 75% is not
        a real match, a conclusive 55% is. Within each group, score
        descending.
    """
    unique = {}
    for result in _all_evaluations:
        key = result.get("url") or f"{result['job_title']}|{result['company']}"
        unique[key] = result  # a later evaluation of the same job wins

    ranked = list(unique.values())
    ranked.sort(key=lambda r: (r.get("inconclusive", True), -r["score_percent"]))
    return ranked


def get_ranked_jobs_payload() -> list:
    """
    The shape the deliverable builders (Gmail draft + Telegram message)
    share for the ranked list: strips each evaluation down to only the
    fields email/telegram rendering needs. Both transport tools call THIS
    so the payload can never drift between channels.
    """
    return [
        {
            "job_title": r["job_title"],
            "company": r["company"],
            "score_percent": r["score_percent"],
            "url": r.get("url", ""),
            "inconclusive": r.get("inconclusive", False),
            "skills_detail": {
                "matching": r["matching_skills"],
                "missing": r["missing_skills"],
            },
        }
        for r in get_ranked_evaluations()
    ]


def _lookup_stored(job: dict):
    """
    Recover the FULL posting for an agent retype from this run's search
    results. Small models reliably drift from the documented single-job /
    url-only contract, so resolve by (1) url, (2) title+company, and
    finally (3) a UNIQUE title match when no company was given. The
    resolved posting is preferred over whatever the agent may have re-typed
    — scoring quality must never depend on how faithfully it copied text.
    """
    job_url = job.get("url", "")
    job_title = job.get("title", "")
    company = job.get("company", "")

    if job_url:
        for stored_job in job_search._last_search_results:
            if stored_job.get("url") == job_url:
                return stored_job

    if job_title and company:
        for stored_job in job_search._last_search_results:
            if stored_job.get("title") == job_title and stored_job.get("company") == company:
                return stored_job

    if job_title and not company:
        matches = [
            s for s in job_search._last_search_results if s.get("title") == job_title
        ]
        if len(matches) == 1:
            return matches[0]

    return None


def _available_postings() -> str:
    if not job_search._last_search_results:
        return ""
    shown = job_search._last_search_results[-5:]
    return (
        " Available postings you can pass by url: "
        + "; ".join(
            f"{s.get('title','')} @ {s.get('company','')} (url: {s.get('url','')})"
            for s in shown
        )
    )


def _evaluate_single(job: dict) -> dict:
    """
    Score ONE job. Returns either a full result_dict (already appended to
    _all_evaluations) or an {"error": ...} dict. Used directly for a single
    job input, and once per entry when the model passes a batch.
    """
    global _all_evaluations

    job_title = job.get("title", "")
    job_description = job.get("description", "")
    company = job.get("company", "")
    job_url = job.get("url", "")

    stored = _lookup_stored(job)
    if stored is not None:
        if not job_url:
            job_url = stored.get("url", "")
        if not job_title:
            job_title = stored.get("title", "")
        if not company:
            company = stored.get("company", "")
        job_description = stored.get("description", "")

    if not job_title or not job_description:
        return {
            "error": "Missing title or description, and no matching posting "
            "was found to look up the full description."
            + _available_postings()
        }

    job_req = extract_job_requirements(job_title, job_description)
    result = calculate_compatibility(_current_cv_info, job_req)

    matching_skill_names = [m["job_skill"] for m in result["skills"]["matching"]]
    missing_skill_names = result["skills"]["missing"]
    inconclusive = not matching_skill_names and not missing_skill_names

    if _current_cv_info.mail:
        # Record EVERY scored job — even when no URL is available (the agent
        # often retypes postings and drops the url). record_seen keys by
        # `missing|title|company` in that case, and search_real_jobs filters
        # on both forms, so a URL-less evaluation is still remembered.
        record_seen(
            candidate_email=_current_cv_info.mail,
            job_url=job_url,
            job_title=job_title,
            company=company,
            score_percent=result["score_percent"],
            matching_skills=matching_skill_names,
            missing_skills=missing_skill_names,
        )

    result_dict = {
        "job_title": job_title,
        "company": company,
        "url": job_url,
        "score_percent": result["score_percent"],
        "matching_skills": result["skills"]["matching"],
        "missing_skills": result["skills"]["missing"],
        "inconclusive": inconclusive,
        "experience_match_score": f"{result['experience']['score']*100:.0f}%",
        "education_match_score": f"{result['education']['score']*100:.0f}%",
    }

    _all_evaluations.append(result_dict)

    return result_dict


@tool
def evaluate_job_match(job_json: str) -> str:
    """
    Evaluates how well the candidate's profile matches a specific job posting.

    Input: a JSON string identifying ONE job posting. You may pass the
    full posting as returned by search_jobs_for_agent, OR just its "url" —
    the tool looks the full posting back up itself from the search results
    when you give a url (or a title+company pair). A list of jobs (or
    {{"jobs": [...]}}) is also accepted and each entry is evaluated — the
    response is then {{"results": [...]}}.

    Output: JSON string with score_percent, matching skills, missing skills,
    experience and education match details.

    Example inputs:
    {{"url": "https://..."}}
    {{"title": "AI Engineer", "company": "Acme", "description": "We need..."}}
    """
    if _current_cv_info is None:
        return json.dumps({"error": "No candidate profile loaded."})

    try:
        parsed = json.loads(job_json)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON input: {job_json[:100]}"})

    # The small models reliably batch postings instead of calling once per
    # job (the prompt forbids it, but they do it anyway and then loop on the
    # repeated error). Accept one job, a bare list, or {"jobs": [...]} so a
    # single call resolves cleanly instead of burning the iteration budget.
    if isinstance(parsed, list):
        entries = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("jobs"), list):
        entries = parsed["jobs"]
    else:
        entries = [parsed]

    results = [_evaluate_single(entry) for entry in entries]

    if len(entries) == 1:
        return json.dumps(results[0], indent=2)
    return json.dumps({"results": results}, indent=2)