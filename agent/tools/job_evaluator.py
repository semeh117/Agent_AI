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


@tool
def evaluate_job_match(job_json: str) -> str:
    """
    Evaluates how well the candidate's profile matches a specific job posting.

    Input: a JSON string representing ONE job posting, exactly as returned
    by search_real_jobs (with fields: title, company, description).
    Output: JSON string with score_percent, matching skills, missing skills,
    experience and education match details.

    Example input:
    {{"title": "AI Engineer", "company": "Acme", "description": "We need..."}}
    """
    global _all_evaluations

    if _current_cv_info is None:
        return json.dumps({"error": "No candidate profile loaded."})

    try:
        job = json.loads(job_json)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON input: {job_json[:100]}"})

    job_title = job.get("title", "")
    job_description = job.get("description", "")
    company = job.get("company", "")
    job_url = job.get("url", "")

    # If the agent didn't pass a URL (common — it retypes job details by
    # hand and sometimes drops fields), look it up from the real search
    # results still held in memory, matched by title + company.
    if not job_url:
        for stored_job in job_search._last_search_results:
            if stored_job.get("title") == job_title and stored_job.get("company") == company:
                job_url = stored_job.get("url", "")
                break

    if not job_title or not job_description:
        return json.dumps({"error": "Missing title or description in job JSON."})

    job_req = extract_job_requirements(job_title, job_description)
    result = calculate_compatibility(_current_cv_info, job_req)

    matching_skill_names = [m["job_skill"] for m in result["skills"]["matching"]]
    missing_skill_names = result["skills"]["missing"]

    if _current_cv_info.mail and job_url:
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
        "experience_match_score": f"{result['experience']['score']*100:.0f}%",
        "education_match_score": f"{result['education']['score']*100:.0f}%",
    }

    _all_evaluations.append(result_dict)

    return json.dumps(result_dict, indent=2)