"""
Agent-facing tool: evaluates ONE job posting against the candidate's
already-extracted profile. Wraps job_parser.py (requirement extraction)
+ matcher.py (scoring) into a single tool call the agent can invoke
per job it finds.
"""

import json
from langchain_core.tools import tool
from core.job_parser import extract_job_requirements
from core.matcher import calculate_compatibility

_current_cv_info = None


def set_candidate_profile(cv_info):
    global _current_cv_info
    _current_cv_info = cv_info

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
    if _current_cv_info is None:
        return json.dumps({"error": "No candidate profile loaded."})

    try:
        job = json.loads(job_json)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON input: {job_json[:100]}"})

    job_title = job.get("title", "")
    job_description = job.get("description", "")
    company = job.get("company", "")

    if not job_title or not job_description:
        return json.dumps({"error": "Missing title or description in job JSON."})

    job_req = extract_job_requirements(job_title, job_description)
    result = calculate_compatibility(_current_cv_info, job_req)

    return json.dumps({
        "job_title": job_title,
        "company": company,
        "score_percent": result["score_percent"],
        "matching_skills": result["skills"]["matching"],
        "missing_skills": result["skills"]["missing"],
        "experience_match_score": f"{result['experience']['score']*100:.0f}%",
        "education_match_score": f"{result['education']['score']*100:.0f}%",
    }, indent=2)