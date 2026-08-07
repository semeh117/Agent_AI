"""
Agent-facing tool: generates a cover letter for ONE job the agent has
already evaluated as the top match. Wraps pipeline/cover_letter.py's
generate_cover_letter() so the agent can call it as one more tool-use
step in its own reasoning, after it has decided ranking is complete.
 
Same module-level candidate-profile pattern as job_evaluator.py: the
agent's tool-calling can only pass simple JSON/string arguments, not a
full CVInfo object, so set_candidate_profile() (already called once by
agent.py before the agent runs) is reused here too.
"""
 
import json
from langchain_core.tools import tool
from pipeline.cover_letter import generate_cover_letter
import agent.tools.job_evaluator as job_evaluator 

@tool
def write_cover_letter(evaluated_job_json: str) -> str:
    """
    Writes a cover letter for a SINGLE job, using the candidate's profile
    and that job's match details. Call this AFTER evaluate_job_match has
    scored the job you want a cover letter for — typically the highest
    scoring one.
 
    Input: a JSON string with the SAME shape evaluate_job_match returns:
    job_title, company, score_percent, matching_skills, missing_skills.
    You can also include "description" if you have it (from the original
    search_real_jobs result) for a more grounded letter — if omitted,
    the letter will note less detail about the role itself.
 
    Output: the cover letter as plain text (not JSON — read it directly).
    """
    if job_evaluator._current_cv_info is None:
        return "Error: No candidate profile loaded."
 
    try:
        job = json.loads(evaluated_job_json)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON input: {evaluated_job_json[:100]}"
 
    required = ["job_title", "company", "score_percent", "matching_skills", "missing_skills"]
    missing_fields = [f for f in required if f not in job]
    if missing_fields:
        return f"Error: Missing required fields: {', '.join(missing_fields)}"
 
    # Adapt to the shape generate_cover_letter() expects (same as
    # job_matching_pipeline.py's results[0] shape).
    top_result_for_letter = {
        "job_title": job["job_title"],
        "company": job["company"],
        "description": job.get("description", ""),
        "score_percent": job["score_percent"],
        "skills_detail": {
            "matching": job["matching_skills"],
            "missing": job["missing_skills"],
        },
    }
 
    return generate_cover_letter(job_evaluator._current_cv_info, top_result_for_letter)