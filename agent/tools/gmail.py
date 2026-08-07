"""
Agent-facing tool: creates a Gmail draft containing a job's cover letter
(and optionally its match details) addressed to the candidate. Wraps
pipeline/send_results_email.py's create_results_draft() so the agent can
call it as a tool-use step, after it has decided a cover letter is ready
to send.
 
Same module-level candidate-profile pattern as job_evaluator.py and
cover_letter_tool.py — the agent needs the candidate's email address,
which lives on the CVInfo object (cv_info.mail), not something the LLM
would type out itself.
"""
 
import json
from langchain_core.tools import tool
from pipeline.send_results_email import create_results_draft
import agent.tools.job_evaluator as job_evaluator

@tool
def send_results_draft(job_and_letter_json: str) -> str:
    """
    Creates a Gmail DRAFT (does not send) addressed to the candidate's
    own email, containing a job's details and its cover letter. Call
    this AFTER write_cover_letter has produced a letter you want to
    deliver — this is the final step, once you have both a scored job
    and its cover letter ready.
 
    Input: a JSON string with: job_title, company, score_percent,
    matching_skills, missing_skills, url (optional), and cover_letter
    (the full text from write_cover_letter's output).
 
    Output: a short confirmation string, or an error message if the
    candidate's email address wasn't found or the draft couldn't be
    created (e.g. missing/invalid Gmail credentials).
    """
    if job_evaluator._current_cv_info is None:
        return "Error: No candidate profile loaded."
 
    cv_info = job_evaluator._current_cv_info
    if not cv_info.mail:
        return ("Error: No email address found on the candidate's profile "
                "(cv_info.mail is empty) — cannot create a draft without a "
                "recipient address.")
 
    try:
        data = json.loads(job_and_letter_json)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON input: {job_and_letter_json[:100]}"
 
    required = ["job_title", "company", "score_percent", "matching_skills",
                "missing_skills", "cover_letter"]
    missing_fields = [f for f in required if f not in data]
    if missing_fields:
        return f"Error: Missing required fields: {', '.join(missing_fields)}"
 
    # create_results_draft() expects ranked_jobs as a LIST (it was built
    # for job_matching_pipeline.py's multi-job results) — wrap this single
    # job in a one-item list so the email body still renders correctly.
    single_job_as_list = [{
        "job_title": data["job_title"],
        "company": data["company"],
        "score_percent": data["score_percent"],
        "url": data.get("url", ""),
        "skills_detail": {
            "matching": data["matching_skills"],
            "missing": data["missing_skills"],
        },
    }]
 
    try:
        draft = create_results_draft(
            cv_info,
            single_job_as_list,
            data["cover_letter"],
            to_email=cv_info.mail,
        )
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Gmail draft creation failed: {str(e)[:200]}"
 
    return f"Draft created successfully (id: {draft['id']}). The candidate can review and send it from Gmail > Drafts."    