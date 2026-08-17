""" PATH A : determenistic post agent orchestration .
the ReAct agent's loop job stops at search + evaluate + rank (agent_core.py).
Everything after that - picking the top job , writing it's cover letter , creating the Gmail draft - is guaranteed plain Python , not something the llm decides to do 
as a further tool call.

Why: a small 7B agent reliably searches and scores jobs, but is NOT
reliable at correctly sequencing and firing multi-step tool calls for
the delivery steps (cover letter -> draft)
"""

import json
from agent.agent_core import run_agent_job_matching
from agent.tools.job_evaluator import get_ranked_evaluations
from pipeline.cover_letter import generate_cover_letter
from pipeline.send_results_email import create_results_draft

def _find_job_description(intermediate_steps :list, job_title :str , company :str) -> dict | None:
    """
    search_real_jobs's own observation has the full description;
    evaluate_job_match's doesn't. Looks it up by title+company match.    """
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        if tool_name != "search_real_jobs":
            continue
        try:
            jobs = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            continue
        for job in jobs:
            if job.get("job") == job_title and job.get("company") == company:
                return job.get("description")
    return ""
def run_agent1_full_pipeline(cv_info ,results_count :int =3) -> dict :
    """
        Runs the search+evaluate agent, then deterministically: picks the
    top job, writes its cover letter, builds the FULL ranked list from
    every evaluate_job_match call this run, and creates the Gmail draft
    — all as guaranteed Python calls, no LLM tool-calling involved past
    the ranking stage."""

    result = run_agent_job_matching(cv_info, results_count=results_count)

    ranked_jobs = get_ranked_evaluations()
    top_evaluation = ranked_jobs[0] if ranked_jobs else None
    if top_evaluation is None:
        result["cover_letter"] = None
        result["draft_error"] = "No valid job evaluations found."
        return result
    description = _find_job_description(result["intermediate_steps"], top_evaluation["job_title"], top_evaluation["company"])

    top_result_for_letter = {
        
        "job_title": top_evaluation["job_title"],
        "company": top_evaluation["company"],
        "description": description,
        "score_percent": top_evaluation["score_percent"],
        "url": top_evaluation.get("url"),
        "skills_detail": {
            "matching" : top_evaluation.get("matching_skills"),
            "missing" : top_evaluation.get("missing_skills"),

        },
    }
    result["cover_letter"] = generate_cover_letter(cv_info, top_result_for_letter)
    result["cover_letter_job"] = top_result_for_letter

    if not cv_info.mail:
        result["draft_error"] = "No email address provided in CV info."
        return result
    ranked_jobs_for_email = [
        {
            "job_title": r["job_title"],
            "company": r["company"],
            "score_percent": r["score_percent"],
            "url": r.get("url", ""),
            "inconclusive": r.get("inconclusive", False),
            "skills_detail": {"matching": r["matching_skills"], "missing": r["missing_skills"]},
        }
        for r in ranked_jobs
    ]

    try:
        result["draft"] = create_results_draft(
            cv_info, ranked_jobs_for_email, result["cover_letter"], to_email=cv_info.mail
        )
    except Exception as e:
        result["draft_error"] = str(e)

    return result
    