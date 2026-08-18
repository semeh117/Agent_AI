"""
Agent-facing tool: generates a cover letter for ONE job the agent has
already evaluated as the top match. Wraps pipeline/cover_letter.py's
generate_cover_letter() so the agent can call it as one more tool-use
step in its own reasoning, after it has decided ranking is complete.

Same module-level candidate-profile pattern as job_evaluator.py: the
agent's tool-calling can only pass simple JSON/string arguments, not a
full CVInfo object, so set_candidate_profile() (already called once by
agent.py before the agent runs) is reused here too.

IMPORTANT: the generated letter (and the job dict it was written for)
are ALSO saved into module-level state here (_last_cover_letter,
_last_cover_letter_job), NOT just returned as text. This is deliberate:
send_results_draft (gmail_tool.py) reads the letter back from this
module state instead of requiring the LLM to copy/re-serialize the
full letter text into a new JSON string. Making the LLM hand-construct
valid JSON containing a multi-paragraph, punctuation-heavy string is a
fragile, error-prone pattern — that's exactly what broke in testing
(the model produced malformed JSON trying to embed the letter). Reading
it back from state sidesteps that failure mode entirely.
"""

import json
from langchain_core.tools import tool
from pipeline.cover_letter import generate_cover_letter
import agent.tools.job_evaluator as job_evaluator  # import the MODULE, not the
# variable directly — job_evaluator._current_cv_info is only set correctly
# at RUNTIME (after set_candidate_profile() runs), and importing the bare
# name would freeze a stale None copied at import time instead of staying
# linked to the module's actual current value.
from search import job_search

_last_cover_letter = None
_last_cover_letter_job = None


def _lookup_evaluation(job: dict):
    """
    The model sometimes passes only a url (or a title+company pair) to
    write_cover_letter — the same drift it shows with evaluate_job_match.
    Recover the FULL evaluation recorded earlier this run (job_title,
    company, score_percent, matching/missing skills, ...) so the letter can
    be built without the model re-serializing it. Resolves by url, then by
    title+company, then by a UNIQUE title when no company was given.
    """
    job_url = job.get("url", "")
    job_title = job.get("title", "") or job.get("job_title", "")
    company = job.get("company", "")

    if job_url:
        for ev in job_evaluator._all_evaluations:
            if ev.get("url") == job_url:
                return ev

    if job_title:
        candidates = [
            ev for ev in job_evaluator._all_evaluations
            if ev.get("job_title") == job_title
        ]
        if company:
            candidates = [ev for ev in candidates if ev.get("company") == company]
        if len(candidates) == 1:
            return candidates[0]

    return None


def _fill_stored_description(job: dict) -> dict:
    """Attach the FULL stored posting description to an evaluation-shaped job
    so the letter can be grounded in the actual role. Evaluation results
    carry no description; the stored search results do."""
    if job.get("description"):
        return job
    job_url = job.get("url", "")
    job_title = job.get("job_title", "")
    company = job.get("company", "")
    for stored in job_search._last_search_results:
        if (job_url and stored.get("url") == job_url) or (
            not job_url and not company and stored.get("title") == job_title
        ):
            return {**job, "description": stored.get("description", "")}
        if not job_url and company and (
            stored.get("title") == job_title and stored.get("company") == company
        ):
            return {**job, "description": stored.get("description", "")}
    return job


def _normalize_matching_skills(matching_skills) -> list:
    """
    generate_cover_letter() expects matching skills as
    [{"job_skill": "Python", "matched_via": "Python"}, ...] — that's
    matcher.py's real output shape, and evaluate_job_match's JSON
    output preserves it correctly. But when the LLM re-types this same
    data as the input to THIS tool, it sometimes flattens it to plain
    strings instead (["Python", "LangChain"]), simplifying the nested
    structure on its own. Accept either shape here rather than crashing.
    """
    normalized = []
    for item in matching_skills:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            # Plain string — wrap it into the expected dict shape,
            # using the string itself for both fields since we don't
            # know what it was actually matched via.
            normalized.append({"job_skill": str(item), "matched_via": str(item)})
    return normalized


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
    the letter will note less detail about the role itself. Passing just
    the job's "url" also works: the full evaluation recorded earlier this
    run is looked up automatically.

    Output: a short confirmation that the letter was written (NOT the
    full letter text — you do not need to copy or repeat it anywhere).
    The letter is saved automatically; send_results_draft will use it
    directly without you needing to pass it along.
    """
    global _last_cover_letter, _last_cover_letter_job

    if job_evaluator._current_cv_info is None:
        return "Error: No candidate profile loaded."

    try:
        job = json.loads(evaluated_job_json)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON input: {evaluated_job_json[:100]}"

    required = ["job_title", "company", "score_percent", "matching_skills", "missing_skills"]
    missing_fields = [f for f in required if f not in job]

    if missing_fields:
        # The model sometimes hands over just a url (or a title+company
        # pair) — recover the evaluated job from this run's records rather
        # than rejecting the call and letting the agent loop on it.
        resolved = _lookup_evaluation(job)
        if resolved is not None:
            job = _fill_stored_description(resolved)
            missing_fields = [f for f in required if f not in job]
        if missing_fields:
            return f"Error: Missing required fields: {', '.join(missing_fields)}"

    # Adapt to the shape generate_cover_letter() expects (same as
    # job_matching_pipeline.py's results[0] shape). matching_skills
    # SHOULD arrive as [{"job_skill": ..., "matched_via": ...}], the
    # exact shape evaluate_job_match returns — but in testing, the LLM
    # sometimes flattens this to plain strings (["Python", "LangChain"])
    # when hand-typing the JSON for this call, likely simplifying the
    # nested structure on its own. Normalize defensively so either shape
    # works, instead of crashing on a TypeError when it drifts.
    matching_skills = _normalize_matching_skills(job["matching_skills"])

    top_result_for_letter = {
        "job_title": job["job_title"],
        "company": job["company"],
        "description": job.get("description", ""),
        "score_percent": job["score_percent"],
        "url": job.get("url", ""),
        "skills_detail": {
            "matching": matching_skills,
            "missing": job["missing_skills"],
        },
    }

    letter = generate_cover_letter(job_evaluator._current_cv_info, top_result_for_letter)

    _last_cover_letter = letter
    _last_cover_letter_job = top_result_for_letter

    return (f"Cover letter written successfully for {job['job_title']} @ {job['company']}. "
            f"It is saved and ready — call send_results_draft next (you do not need to "
            f"include the letter text in that call).")