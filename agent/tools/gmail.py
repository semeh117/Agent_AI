"""
Agent-facing tool: creates a Gmail draft containing a job's cover letter
addressed to the candidate. Wraps pipeline/send_results_email.py's
create_results_draft() so the agent can call it as a tool-use step,
after write_cover_letter has produced a letter.

Takes NO meaningful input — it reads the job details and cover letter
straight from cover_letter_tool.py's module state (_last_cover_letter,
_last_cover_letter_job), which write_cover_letter populates. This is
deliberate: earlier testing showed the LLM producing malformed JSON when
asked to re-embed a full, multi-paragraph cover letter as a string field
inside a new tool call. Reading the letter back from state instead of
asking the model to retype/re-serialize it removes that failure mode
entirely — there is nothing left for the model to get wrong here except
deciding to call this tool at the right time.

Same module-level candidate-profile pattern as job_evaluator.py and
cover_letter_tool.py — the agent needs the candidate's email address,
which lives on the CVInfo object (cv_info.mail), not something the LLM
would type out itself.
"""

from langchain_core.tools import tool
from pipeline.send_results_email import create_results_draft
import agent.tools.job_evaluator as job_evaluator  # module import, see
# cover_letter_tool.py for why this must NOT be `from ... import _current_cv_info`
import agent.tools.cover_letter as cover_letter   # same reasoning:
# need the LIVE value of _last_cover_letter / _last_cover_letter_job,
# which is only set after write_cover_letter actually runs.


@tool
def send_results_draft(_unused_input: str = "") -> str:
    """
    Creates a Gmail DRAFT (does not send) addressed to the candidate's
    own email, containing the job and cover letter that write_cover_letter
    most recently produced. Call this AFTER write_cover_letter has run
    successfully — this is the final step.

    Input: not needed — pass an empty string. This tool automatically
    uses the most recently written cover letter and its job; you do not
    need to pass the job details or letter text yourself.

    Output: a short confirmation string, or an error message if no
    cover letter has been written yet, the candidate's email address
    wasn't found, or the draft couldn't be created (e.g. missing/invalid
    Gmail credentials).
    """
    if job_evaluator._current_cv_info is None:
        return "Error: No candidate profile loaded."

    if cover_letter._last_cover_letter is None:
        return "Error: No cover letter has been written yet. Call write_cover_letter first."

    cv_info = job_evaluator._current_cv_info
    if not cv_info.mail:
        return ("Error: No email address found on the candidate's profile "
                "(cv_info.mail is empty) — cannot create a draft without a "
                "recipient address.")

    # create_results_draft() expects ranked_jobs as a LIST (it was built
    # for job_matching_pipeline.py's multi-job results) — get_ranked_jobs_payload()
    # wraps this run's evaluations into that shape, shared with the
    # Telegram tool so both channels render identical content.
    ranked_jobs_for_email = job_evaluator.get_ranked_jobs_payload()

    try:
        draft = create_results_draft(
            cv_info,
            ranked_jobs_for_email,
            cover_letter._last_cover_letter,
            to_email=cv_info.mail,
        )
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Gmail draft creation failed: {str(e)[:200]}"

    return f"Draft created successfully (id: {draft['id']}). The candidate can review and send it from Gmail > Drafts."
