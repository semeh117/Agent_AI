"""
Agent-facing tool: delivers the job matches + cover letter to the
candidate via a Telegram bot message. Wraps pipeline/send_results_telegram.py
so the agent can offer it as an alternative delivery channel on top of the
Gmail draft.

Mirrors gmail.py's design exactly: takes no meaningful input, reads the
job details + cover letter straight from cover_letter_tool.py's module
state (_last_cover_letter, _last_cover_letter_job), so there is no JSON
re-typing of the letter text for the model to get wrong.
"""

from langchain_core.tools import tool
from pipeline.send_results_telegram import create_results_telegram
import agent.tools.job_evaluator as job_evaluator  # module import, see
# cover_letter_tool.py for why this must NOT be `from ... import _current_cv_info`
import agent.tools.cover_letter as cover_letter   # same reasoning: need the
# LIVE value of _last_cover_letter / _last_cover_letter_job


@tool
def send_results_telegram(_unused_input: str = "") -> str:
    """
    Sends the ranked job matches and the cover letter that
    write_cover_letter most recently produced to the candidate's Telegram
    chat as a message. Use this INSTEAD of send_results_draft when the
    user chose Telegram delivery.

    Input: not needed — pass an empty string. This tool automatically
    uses the most recently written cover letter and its job; you do not
    need to pass the job details or letter text yourself.

    Output: a short confirmation string, or an error message if no
    cover letter has been written yet or the Telegram message could not
    be sent (e.g. missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env).
    """
    if job_evaluator._current_cv_info is None:
        return "Error: No candidate profile loaded."

    if cover_letter._last_cover_letter is None:
        return "Error: No cover letter has been written yet. Call write_cover_letter first."

    ranked_jobs = job_evaluator.get_ranked_jobs_payload()

    try:
        create_results_telegram(
            job_evaluator._current_cv_info,
            ranked_jobs,
            cover_letter._last_cover_letter,
        )
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Telegram message failed: {str(e)[:200]}"

    return "Telegram message sent successfully to the candidate's chat."