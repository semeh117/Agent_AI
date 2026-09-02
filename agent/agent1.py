"""
agent1.py
--------------
PATH B — fully agent-driven. The LLM itself decides, as further steps
in its own ReAct loop, when to write the cover letter, when to ask the
candidate which delivery channel they prefer (Gmail draft or Telegram
message), and when to deliver — the same way it already decides to
search and evaluate.

OBSERVED FAILURE MODE (kept here deliberately, not smoothed over): in
testing with qwen-2.5-7b-instruct, the model has twice produced a Final
Answer confidently claiming "a Gmail draft has been created" while its
own intermediate_steps show send_results_draft was never actually
called. The verification block in run_agent1() below is a
fallback that catches this and completes the step for real — meaning
this path currently only works BECAUSE of that fallback, not because
the agent reliably follows the prompted workflow. Compare against
agent1_deterministic.py, which has no equivalent failure mode since
nothing past ranking is left to the LLM's judgment.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from config import get_agent_llm
from agent.agent_core import TOOLS
from agent.react_output_parser import RequiredToolsVerifyingParser
import agent.tools.job_evaluator as job_evaluator  # module import — need the
# LIVE _current_cv_info / _all_evaluations, only set at runtime.
from agent.tools.job_evaluator import set_candidate_profile
from agent.tools.cover_letter import write_cover_letter
import agent.tools.cover_letter as cover_letter_tool
from agent.tools.gmail import send_results_draft
from agent.tools.telegram_tool import send_results_telegram
from agent.tools.delivery_choice import ask_user_delivery_channel
import agent.tools.delivery_choice as delivery_choice
from search import job_search

TOOLS_WITH_COVER_LETTER_AND_DELIVERY = (
    TOOLS
    + [write_cover_letter, ask_user_delivery_channel, send_results_draft, send_results_telegram]
)

REACT_PROMPT_TEMPLATE_WITH_COVER_LETTER_AND_EMAIL = """You are an expert career assistant AI helping a candidate
find and evaluate real job opportunities based on their profile, then
prepare and deliver a cover letter for their best match.

You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a short summary confirming the ranked jobs, that a cover
letter was written for the top match, and that a Gmail draft was created
for the candidate (or a clear note if any step could not be completed).

CRITICAL RULE — READ BEFORE DOING ANYTHING ELSE:
Never write "Final Answer" until you have actually called search_jobs_for_agent,
evaluate_job_match (once per job), write_cover_letter, and
send_results_draft as real tool Actions and read their real Observations.
Writing job descriptions, skill comparisons, or scores in prose WITHOUT a
matching Action/Observation pair is strictly forbidden — every score_percent
you mention must come from an evaluate_job_match Observation, never
invented or estimated by you. If you have not yet called a tool in this
response, stop after a single Action Input line — do not continue writing
past it, and do not write "Final Answer" in the same response as your
first Thought.

WORKFLOW:
1. Call search_jobs_for_agent with a query built from the candidate's profile.
   It returns a short status note, then a JSON list of job objects.
   If the note says jobs were filtered out because you already evaluated
   them in a previous run, or that fewer postings came back than
   requested, do not proceed with a short list — refine the query
   (different skills, a different job title, broader/narrower phrasing)
   and call search_jobs_for_agent again. Do this at most twice per search
   need; if still short after that, proceed with whatever fresh postings
   you have.
2. Call evaluate_job_match once per job, each time with a SINGLE job
   object as a JSON string — never the whole list, never batched. Pass the
   job's "url" from the search result (evaluate_job_match fetches the full
   posting itself from the search it was returned by) — do NOT copy or
   re-type the job description or any other field; a short {{"url":
   "https://..."}} input is all it needs.
3. Look at every evaluation's "inconclusive" field. inconclusive: true
   means the posting had no extractable requirements, so its
   score_percent is meaningless even if it shows 100% — it is not a
   real match. If fewer than 2 evaluations came back
   inconclusive: false, say so explicitly (e.g. "I've already analyzed
   and ranked these jobs, but most were inconclusive — searching for
   more similar postings before finalizing"), then call search_jobs_for_agent
   again with a refined query based on the strongest genuine job's title
   or the candidate's top skills, and evaluate each new job the same
   way. Do this at most once more, even if still short of 2.
4. Combine every evaluation from all rounds. Rank all
   inconclusive: false evaluations above any inconclusive: true ones,
   regardless of raw score_percent, then sort each group by
   score_percent descending. Double check the ordering.
5. Call write_cover_letter exactly once, for the top-ranked job (prefer
   a non-inconclusive one — only use an inconclusive job if literally
   everything evaluated came back inconclusive). Pass that job's
   evaluate_job_match result as a JSON string. write_cover_letter saves
   the letter automatically — do not copy or repeat its text yourself.
6. Call ask_user_delivery_channel exactly once, with an empty string as
   input. It asks the human at the console whether they want the results
   as a Gmail draft or a Telegram message, and returns the chosen channel.
7. Deliver on the chosen channel: if the user chose Gmail, call
   send_results_draft exactly once; if they chose Telegram, call
   send_results_telegram exactly once. Pass an empty string — both read
   the saved letter automatically.
8. Write your Final Answer: confirm the ranked list with percentages
   (noting which were inconclusive), that a cover letter was generated
   for the top genuine match, and that it was delivered on the channel
   the user chose — or explain clearly which step failed if a tool
   returned an error.

NOTES:
- evaluate_job_match takes one job as a JSON string, not the search
  results list. A short {{"url": "..."}} is sufficient — it will fetch the
  full posting itself, so never re-type the description.
- write_cover_letter is called EXACTLY ONCE per run, only for the top
  job — never once per job in the list.
- ask_user_delivery_channel is called EXACTLY ONCE, after the cover
  letter, before delivery.
- Exactly ONE deliver tool runs per run: send_results_draft (Gmail) or
  send_results_telegram (Telegram) — whichever matches the user's answer.
- Never hand-copy the cover letter text into a deliver tool's input.
- If a tool returns a message starting with "Error:", do not retry more
  than once, and report the failure honestly instead of claiming success.

EXAMPLE OF CORRECT FORMAT (abbreviated — follow this exact pattern):
Thought: I need to find jobs matching this profile first.
Action: search_jobs_for_agent
Action Input: {{"query": "AI Engineer Python LangChain", "results_count": 3}}
Observation: Returning 3 fresh posting(s).

[{{"title": "AI Engineer", "company": "Acme", "description": "..."}}, ...]
Thought: Now I'll evaluate the first job.
Action: evaluate_job_match
Action Input: {{"url": "https://himalayas.app/jobs/ai-engineer"}}
Observation: {{"job_title": "AI Engineer", "company": "Acme", "score_percent": 82.0, "inconclusive": false, ...}}
Thought: Now the next job.
Action: evaluate_job_match
Action Input: {{"url": "https://himalayas.app/jobs/..."}}
Observation: {{...}}
(continue until every job from the search is evaluated, then apply step 3
if too many were inconclusive, then rank, then write_cover_letter, then
ask_user_delivery_channel, then the deliver tool matching the answer)
Thought: I now know the final answer
Final Answer: ...

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

def build_agent1_executor(verbose: bool = True) -> AgentExecutor:
    # Use the same dedicated orchestration provider/model as Agent 2.
    llm = get_agent_llm(temperature=0.0)
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE_WITH_COVER_LETTER_AND_EMAIL)
    agent = create_react_agent(llm=llm, tools=TOOLS_WITH_COVER_LETTER_AND_DELIVERY, prompt=prompt,
                               output_parser=RequiredToolsVerifyingParser(
                                   required_tools={"write_cover_letter", "ask_user_delivery_channel"},
                                   only_if_any={"evaluate_job_match"},
                                   any_one_of=[{"send_results_draft", "send_results_telegram"}],
                               ))

    return AgentExecutor(
        agent=agent,
        tools=TOOLS_WITH_COVER_LETTER_AND_DELIVERY,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=20,
        return_intermediate_steps=True,
    )


def run_agent1(cv_info, results_count: int = 3) -> dict:
    """
    cv_info must have cv_info.mail set for the Gmail draft step to
    succeed — see agent/tools/gmail.py's error handling if it's missing.
    """
    set_candidate_profile(cv_info)
    delivery_choice._last_delivery_channel = None  # fresh user answer each run

    executor = build_agent1_executor(verbose=True)

    profile_summary = (
        f"Most recent title: {cv_info.job_titles[0] if cv_info.job_titles else 'N/A'}. "
        f"Key skills: {', '.join(cv_info.skills[:10])}. "
        f"Experience: {cv_info.experience_years} years. "
        f"Education: {cv_info.highest_education_level}."
    )

    question = (
        f"Here is the candidate's profile:\n{profile_summary}\n\n"
        f"Find {results_count} real jobs matching this profile, evaluate each "
        f"one, rank them, write a cover letter for the best match, ask the "
        f"user whether they want it as a Gmail draft or a Telegram message, "
        f"and deliver it on the chosen channel."
    )

    result = executor.invoke({"input": question})

    return _complete_deterministically(result)


def _complete_deterministically(result: dict) -> dict:
    """
    Do not trust the Final Answer — see the module docstring for why the
    model reliably claims success it never performed. Instead, rebuild the
    outcome from ground truth the tools recorded during the run:

      1. Rank every evaluation DESCENDING by score_percent (in code, so the
         order and the chosen #1 are never left to the LLM's arithmetic).
      2. Guarantee a cover letter exists for the TRUE top match, writing it
         now if the agent skipped it or wrote one for the wrong job.
      3. Guarantee the final cover letter was delivered on the chosen channel.
         If step 2 replaced a wrong letter after an earlier delivery, deliver
         the corrected letter again instead of treating the stale delivery as
         valid.

    Appends a deterministic ranked list and any corrective notes to the
    output so the final answer always shows jobs in descending score order.
    """
    import json

    notes = []

    ranked = job_evaluator.get_ranked_evaluations()

    if not ranked:
        result["output"] += (
            "\n\n[NOTE: no jobs were actually evaluated, so no cover letter or "
            "draft was possible — only the search steps ran.]"
        )
        return result

    top = ranked[0]

    # --- 1. Deterministic ranked summary (guaranteed descending) ---
    summary_lines = ["", "Final ranked jobs (computed deterministically):"]
    for i, job in enumerate(ranked, start=1):
        line = f"{i}. {job['job_title']} @ {job['company']} — {job['score_percent']}%"
        if job.get("inconclusive"):
            line += " (inconclusive)"
        summary_lines.append(line)
    result["output"] += "\n".join(summary_lines)

    # --- 2. Cover letter for the TRUE top match ---
    letter = cover_letter_tool._last_cover_letter
    letter_job = cover_letter_tool._last_cover_letter_job
    wrote_for_top = (
        letter is not None
        and letter_job is not None
        and letter_job.get("job_title") == top["job_title"]
        and letter_job.get("company") == top["company"]
    )
    cover_letter_was_replaced = False
    if not wrote_for_top:
        if letter is not None:
            notes.append(
                f"The agent wrote a cover letter for a job that was NOT the highest "
                f"scorer — rewriting it for the true top match: {top['job_title']} "
                f"@ {top['company']} ({top['score_percent']}%)."
            )
        else:
            notes.append(
                f"The agent claimed a cover letter was written but never actually "
                f"called write_cover_letter — generating it now for the true top "
                f"match: {top['job_title']} @ {top['company']} "
                f"({top['score_percent']}%)."
            )

        description = ""
        for stored in job_search._last_search_results:
            if (stored.get("title") == top["job_title"]
                    and stored.get("company") == top["company"]):
                description = stored.get("description", "")
                break

        letter_call = json.dumps({
            "job_title": top["job_title"],
            "company": top["company"],
            "score_percent": top["score_percent"],
            "url": top.get("url", ""),
            "matching_skills": top["matching_skills"],
            "missing_skills": top["missing_skills"],
            "description": description,
        })
        cover_observation = write_cover_letter.func(letter_call)
        notes.append(f"Cover letter result: {cover_observation}")

        corrected_job = cover_letter_tool._last_cover_letter_job
        cover_letter_was_replaced = (
            cover_letter_tool._last_cover_letter is not None
            and corrected_job is not None
            and corrected_job.get("job_title") == top["job_title"]
            and corrected_job.get("company") == top["company"]
        )

    # --- 3. Deliver on the channel the USER chose ---
    # Best-effort resolution: if the agent actually called
    # ask_user_delivery_channel, its answer was recorded in module state —
    # use that. If the agent skipped the question, fall back to Gmail (the
    # original default) rather than inventing a preference.
    channel = delivery_choice._last_delivery_channel or "gmail"
    channel_tool = {
        "gmail": "send_results_draft",
        "telegram": "send_results_telegram",
    }.get(channel, "send_results_draft")
    other_tool = (
        "send_results_telegram"
        if channel_tool == "send_results_draft"
        else "send_results_draft"
    )

    def _delivered(tool_name: str, observation) -> bool:
        return str(observation or "").startswith("Error:") is False

    delivery_calls = [
        (getattr(a, "tool", None), obs)
        for a, obs in result.get("intermediate_steps", [])
        if getattr(a, "tool", None) in ("send_results_draft", "send_results_telegram")
    ]

    to_call = send_results_draft if channel_tool == "send_results_draft" else send_results_telegram

    final_letter_job = cover_letter_tool._last_cover_letter_job
    final_letter_is_for_top = (
        cover_letter_tool._last_cover_letter is not None
        and final_letter_job is not None
        and final_letter_job.get("job_title") == top["job_title"]
        and final_letter_job.get("company") == top["company"]
    )

    if not final_letter_is_for_top:
        notes.append(
            "No valid cover letter exists for the true top job, so nothing "
            "was delivered."
        )
    else:
        right_channel_ok = any(
            tool == channel_tool and _delivered(tool, obs)
            for tool, obs in delivery_calls
        )
        right_channel_failed = any(
            tool == channel_tool and not _delivered(tool, obs)
            for tool, obs in delivery_calls
        )
        wrong_channel_delivered = any(
            tool == other_tool and _delivered(tool, obs)
            for tool, obs in delivery_calls
        )

        if cover_letter_was_replaced:
            stale_delivery_note = (
                " An earlier delivery may still contain the previous job's "
                "letter; it was not deleted automatically."
                if right_channel_ok or wrong_channel_delivered
                else ""
            )
            notes.append(
                f"The cover letter changed after deterministic ranking, so the "
                f"corrected letter was delivered via {channel}. Result: "
                f"{to_call.func('')}{stale_delivery_note}"
            )
        elif right_channel_ok:
            pass  # already delivered on the right channel — nothing to fix
        elif wrong_channel_delivered:
            notes.append(
                f"The candidate asked for delivery via {channel}, but the agent "
                f"delivered via the other channel — re-delivering on {channel} "
                f"now. Result: {to_call.func('')}"
            )
        elif right_channel_failed:
            notes.append(
                f"The agent's {channel_tool} call failed — retrying on {channel} "
                f"automatically. Result: {to_call.func('')}"
            )
        else:
            notes.append(
                f"The agent never actually called {channel_tool} — delivering via "
                f"{channel} now. Result: {to_call.func('')}"
            )

    if notes:
        result["output"] += "\n\n" + "\n".join(f"[NOTE: {n}]" for n in notes)

    return result
