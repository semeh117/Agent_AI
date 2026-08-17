"""
agent2_full_auto.py
--------------
PATH B — fully agent-driven. The LLM itself decides, as further steps
in its own ReAct loop, when to write the cover letter and when to
create the Gmail draft — the same way it already decides to search and
evaluate.

OBSERVED FAILURE MODE (kept here deliberately, not smoothed over): in
testing with qwen-2.5-7b-instruct, the model has twice produced a Final
Answer confidently claiming "a Gmail draft has been created" while its
own intermediate_steps show send_results_draft was never actually
called. The verification block in run_agent2_full_auto() below is a
fallback that catches this and completes the step for real — meaning
this path currently only works BECAUSE of that fallback, not because
the agent reliably follows the prompted workflow. Compare against
agent1_deterministic.py, which has no equivalent failure mode since
nothing past ranking is left to the LLM's judgment.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from config import get_llm
from agent.agent_core import TOOLS
from agent.react_output_parser import TolerantReActSingleInputOutputParser
import agent.tools.job_evaluator as job_evaluator  # module import — need the
# LIVE _current_cv_info / _all_evaluations, only set at runtime.
from agent.tools.job_evaluator import set_candidate_profile
from agent.tools.cover_letter import write_cover_letter
import agent.tools.cover_letter as cover_letter_tool
from agent.tools.gmail import send_results_draft
from search import job_search

TOOLS_WITH_COVER_LETTER_AND_EMAIL = TOOLS + [write_cover_letter, send_results_draft]

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
Never write "Final Answer" until you have actually called search_real_jobs,
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
1. Call search_real_jobs with a query built from the candidate's profile.
   It returns a JSON list of job objects.
2. Call evaluate_job_match once per job, each time with a SINGLE job
   object as a JSON string — never the whole list, never batched.
3. Look at every evaluation's "inconclusive" field. inconclusive: true
   means the posting had no extractable requirements, so its
   score_percent is meaningless even if it shows 100% — it is not a
   real match. If fewer than 2 evaluations came back
   inconclusive: false, say so explicitly (e.g. "I've already analyzed
   and ranked these jobs, but most were inconclusive — searching for
   more similar postings before finalizing"), then call search_real_jobs
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
6. Call send_results_draft exactly once, with an empty string as input,
   only after write_cover_letter has succeeded. It reads the saved
   letter automatically.
7. Write your Final Answer: confirm the ranked list with percentages
   (noting which were inconclusive), that a cover letter was generated
   for the top genuine match, and that a Gmail draft was created — or
   explain clearly which step failed if write_cover_letter or
   send_results_draft returned an error.

NOTES:
- evaluate_job_match takes one job as a JSON string, not the search
  results list.
- write_cover_letter and send_results_draft are each called EXACTLY
  ONCE per run, only for the top job — never once per job in the list.
- Never hand-copy the cover letter text into send_results_draft's input.
- If write_cover_letter or send_results_draft returns a message starting
  with "Error:", do not retry more than once, and report the failure
  honestly instead of claiming success.

EXAMPLE OF CORRECT FORMAT (abbreviated — follow this exact pattern):
Thought: I need to find jobs matching this profile first.
Action: search_real_jobs
Action Input: {{"query": "AI Engineer Python LangChain", "results_count": 3}}
Observation: [{{"title": "AI Engineer", "company": "Acme", "description": "..."}}, ...]
Thought: Now I'll evaluate the first job.
Action: evaluate_job_match
Action Input: {{"title": "AI Engineer", "company": "Acme", "description": "..."}}
Observation: {{"job_title": "AI Engineer", "company": "Acme", "score_percent": 82.0, "inconclusive": false, ...}}
Thought: Now the next job.
Action: evaluate_job_match
Action Input: {{"title": "...", "company": "...", "description": "..."}}
Observation: {{...}}
(continue until every job from the search is evaluated, then apply step 3
if too many were inconclusive, then rank, then write_cover_letter, then
send_results_draft)
Thought: I now know the final answer
Final Answer: ...

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

def build_agent2_executor(verbose: bool = True) -> AgentExecutor:
    llm = get_llm(temperature=0.0)
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE_WITH_COVER_LETTER_AND_EMAIL)
    agent = create_react_agent(llm=llm, tools=TOOLS_WITH_COVER_LETTER_AND_EMAIL, prompt=prompt,
                               output_parser=TolerantReActSingleInputOutputParser())

    return AgentExecutor(
        agent=agent,
        tools=TOOLS_WITH_COVER_LETTER_AND_EMAIL,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=20,
        return_intermediate_steps=True,
    )


def run_agent2_full_auto(cv_info, results_count: int = 3) -> dict:
    """
    cv_info must have cv_info.mail set for the Gmail draft step to
    succeed — see agent/tools/gmail.py's error handling if it's missing.
    """
    set_candidate_profile(cv_info)

    executor = build_agent2_executor(verbose=True)

    profile_summary = (
        f"Most recent title: {cv_info.job_titles[0] if cv_info.job_titles else 'N/A'}. "
        f"Key skills: {', '.join(cv_info.skills[:10])}. "
        f"Experience: {cv_info.experience_years} years. "
        f"Education: {cv_info.highest_education_level}."
    )

    question = (
        f"Here is the candidate's profile:\n{profile_summary}\n\n"
        f"Find {results_count} real jobs matching this profile, evaluate each "
        f"one, rank them, write a cover letter for the best match, and "
        f"send it to the candidate as a Gmail draft."
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
      3. Guarantee the Gmail draft was actually created, creating it now if
         the agent claimed it without calling the tool.

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
        notes.append(f"Cover letter result: {write_cover_letter.func(letter_call)}")

    # --- 3. Guarantee the Gmail draft was actually created ---
    draft_observation = None
    for action, observation in result.get("intermediate_steps", []):
        if getattr(action, "tool", None) == "send_results_draft":
            draft_observation = observation
            break

    if draft_observation is None:
        if cover_letter_tool._last_cover_letter is not None:
            notes.append(
                "The agent claimed a Gmail draft was created but never actually "
                f"called send_results_draft — creating it now. Result: "
                f"{send_results_draft.func('')}"
            )
        else:
            notes.append("No cover letter exists, so no draft could be created.")
    elif str(draft_observation).startswith("Error:"):
        notes.append(
            f"The agent's send_results_draft call failed ({draft_observation}) — "
            f"retrying automatically. Result: {send_results_draft.func('')}"
        )

    if notes:
        result["output"] += "\n\n" + "\n".join(f"[NOTE: {n}]" for n in notes)

    return result