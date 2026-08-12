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
from agent.tools.job_evaluator import set_candidate_profile
from agent.tools.cover_letter import write_cover_letter
import agent.tools.cover_letter as cover_letter_tool
from agent.tools.gmail import send_results_draft

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

STRICT WORKFLOW — follow this exactly:
1. Call search_real_jobs with a relevant query based on the candidate's
   profile. It returns a JSON LIST of job objects.
2. For EACH job object in the list, call evaluate_job_match with that
   SINGLE job object as a JSON string. Call it once per job — do not
   batch multiple jobs into one call.
3. Collect the score_percent from every evaluate_job_match result, and
   sort the jobs by score_percent in DESCENDING order. Double check the
   numeric order is correct — do not rely on the order you evaluated
   them in.
4. Once you know which job scored HIGHEST, call write_cover_letter with
   that job's evaluate_job_match result (as a JSON string) as input.
   Only call this ONCE, for the single highest-scoring job — not for
   every job. write_cover_letter saves the letter automatically — you do
   NOT need to copy or repeat the letter text anywhere yourself.
5. Call send_results_draft with an empty string as input (it needs no
   real input — it automatically uses the letter write_cover_letter just
   saved). Only call this ONCE, after write_cover_letter has succeeded.
6. Write your Final Answer: confirm the ranked list with percentages,
   that a cover letter was generated for the top match, and that a
   Gmail draft was created — or explain clearly which step failed if
   write_cover_letter or send_results_draft returned an error.

IMPORTANT:
- evaluate_job_match expects a single job as a JSON string, NOT the
  entire search results list.
- write_cover_letter and send_results_draft should each be called
  EXACTLY ONCE per run, only for the top-ranked job — never once per
  job in the list.
- Do NOT attempt to copy the cover letter's text into send_results_draft's
  input — it takes no meaningful input and reads the letter automatically.
- If write_cover_letter or send_results_draft returns a message
  starting with "Error:", do not retry it more than once, and explain
  the failure honestly in your Final Answer rather than pretending it
  succeeded.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def build_agent2_executor(verbose: bool = True) -> AgentExecutor:
    llm = get_llm(temperature=0.0)
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE_WITH_COVER_LETTER_AND_EMAIL)
    agent = create_react_agent(llm=llm, tools=TOOLS_WITH_COVER_LETTER_AND_EMAIL, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS_WITH_COVER_LETTER_AND_EMAIL,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=15,
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

    # VERIFICATION — do not trust the Final Answer as proof send_results_draft
    # actually ran. See module docstring for why this exists.
    draft_tool_called = any(
        getattr(action, "tool", None) == "send_results_draft"
        for action, _ in result.get("intermediate_steps", [])
    )

    if not draft_tool_called:
        if cover_letter_tool._last_cover_letter is not None:
            fallback_result = send_results_draft.func("")
            result["output"] += (
                f"\n\n[NOTE: the agent's Final Answer above claimed a Gmail draft "
                f"step, but did not actually call send_results_draft — this was "
                f"caught and completed automatically as a fallback. Result: "
                f"{fallback_result}]"
            )
        else:
            result["output"] += (
                "\n\n[NOTE: no cover letter was written and no Gmail draft was "
                "created, despite what the Final Answer above may claim.]"
            )

    return result