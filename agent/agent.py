"""
agent.py
--------------
ReAct agent: given a candidate's already-extracted profile, searches for
real jobs and evaluates each one, then produces a ranked recommendation
with reasoning — not just raw scores.

CV extraction stays OUTSIDE the agent (deterministic preprocessing,
happens once in app-level code before the agent runs) — the agent's job
starts at "search for jobs" and ends at "explain the recommendation."
"""


from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from config import get_llm
from search.job_search import search_real_jobs


from agent.tools.job_evaluator import evaluate_job_match, set_candidate_profile
from agent.tools.cover_letter import write_cover_letter
import agent.tools.cover_letter as cover_letter_tool  # for _last_cover_letter / _last_cover_letter_job
from agent.tools.gmail import send_results_draft 

TOOLS = [search_real_jobs, evaluate_job_match]
TOOLS_WITH_COVER_LETTER_AND_EMAIL = TOOLS + [write_cover_letter, send_results_draft]


REACT_PROMPT_TEMPLATE = """You are an expert career assistant AI helping a candidate
find and evaluate real job opportunities based on their profile.

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
Final Answer: ranked list of jobs from best to worst match with scores and
a short explanation of the top match's strengths and main skill gap.

STRICT WORKFLOW — follow this exactly:
1. Call search_real_jobs with a relevant query based on the candidate's
   profile. It returns a JSON LIST of job objects.
2. For EACH job object in the list, call evaluate_job_match with that
   SINGLE job object as a JSON string. Call it once per job — do not
   batch multiple jobs into one call.
3. Collect the score_percent from every evaluate_job_match result.
4. Sort the jobs by score_percent in DESCENDING order (highest match
   first, lowest match last) BEFORE writing your Final Answer. Double
   check the numeric order is correct — do not rely on the order you
   evaluated them in.
5. Write your Final Answer: the SORTED ranked list with percentages,
   plus 2-3 sentences explaining the top match's strengths and its main
   missing skill.

IMPORTANT: evaluate_job_match expects a single job as a JSON string,
NOT the entire search results list. Extract one job at a time from the
search results and pass it individually.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


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


def build_agent_executor(verbose: bool = True) -> AgentExecutor:
    llm = get_llm(temperature=0.0)
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
    )


def build_agent_executor_with_cover_letter_and_email(verbose: bool = True) -> AgentExecutor:
    """
    Same as build_agent_executor(), but wired to the extended tool set
    and prompt — the agent itself decides when to write the cover letter
    and when to create the Gmail draft, as two more of its own tool
    calls, rather than those steps happening automatically in Python
    after the agent finishes (see run_agent_job_matching_with_cover_letter()
    for that alternative design).

    max_iterations is raised from 10 to 15 since this workflow has two
    more mandatory steps (write_cover_letter, send_results_draft) beyond
    search + N evaluations — the original cap was sized for the shorter
    workflow and could cut this one off before it reaches the email step.
    """
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


def run_agent_job_matching(cv_info, results_count: int = 3) -> dict:
    """
    cv_info: a CVInfo object, already extracted via cv_parser.py
    (deterministic step, done BEFORE calling this function).
    """
    set_candidate_profile(cv_info)  # make it available to evaluate_job_match

    executor = build_agent_executor(verbose=True)

    profile_summary = (
        f"Most recent title: {cv_info.job_titles[0] if cv_info.job_titles else 'N/A'}. "
        f"Key skills: {', '.join(cv_info.skills[:10])}. "
        f"Experience: {cv_info.experience_years} years. "
        f"Education: {cv_info.highest_education_level}."
    )

    question = (
        f"Here is the candidate's profile:\n{profile_summary}\n\n"
        f"Find {results_count} real jobs matching this profile, evaluate each "
        f"one, and give a ranked recommendation."
    )

    return executor.invoke({"input": question})


def run_agent_job_matching_full_auto(cv_info, results_count: int = 3) -> dict:
    """
    Runs the FULLY agent-driven version: the LLM itself decides to call
    write_cover_letter and send_results_draft as two more steps in its
    own ReAct loop, after search + evaluate + rank — rather than those
    steps being guaranteed Python calls after the agent finishes (compare
    against run_agent_job_matching_with_cover_letter(), which uses the
    latter approach).

    This is the design meant by "add Gmail as a tool the agent itself
    decides to use" — the agent reasons its way through writing the
    letter and creating the draft, the same way it already reasons its
    way through searching and evaluating jobs.

    cv_info: a CVInfo object, already extracted via cv_parser.py
    (deterministic step, done BEFORE calling this function). Must have
    cv_info.mail set for the Gmail draft step to succeed — see
    agent/tools/gmail_tool.py's error handling if it's missing.
    """
    set_candidate_profile(cv_info)  # available to evaluate_job_match,
    # write_cover_letter, and send_results_draft — all three read the
    # same module-level profile.

    executor = build_agent_executor_with_cover_letter_and_email(verbose=True)

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
 
    # VERIFICATION STEP — do not trust the LLM's Final Answer as proof
    # send_results_draft actually ran. In testing, the model sometimes
    # generated "a Gmail draft has been created and sent" in its Final
    # Answer despite NEVER calling send_results_draft at all (visible by
    # checking intermediate_steps — no Action: send_results_draft entry
    # present). This checks the agent's actual tool-call history, not
    # its self-reported narration, and calls the tool directly as a
    # guaranteed fallback if the agent skipped it.
    draft_tool_called = any(
        getattr(action, "tool", None) == "send_results_draft"
        for action, _ in result.get("intermediate_steps", [])
    )
 
    if not draft_tool_called:
        letter_written = cover_letter_tool._last_cover_letter is not None
        if letter_written:
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
 


def _find_top_evaluation(intermediate_steps: list) -> dict | None:
    """
    Walks the agent's tool-call history, keeps only evaluate_job_match
    results, and returns the one with the highest score_percent.

    Real version of the pattern demoed in dev/ earlier — same idea
    (search intermediate_steps for evaluate_job_match calls, parse the
    JSON, take the max), but handles malformed/error entries instead of
    assuming every call succeeded.
    """
    import json

    best = None
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        if tool_name != "evaluate_job_match":
            continue
        try:
            result = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            continue
        if "error" in result:
            continue
        if best is None or result["score_percent"] > best["score_percent"]:
            best = result
    return best


def _find_job_description(intermediate_steps: list, job_title: str, company: str) -> str:
    """
    evaluate_job_match's own output doesn't include the job description
    (job_evaluator.py never asked for one back) — but the original
    search_real_jobs call earlier in the SAME intermediate_steps list
    does have it. This looks it up by matching title+company, so the
    cover letter has real job text to work from instead of nothing.
    Returns "" if not found (e.g. title/company text didn't match
    exactly), which generate_cover_letter() will just render as an
    empty Description line rather than crash.
    """
    import json

    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        if tool_name != "search_real_jobs":
            continue
        try:
            jobs = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if job.get("title") == job_title and job.get("company") == company:
                return job.get("description", "")
    return ""


def run_agent_job_matching_with_cover_letter(cv_info, results_count: int = 3) -> dict:
    """
    Runs the agent exactly as run_agent_job_matching() does (search,
    evaluate, rank, summarize), then ADDITIONALLY generates a cover
    letter for whichever job the agent scored highest — by reading it
    back out of intermediate_steps, not by re-running or re-asking the
    LLM to produce it as part of the same Final Answer.

    This exists specifically to compare against the pipeline version
    (pipeline/job_matching_pipeline.py + pipeline/cover_letter.py) —
    same end result, sourced from the agent's own ranking instead of
    the deterministic pipeline's ranking.

    Returns a dict with everything run_agent_job_matching() returns,
    plus "cover_letter" (str) and "cover_letter_job" (dict, the job the
    letter was written for) — both None if no successful evaluation was
    found in the agent's trace.
    """
    from pipeline.cover_letter import generate_cover_letter

    result = run_agent_job_matching(cv_info, results_count=results_count)

    top_evaluation = _find_top_evaluation(result["intermediate_steps"])
    if top_evaluation is None:
        result["cover_letter"] = None
        result["cover_letter_job"] = None
        return result

    description = _find_job_description(
        result["intermediate_steps"],
        top_evaluation["job_title"],
        top_evaluation["company"],
    )

    # Adapt evaluate_job_match's flat shape into the shape
    # generate_cover_letter() actually expects (same shape as
    # job_matching_pipeline.py's results[0]). job_evaluator.py's
    # matching_skills is already [{"job_skill": ..., "matched_via": ...}]
    # (straight from matcher.py's skills_result["matching"]), so this is
    # a direct pass-through, not a real format conversion.
    top_result_for_letter = {
        "job_title": top_evaluation["job_title"],
        "company": top_evaluation["company"],
        "description": description,
        "score_percent": top_evaluation["score_percent"],
        "skills_detail": {
            "matching": top_evaluation["matching_skills"],
            "missing": top_evaluation["missing_skills"],
        },
    }

    result["cover_letter"] = generate_cover_letter(cv_info, top_result_for_letter)
    result["cover_letter_job"] = top_result_for_letter
    return result