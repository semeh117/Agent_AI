""" 
shared foundation for both agents designs (agent1.py and agent2.py) 
the search + evaluate ReAct loop that both paths run identically .Cover-letter and an email/draft delvery are where two paths diverge
"""
from langchain.agents import  AgentExecutor , creaate_react_agent
from langchain_core.tools import PromptTemplate
from numpy.testing import verbose
from config import get_llm
from search.job_search import search_real_jobs
from agent.tools.job_evaluator import evaluate_job_match, set_candidate_profile

TOOLS =[search_real_jobs, evaluate_job_match]
REACT_PROMPT_TEMPLATE = """
You are an expert career-matching assistant. Your task is to find real job opportunities and evaluate how well each opportunity matches the candidate's profile.

## Available tools

{tools}

Available tool names:
{tool_names}

## Candidate request

{input}

## Required tool workflow

Follow this workflow exactly:

### Step 1 — Search for jobs

Call `search_real_jobs` exactly as needed to obtain relevant real job opportunities.

Build the search query from the candidate's profile and request. Prioritize relevant job titles, required skills, experience level, location/work arrangement, industry, and other explicitly stated constraints.

The tool returns a JSON list of job objects.

### Step 2 — Evaluate every job individually

For every job object returned by `search_real_jobs`:

1. Extract exactly one job object.
2. Serialize that single job object as a valid JSON string.
3. Call `evaluate_job_match` once for that job.
4. Do not pass the entire search-results list.
5. Do not combine multiple jobs into one evaluation call.
6. Record the returned `score_percent` together with the corresponding job.

Repeat until every returned job has been evaluated.

### Step 3 — Validate the results

Before producing the final answer:

* Ensure every searched job has a corresponding evaluation.
* Ensure each evaluation is associated with the correct job.
* Use the `score_percent` returned by `evaluate_job_match`; do not invent, modify, or recalculate scores unless the tool explicitly requires it.
* Sort all evaluated jobs by `score_percent` in descending numerical order.
* Double-check the ordering before responding.
* Do not preserve the search-result order or evaluation order unless it happens to match the ranking.

If two jobs have the same score, use relevance to the candidate's stated preferences as the secondary ranking criterion.

## Tool interaction format

Use this format for tool reasoning:

Question: the input question you must answer
Thought: briefly determine the next required step
Action: one of [{tool_names}]
Action Input: the input to the selected action
Observation: the tool result

Repeat the Thought → Action → Action Input → Observation cycle as necessary.

Do not skip the required individual job evaluations.

After all jobs have been evaluated:

Thought: I now know the final answer

## Final answer requirements

Return a ranked list from highest match to lowest match.

For each job, include:

1. Rank
2. Job title
3. Company
4. Match score (`score_percent`)
5. Location/work arrangement, if available
6. Job URL, if available
7. A concise reason for the ranking

Then provide a short analysis of the #1 match containing:

* Its strongest alignment with the candidate's profile
* The most important missing or weaker skill/requirement
* Any notable consideration the candidate should know before applying

Keep the final answer concise, factual, and decision-oriented.

Never claim a job is real, available, or currently accepting applications unless that information is supported by the search-tool result.

Begin  !

Question: {input}
{agent_scratchpad}

"""
def build_agent_executor(verbose: bool = True) -> AgentExecutor:
    llm = get_llm(temperature=0.0)
    prompt = PromptTemplate.from_Template(REACT_PROMPT_TEMPLATE)
    agent = creaate_react_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=TOOLS, 
        verbose=verbose,
        handle_parsing_errors=True,  # if the LLM returns invalid JSON, don't crash; instead, return an error message
        max_iterations=10,  # prevent infinite loops if the LLM gets stuck
        return_intermediate_steps=True,  # for debugging and analysis; can be set to False in production
        )
    
def run_agent_job_matching(cv_info: dict, results_count : int=3 ) -> dict :
    """
    cv_info : a CVINfo object , already extracted via cv_prser.py 
    (deterministic steps ,done BEFORE calling this function)."""   

    set_candidate_profile(cv_info)

    executor =build_agent_executor(verbose=True)

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