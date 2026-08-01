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
from job_search import search_real_jobs


from tools.job_evaluator import evaluate_job_match, set_candidate_profile

TOOLS = [search_real_jobs, evaluate_job_match]


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
3. Collect all scores from evaluate_job_match results and rabk the jobs from best to worst match. :
4. how much experience needs to be gained to meet the job requirements for example if the candidate has 2 years of experience and the job requires 5 years of experience then the candidate needs to gain 3 more years of experience to meet the job requirements.
5. Write your Final Answer: the ranked list with percentages , plus 2-3
   sentences explaining the top match and what the main missing skill is.

IMPORTANT: evaluate_job_match expects a single job as a JSON string,
NOT the entire search results list. Extract one job at a time from the
search results and pass it individually.

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