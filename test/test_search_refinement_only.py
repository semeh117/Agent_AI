# test/test_search_refinement_only.py
"""
test_search_refinement_only.py
--------------
Cheap, narrowly-scoped test: does the LLM actually refine its query and
search again after search_jobs_for_agent reports a shortfall (jobs
filtered out as already-seen, fewer returned than requested)?

Deliberately does NOT wire up evaluate_job_match, write_cover_letter, or
send_results_draft — those are the expensive, token-heavy steps (full
job descriptions get fed to the LLM for requirement extraction) and are
irrelevant to what this test is checking. This keeps each run cheap
enough to use against a free-tier API without hitting token/rate limits,
and reuses whatever seen-jobs history already exists for the candidate
instead of needing to freshly exhaust the pool via full pipeline runs.

Prints every Action/Action Input/Observation so you can see directly
whether a second search_jobs_for_agent call happened with a different
query after a shortfall note.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from config import get_llm
from agent.tools.job_search_tool import search_jobs_for_agent
from agent.react_output_parser import TolerantReActSingleInputOutputParser
from search.job_search import set_candidate_email

SEARCH_ONLY_TOOLS = [search_jobs_for_agent]

SEARCH_ONLY_PROMPT = """You are a job search assistant. Your ONLY task right now is to
find fresh (not already-evaluated) job postings for the candidate — do
not evaluate, score, or write anything about them, just find them.

You have access to the following tools:

{tools}

Use this format strictly:

Question: the input question you must answer
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: a short summary of the fresh job titles found (just titles
and companies, nothing else).

RULES:
- Call search_jobs_for_agent with a query built from the candidate's
  profile below.
- Read the Observation's note carefully. If it says jobs were filtered
  out because you already evaluated them, or that fewer postings came
  back than requested, DO NOT stop there. Refine your query (different
  skill keywords, a different job title angle) and call
  search_jobs_for_agent again. Try at most 3 times total.
- Once you have at least 3 fresh postings (or you've tried 3 times),
  write your Final Answer listing what you found.

Candidate profile: {input}

Question: {input}
Thought:{agent_scratchpad}"""


def build_search_only_executor(verbose: bool = True) -> AgentExecutor:
    llm = get_llm(temperature=0.0)
    prompt = PromptTemplate.from_template(SEARCH_ONLY_PROMPT)
    agent = create_react_agent(
        llm=llm, tools=SEARCH_ONLY_TOOLS, prompt=prompt,
        output_parser=TolerantReActSingleInputOutputParser()
    )
    return AgentExecutor(
        agent=agent,
        tools=SEARCH_ONLY_TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=6,  # generous enough for up to 3 search attempts + reasoning
        return_intermediate_steps=True,
    )


def main():
    candidate_email = input(
        "Candidate email to test with (use one that already has search "
        "history, so a shortfall is likely on the first try): "
    ).strip()
    profile_summary = input(
        "Profile summary to search with (e.g. 'Python Scikit-learn PyTorch "
        "XGBoost Hugging Face Transformers RAG pipelines'): "
    ).strip()

    set_candidate_email(candidate_email)

    executor = build_search_only_executor(verbose=True)
    result = executor.invoke({"input": profile_summary})

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["output"])

    print("\n" + "=" * 70)
    print("SEARCH CALL COUNT")
    print("=" * 70)
    search_calls = [
        step for step in result.get("intermediate_steps", [])
        if getattr(step[0], "tool", None) == "search_jobs_for_agent"
    ]
    print(f"search_jobs_for_agent was called {len(search_calls)} time(s) this run.")
    for i, (action, observation) in enumerate(search_calls, start=1):
        print(f"\n  Call {i} — Action Input: {action.tool_input}")
        note_line = str(observation).split("\n")[0]
        print(f"  Call {i} — Note: {note_line}")

    if len(search_calls) > 1:
        print("\n[RESULT] Model refined its query and searched again after "
              "seeing a shortfall/filtered note — refinement behavior confirmed.")
    else:
        print("\n[RESULT] Model only searched once. Either no shortfall occurred "
              "(check the Note above), or the model didn't act on it — check the "
              "full trace above to see which.")


if __name__ == "__main__":
    main()