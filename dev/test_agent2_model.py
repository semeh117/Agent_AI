"""Live Agent 2 orchestration test with a deterministic fake LinkedIn tool.

This tests the configured agent model's ReAct/tool-following behavior without
opening Chrome, scraping LinkedIn, parsing jobs, or loading embeddings.
"""

from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from agent.agent2 import AGENT2_SYSTEM_PROMPT
from config import get_agent_llm


@tool("match_linkedin_jobs_for_agent")
def fake_linkedin_match(query: str) -> str:
    """Return a fixed ranked job list for a plain LinkedIn query."""

    return json.dumps(
        {
            "query": query,
            "parsed_count": 1,
            "skipped_count": 0,
            "skipped_jobs": [],
            "ranked_jobs": [
                {
                    "rank": 1,
                    "title": "AI Engineer",
                    "company": "Example AI",
                    "final_score": 82.5,
                    "skills_score": 85.0,
                    "experience_score": 75.0,
                    "education_score": 87.5,
                    "url": "https://example.com/jobs/1",
                }
            ],
        }
    )


def main() -> int:
    tools = [fake_linkedin_match]
    model = get_agent_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                AGENT2_SYSTEM_PROMPT,
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(
        llm=model,
        tools=tools,
        prompt=prompt,
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True,
    )

    result = executor.invoke(
        {
            "input": (
                "Candidate profile: AI Engineer; skills Python, Machine Learning, "
                "Docker; 3 years experience; Bachelor's degree. Find one job."
            )
        }
    )

    steps = result.get("intermediate_steps", [])
    if len(steps) != 1:
        raise AssertionError(f"Expected exactly one tool call, received {len(steps)}.")
    if steps[0][0].tool != "match_linkedin_jobs_for_agent":
        raise AssertionError(f"Unexpected tool: {steps[0][0].tool}")
    if "AI Engineer" not in result.get("output", ""):
        raise AssertionError("Final answer did not contain the ranked job title.")

    print("\n[PASS] GPT-OSS 120B called the matching tool exactly once.")
    print("[PASS] It consumed the observation and returned the ranked job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
