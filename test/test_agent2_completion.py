# test/test_agent2_completion.py
"""
test_agent2_completion.py
----------------------
Verifies _complete_deterministically(), the safety net in
agent2_full_auto.py that must guarantee three things regardless of how
unreliably the LLM behaved:

  1. Evaluations are ranked DESCENDING by score_percent (the exact failure
     in a real run: jobs shown in scrambled order and the wrong job chosen
     as top match).
  2. A cover letter exists for the TRUE top match (written automatically if
     the agent skipped it or wrote one for a lower-scoring job).
  3. The Gmail draft is actually created (created automatically if the agent
     only claimed to have done so).

LLM and Gmail calls are faked here — only the completion logic is tested.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.agent2_full_auto as full_auto
import agent.tools.job_evaluator as job_evaluator
import agent.tools.cover_letter as cover_letter_tool
from search import job_search

EVALS = [
    {"job_title": "AI Engineer (LLMs)", "company": "Acme", "score_percent": 64.0,
     "url": "u1", "matching_skills": [], "missing_skills": [], "inconclusive": True},
    {"job_title": "FDE Teaching Expert", "company": "Learn", "score_percent": 53.8,
     "url": "u2", "matching_skills": [], "missing_skills": [], "inconclusive": True},
    {"job_title": "AI Engineer - Equity", "company": "Startup", "score_percent": 67.6,
     "url": "u3", "matching_skills": [], "missing_skills": [], "inconclusive": False},
]


class FakeTool:
    def __init__(self, func):
        self.func = func


def reset_state():
    job_evaluator._all_evaluations = list(EVALS)
    cover_letter_tool._last_cover_letter = None
    cover_letter_tool._last_cover_letter_job = None
    job_search._last_search_results = [
        {"title": EVALS[2]["job_title"], "company": EVALS[2]["company"],
         "description": "build agentic systems"},
    ]


def make_agent_did_nothing():
    reset_state()
    written = []
    drafted = []

    def fake_write(call_json):
        written.append(call_json)
        # mimic the real tool: it stores the letter + job for send_results_draft
        cover_letter_tool._last_cover_letter = "fake letter text"
        cover_letter_tool._last_cover_letter_job = {
            "job_title": EVALS[2]["job_title"], "company": EVALS[2]["company"]}
        return "fake letter written"

    full_auto.write_cover_letter = FakeTool(fake_write)
    full_auto.send_results_draft = FakeTool(
        lambda _: drafted.append(1) or "fake draft created"
    )
    result = {"output": "Final Answer: I wrote the cover letter and created the draft.", "intermediate_steps": []}
    out = full_auto._complete_deterministically(result)
    return out, written, drafted


def test_ranked_list_is_descending():
    out, _, _ = make_agent_did_nothing()
    output = out["output"]
    idx_676 = output.index("67.6%")
    idx_640 = output.index("64.0%")
    idx_538 = output.index("53.8%")
    assert idx_676 < idx_640 < idx_538, "ranked list must be descending"


def test_writes_letter_for_true_top_match():
    out, written, _ = make_agent_did_nothing()
    assert len(written) == 1
    import json
    call = json.loads(written[0])
    assert call["job_title"] == "AI Engineer - Equity"
    assert call["score_percent"] == 67.6
    assert "build agentic systems" in call["description"]


def test_creates_draft_when_agent_claimed_it():
    out, _, drafted = make_agent_did_nothing()
    assert len(drafted) == 1
    assert "creating it now" in out["output"]
    assert "fake draft created" in out["output"]


def test_no_duplicate_when_agent_did_everything_right():
    reset_state()
    cover_letter_tool._last_cover_letter = "existing letter"
    cover_letter_tool._last_cover_letter_job = {
        "job_title": "AI Engineer - Equity", "company": "Startup"}
    drafted = []
    full_auto.send_results_draft = FakeTool(lambda _: drafted.append(1))
    action = type("A", (), {"tool": "send_results_draft"})()
    result = {
        "output": "Final Answer: done.",
        "intermediate_steps": [(action, "Draft created (id: 123)")],
    }
    out = full_auto._complete_deterministically(result)
    assert len(drafted) == 0, "must not create a second draft"
    assert "ERROR" not in out["output"].upper()


def test_no_evaluations_is_handled():
    job_evaluator._all_evaluations = []
    result = {"output": "x", "intermediate_steps": []}
    out = full_auto._complete_deterministically(result)
    assert "no jobs were actually evaluated" in out["output"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All completion tests passed.")
