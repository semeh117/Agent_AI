# test/test_react_output_parser.py
"""
test_react_output_parser.py
----------------------
Verifies the TolerantReActSingleInputOutputParser handles the failure mode
that produced "Parsing LLM output produced both a final answer and a
parse-able action" in real runs: a model emitting BOTH a "Final Answer:" and
a parseable "Action:/Action Input:" in one response.

The tolerant parser must resolve that case as an Action (continue the loop)
instead of raising OutputParserException and forcing a noisy retry — while
still behaving exactly like the stock parser for clean inputs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.agents import AgentAction, AgentFinish
from agent.react_output_parser import TolerantReActSingleInputOutputParser

parser = TolerantReActSingleInputOutputParser()


def test_clean_action():
    out = parser.parse(
        'Thought: I should search.\nAction: search_real_jobs\nAction Input: {"query": "AI"}'
    )
    assert isinstance(out, AgentAction)
    assert out.tool == "search_real_jobs"


def test_clean_final_answer():
    out = parser.parse(
        'Thought: I am done.\nFinal Answer: The ranked jobs are A, B, C.'
    )
    assert isinstance(out, AgentFinish)
    assert "A, B, C" in out.return_values["output"]


def test_both_final_answer_and_action_continues_loop():
    """The exact failure from real runs — must return an Action, not raise."""
    out = parser.parse(
        'Thought: I now know the final answer\n'
        'Final Answer: I analyzed the jobs and ranked them... I will now '
        'write the cover letter.\n'
        'Thought: Action: write_cover_letter\n'
        'Action Input: {"job_title": "AI Engineer"}'
    )
    assert isinstance(out, AgentAction)
    assert out.tool == "write_cover_letter"


def test_stock_parser_raises_but_tolerant_does_not():
    """Prove the divergence: the stock parser raises on the same input."""
    from langchain.agents.output_parsers.react_single_input import (
        ReActSingleInputOutputParser,
    )
    from langchain_core.exceptions import OutputParserException

    text = (
        'Thought: I now know the final answer\n'
        'Final Answer: done.\n'
        'Thought: Action: write_cover_letter\n'
        'Action Input: {}'
    )
    try:
        ReActSingleInputOutputParser().parse(text)
    except OutputParserException:
        pass
    else:
        raise AssertionError("Stock parser was expected to raise on 'both' input")

    out = parser.parse(text)
    assert isinstance(out, AgentAction)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All parser tests passed.")
