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
from langchain_core.exceptions import OutputParserException
from agent.react_output_parser import (
    TolerantReActSingleInputOutputParser,
    RequiredToolsVerifyingParser,
)

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


def _fresh_verifying_parser():
    return RequiredToolsVerifyingParser(
        required_tools={"write_cover_letter", "send_results_draft"},
        only_if_any={"evaluate_job_match"},
    )


PARSER_STEP = 'Thought: I\'ll search.\nAction: search_jobs_for_agent\nAction Input: {{"query": "AI"}}'
EVAL_STEP = 'Thought: Now evaluate.\nAction: evaluate_job_match\nAction Input: {{"title": "AI Engineer"}}'
LETTER_STEP = 'Thought: Now the letter.\nAction: write_cover_letter\nAction Input: {{"job_title": "AI Engineer"}}'
DRAFT_STEP = 'Thought: Now the draft.\nAction: send_results_draft\nAction Input: ""'
FINAL_ANSWER = 'Thought: I know the answer.\nFinal Answer: Draft created for the top match.'


def test_verifying_parser_rejects_premature_final_answer():
    """The qwen failure: Final Answer WITHOUT calling required tools,
    after having evaluated jobs. Must be rejected and sent back to the LLM."""
    vp = _fresh_verifying_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    result = vp.parse(EVAL_STEP)  # a second eval is fine, still tracking
    assert isinstance(result, AgentAction)

    try:
        vp.parse(FINAL_ANSWER)
    except OutputParserException as e:
        assert e.send_to_llm is True
        assert "write_cover_letter" in e.observation
        assert "send_results_draft" in e.observation
    else:
        raise AssertionError("Premature Final Answer was not rejected")


def test_verifying_parser_accepts_final_after_required_tools():
    """The happy path: only after write_cover_letter AND send_results_draft
    have been called does the parser accept the Final Answer."""
    vp = _fresh_verifying_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    vp.parse(LETTER_STEP)
    vp.parse(DRAFT_STEP)
    out = vp.parse(FINAL_ANSWER)
    assert isinstance(out, AgentFinish)
    assert "Draft created" in out.return_values["output"]


def test_verifying_parser_does_not_require_tools_if_nothing_evaluated():
    """If NO job was evaluated (e.g. search returned nothing), the parser
    must NOT demand a cover letter — that would loop forever."""
    vp = _fresh_verifying_parser()
    vp.parse(PARSER_STEP)
    out = vp.parse(FINAL_ANSWER)
    assert isinstance(out, AgentFinish)


def test_verifying_parser_rejects_final_if_letter_called_but_no_draft():
    """Even after write_cover_letter, a Final Answer without
    send_results_draft must still be rejected."""
    vp = _fresh_verifying_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    vp.parse(LETTER_STEP)
    try:
        vp.parse(FINAL_ANSWER)
    except OutputParserException as e:
        assert "send_results_draft" in e.observation
    else:
        raise AssertionError("Final Answer accepted without send_results_draft")


def test_verifying_parser_stops_nagging_after_budget():
    """The reject loop must not run forever (that is what spammed the
    corrective message across max_iterations and blew the context window).
    After max_rejections misses, the parser must ACCEPT the Final Answer
    so the deterministic fallback can complete the steps."""
    vp = _fresh_verifying_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)

    for _ in range(2):
        try:
            vp.parse(FINAL_ANSWER)
        except OutputParserException:
            pass
        else:
            raise AssertionError("Expected rejection before budget was exhausted")

    out = vp.parse(FINAL_ANSWER)
    assert isinstance(out, AgentFinish), "Parser must give up after max_rejections"


def test_verifying_parser_rejection_replays_truncated_output():
    """llm_output replayed on rejection must be capped, so the next prompt
    does not double-include a huge model dump (context overflow)."""
    vp = _fresh_verifying_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    try:
        vp.parse(FINAL_ANSWER + "\n" + "x" * 5000)
    except OutputParserException as e:
        assert len(e.llm_output) <= 2000
    else:
        raise AssertionError("Premature Final Answer was not rejected")


def _fresh_any_one_of_parser():
    """The delivery-channel parser config: write_cover_letter +
    ask_user_delivery_channel required, and AT LEAST ONE of
    send_results_draft / send_results_telegram."""
    return RequiredToolsVerifyingParser(
        required_tools={"write_cover_letter", "ask_user_delivery_channel"},
        only_if_any={"evaluate_job_match"},
        any_one_of=[{"send_results_draft", "send_results_telegram"}],
    )


LETTER_STEP_DELIVERY = 'Thought: Now the letter.\nAction: write_cover_letter\nAction Input: {{"job_title": "AI Engineer"}}'
ASK_STEP = 'Thought: Ask the user.\nAction: ask_user_delivery_channel\nAction Input: ""'
FINAL_ANSWER_DELIVERY = 'Thought: I know the answer.\nFinal Answer: Delivered the results.'


def test_any_of_rejects_final_with_no_delivery_tool():
    """write_cover_letter + ask happened but NO delivery tool — reject."""
    vp = _fresh_any_one_of_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    vp.parse(LETTER_STEP_DELIVERY)
    vp.parse(ASK_STEP)
    try:
        vp.parse(FINAL_ANSWER_DELIVERY)
    except OutputParserException as e:
        assert "send_results_draft" in e.observation
        assert "send_results_telegram" in e.observation
    else:
        raise AssertionError("Final Answer accepted without any delivery tool")


def test_any_of_accepts_final_after_gmail_delivery():
    """Delivering via send_results_draft satisfies the any-of group."""
    vp = _fresh_any_one_of_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    vp.parse(LETTER_STEP_DELIVERY)
    vp.parse(ASK_STEP)
    vp.parse('Thought: Deliver.\nAction: send_results_draft\nAction Input: ""')
    out = vp.parse(FINAL_ANSWER_DELIVERY)
    assert isinstance(out, AgentFinish)


def test_any_of_accepts_final_after_telegram_delivery():
    """Delivering via send_results_telegram also satisfies the any-of group."""
    vp = _fresh_any_one_of_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    vp.parse(LETTER_STEP_DELIVERY)
    vp.parse(ASK_STEP)
    vp.parse('Thought: Deliver.\nAction: send_results_telegram\nAction Input: ""')
    out = vp.parse(FINAL_ANSWER_DELIVERY)
    assert isinstance(out, AgentFinish)


def test_any_of_requires_ask_user_tool():
    """Even with a delivery tool called, skipping ask_user_delivery_channel
    must be rejected — the user's preference is part of the contract."""
    vp = _fresh_any_one_of_parser()
    vp.parse(PARSER_STEP)
    vp.parse(EVAL_STEP)
    vp.parse(LETTER_STEP_DELIVERY)
    vp.parse('Thought: Deliver.\nAction: send_results_draft\nAction Input: ""')
    try:
        vp.parse(FINAL_ANSWER_DELIVERY)
    except OutputParserException as e:
        assert "ask_user_delivery_channel" in e.observation
    else:
        raise AssertionError("Final Answer accepted without ask_user_delivery_channel")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All parser tests passed.")
