"""
react_output_parser.py
----------------------
Tolerant ReAct output parser for small/imperfect LLMs (qwen-2.5-7b, etc).

The stock ReActSingleInputOutputParser raises OutputParserException whenever
the model emits BOTH a "Final Answer:" and a parseable "Action: ... Action
Input: ..." in a single response — a common slip for these models. With
handle_parsing_errors=True, AgentExecutor then feeds the whole error back to
the model and asks it to retry, producing noisy, slow runs like:

    Parsing LLM output produced both a final answer and a parse-able action::
    ...
    Thought: It seems there was an issue... I will attempt to call it again.

This parser instead resolves the ambiguous case deterministically: when both
are present, it treats the response as an Action and continues the loop, on
the assumption that a model which wrote a Final Answer AND kept going has not
actually finished. The true final answer is then reached on a later, clean
iteration — at worst costing one extra loop step instead of a retry.
"""

import re

from langchain_core.agents import AgentAction
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)

FINAL_ANSWER_ACTION = "Final Answer:"
ACTION_REGEX = (
    r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
)


class TolerantReActSingleInputOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        includes_answer = FINAL_ANSWER_ACTION in text
        action_match = re.search(ACTION_REGEX, text, re.DOTALL)
        if action_match and includes_answer:
            # The model wrote a premature "Final Answer" and then kept going
            # with an Action — it isn't finished. Prefer the Action so the
            # loop continues cleanly instead of erroring and forcing a retry.
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip(" ")
            action_input = action_input.strip('"')
            return AgentAction(action, action_input, text)
        return super().parse(text)
