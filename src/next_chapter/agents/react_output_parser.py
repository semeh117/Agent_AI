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

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
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


class RequiredToolsVerifyingParser(TolerantReActSingleInputOutputParser):
    """
    Enforces that a Final Answer is only accepted AFTER every tool in
    required_tools has actually been called AND its observation seen.

    Fixes the observed qwen-2.5-7b failure: the model writes a confident
    Final Answer ("a Gmail draft has been created") without ever emitting an
    Action for write_cover_letter / send_results_draft. This parser tracks
    every tool the agent has truly acted on (from parse() return values) and
    raises OutputParserException with send_to_llm=True on a premature
    AgentFinish. Because every AgentExecutor here is built with
    handle_parsing_errors=True, that exception becomes the next loop
    Observation — the agent is told which tool(s) it skipped and continues,
    so the missing tool IS called by the agent loop itself, not by the
    Python fallback.

    only_if_any: the set of tool names that, if any has been called, makes
    the requirement apply. Used to avoid demanding write_cover_letter on a
    run where NOTHING was evaluated (e.g. search returned nothing) — there
    the skip causes an infinite loop until max_iterations.

    any_one_of: a list of tool-name groups; at least ONE tool from each
    group must have been called. Used e.g. when delivery can be EITHER a
    Gmail draft OR a Telegram message — the chain requires
    [{send_results_draft, send_results_telegram}], not both.
    """

    required_tools: set = set()
    only_if_any: set = set()
    any_one_of: list = []
    max_rejections: int = 2

    def __init__(self, required_tools, only_if_any, any_one_of=None,
                 max_rejections: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.required_tools = set(required_tools)
        self.only_if_any = set(only_if_any)
        self.any_one_of = [set(group) for group in (any_one_of or [])]
        self.max_rejections = max_rejections
        self._called_tools: set[str] = set()
        self._rejections_left = max_rejections

    def parse(self, text: str):
        parsed = super().parse(text)

        if isinstance(parsed, AgentAction):
            self._called_tools.add(parsed.tool)
            return parsed

        if not isinstance(parsed, AgentFinish):
            return parsed

        # A Final Answer was produced. Accept it only if the run actually
        # reached the evaluation stage AND every required tool was called.
        reached_stage = bool(self.only_if_any) and bool(self._called_tools & self.only_if_any)
        missing = self.required_tools - self._called_tools
        missing_any_groups = [
            sorted(group)
            for group in self.any_one_of
            if not (self._called_tools & group)
        ]

        if (reached_stage and (missing or missing_any_groups)
                and self._rejections_left > 0):
            self._rejections_left -= 1
            problem_names = sorted(missing)
            for group in missing_any_groups:
                problem_names.append(" or ".join(group))
            missing_names = ", ".join(
                str(name) if isinstance(name, str) else f"({name})"
                for name in problem_names
            )
            raise OutputParserException(
                "You gave a Final Answer without actually calling the tool(s): "
                f"{missing_names}.",
                observation=(
                    "Your Final Answer mentioned work that was never performed. "
                    "The run transcript shows you called none of these required "
                    f"tool(s): {missing_names}. "
                    "Continue NOW: emit exactly one Action/Action Input for each "
                    "missing tool, in order, wait for each Observation, and only "
                    "afterwards give your Final Answer."
                ),
                llm_output=text[:2000],
                send_to_llm=True,
            )

        # Accepted (tools all called, stage not reached, or rejection budget
        # exhausted) — reset per-run state and let the loop finish.
        self._called_tools.clear()
        self._rejections_left = self.max_rejections
        return parsed
