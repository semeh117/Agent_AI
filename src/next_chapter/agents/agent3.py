"""Agent 3: a classic ReAct CV-to-LinkedIn workflow.

Agent 3 exposes scraping, evaluation, skill-gap analysis, and cover-letter
generation as separate actions. The model follows the textual
Thought/Action/Action Input/Observation protocol provided by LangChain's
``create_react_agent``. Delivery remains an application-level action performed
only after the user supplies a Gmail or Telegram choice.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

import next_chapter.agents.tools.cover_letter as cover_letter_tool
import next_chapter.agents.tools.interview_preparation as interview_tool
import next_chapter.agents.tools.job_evaluator as job_evaluator
from next_chapter.agents.react_output_parser import (
    RequiredToolsVerifyingParser,
    TolerantReActSingleInputOutputParser,
)
from next_chapter.agents.tools.agent3_react_tools import (
    Agent3RunContext,
    build_agent3_react_tools,
)
from next_chapter.agents.tools.interview_preparation import (
    generate_interview_preparation_pdf,
)
from next_chapter.config import get_agent3_llm


AGENT3_TOOL_NAMES = (
    "search_linkedin_jobs",
    "evaluate_linkedin_results",
    "analyze_skill_gaps",
    "write_cover_letter_for_job",
)


AGENT3_REACT_PROMPT = """You are an expert career-matching assistant. Use a
classic ReAct loop to find and evaluate LinkedIn jobs for the candidate and
write one grounded cover letter for the best valid match.

You have access to these tools:

{tools}

Use this exact format:

Question: the request you must complete
Thought: briefly state what evidence or action is needed next
Available Action names: {tool_names}
Action: write exactly one available action name, with no explanation
Action Input: one plain-text input for that tool
Observation: the real tool result
... repeat Thought/Action/Action Input/Observation as needed
Thought: I now know the final answer
Final Answer: a concise factual result based only on tool observations

Required workflow:

1. Build a concise query from one realistic role and up to three strong skills.
2. Call search_linkedin_jobs. Its input is only the plain query.
3. Inspect the returned titles. A second, refined search is allowed only when
   the first result set is empty, too small, or clearly irrelevant. Never make
   more than two searches.
4. Call evaluate_linkedin_results exactly once after the final search. Pass
   the word none as its input. The tool parses requirements, calculates the
   existing skill/experience/education scores, ranks jobs, and saves them.
5. Preserve the returned ranking and scores. Never calculate your own scores.
6. If ranked jobs exist, call analyze_skill_gaps exactly once with the word
   none, then call write_cover_letter_for_job exactly once with only the URL
   of the first ranked job.
7. If a tool returns an error, report it honestly. Never claim an action
   succeeded unless its Observation confirms success.
8. Delivery is outside this ReAct loop. Do not ask for Gmail or Telegram and
   do not claim anything was delivered.

Scoring contract:

final_score = skills_score * 0.5
            + experience_score * 0.3
            + education_score * 0.2

An unspecified experience or education requirement receives no penalty, but
it is not a verified match. ESCO normalization counts describe vocabulary
coverage, not candidate-job matches.

In the Final Answer, preserve ranking order and include title, company, final
score, component scores, URL, application_id when available, and the strongest
match or biggest constraint. Summarize recurring skill gaps from the tool's
observation and confirm which job received the cover letter.

Never put an Action and a Final Answer in the same response. Stop after one
Action Input and wait for its Observation.

Your first response must follow this shape (choose the query from the actual
candidate profile):
Thought: I need to search for relevant LinkedIn jobs.
Action: search_linkedin_jobs
Action Input: Data Scientist Python scikit-learn XGBoost

Begin.

Question: {input}
Thought:{agent_scratchpad}"""


INTERVIEW_REACT_PROMPT = """You prepare an interview pack for one saved job.

Tools:
{tools}

Use this exact format:
Question: the request
Thought: what action is required
Available Action names: {tool_names}
Action: write exactly one available action name
Action Input: the saved application ID
Observation: the real tool result
Thought: I now know the final answer
Final Answer: report the job, company, PDF path, provider, and model, or the
error returned by the tool

Call generate_interview_preparation_pdf exactly once. Do not claim that a PDF
exists unless the Observation contains a pdf_path.

Question: {input}
Thought:{agent_scratchpad}"""


def _reset_agent3_state(cv_info: Any) -> None:
    """Reset shared legacy service state used by cover and delivery adapters."""

    job_evaluator.set_candidate_profile(cv_info)
    cover_letter_tool._last_cover_letter = None
    cover_letter_tool._last_cover_letter_job = None
    interview_tool.reset_interview_preparation_state()


def build_agent3_prompt() -> PromptTemplate:
    """Build Agent 3's textual Thought/Action/Observation prompt."""

    return PromptTemplate.from_template(AGENT3_REACT_PROMPT)


def build_agent3_executor(
    context: Agent3RunContext,
    verbose: bool = True,
) -> AgentExecutor:
    """Create Agent 3's classic ReAct executor for one isolated run."""

    tools = build_agent3_react_tools(context)
    agent = create_react_agent(
        llm=get_agent3_llm(temperature=0.0),
        tools=tools,
        prompt=build_agent3_prompt(),
        output_parser=RequiredToolsVerifyingParser(
            required_tools={"analyze_skill_gaps", "write_cover_letter_for_job"},
            only_if_any={"evaluate_linkedin_results"},
            tool_aliases={
                "search_linkinned_jobs": "search_linkedin_jobs",
                "search_linked_in_jobs": "search_linkedin_jobs",
            },
            valid_tools=AGENT3_TOOL_NAMES,
        ),
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
    )


def _profile_question(
    cv_info: Any,
    results_count: int,
    location: str,
    query: str = "",
) -> str:
    profile_summary = (
        "Most recent title: "
        f"{cv_info.job_titles[0] if cv_info.job_titles else 'N/A'}. "
        f"Other roles: {', '.join(cv_info.job_titles[1:5]) or 'N/A'}. "
        f"Key skills: {', '.join(cv_info.skills[:15])}. "
        f"Experience: {cv_info.experience_years} years. "
        f"Education: {cv_info.highest_education_level or 'N/A'}. "
        f"Requested location: {location or 'any location'}."
    )
    requested_query = str(query or "").strip()
    query_instruction = (
        f"The user supplied this search query; use it exactly for the first "
        f"search: {requested_query}. "
        if requested_query
        else "Build the first search query from the candidate profile. "
    )
    return (
        f"Here is the candidate's profile:\n{profile_summary}\n\n"
        f"{query_instruction}"
        f"Find up to {results_count} LinkedIn jobs, evaluate and rank them, "
        "analyze recurring skill gaps, and write one cover letter for the "
        "first valid ranked job."
    )


def _react_trace(agent_result: dict[str, Any]) -> list[dict[str, str]]:
    """Return an auditable action/observation trace without model thoughts."""

    return [
        {
            "tool": str(action.tool),
            "input": str(action.tool_input),
            "observation": str(observation),
        }
        for action, observation in agent_result.get("intermediate_steps", [])
    ]


def _normalize_delivery_channel(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {"email": "gmail", "mail": "gmail", "tg": "telegram"}
    channel = str(value).strip().casefold()
    channel = aliases.get(channel, channel)
    if channel not in {"gmail", "telegram"}:
        raise ValueError("delivery_channel must be 'gmail' or 'telegram'.")
    return channel


def _finalize_agent3_result(
    agent_result: dict[str, Any],
    context: Agent3RunContext,
) -> dict[str, Any]:
    """Validate the ReAct run and expose it for a later approved delivery.

    This function never repairs missing ReAct actions. It only attaches the
    ground-truth tool state and reports validation failures.
    """

    match_result = context.match_result or {}
    ranked_jobs = list(match_result.get("ranked_jobs", []))
    agent_result["ranked_jobs"] = ranked_jobs
    agent_result["tracked_applications"] = list(
        match_result.get("tracked_applications", [])
    )
    agent_result["candidate_id"] = match_result.get("candidate_id", "")
    agent_result["match_result"] = match_result
    agent_result["workflow_type"] = "agent3"
    agent_result["skill_gap_analysis"] = context.skill_gap_analysis
    agent_result["react_trace"] = _react_trace(agent_result)
    if match_result.get("persistence_warning"):
        agent_result["persistence_warning"] = match_result["persistence_warning"]

    called = [step["tool"] for step in agent_result["react_trace"]]
    required = ["search_linkedin_jobs", "evaluate_linkedin_results"]
    if ranked_jobs:
        required.extend(["analyze_skill_gaps", "write_cover_letter_for_job"])
    missing = [name for name in required if name not in called]

    order_error = False
    if not missing:
        positions = [called.index(name) for name in required]
        order_error = positions != sorted(positions)

    top_job = ranked_jobs[0] if ranked_jobs else None
    cover_job = cover_letter_tool._last_cover_letter_job
    cover_matches_top = bool(
        top_job
        and cover_job
        and str(cover_job.get("url") or "") == str(top_job.get("url") or "")
        and cover_letter_tool._last_cover_letter
    )
    if (
        ranked_jobs
        and not cover_matches_top
        and "write_cover_letter_for_job" not in missing
    ):
        missing.append("write_cover_letter_for_job_for_top_ranked_job")

    validation_ok = not missing and not order_error
    agent_result["react_validation"] = {
        "status": "passed" if validation_ok else "failed",
        "called_tools": called,
        "missing_tools": missing,
        "order_valid": not order_error,
    }
    agent_result["cover_letter"] = (
        cover_letter_tool._last_cover_letter if cover_matches_top else None
    )
    if cover_matches_top and top_job:
        agent_result["cover_letter_job"] = {
            "job_title": top_job.get("job_title", ""),
            "company": top_job.get("company", ""),
            "url": top_job.get("url", ""),
            "final_score": top_job.get("final_score", 0.0),
        }

    if not validation_ok:
        agent_result["status"] = "incomplete"
        agent_result["delivery"] = {
            "status": "skipped",
            "error": "The ReAct workflow did not complete its required actions.",
        }
        return agent_result

    if not ranked_jobs:
        agent_result["status"] = "completed"
        agent_result["delivery"] = {
            "status": "skipped",
            "error": "No ranked LinkedIn job was available.",
        }
        return agent_result

    agent_result["status"] = "awaiting_delivery"
    agent_result["delivery"] = {"status": "awaiting_choice"}
    return agent_result


def _invoke_agent3(
    cv_info: Any,
    results_count: int,
    location: str,
    verbose: bool,
    query: str = "",
    search_pool_size: int | None = None,
) -> tuple[dict[str, Any], Agent3RunContext]:
    context = Agent3RunContext(
        cv_info=cv_info,
        location=location,
        results_count=results_count,
        search_pool_size=search_pool_size,
    )
    executor = build_agent3_executor(context, verbose=verbose)
    question = _profile_question(
        cv_info,
        context.results_count,
        context.location,
        query=query,
    )
    try:
        result = executor.invoke({"input": question})
    except Exception as exc:
        result = {
            "output": f"Agent 3 stopped before completing the ReAct loop: {exc}",
            "intermediate_steps": [],
            "orchestration_error": f"{type(exc).__name__}: {exc}",
        }
    return result, context


def run_agent3_full_auto(
    cv_info: Any,
    results_count: int = 3,
    location: str = "",
    delivery_channel: str | None = None,
    verbose: bool = True,
    query: str = "",
    search_pool_size: int | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Run pure ReAct work, then deliver only when a channel is supplied."""

    channel = _normalize_delivery_channel(delivery_channel)
    _reset_agent3_state(cv_info)
    result, context = _invoke_agent3(
        cv_info,
        results_count,
        location,
        verbose,
        query=query,
        search_pool_size=search_pool_size,
    )
    result = _finalize_agent3_result(result, context)
    result["workflow_id"] = workflow_id or str(uuid4())
    result["cv_info"] = cv_info
    if channel is not None and result.get("status") == "awaiting_delivery":
        result = deliver_agent3_result(result, channel)
    return result


def run_agent3_full_auto_from_pdf(
    pdf_source: Any,
    results_count: int = 3,
    location: str = "",
    use_cache: bool = True,
    delivery_channel: str | None = None,
    verbose: bool = True,
    query: str = "",
    search_pool_size: int | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Parse a CV PDF, run Agent 3, and optionally deliver approved results."""

    from next_chapter.parsing.agent2_document_extractor import (
        extract_cv_document_agent2,
    )
    from next_chapter.parsing.agent2_parser import extract_cv_info_agent2

    cv_document = extract_cv_document_agent2(pdf_source)
    cv_info = extract_cv_info_agent2(
        cv_document.text,
        layout_text=cv_document.layout_text,
        cache_identity=f"{cv_document.extraction_version}:{cv_document.content_hash}",
        use_cache=use_cache,
    )
    result = run_agent3_full_auto(
        cv_info,
        results_count=results_count,
        location=location,
        delivery_channel=delivery_channel,
        verbose=verbose,
        query=query,
        search_pool_size=search_pool_size,
        workflow_id=workflow_id,
    )
    result["cv_info"] = cv_info
    result["cv_extraction_backend"] = cv_document.backend
    result["cv_extraction_warnings"] = list(cv_document.warnings)
    return result


def deliver_agent3_result(
    result: dict[str, Any],
    delivery_channel: str,
    cover_letter: str | None = None,
) -> dict[str, Any]:
    """Deliver a reviewed Agent 3 result without rerunning its ReAct loop."""

    if result.get("workflow_type") != "agent3":
        raise ValueError("The supplied result is not an Agent 3 workflow.")
    delivery = result.get("delivery", {})
    if result.get("status") == "completed" and delivery.get("status") == "completed":
        raise ValueError("This Agent 3 result has already been delivered.")
    if result.get("status") not in {"awaiting_delivery", "delivery_failed"}:
        raise ValueError("This Agent 3 result is not ready for delivery.")

    channel = _normalize_delivery_channel(delivery_channel)
    if channel is None:
        raise ValueError("A delivery channel is required.")
    letter_source = (
        cover_letter if cover_letter is not None else result.get("cover_letter") or ""
    )
    letter = str(letter_source).strip()
    if not letter:
        raise ValueError("A reviewed cover letter is required for delivery.")
    ranked_jobs = list(result.get("ranked_jobs", []))
    if not ranked_jobs:
        raise ValueError("At least one ranked job is required for delivery.")

    from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo

    cv_info = Agent2CVInfo.model_validate(result.get("cv_info"))
    updated = deepcopy(result)
    updated["cover_letter"] = letter
    try:
        if channel == "gmail":
            if not cv_info.mail:
                raise ValueError("The reviewed CV profile has no email address.")
            from next_chapter.delivery.gmail import create_results_draft

            receipt = create_results_draft(
                cv_info,
                ranked_jobs,
                letter,
                to_email=cv_info.mail,
            )
            observation = f"Gmail draft created successfully (id: {receipt['id']})."
        else:
            from next_chapter.delivery.telegram import create_results_telegram

            receipt = create_results_telegram(
                cv_info,
                ranked_jobs,
                letter,
                delivery_id=str(result.get("workflow_id") or ""),
            )
            observation = (
                "Telegram results sent successfully "
                f"({len(receipt.get('messages', []))} message part(s))."
            )
    except Exception as exc:
        updated["status"] = "delivery_failed"
        updated["delivery"] = {
            "channel": channel,
            "status": "failed",
            "error": str(exc),
        }
        return updated

    updated["status"] = "completed"
    updated["delivery"] = {
        "channel": channel,
        "status": "completed",
        "observation": observation,
    }
    return updated


def _build_interview_executor(verbose: bool) -> AgentExecutor:
    tools = [generate_interview_preparation_pdf]
    prompt = PromptTemplate.from_template(INTERVIEW_REACT_PROMPT)
    agent = create_react_agent(
        llm=get_agent3_llm(temperature=0.0),
        tools=tools,
        prompt=prompt,
        output_parser=TolerantReActSingleInputOutputParser(),
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=4,
        return_intermediate_steps=True,
    )


def run_agent3_interview_preparation(
    application_id: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a separate one-tool ReAct request for a saved application."""

    resolved_id = str(application_id or "").strip()
    if not resolved_id:
        raise ValueError("application_id is required for interview preparation.")

    interview_tool.reset_interview_preparation_state()
    executor = _build_interview_executor(verbose)
    request = (
        "Prepare interview material for the saved application with ID "
        f"{resolved_id}."
    )
    try:
        result = executor.invoke({"input": request})
    except Exception as exc:
        result = {
            "output": f"Interview ReAct run failed: {exc}",
            "intermediate_steps": [],
            "orchestration_error": f"{type(exc).__name__}: {exc}",
        }

    result["react_trace"] = _react_trace(result)
    preparation = interview_tool.get_last_interview_preparation()
    called = [step["tool"] for step in result["react_trace"]]
    if "generate_interview_preparation_pdf" not in called or preparation is None:
        result["interview_preparation"] = None
        result["interview_error"] = (
            "The ReAct executor did not successfully generate the interview PDF."
        )
        return result
    result["interview_preparation"] = preparation
    return result


run_agent3 = run_agent3_full_auto_from_pdf
