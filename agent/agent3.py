"""Agent 3: complete CV-to-LinkedIn matching and delivery workflow.

The public full-auto entry point parses the CV, lets the orchestration model
build one LinkedIn query, then scrapes, parses, scores, and ranks the jobs.
The agent's native tool loop generates a cover letter for the true top score,
asks the user to choose Gmail or Telegram, and calls the selected delivery tool.
Ranked jobs are saved to the shared Agent 2 application tracker inside the
matching tool. Interview preparation is a separate, user-triggered request
(``run_agent3_interview_preparation``) that shares Agent 2's service code.
"""

import json

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import agent.tools.cover_letter as cover_letter_tool
import agent.tools.delivery_choice as delivery_choice
import agent.tools.job_evaluator as job_evaluator
from agent.tools.cover_letter import write_cover_letter
from agent.tools.delivery_choice import ask_user_delivery_channel
from agent.tools.gmail import send_results_draft
import agent.tools.interview_preparation as interview_tool
from agent.tools.interview_preparation import generate_interview_preparation_pdf
from agent.tools.linkedin_match_tool import (
    get_last_match_result,
    get_last_tracked_applications,
    match_linkedin_jobs_for_agent,
    set_candidate_profile,
    sync_linkedin_results_for_shared_tools,
)
from agent.tools.telegram_tool import send_results_telegram
from config import get_agent_llm


def _reset_agent3_completion_state(cv_info, delivery_channel: str | None = None) -> None:
    """Reset shared tool state before one isolated Agent 3 run."""

    job_evaluator.set_candidate_profile(cv_info)
    cover_letter_tool._last_cover_letter = None
    cover_letter_tool._last_cover_letter_job = None
    delivery_choice._last_delivery_channel = delivery_channel
    interview_tool.reset_interview_preparation_state()


TOOLS = [
    match_linkedin_jobs_for_agent,
    write_cover_letter,
    ask_user_delivery_channel,
    send_results_draft,
    send_results_telegram,
    generate_interview_preparation_pdf,
]


AGENT3_SYSTEM_PROMPT = """You are an expert career-matching and delivery assistant. Complete
the entire LinkedIn workflow through the provided tools. Do not end after ranking.

## Required workflow

1. Study the candidate profile carefully.
2. Build one concise LinkedIn query containing:
   - one realistic target job title supported by the CV;
   - up to three of the candidate's strongest relevant skills.
3. Call `match_linkedin_jobs_for_agent` exactly once.
4. Pass the plain query in the tool's `query` argument. Do not include the
   location, result count, explanations, markdown, or quotation marks.
5. Read the returned JSON observation. The tool has already scraped LinkedIn,
   parsed every usable job, calculated all scores, sorted the jobs, and saved
   them to the application tracker (`tracked_applications`). Do not call any
   tool to save them again.
6. Do not recalculate scores and do not change the returned ranking order.
7. If the matching observation contains an `error`, report it honestly and stop.
8. If at least one job was ranked, call `write_cover_letter` exactly once for
   the first job in the returned ranking. Pass that job's URL directly in the
   tool's `url` argument, for example `{{"url": "https://..."}}`. Do not use
   `evaluated_job_json`. The shared tool recovers the complete stored LinkedIn
   job and uses the dedicated writer.
9. After the cover letter succeeds, call `ask_user_delivery_channel` exactly
   once. Pass an empty string and wait for its Gmail or Telegram choice.
10. Call exactly one matching delivery tool:
    - Gmail: `send_results_draft`
    - Telegram: `send_results_telegram`
    Pass an empty string. Never call both delivery tools.
11. If any completion or delivery tool returns an error, report it honestly.
    Never claim that a letter, draft, or message exists unless its tool
    observation explicitly confirms success.

## Optional interview preparation (user-triggered only)

`generate_interview_preparation_pdf` is NOT part of the workflow above. Call it
only when the user's message explicitly asks for interview preparation for one
saved application and provides its `application_id`. Never call it during the
job-search, cover-letter, or delivery steps, and never call it for every ranked
job. Never claim an interview PDF exists unless its observation contains
`pdf_path`. When the user asks only for interview preparation, call that tool
exactly once and report its result; do not run the LinkedIn workflow.

## Scoring contract

The tool calculates:

final_score = skills_score * 0.5
            + experience_score * 0.3
            + education_score * 0.2

`skills_score` is the percentage of scored required-skill units satisfied.
Exact, reviewed alias, ESCO-concept, and thresholded embedding-cosine decisions
can satisfy each unit. It is not one global CV/job cosine value. Experience and
education are deterministic eligibility scores. Use only returned values.

ESCO ``candidate_mapped`` and ``required_mapped`` values describe coverage of
the ESCO vocabulary only. Never describe them as candidate-job matches. When
experience or education has a ``No ... requirement stated`` note, its 100 score
means no penalty was applied; call the requirement "not specified", never a
strong or verified match.

Never provide the final answer before completing all applicable tool calls and
reading every observation.

## Final answer requirements

For every ranked job, include:

1. Rank
2. Job title and company
3. Final score
4. Skills, experience, and education component scores
5. Job URL
6. The tracker `application_id` when `tracked_applications` provides one
7. A concise factual note about the strongest match or biggest constraint

Mention skipped or inconclusive jobs when the tool reports them, and mention a
`persistence_warning` if present. Confirm which top job received the cover
letter and whether Gmail or Telegram delivery succeeded. Keep the answer
concise and preserve descending `final_score` order."""


def build_agent3_prompt() -> ChatPromptTemplate:
    """Build and validate Agent 3's native tool-calling prompt."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", AGENT3_SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_agent3_executor(verbose: bool = True) -> AgentExecutor:
    """Create Agent 3's native tool-calling executor."""

    llm = get_agent_llm(temperature=0.0)
    prompt = build_agent3_prompt()
    agent = create_tool_calling_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
    )


def run_agent3_job_matching(
    cv_info,
    results_count: int = 3,
    location: str = "",
) -> dict:
    """Run Agent 3 with an already-parsed ``CVInfo`` object."""

    set_candidate_profile(
        cv_info,
        location=location,
        results_count=results_count,
    )
    executor = build_agent3_executor(verbose=True)

    profile_summary = (
        f"Most recent title: "
        f"{cv_info.job_titles[0] if cv_info.job_titles else 'N/A'}. "
        f"Other roles: {', '.join(cv_info.job_titles[1:5]) or 'N/A'}. "
        f"Key skills: {', '.join(cv_info.skills[:15])}. "
        f"Experience: {cv_info.experience_years} years. "
        f"Education: {cv_info.highest_education_level or 'N/A'}. "
        f"Requested location: {location or 'any location'}."
    )
    question = (
        f"Here is the candidate's profile:\n{profile_summary}\n\n"
        f"Find up to {results_count} LinkedIn jobs matching this profile and "
        "complete the full workflow: rank them, write one cover letter for the "
        "true top job, ask me to choose Gmail or Telegram, and deliver it."
    )
    return executor.invoke({"input": question})


def complete_agent3_delivery(
    agent_result: dict,
    cv_info,
    match_result: dict,
    delivery_channel: str | None = None,
) -> dict:
    """Verify the agent completed its tool workflow and repair missed calls.

    Normal execution has already generated and delivered inside the agent's
    native tool loop. This deterministic guard never duplicates a successful
    delivery; it only performs a tool call the model omitted.
    """

    ranked_jobs = sorted(
        match_result.get("ranked_jobs", []),
        key=lambda job: (bool(job.get("inconclusive")), -float(job["final_score"])),
    )
    agent_result["ranked_jobs"] = ranked_jobs
    # Tracker rows were written deterministically inside the matching tool.
    agent_result["tracked_applications"] = match_result.get(
        "tracked_applications", get_last_tracked_applications()
    )
    agent_result["candidate_id"] = match_result.get("candidate_id", "")
    if match_result.get("persistence_warning"):
        agent_result["persistence_warning"] = match_result["persistence_warning"]
    sync_linkedin_results_for_shared_tools(
        {**match_result, "ranked_jobs": ranked_jobs}
    )

    if not ranked_jobs:
        agent_result["cover_letter"] = None
        agent_result["delivery"] = {
            "status": "skipped",
            "error": "No ranked LinkedIn job was available.",
        }
        agent_result["output"] += (
            "\n\nNo cover letter or delivery was possible because no job was ranked."
        )
        return agent_result

    top_job = ranked_jobs[0]
    top_identity = (top_job.get("job_title"), top_job.get("company"), top_job.get("url"))
    cover_identity = None
    if cover_letter_tool._last_cover_letter_job:
        cover_identity = (
            cover_letter_tool._last_cover_letter_job.get("job_title"),
            cover_letter_tool._last_cover_letter_job.get("company"),
            cover_letter_tool._last_cover_letter_job.get("url"),
        )
    if cover_letter_tool._last_cover_letter is None or cover_identity != top_identity:
        cover_reference = {"url": top_job.get("url", "")}
        if not cover_reference["url"]:
            cover_reference = {
                "title": top_job.get("job_title", ""),
                "company": top_job.get("company", ""),
            }
        cover_observation = write_cover_letter.func(
            json.dumps(cover_reference, ensure_ascii=False)
        )
    else:
        cover_observation = "Cover letter was generated inside the agent tool loop."

    if cover_letter_tool._last_cover_letter is None:
        agent_result["cover_letter"] = None
        agent_result["delivery"] = {
            "status": "skipped",
            "error": cover_observation,
        }
        agent_result["output"] += (
            f"\n\nCover-letter generation failed for {top_job['job_title']} @ "
            f"{top_job['company']}: {cover_observation}"
        )
        return agent_result

    agent_result["cover_letter"] = cover_letter_tool._last_cover_letter
    agent_result["cover_letter_job"] = {
        "job_title": top_job["job_title"],
        "company": top_job["company"],
        "url": top_job.get("url", ""),
        "final_score": top_job["final_score"],
    }

    aliases = {"email": "gmail", "mail": "gmail", "tg": "telegram"}
    if delivery_channel is not None:
        requested_channel = delivery_channel.strip().lower()
        delivery_choice._last_delivery_channel = aliases.get(
            requested_channel, requested_channel
        )
    if delivery_choice._last_delivery_channel not in {"gmail", "telegram"}:
        channel_observation = ask_user_delivery_channel.func("")
        if delivery_choice._last_delivery_channel not in {"gmail", "telegram"}:
            raise ValueError(f"Invalid delivery choice returned: {channel_observation}")

    delivery_channel = delivery_choice._last_delivery_channel
    if delivery_channel not in {"gmail", "telegram"}:
        raise ValueError("delivery_channel must be 'gmail' or 'telegram'.")

    delivery_tool = (
        "send_results_draft"
        if delivery_channel == "gmail"
        else "send_results_telegram"
    )
    successful_observation = None
    for action, observation in agent_result.get("intermediate_steps", []):
        if action.tool != delivery_tool:
            continue
        observation_text = str(observation)
        if not observation_text.lstrip().lower().startswith("error:"):
            successful_observation = observation_text

    if successful_observation is None:
        if delivery_channel == "gmail":
            delivery_observation = send_results_draft.func("")
        else:
            delivery_observation = send_results_telegram.func("")
    else:
        delivery_observation = successful_observation

    delivery_failed = str(delivery_observation).lstrip().lower().startswith("error:")
    agent_result["delivery"] = {
        "channel": delivery_channel,
        "status": "failed" if delivery_failed else "completed",
        "observation": delivery_observation,
    }
    if delivery_failed:
        agent_result["delivery"]["error"] = delivery_observation
    if agent_result["delivery"]["status"] != "completed":
        agent_result["output"] += (
            f"\n\nCover letter generated for {top_job['job_title']} @ "
            f"{top_job['company']}, but {delivery_channel} delivery failed: "
            f"{delivery_observation}"
        )
        return agent_result

    confirmation = (
        "Gmail draft created successfully."
        if delivery_channel == "gmail"
        else "Telegram results sent successfully."
    )
    agent_result["output"] += (
        f"\n\nCover letter generated for {top_job['job_title']} @ "
        f"{top_job['company']}. {confirmation}"
    )
    return agent_result


def run_agent3_full_auto(
    cv_info,
    results_count: int = 3,
    location: str = "",
    delivery_channel: str | None = None,
) -> dict:
    """Run matching, cover generation, channel choice, and delivery."""

    aliases = {"email": "gmail", "mail": "gmail", "tg": "telegram"}
    if delivery_channel is not None:
        delivery_channel = aliases.get(
            delivery_channel.strip().lower(), delivery_channel.strip().lower()
        )
        if delivery_channel not in {"gmail", "telegram"}:
            raise ValueError("delivery_channel must be 'gmail' or 'telegram'.")
    _reset_agent3_completion_state(cv_info, delivery_channel)

    try:
        agent_result = run_agent3_job_matching(
            cv_info,
            results_count=results_count,
            location=location,
        )
    except Exception as exc:
        # A native tool-calling provider can reject one malformed tool call
        # after LinkedIn matching has already completed. Preserve the valid,
        # deterministic ranking and let complete_agent3_delivery repair the
        # remaining cover/delivery steps instead of discarding the whole run.
        match_result = get_last_match_result()
        if not match_result or not match_result.get("ranked_jobs"):
            raise
        error_text = f"{type(exc).__name__}: {exc}"
        agent_result = {
            "output": (
                "Agent 3's tool loop stopped after producing a valid ranking "
                f"({error_text}). The deterministic completion guard continued "
                "the cover-letter and delivery steps."
            ),
            "intermediate_steps": [],
            "orchestration_error": error_text,
        }
    else:
        match_result = get_last_match_result() or {"ranked_jobs": []}

    return complete_agent3_delivery(
        agent_result,
        cv_info,
        match_result,
        delivery_channel=delivery_channel,
    )


def run_agent3_full_auto_from_pdf(
    pdf_source,
    results_count: int = 3,
    location: str = "",
    use_cache: bool = True,
    delivery_channel: str | None = None,
) -> dict:
    """Run Agent 3's complete workflow starting from a CV PDF source."""

    from core.agent2_document_extractor import extract_cv_document_agent2
    from core.agent2_parser import extract_cv_info_agent2

    cv_document = extract_cv_document_agent2(pdf_source)
    cv_info = extract_cv_info_agent2(
        cv_document.pypdf_text,
        layout_text=cv_document.markdown,
        cache_identity=(
            f"{cv_document.extraction_version}:{cv_document.content_hash}"
        ),
        use_cache=use_cache,
    )
    result = run_agent3_full_auto(
        cv_info,
        results_count=results_count,
        location=location,
        delivery_channel=delivery_channel,
    )
    result["cv_info"] = cv_info
    result["cv_extraction_backend"] = cv_document.backend
    result["cv_extraction_warnings"] = list(cv_document.warnings)
    return result


def run_agent3_interview_preparation(
    application_id: str,
    verbose: bool = True,
) -> dict:
    """Ask the ReAct executor to prepare one saved application for interview.

    This is a separate, user-triggered request. It never runs the LinkedIn
    workflow and needs no CV: the candidate profile is already in SQLite. The
    tool result is attached deterministically so callers do not have to parse
    the model's prose to find the PDF path.
    """

    resolved_id = str(application_id or "").strip()
    if not resolved_id:
        raise ValueError("application_id is required for interview preparation.")

    interview_tool.reset_interview_preparation_state()
    executor = build_agent3_executor(verbose=verbose)
    request = (
        "The user explicitly requests interview preparation for the tracked "
        f"application with application_id \"{resolved_id}\". Call "
        "`generate_interview_preparation_pdf` exactly once with that ID, then "
        "report the job title, company, PDF path, and model from its "
        "observation, or the error it returned. Do not search LinkedIn, write "
        "a cover letter, or deliver anything."
    )
    agent_result = executor.invoke({"input": request})

    preparation = interview_tool.get_last_interview_preparation()
    if preparation is None:
        # The model skipped or mangled the call; run the tool deterministically
        # so the user still receives an honest result.
        observation = generate_interview_preparation_pdf.func(resolved_id)
        preparation = interview_tool.get_last_interview_preparation()
        if preparation is None:
            agent_result["interview_preparation"] = None
            agent_result["interview_error"] = json.loads(observation).get(
                "error", observation
            )
            return agent_result
        agent_result["output"] += (
            "\n\nInterview preparation completed by the deterministic guard: "
            f"{preparation['pdf_path']}"
        )
    agent_result["interview_preparation"] = preparation
    return agent_result


# The natural Agent 3 entry point now runs the complete workflow. Call
# run_agent3_job_matching explicitly when ranking-only behavior is desired.
run_agent3 = run_agent3_full_auto_from_pdf
