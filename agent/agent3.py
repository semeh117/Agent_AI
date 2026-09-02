"""Agent 3: complete CV-to-LinkedIn-to-Gmail workflow.

The public full-auto entry point parses the CV, lets the orchestration model
build one LinkedIn query, then scrapes, parses, scores, and ranks the jobs.
Finally, deterministic Python selects the true top score, generates one cover
letter with the dedicated writer model, and creates a Gmail draft.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools.linkedin_match_tool import (
    get_last_match_result,
    match_linkedin_jobs_for_agent,
    set_candidate_profile,
)
from config import get_agent_llm


TOOLS = [match_linkedin_jobs_for_agent]


AGENT3_SYSTEM_PROMPT = """You are an expert career-matching assistant. Your task is to build a
relevant LinkedIn search query from the candidate's CV profile, call the
LinkedIn matching tool, and present its weighted job ranking accurately.

## Required workflow

1. Study the candidate profile carefully.
2. Build one concise LinkedIn query containing:
   - one realistic target job title supported by the CV;
   - up to three of the candidate's strongest relevant skills.
3. Call `match_linkedin_jobs_for_agent` exactly once.
4. Pass the plain query in the tool's `query` argument. Do not include the
   location, result count, explanations, markdown, or quotation marks.
5. Read the returned JSON observation. The tool has already scraped LinkedIn,
   parsed every usable job, calculated all scores, and sorted the jobs.
6. Do not recalculate scores and do not change the returned ranking order.
7. If the observation contains an `error`, report it honestly and stop.

## Scoring contract

The tool calculates:

final_score = skills_score * 0.5
            + experience_score * 0.3
            + education_score * 0.2

`skills_score` is embedding cosine similarity. Experience and education are
deterministic eligibility scores. Use only the values returned by the tool.

Never provide the final answer before calling the tool and reading its result.

## Final answer requirements

For every ranked job, include:

1. Rank
2. Job title and company
3. Final score
4. Skills, experience, and education component scores
5. Job URL
6. A concise factual note about the strongest match or biggest constraint

Mention skipped or inconclusive jobs when the tool reports them. Keep the
answer concise and preserve descending `final_score` order."""


def build_agent3_executor(verbose: bool = True) -> AgentExecutor:
    """Create Agent 3's native tool-calling executor."""

    llm = get_agent_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT3_SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
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
        max_iterations=5,
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
        "return the weighted ranked recommendation."
    )
    return executor.invoke({"input": question})


def complete_agent3_delivery(
    agent_result: dict,
    cv_info,
    match_result: dict,
    delivery_channel: str | None = None,
) -> dict:
    """Generate one letter for the true top job and deliver it reliably.

    This stage is deliberately deterministic. The agent model never receives
    the full job descriptions or cover-letter text, and Gmail/Telegram do not
    consume LLM tokens.
    """

    from pipeline.cover_letter import generate_cover_letter
    from pipeline.send_results_email import create_results_draft
    from pipeline.send_results_telegram import create_results_telegram

    ranked_jobs = sorted(
        match_result.get("ranked_jobs", []),
        key=lambda job: (bool(job.get("inconclusive")), -float(job["final_score"])),
    )
    agent_result["ranked_jobs"] = ranked_jobs

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
    try:
        cover_letter = generate_cover_letter(cv_info, top_job)
    except Exception as exc:
        agent_result["cover_letter"] = None
        agent_result["delivery"] = {
            "status": "skipped",
            "error": f"Cover-letter generation failed: {exc}",
        }
        agent_result["output"] += (
            f"\n\nCover-letter generation failed for {top_job['job_title']} @ "
            f"{top_job['company']}: {exc}"
        )
        return agent_result

    agent_result["cover_letter"] = cover_letter
    agent_result["cover_letter_job"] = {
        "job_title": top_job["job_title"],
        "company": top_job["company"],
        "url": top_job.get("url", ""),
        "final_score": top_job["final_score"],
    }

    if delivery_channel is None:
        from agent.tools.delivery_choice import ask_user_delivery_channel

        delivery_channel = ask_user_delivery_channel.func("").lower()
    else:
        delivery_channel = delivery_channel.strip().lower()

    aliases = {"email": "gmail", "mail": "gmail", "tg": "telegram"}
    delivery_channel = aliases.get(delivery_channel, delivery_channel)
    if delivery_channel not in {"gmail", "telegram"}:
        raise ValueError("delivery_channel must be 'gmail' or 'telegram'.")

    try:
        if delivery_channel == "gmail":
            if not cv_info.mail:
                raise ValueError("The parsed CV has no email address for the draft recipient.")
            delivery_resource = create_results_draft(
                cv_info,
                ranked_jobs,
                cover_letter,
                to_email=cv_info.mail,
            )
            confirmation = (
                f"Gmail draft created successfully (id: "
                f"{delivery_resource.get('id', 'unknown')})."
            )
        else:
            delivery_resource = create_results_telegram(
                cv_info,
                ranked_jobs,
                cover_letter,
            )
            confirmation = "Telegram results sent successfully."
    except Exception as exc:
        agent_result["delivery"] = {
            "channel": delivery_channel,
            "status": "failed",
            "error": str(exc),
        }
        agent_result["output"] += (
            f"\n\nCover letter generated for {top_job['job_title']} @ "
            f"{top_job['company']}, but {delivery_channel} delivery failed: {exc}"
        )
        return agent_result

    agent_result["delivery"] = {
        "channel": delivery_channel,
        "status": "completed",
        "resource": delivery_resource,
    }
    agent_result["output"] += (
        f"\n\nCover letter generated for {top_job['job_title']} @ "
        f"{top_job['company']}. {confirmation}"
    )
    return agent_result


def run_agent3_full_auto(
    cv_info,
    results_count: int = 3,
    location: str = "",
    delivery_channel: str = "gmail",
) -> dict:
    """Run matching, top-job cover generation, and Gmail draft creation."""

    agent_result = run_agent3_job_matching(
        cv_info,
        results_count=results_count,
        location=location,
    )
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
) -> dict:
    """Run Agent 3's complete workflow starting from a CV PDF source."""

    from core.cv_parser import extract_cv_info, extract_text_from_pdf

    cv_text = extract_text_from_pdf(pdf_source)
    cv_info = extract_cv_info(cv_text, use_cache=use_cache)
    result = run_agent3_full_auto(
        cv_info,
        results_count=results_count,
        location=location,
        delivery_channel="gmail",
    )
    result["cv_info"] = cv_info
    return result


# The natural Agent 3 entry point now runs the complete workflow. Call
# run_agent3_job_matching explicitly when ranking-only behavior is desired.
run_agent3 = run_agent3_full_auto_from_pdf
