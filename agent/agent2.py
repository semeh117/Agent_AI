"""Agent 2's explicit LangGraph workflow.

The graph owns control flow only. Document extraction, parsing, LinkedIn
search, cosine scoring, cover-letter generation, and delivery remain ordinary
project functions with clear inputs and outputs. This keeps matching
deterministic while giving the future Streamlit UI checkpointed progress and a
real pause/resume point before Gmail or Telegram delivery.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from config import get_agent_llm
from core.agent2_cv_parser import Agent2CVInfo
from pipeline.linkedin_cosine_pipeline import (
    build_linkedin_query,
    match_linkedin_jobs,
)
from storage.agent2_checkpointer import Agent2SqliteSaver

__all__ = [
    "Agent2State",
    "resume_agent2_workflow",
    "run_agent2",
    "run_agent2_full_auto",
    "run_agent2_full_auto_from_pdf",
]


# ---------------------------------------------------------------------------
# Graph state and validation models
# ---------------------------------------------------------------------------


class Agent2State(TypedDict, total=False):
    """Serializable state shared by every Agent 2 graph node."""

    workflow_id: str
    pdf_source: str
    cv_info: dict[str, Any]
    location: str
    results_count: int
    use_cache: bool
    query: str
    query_source: str
    match_result: dict[str, Any]
    ranked_jobs: list[dict[str, Any]]
    top_job: dict[str, Any]
    cover_letter: str
    cover_letter_job: dict[str, Any]
    delivery_channel: str
    delivery: dict[str, Any]
    candidate_id: str
    tracked_applications: list[dict[str, Any]]
    cv_extraction_backend: str
    cv_extraction_warnings: list[str]
    completed_steps: list[str]
    warnings: list[str]
    error: Optional[str]
    status: str
    output: str


class _LinkedInQuery(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=180,
        description=(
            "One plain LinkedIn query: a realistic job title followed by no "
            "more than three of the candidate's strongest relevant skills."
        ),
    )


_CHANNEL_ALIASES = {
    "gmail": "gmail",
    "email": "gmail",
    "mail": "gmail",
    "telegram": "telegram",
    "tg": "telegram",
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _as_cv_info(value: Any) -> Agent2CVInfo:
    if isinstance(value, Agent2CVInfo):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return Agent2CVInfo.model_validate(value)


def _steps(state: Agent2State, step: str) -> list[str]:
    return [*state.get("completed_steps", []), step]


def _warnings(state: Agent2State, warning: str) -> list[str]:
    return [*state.get("warnings", []), warning]


def _failure(state: Agent2State, step: str, exc: Exception | str) -> Agent2State:
    message = str(exc).strip() or type(exc).__name__
    return {
        "status": "failed",
        "error": f"{step} failed: {message}",
        "completed_steps": _steps(state, f"{step}_failed"),
    }


def _clean_query(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:linkedin\s+)?(?:search\s+)?query\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = " ".join(text.strip(" `\"'").split())
    return text[:180].strip()


# ---------------------------------------------------------------------------
# Workflow nodes
# ---------------------------------------------------------------------------


def _load_cv_node(state: Agent2State) -> Agent2State:
    """Load a supplied profile or parse a PDF through Agent 2's hybrid path."""

    try:
        if state.get("cv_info"):
            cv_info = _as_cv_info(state["cv_info"])
            return {
                "cv_info": cv_info.model_dump(),
                "cv_extraction_backend": "preparsed",
                "cv_extraction_warnings": [],
                "status": "cv_parsed",
                "error": None,
                "completed_steps": _steps(state, "cv_parsed"),
            }

        pdf_source = str(state.get("pdf_source") or "").strip()
        if not pdf_source:
            raise ValueError("A CV PDF or parsed CV profile is required.")

        from core.agent2_document_extractor import extract_cv_document_agent2
        from core.agent2_parser import extract_cv_info_agent2

        document = extract_cv_document_agent2(
            pdf_source,
            use_cache=bool(state.get("use_cache", True)),
        )
        cv_info = extract_cv_info_agent2(
            document.pypdf_text,
            layout_text=document.markdown,
            cache_identity=f"{document.extraction_version}:{document.content_hash}",
            use_cache=bool(state.get("use_cache", True)),
        )
        return {
            "cv_info": cv_info.model_dump(),
            "cv_extraction_backend": document.backend,
            "cv_extraction_warnings": list(document.warnings),
            "status": "cv_parsed",
            "error": None,
            "completed_steps": _steps(state, "cv_parsed"),
        }
    except Exception as exc:
        return _failure(state, "cv_parsing", exc)


def _build_query_node(state: Agent2State) -> Agent2State:
    """Use the orchestration model once, with a deterministic fallback."""

    try:
        cv_info = _as_cv_info(state["cv_info"])
        fallback = build_linkedin_query(cv_info, max_skills=3)
        title = cv_info.job_titles[0] if cv_info.job_titles else "Not specified"
        headline = str(getattr(cv_info, "headline", "") or "").strip()
        if headline:
            title = f"{title} (CV headline / target role: {headline})"
        prompt = (
            "Build one concise LinkedIn job-search query for this candidate. "
            "Use one realistic target title supported by the CV and at most "
            "three strong relevant skills. Return only the structured query; "
            "do not add a location, explanation, Boolean operators, markdown, "
            "or quotation marks.\n\n"
            f"Most recent role: {title}\n"
            f"Other roles: {', '.join(cv_info.job_titles[1:5]) or 'None'}\n"
            f"Candidate skills: {', '.join(cv_info.skills[:25])}"
        )

        try:
            response = (
                get_agent_llm(temperature=0.0)
                .with_structured_output(_LinkedInQuery)
                .invoke(prompt)
            )
            if isinstance(response, _LinkedInQuery):
                query = response.query
            elif isinstance(response, dict):
                query = response.get("query", "")
            else:
                query = getattr(response, "query", "")
            query = _clean_query(query)
            if not query:
                raise ValueError("The orchestration model returned an empty query.")
            query_source = "agent_llm"
            warnings = state.get("warnings", [])
        except Exception as exc:
            query = fallback
            query_source = "deterministic_fallback"
            warnings = _warnings(
                state,
                "The orchestration model could not build the query; Agent 2 "
                f"used its deterministic fallback ({type(exc).__name__}: {exc}).",
            )

        return {
            "query": query,
            "query_source": query_source,
            "warnings": warnings,
            "status": "query_ready",
            "error": None,
            "completed_steps": _steps(state, "query_built"),
        }
    except Exception as exc:
        return _failure(state, "query_generation", exc)


def _match_jobs_node(state: Agent2State) -> Agent2State:
    """Run scraping, validation, parsing, cosine scoring, and ranking."""

    try:
        result = match_linkedin_jobs(
            cv_info=_as_cv_info(state["cv_info"]),
            query=state["query"],
            location=str(state.get("location") or ""),
            max_jobs=int(state.get("results_count", 3)),
            use_cache=bool(state.get("use_cache", True)),
            posted_within_hours=24,
            exclude_previously_tracked=True,
        )
        ranked_jobs = list(result.get("ranked_jobs", []))
        return {
            "match_result": result,
            "ranked_jobs": ranked_jobs,
            "top_job": ranked_jobs[0] if ranked_jobs else {},
            "status": "jobs_ranked" if ranked_jobs else "no_ranked_jobs",
            "error": None,
            "completed_steps": _steps(state, "jobs_ranked"),
        }
    except Exception as exc:
        return _failure(state, "linkedin_matching", exc)


def _persist_recommendations_node(state: Agent2State) -> Agent2State:
    """Store ranked recommendations for the future application dashboard."""

    try:
        from services.application_tracker import save_application

        candidate = _as_cv_info(state["cv_info"])
        candidate_id: Optional[str] = None
        tracked: list[dict[str, Any]] = []
        for job in state.get("ranked_jobs", []):
            record = save_application(
                candidate,
                job,
                status="discovered",
                candidate_id=candidate_id,
            )
            candidate_id = record.candidate_id
            tracked.append(
                {
                    "application_id": record.application_id,
                    "job_id": record.job_id,
                    "url": record.url,
                }
            )
        return {
            "candidate_id": candidate_id or "",
            "tracked_applications": tracked,
            "status": "recommendations_saved",
            "error": None,
            "completed_steps": _steps(state, "recommendations_saved"),
        }
    except Exception as exc:
        # Tracking must not discard otherwise valid recommendations.
        return {
            "tracked_applications": [],
            "warnings": _warnings(
                state,
                "Ranked jobs could not be saved to the application tracker "
                f"({type(exc).__name__}: {exc}).",
            ),
            "status": "recommendations_not_saved",
            "error": None,
            "completed_steps": _steps(state, "recommendations_persistence_failed"),
        }


def _cover_letter_node(state: Agent2State) -> Agent2State:
    """Generate exactly one letter for the deterministic top-ranked job."""

    try:
        from pipeline.cover_letter import generate_cover_letter

        top_job = state["top_job"]
        cover_letter = generate_cover_letter(_as_cv_info(state["cv_info"]), top_job)
        if not str(cover_letter or "").strip():
            raise ValueError("The cover-letter model returned empty content.")
        cover_letter = str(cover_letter).strip()
        warnings = state.get("warnings", [])
        if state.get("candidate_id"):
            try:
                from services.application_tracker import save_application

                save_application(
                    _as_cv_info(state["cv_info"]),
                    top_job,
                    status="discovered",
                    cover_letter=cover_letter,
                    candidate_id=state["candidate_id"],
                )
            except Exception as exc:
                warnings = _warnings(
                    state,
                    "The cover letter was generated but could not be stored "
                    f"in the application tracker ({type(exc).__name__}: {exc}).",
                )
        return {
            "cover_letter": cover_letter,
            "cover_letter_job": {
                "job_title": top_job.get("job_title", ""),
                "company": top_job.get("company", ""),
                "url": top_job.get("url", ""),
                "final_score": top_job.get("final_score", 0.0),
            },
            "status": "cover_letter_ready",
            "error": None,
            "warnings": warnings,
            "completed_steps": _steps(state, "cover_letter_generated"),
        }
    except Exception as exc:
        return _failure(state, "cover_letter_generation", exc)


def _delivery_choice_node(state: Agent2State) -> Agent2State:
    """Pause safely until the user chooses Gmail or Telegram."""

    supplied = str(state.get("delivery_channel") or "").strip().casefold()
    if supplied:
        choice = supplied
    else:
        top_job = state.get("top_job", {})
        choice = interrupt(
            {
                "type": "delivery_choice",
                "question": (
                    "How do you want the ranked results and top-job cover "
                    "letter delivered?"
                ),
                "options": ["gmail", "telegram"],
                "top_job": {
                    "job_title": top_job.get("job_title", ""),
                    "company": top_job.get("company", ""),
                    "final_score": top_job.get("final_score", 0.0),
                },
            }
        )

    channel = _CHANNEL_ALIASES.get(str(choice or "").strip().casefold())
    if channel is None:
        raise ValueError("Delivery choice must be Gmail or Telegram.")
    return {
        "delivery_channel": channel,
        "status": "delivery_selected",
        "error": None,
        "completed_steps": _steps(state, "delivery_selected"),
    }


def _delivery_node(state: Agent2State) -> Agent2State:
    """Execute exactly one approved external delivery action."""

    channel = state["delivery_channel"]
    cv_info = _as_cv_info(state["cv_info"])
    ranked_jobs = state.get("ranked_jobs", [])
    cover_letter = state.get("cover_letter", "")

    try:
        if channel == "gmail":
            if not cv_info.mail:
                raise ValueError(
                    "The parsed CV has no email address for the Gmail draft."
                )
            from pipeline.send_results_email import create_results_draft

            resource = create_results_draft(
                cv_info,
                ranked_jobs,
                cover_letter,
                to_email=cv_info.mail,
            )
            delivery = {
                "channel": "gmail",
                "status": "completed",
                "draft_id": resource.get("id", "unknown"),
            }
        elif channel == "telegram":
            from pipeline.send_results_telegram import create_results_telegram

            resource = create_results_telegram(cv_info, ranked_jobs, cover_letter)
            delivery = {
                "channel": "telegram",
                "status": "completed",
                "messages_sent": len(resource.get("messages", [])),
            }
        else:
            raise ValueError("Delivery channel must be Gmail or Telegram.")

        return {
            "delivery": delivery,
            "status": "completed",
            "error": None,
            "completed_steps": _steps(state, "results_delivered"),
        }
    except Exception as exc:
        return {
            "delivery": {
                "channel": channel,
                "status": "failed",
                "error": str(exc),
            },
            **_failure(state, "delivery", exc),
        }


def _final_output(state: Agent2State) -> str:
    ranked_jobs = state.get("ranked_jobs", [])
    lines: list[str] = []

    if ranked_jobs:
        lines.append("Agent 2 LinkedIn recommendations:")
        for index, job in enumerate(ranked_jobs, start=1):
            lines.extend(
                [
                    "",
                    f"{index}. {job.get('job_title', 'Unknown role')} @ "
                    f"{job.get('company', 'Unknown company')}",
                    f"   Final: {job.get('final_score', 0.0)}% | "
                    f"Skills: {job.get('skills_score', 0.0)}% | "
                    f"Experience: {job.get('experience_score', 0.0)}% | "
                    f"Education: {job.get('education_score', 0.0)}%",
                    f"   URL: {job.get('url', '')}",
                ]
            )
        skipped = state.get("match_result", {}).get("skipped_count", 0)
        if skipped:
            lines.extend(["", f"Skipped jobs: {skipped}"])
        tracked_count = len(state.get("tracked_applications", []))
        if tracked_count:
            lines.extend(
                ["", f"Saved {tracked_count} recommendation(s) to the application tracker."]
            )
    else:
        lines.append("Agent 2 produced no ranked LinkedIn jobs.")

    cover_job = state.get("cover_letter_job", {})
    if state.get("cover_letter") and cover_job:
        lines.extend(
            [
                "",
                "Cover letter generated for "
                f"{cover_job.get('job_title', '')} @ {cover_job.get('company', '')}.",
            ]
        )

    delivery = state.get("delivery", {})
    if delivery.get("status") == "completed":
        if delivery.get("channel") == "gmail":
            lines.append(
                "Gmail draft created successfully "
                f"(id: {delivery.get('draft_id', 'unknown')})."
            )
        else:
            lines.append(
                "Telegram delivery completed successfully "
                f"({delivery.get('messages_sent', 0)} message(s))."
            )
    elif delivery.get("status") == "failed":
        lines.append(
            f"{delivery.get('channel', 'Delivery').title()} delivery failed: "
            f"{delivery.get('error', 'unknown error')}"
        )

    if state.get("error") and delivery.get("status") != "failed":
        lines.extend(["", f"Workflow error: {state['error']}"])
    for warning in state.get("warnings", []):
        lines.extend(["", f"Warning: {warning}"])
    return "\n".join(lines).strip()


def _finalize_node(state: Agent2State) -> Agent2State:
    return {
        "output": _final_output(state),
        "completed_steps": _steps(state, "workflow_finalized"),
    }


def _route_standard(state: Agent2State) -> str:
    return "finalize" if state.get("error") else "continue"


def _route_after_matching(state: Agent2State) -> str:
    if state.get("error") or not state.get("ranked_jobs"):
        return "finalize"
    return "cover_letter"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_agent2_graph(checkpointer=None):
    builder = StateGraph(Agent2State)
    builder.add_node("load_cv", _load_cv_node)
    builder.add_node("build_query", _build_query_node)
    builder.add_node("match_jobs", _match_jobs_node)
    builder.add_node("persist_recommendations", _persist_recommendations_node)
    builder.add_node("generate_cover_letter", _cover_letter_node)
    builder.add_node("choose_delivery", _delivery_choice_node)
    builder.add_node("deliver", _delivery_node)
    builder.add_node("finalize", _finalize_node)

    builder.add_edge(START, "load_cv")
    builder.add_conditional_edges(
        "load_cv",
        _route_standard,
        {"continue": "build_query", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "build_query",
        _route_standard,
        {"continue": "match_jobs", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "match_jobs",
        _route_after_matching,
        {"cover_letter": "persist_recommendations", "finalize": "finalize"},
    )
    builder.add_edge("persist_recommendations", "generate_cover_letter")
    builder.add_conditional_edges(
        "generate_cover_letter",
        _route_standard,
        {"continue": "choose_delivery", "finalize": "finalize"},
    )
    builder.add_edge("choose_delivery", "deliver")
    builder.add_edge("deliver", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer or Agent2SqliteSaver())


@lru_cache(maxsize=1)
def _get_agent2_graph():
    return _build_agent2_graph()


# ---------------------------------------------------------------------------
# Public workflow runtime
# ---------------------------------------------------------------------------


def _normalize_channel(value: Optional[str]) -> str:
    if value is None:
        return ""
    channel = _CHANNEL_ALIASES.get(value.strip().casefold())
    if channel is None:
        raise ValueError("delivery_channel must be 'gmail' or 'telegram'.")
    return channel


def _interrupt_payload(
    result: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    interruptions = result.get("__interrupt__") or ()
    if interruptions:
        first = interruptions[0]
        value = getattr(first, "value", first)
        return value if isinstance(value, dict) else {"question": str(value)}

    # Some LangGraph versions keep interrupts only on the checkpoint task
    # instead of including ``__interrupt__`` in ``invoke()``'s state result.
    if config is not None:
        snapshot = _get_agent2_graph().get_state(config)
        for task in getattr(snapshot, "tasks", ()):
            task_interrupts = getattr(task, "interrupts", ())
            if not task_interrupts:
                continue
            first = task_interrupts[0]
            value = getattr(first, "value", first)
            return value if isinstance(value, dict) else {"question": str(value)}
    return None


def _ask_delivery_cli(payload: Optional[dict[str, Any]]) -> str:
    question = (payload or {}).get(
        "question",
        "How do you want the results and cover letter delivered?",
    )
    while True:
        raw = input(f"{question} (gmail / telegram): ").strip().casefold()
        channel = _CHANNEL_ALIASES.get(raw)
        if channel:
            return channel
        print(f"  '{raw}' is not valid. Type 'gmail' or 'telegram'.")


def _public_result(
    result: dict[str, Any],
    workflow_id: str,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    public = dict(result)
    public["workflow_id"] = workflow_id
    if public.get("cv_info") and isinstance(public["cv_info"], dict):
        public["cv_info"] = _as_cv_info(public["cv_info"])
    payload = _interrupt_payload(public, config=config)
    if payload is not None:
        public["status"] = "awaiting_delivery"
        public["interrupt"] = payload
    return public


def _invoke_workflow(
    *,
    cv_info: Any = None,
    pdf_source: str = "",
    results_count: int = 3,
    location: str = "",
    use_cache: bool = True,
    delivery_channel: Optional[str] = None,
    interactive_delivery: bool = True,
    workflow_id: Optional[str] = None,
) -> dict[str, Any]:
    if results_count <= 0:
        raise ValueError("results_count must be greater than zero.")
    if cv_info is None and not str(pdf_source or "").strip():
        raise ValueError("A CV PDF or parsed CV profile is required.")

    resolved_id = workflow_id or str(uuid4())
    state: Agent2State = {
        "workflow_id": resolved_id,
        "pdf_source": str(pdf_source or ""),
        "location": str(location or "").strip(),
        "results_count": min(int(results_count), 20),
        "use_cache": bool(use_cache),
        "delivery_channel": _normalize_channel(delivery_channel),
        "completed_steps": [],
        "warnings": [],
        "error": None,
        "status": "started",
    }
    if cv_info is not None:
        state["cv_info"] = _as_cv_info(cv_info).model_dump()

    config = {"configurable": {"thread_id": resolved_id}}
    result = _get_agent2_graph().invoke(state, config=config)
    payload = _interrupt_payload(result, config=config)
    if payload is not None and interactive_delivery:
        choice = _ask_delivery_cli(payload)
        result = _get_agent2_graph().invoke(
            Command(resume=choice),
            config=config,
        )
    return _public_result(result, resolved_id, config=config)


def run_agent2_full_auto(
    cv_info: Any,
    results_count: int = 3,
    location: str = "",
    delivery_channel: Optional[str] = None,
    *,
    interactive_delivery: bool = True,
    workflow_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run Agent 2 from an existing parsed CV profile."""

    return _invoke_workflow(
        cv_info=cv_info,
        results_count=results_count,
        location=location,
        delivery_channel=delivery_channel,
        interactive_delivery=interactive_delivery,
        workflow_id=workflow_id,
    )


def run_agent2_full_auto_from_pdf(
    pdf_source: Any,
    results_count: int = 3,
    location: str = "",
    use_cache: bool = True,
    delivery_channel: Optional[str] = None,
    *,
    interactive_delivery: bool = True,
    workflow_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run the complete LangGraph workflow from one CV PDF."""

    return _invoke_workflow(
        pdf_source=str(pdf_source),
        results_count=results_count,
        location=location,
        use_cache=use_cache,
        delivery_channel=delivery_channel,
        interactive_delivery=interactive_delivery,
        workflow_id=workflow_id,
    )


def resume_agent2_workflow(
    workflow_id: str,
    delivery_channel: str,
) -> dict[str, Any]:
    """Resume a Streamlit-paused workflow with the user's delivery choice."""

    resolved_id = str(workflow_id or "").strip()
    if not resolved_id:
        raise ValueError("workflow_id is required to resume Agent 2.")
    channel = _normalize_channel(delivery_channel)
    config = {"configurable": {"thread_id": resolved_id}}
    result = _get_agent2_graph().invoke(
        Command(resume=channel),
        config=config,
    )
    return _public_result(result, resolved_id, config=config)


# Natural Agent 2 entry point retained for existing callers.
run_agent2 = run_agent2_full_auto_from_pdf
