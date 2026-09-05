"""Agent 2's on-demand interview-preparation LangGraph.

This graph is deliberately separate from the job-search graph in
``agent/agent2.py``. Interview packs cost thousands of LLM tokens, so they are
generated only when a user selects one saved application and asks for it (a
Streamlit button later). The graph owns control flow only; loading, prompt
construction, Groq generation, PDF rendering, and persistence are the shared
functions in ``services/interview_preparation.py`` that Agent 3's tool also
uses.

    START -> load_application -> validate_application
          -> generate_interview_content -> render_and_persist_pdf
          -> finalize -> END
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from services.application_tracker import (
    ApplicationRecord,
    get_application,
    get_candidate_profile,
)
from services.interview_preparation import (
    InterviewPreparationContent,
    generate_interview_content,
    get_interview_model_info,
    persist_interview_preparation,
    render_interview_preparation_pdf,
)
from storage.agent2_checkpointer import Agent2SqliteSaver

__all__ = [
    "Agent2InterviewState",
    "run_agent2_interview_preparation",
]


class Agent2InterviewState(TypedDict, total=False):
    """Serializable state shared by every interview graph node."""

    workflow_id: str
    application_id: str
    # Runtime options. Empty strings mean "use the configured defaults".
    database_path: str
    output_directory: str
    application: dict[str, Any]
    candidate_profile: dict[str, Any]
    interview_content: dict[str, Any]
    preparation_id: str
    pdf_path: str
    provider: str
    model: str
    created_at: str
    status: str
    completed_steps: list[str]
    warnings: list[str]
    error: Optional[str]
    output: str


# The interview LLM cannot be serialized into a checkpoint, so a test double
# is injected through this module-level hook instead of graph state. ``None``
# means "build the configured Groq model" inside the shared service.
_llm_override: Any = None


def _steps(state: Agent2InterviewState, step: str) -> list[str]:
    return [*state.get("completed_steps", []), step]


def _failure(
    state: Agent2InterviewState,
    step: str,
    exc: Exception | str,
) -> Agent2InterviewState:
    message = str(exc).strip() or type(exc).__name__
    return {
        "status": "failed",
        "error": f"{step} failed: {message}",
        "completed_steps": _steps(state, f"{step}_failed"),
    }


def _database_path(state: Agent2InterviewState) -> Optional[str]:
    return str(state.get("database_path") or "").strip() or None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _load_application_node(state: Agent2InterviewState) -> Agent2InterviewState:
    """Load the saved application and stored CV profile through the tracker."""

    try:
        application_id = str(state.get("application_id") or "").strip()
        if not application_id:
            raise ValueError("application_id is required.")
        database_path = _database_path(state)
        application = get_application(application_id, database_path=database_path)
        profile = get_candidate_profile(
            application.candidate_id,
            database_path=database_path,
        )
        return {
            "application": application.model_dump(mode="json"),
            "candidate_profile": profile,
            "status": "application_loaded",
            "error": None,
            "completed_steps": _steps(state, "application_loaded"),
        }
    except Exception as exc:
        return _failure(state, "load_application", exc)


def _validate_application_node(
    state: Agent2InterviewState,
) -> Agent2InterviewState:
    """Refuse to spend interview tokens on incomplete application data."""

    application = state.get("application") or {}
    profile = state.get("candidate_profile") or {}
    problems: list[str] = []
    if not application:
        problems.append("the application record is missing")
    if not profile:
        problems.append("the candidate profile is missing")
    if not str(application.get("job_title") or "").strip():
        problems.append("the job title is missing")
    if not str(application.get("company") or "").strip():
        problems.append("the company is missing")
    if not str(application.get("description") or "").strip():
        problems.append("the job description is missing")
    details = application.get("match_details")
    if not isinstance(details, dict) or not (
        "matching" in details or "missing" in details
    ):
        problems.append("the matching details are missing")

    if problems:
        return _failure(
            state,
            "validate_application",
            "Interview preparation needs complete application data: "
            + "; ".join(problems)
            + ".",
        )
    return {
        "status": "application_validated",
        "error": None,
        "completed_steps": _steps(state, "application_validated"),
    }


def _generate_content_node(state: Agent2InterviewState) -> Agent2InterviewState:
    """Call the shared structured Groq generation exactly once."""

    try:
        application = ApplicationRecord.model_validate(state["application"])
        content = generate_interview_content(
            application,
            state["candidate_profile"],
            llm=_llm_override,
        )
        model_info = get_interview_model_info()
        return {
            "interview_content": content.model_dump(mode="json"),
            "provider": model_info["provider"],
            "model": model_info["model"],
            "status": "content_generated",
            "error": None,
            "completed_steps": _steps(state, "interview_content_generated"),
        }
    except Exception as exc:
        return _failure(state, "generate_interview_content", exc)


def _render_and_persist_node(state: Agent2InterviewState) -> Agent2InterviewState:
    """Render the PDF with the shared renderer and store the version row."""

    try:
        application = ApplicationRecord.model_validate(state["application"])
        content = InterviewPreparationContent.model_validate(
            state["interview_content"]
        )
        preparation_id = str(uuid4())
        output_directory = str(state.get("output_directory") or "").strip() or None
        pdf_path = render_interview_preparation_pdf(
            preparation_id=preparation_id,
            application=application,
            content=content,
            output_directory=output_directory,
        )
        record = persist_interview_preparation(
            preparation_id=preparation_id,
            application_id=application.application_id,
            content=content,
            pdf_path=pdf_path,
            provider=state.get("provider") or None,
            model=state.get("model") or None,
            database_path=_database_path(state),
        )
        return {
            "preparation_id": record.preparation_id,
            "pdf_path": str(Path(record.pdf_path).resolve()),
            "created_at": record.created_at,
            "status": "completed",
            "error": None,
            "completed_steps": _steps(state, "pdf_rendered_and_persisted"),
        }
    except Exception as exc:
        return _failure(state, "render_and_persist_pdf", exc)


def _final_output(state: Agent2InterviewState) -> str:
    application = state.get("application") or {}
    label = (
        f"{application.get('job_title', 'Unknown role')} @ "
        f"{application.get('company', 'Unknown company')}"
    )
    if state.get("error"):
        lines = [
            "Interview preparation failed.",
            f"Application: {label}" if application else
            f"Application ID: {state.get('application_id', '')}",
            f"Error: {state['error']}",
        ]
    else:
        lines = [
            "Interview preparation generated successfully.",
            f"Application: {label}",
            f"PDF: {state.get('pdf_path', '')}",
            f"Preparation ID: {state.get('preparation_id', '')}",
            f"Model: {state.get('provider', '')}/{state.get('model', '')}",
        ]
    for warning in state.get("warnings", []):
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _finalize_node(state: Agent2InterviewState) -> Agent2InterviewState:
    return {
        "output": _final_output(state),
        "completed_steps": _steps(state, "workflow_finalized"),
    }


def _route(state: Agent2InterviewState) -> str:
    return "finalize" if state.get("error") else "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_agent2_interview_graph(checkpointer=None):
    builder = StateGraph(Agent2InterviewState)
    builder.add_node("load_application", _load_application_node)
    builder.add_node("validate_application", _validate_application_node)
    builder.add_node("generate_interview_content", _generate_content_node)
    builder.add_node("render_and_persist_pdf", _render_and_persist_node)
    builder.add_node("finalize", _finalize_node)

    builder.add_edge(START, "load_application")
    builder.add_conditional_edges(
        "load_application",
        _route,
        {"continue": "validate_application", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "validate_application",
        _route,
        {"continue": "generate_interview_content", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "generate_interview_content",
        _route,
        {"continue": "render_and_persist_pdf", "finalize": "finalize"},
    )
    builder.add_edge("render_and_persist_pdf", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer or Agent2SqliteSaver())


@lru_cache(maxsize=1)
def _get_agent2_interview_graph():
    return _build_agent2_interview_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent2_interview_preparation(
    application_id: str,
    workflow_id: Optional[str] = None,
    *,
    database_path: Optional[str | Path] = None,
    output_directory: Optional[str | Path] = None,
    llm: Any = None,
) -> dict[str, Any]:
    """Generate one interview pack for a saved application through LangGraph.

    The candidate profile is read from SQLite, so no CV is parsed again.
    ``database_path``, ``output_directory`` and ``llm`` exist for tests and the
    UI; leaving them ``None`` uses the configured Groq model and paths.
    """

    global _llm_override

    resolved_application_id = str(application_id or "").strip()
    if not resolved_application_id:
        raise ValueError("application_id is required for interview preparation.")
    resolved_workflow_id = str(workflow_id or "").strip() or f"interview-{uuid4()}"

    state: Agent2InterviewState = {
        "workflow_id": resolved_workflow_id,
        "application_id": resolved_application_id,
        "database_path": str(database_path or ""),
        "output_directory": str(output_directory or ""),
        "completed_steps": [],
        "warnings": [],
        "error": None,
        "status": "started",
    }
    config = {"configurable": {"thread_id": resolved_workflow_id}}
    graph = (
        _build_agent2_interview_graph(Agent2SqliteSaver(database_path))
        if database_path
        else _get_agent2_interview_graph()
    )

    _llm_override = llm
    try:
        result = graph.invoke(state, config=config)
    finally:
        _llm_override = None

    public = dict(result)
    public["workflow_id"] = resolved_workflow_id
    return public
