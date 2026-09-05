"""Agent 3 tool: generate one interview-preparation PDF on explicit request.

Thin wrapper around ``services.interview_preparation`` (the same code Agent
2's interview graph uses). It reimplements no prompt, PDF, or SQL logic. It is
never part of the automatic search -> cover letter -> delivery workflow; the
agent must call it only when a user explicitly asks for interview preparation
for one tracked application.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from services.application_tracker import get_application
from services.interview_preparation import generate_interview_preparation


_last_interview_preparation: dict[str, Any] | None = None


def get_last_interview_preparation() -> dict[str, Any] | None:
    """Return the compact result of the latest successful tool call."""

    return _last_interview_preparation


def reset_interview_preparation_state() -> None:
    global _last_interview_preparation
    _last_interview_preparation = None


@tool
def generate_interview_preparation_pdf(application_id: str) -> str:
    """Generate an interview-preparation PDF for ONE saved application.

    Call this ONLY when the user explicitly asks for interview preparation and
    gives (or you already hold) the tracked ``application_id`` of a saved job
    recommendation. Never call it during the normal job-search, cover-letter,
    or delivery workflow: it spends thousands of LLM tokens per call.

    Input: the ``application_id`` string from the application tracker.

    Output: JSON with ``preparation_id``, ``job_title``, ``company``,
    ``pdf_path``, ``provider`` and ``model`` on success, or an ``error`` field.
    Only report that a PDF exists when this observation contains ``pdf_path``.
    """

    global _last_interview_preparation

    resolved_id = str(application_id or "").strip().strip("\"'")
    if not resolved_id:
        return json.dumps(
            {"error": "application_id is required to prepare an interview."},
            ensure_ascii=False,
        )

    try:
        # Resolve the application first so a wrong ID fails before any LLM
        # tokens are spent; the service loads it again internally, cheaply.
        application = get_application(resolved_id)
        record = generate_interview_preparation(application.application_id)
    except LookupError as exc:
        return json.dumps(
            {"error": str(exc), "application_id": resolved_id},
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "error": f"Interview preparation failed: {type(exc).__name__}: {exc}",
                "application_id": resolved_id,
            },
            ensure_ascii=False,
        )

    result = {
        "preparation_id": record.preparation_id,
        "application_id": record.application_id,
        "job_title": application.job_title,
        "company": application.company,
        "pdf_path": record.pdf_path,
        "provider": record.provider,
        "model": record.model,
        "created_at": record.created_at,
    }
    _last_interview_preparation = result
    return json.dumps(result, indent=2, ensure_ascii=False)
