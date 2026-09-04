"""Business services shared by UI and agent adapters."""

from services.application_tracker import (
    ApplicationRecord,
    ApplicationStatusEvent,
    CandidateRecord,
    TrackerSummary,
    add_application_note,
    get_application,
    get_application_status_history,
    get_tracker_summary,
    get_tracked_job_urls,
    list_applications,
    list_candidates,
    save_application,
    update_application_status,
)

__all__ = [
    "ApplicationRecord",
    "ApplicationStatusEvent",
    "CandidateRecord",
    "TrackerSummary",
    "add_application_note",
    "get_application",
    "get_application_status_history",
    "get_tracker_summary",
    "get_tracked_job_urls",
    "list_applications",
    "list_candidates",
    "save_application",
    "update_application_status",
]
