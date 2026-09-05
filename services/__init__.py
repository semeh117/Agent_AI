"""Business services shared by UI and agent adapters."""

from services.application_tracker import (
    ApplicationRecord,
    ApplicationStatusEvent,
    CandidateRecord,
    TrackerSummary,
    add_application_note,
    get_application,
    get_application_status_history,
    get_candidate_profile,
    get_tracker_summary,
    get_tracked_job_urls,
    list_applications,
    list_candidates,
    save_application,
    update_application_status,
)
from services.interview_preparation import (
    InterviewPreparationContent,
    InterviewPreparationRecord,
    generate_interview_preparation,
    list_interview_preparations,
    render_interview_preparation_pdf,
)

__all__ = [
    "ApplicationRecord",
    "ApplicationStatusEvent",
    "CandidateRecord",
    "TrackerSummary",
    "add_application_note",
    "get_application",
    "get_application_status_history",
    "get_candidate_profile",
    "get_tracker_summary",
    "get_tracked_job_urls",
    "list_applications",
    "list_candidates",
    "save_application",
    "update_application_status",
    "InterviewPreparationContent",
    "InterviewPreparationRecord",
    "generate_interview_preparation",
    "list_interview_preparations",
    "render_interview_preparation_pdf",
]
