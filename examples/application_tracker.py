"""Inspect saved applications or explicitly generate an interview PDF."""
import argparse
from next_chapter.services.application_tracker import get_tracker_summary, list_candidates, list_applications
from next_chapter.services.interview_preparation import generate_interview_preparation


def _show_live_tracker() -> int:
    summary = get_tracker_summary()
    print("AGENT 2 APPLICATION TRACKER")
    print(f"Total applications: {summary.total_applications}")
    print(f"Average score: {summary.average_final_score}")
    print(f"Highest score: {summary.highest_final_score}")
    print(f"Cover letters: {summary.applications_with_cover_letter}")
    active_counts = {
        status: count
        for status, count in summary.status_counts.items()
        if count
    }
    print(f"Statuses: {active_counts}")

    for candidate in list_candidates():
        print(
            f"\n{candidate.full_name or 'Unknown candidate'} "
            f"({candidate.application_count} application(s))"
        )
        for application in list_applications(candidate.candidate_id):
            print(
                f"  [{application.status}] {application.job_title} @ "
                f"{application.company} - {application.final_score}%"
            )
            print(f"    {application.url}")
    return 0

def _prepare_live_interview() -> int:
    applications = list_applications()
    if not applications:
        raise RuntimeError("No tracked applications exist. Run Agent 2 first.")

    print("SAVED APPLICATIONS")
    for index, application in enumerate(applications, start=1):
        print(
            f"  {index}. {application.job_title} @ {application.company} "
            f"({application.final_score}%)"
        )
    raw_choice = input("\nWhich application? (enter number): ").strip()
    selected_index = int(raw_choice) - 1
    if selected_index < 0 or selected_index >= len(applications):
        raise ValueError("The selected application number is out of range.")

    selected = applications[selected_index]
    print(
        f"\nGenerating interview preparation with Groq GPT-OSS 120B for "
        f"{selected.job_title} @ {selected.company}..."
    )
    preparation = generate_interview_preparation(selected.application_id)
    print(f"Preparation ID: {preparation.preparation_id}")
    print(f"Model: {preparation.provider}/{preparation.model}")
    print(f"PDF: {preparation.pdf_path}")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test or inspect Agent 2's application tracker."
    )
    parser.add_argument(
        "--show-live",
        action="store_true",
        help="Read and display the real Agent 2 tracker database.",
    )
    parser.add_argument(
        "--prepare-live",
        action="store_true",
        help="Choose a tracked application and generate its real interview PDF.",
    )
    args = parser.parse_args()
    if args.prepare_live:
        return _prepare_live_interview()
    if args.show_live:
        return _show_live_tracker()

    return _show_live_tracker()


if __name__ == "__main__":
    raise SystemExit(main())
