"""Interactive Agent 2 demo; may call providers and delivery services."""
import argparse
from next_chapter.paths import PROJECT_ROOT
import next_chapter.agents.agent2 as agent2_workflow
import next_chapter.agents.agent2_interview as agent2_interview


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Agent 2's LangGraph workflow.")
    parser.add_argument(
        "--interview",
        metavar="APPLICATION_ID",
        help="Live: run the on-demand interview LangGraph for one saved application.",
    )
    args = parser.parse_args()
    if args.interview:
        result = agent2_interview.run_agent2_interview_preparation(args.interview)
        print(f"Workflow ID: {result['workflow_id']}")
        print(f"Completed nodes: {' -> '.join(result.get('completed_steps', []))}\n")
        print(result["output"])
        return 0 if result.get("status") == "completed" else 1

    cv_folder = PROJECT_ROOT / "cv"
    pdf_files = sorted(cv_folder.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No CV PDF files were found in '{cv_folder}'.")

    print("Available CVs:")
    for index, path in enumerate(pdf_files, start=1):
        print(f"  {index}. {path.name}")

    choice = input("\nWhich CV? (enter number): ").strip()
    selected_index = int(choice) - 1
    if selected_index < 0 or selected_index >= len(pdf_files):
        raise ValueError("The selected CV number is out of range.")

    selected_path = pdf_files[selected_index]
    location = input("LinkedIn location (leave empty for any location): ").strip()

    print("\n=== FULL LINKEDIN AGENT 2 LANGGRAPH WORKFLOW ===")
    result = agent2_workflow.run_agent2_full_auto_from_pdf(
        str(selected_path),
        results_count=3,
        location=location,
    )
    cv_info = result["cv_info"]
    print(
        f"\nCandidate: {cv_info.full_name}, "
        f"{len(cv_info.skills)} skills, "
        f"{cv_info.experience_years} years of experience, "
        f"education={cv_info.highest_education_level or 'not found'}"
    )
    print(f"CV extraction: {result['cv_extraction_backend']}")
    print(f"Workflow ID: {result['workflow_id']}")
    print(f"Completed nodes: {' -> '.join(result.get('completed_steps', []))}\n")
    for warning in result.get("cv_extraction_warnings", []):
        print(f"CV extraction warning: {warning}")
    print(result["output"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
