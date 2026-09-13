"""Interactive classic ReAct Agent 3 demo with optional delivery."""
import argparse
from pathlib import Path
from next_chapter.paths import PROJECT_ROOT
import next_chapter.agents.agent3 as agent3_workflow
from next_chapter.agents.agent3 import run_agent3_full_auto_from_pdf


def _select_cv() -> Path:
    cv_folder = PROJECT_ROOT / "cv"
    pdf_files = sorted(cv_folder.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No CV PDF files were found in '{cv_folder}'.")

    print("Available CVs:")
    for index, path in enumerate(pdf_files, start=1):
        print(f"  {index}. {path.name}")

    selected_index = int(input("\nWhich CV? (enter number): ").strip()) - 1
    if selected_index < 0 or selected_index >= len(pdf_files):
        raise ValueError("The selected CV number is out of range.")
    return pdf_files[selected_index]

def main() -> int:
    parser = argparse.ArgumentParser(description="Test Agent 3's classic ReAct workflow.")
    parser.add_argument(
        "--interview",
        metavar="APPLICATION_ID",
        help="Live: ask Agent 3 to prepare one tracked application for interview.",
    )
    args = parser.parse_args()
    if args.interview:
        result = agent3_workflow.run_agent3_interview_preparation(args.interview)
        print(result["output"])
        preparation = result.get("interview_preparation")
        if preparation:
            print(f"\nPDF: {preparation['pdf_path']}")
        return 0 if preparation else 1

    selected_path = _select_cv()
    location = input("LinkedIn location (leave empty for any location): ").strip()
    delivery = input(
        "Delivery after the ReAct run (gmail / telegram / leave empty): "
    ).strip().lower()
    if delivery not in {"", "gmail", "telegram"}:
        raise ValueError("Delivery must be 'gmail', 'telegram', or empty.")

    print("\n=== CLASSIC REACT LINKEDIN AGENT 3 WORKFLOW ===")
    result = run_agent3_full_auto_from_pdf(
        str(selected_path),
        results_count=3,
        location=location,
        delivery_channel=delivery or None,
    )
    cv_info = result["cv_info"]
    print(
        f"\nCandidate: {cv_info.full_name}, "
        f"{len(cv_info.skills)} skills, "
        f"{cv_info.experience_years} years of experience, "
        f"education={cv_info.highest_education_level or 'not found'}"
    )
    print(f"CV extraction: {result['cv_extraction_backend']}\n")
    for warning in result.get("cv_extraction_warnings", []):
        print(f"CV extraction warning: {warning}")
    print("ReAct action/observation trace:")
    for index, step in enumerate(result.get("react_trace", []), start=1):
        print(f"  {index}. {step['tool']} <- {step['input']}")
    print(result["output"])
    for row in result.get("tracked_applications", []):
        print(f"Tracked application {row['application_id']}: {row['url']}")
    if result.get("persistence_warning"):
        print(f"Warning: {result['persistence_warning']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
