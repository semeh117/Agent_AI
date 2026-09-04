"""Interactive end-to-end demonstration for Agent 3."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from agent.agent3 import run_agent3_full_auto_from_pdf


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
    selected_path = _select_cv()
    location = input("LinkedIn location (leave empty for any location): ").strip()

    print("\n=== FULL LINKEDIN AGENT 3 WORKFLOW ===")
    result = run_agent3_full_auto_from_pdf(
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
    print(f"CV extraction: {result['cv_extraction_backend']}\n")
    for warning in result.get("cv_extraction_warnings", []):
        print(f"CV extraction warning: {warning}")
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
