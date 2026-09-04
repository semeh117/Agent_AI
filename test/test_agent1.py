"""Interactive end-to-end demonstration for Agent 1."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from agent.agent1 import run_agent1
from core.cv_parser import extract_cv_info, extract_text_from_pdf


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

    # Deterministic preprocessing is intentionally outside Agent 1's ReAct loop.
    raw_text = extract_text_from_pdf(str(selected_path))
    cv_info = extract_cv_info(raw_text)
    print(
        f"\nCandidate: {cv_info.full_name}, "
        f"{len(cv_info.skills)} skills, mail={cv_info.mail or 'not found'}"
    )
    print("\n=== FINAL AGENT 1 ANSWER ===")
    result = run_agent1(cv_info)
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
