"""Interactive end-to-end test for the preserved LinkedIn Agent 3."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model responses can contain Unicode punctuation that Windows' legacy
# console encoding cannot print. This affects display only, not agent output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from agent.agent3 import run_agent3_full_auto_from_pdf


CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))

if not pdf_files:
    raise RuntimeError(f"No CV PDF files were found in '{CV_FOLDER}'.")

print("Available CVs:")
for index, path in enumerate(pdf_files, start=1):
    print(f"  {index}. {path.name}")

choice = input("\nWhich CV? (enter number): ").strip()
selected_index = int(choice) - 1
if selected_index < 0 or selected_index >= len(pdf_files):
    raise ValueError("The selected CV number is out of range.")

selected_path = pdf_files[selected_index]
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
    f"{cv_info.experience_years} years of experience\n"
)
print(result["output"])
