# test/test_agent.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from core.cv_parser import extract_text_from_pdf, extract_cv_info
from agent.agent import run_agent_job_matching_full_auto

CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))
print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")
choice = input("\nWhich CV? (enter number): ").strip()
selected_path = pdf_files[int(choice) - 1]

# Deterministic preprocessing — NOT part of the agent
raw_text = extract_text_from_pdf(str(selected_path))
cv_info = extract_cv_info(raw_text)
print(f"\nCandidate: {cv_info.full_name}, {len(cv_info.skills)} skills\n")

result = run_agent_job_matching_full_auto(cv_info, results_count=3)

print("\n\n=== FINAL AGENT ANSWER ===")
result = run_agent_job_matching_full_auto(cv_info, results_count=3)
print(result["output"])


