# test/test_agent1.py
"""
test_agent1.py
--------------
Tests agent1_deterministic.py's full pipeline end-to-end: search +
evaluate (agent-driven) -> pick top job, write cover letter, build full
ranked list, create Gmail draft (all deterministic Python, no LLM
tool-calling past the ranking stage).

Prints every stage's output separately, rather than just the agent's
Final Answer text, so you can see exactly what happened at each step —
including whether the draft actually succeeded, which is the part
agent2's Final Answer proved unreliable at self-reporting.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from core.cv_parser import extract_text_from_pdf, extract_cv_info
from agent.agent1 import run_agent1_full_pipeline

CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))
print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")
choice = input("\nWhich CV? (enter number): ").strip()
selected_path = pdf_files[int(choice) - 1]

results_count_input = input("\nHow many jobs to search for? (default 3): ").strip()
results_count = int(results_count_input) if results_count_input else 3

# Deterministic preprocessing — NOT part of the agent
print(f"\nExtracting CV info from '{selected_path.name}'...")
raw_text = extract_text_from_pdf(str(selected_path))
cv_info = extract_cv_info(raw_text)
print(f"  -> {cv_info.full_name}, {len(cv_info.skills)} skills, "
      f"{cv_info.experience_years} yrs exp, {cv_info.highest_education_level}")
print(f"  -> Email on file: {cv_info.mail or '(none)'}\n")

if not cv_info.mail:
    print("  [WARN] No email extracted from this CV — the draft step will "
          "fail with a clear error further down. That's expected, not a bug.")

print("=" * 70)
print(f"RUNNING AGENT1 FULL PIPELINE ({results_count} jobs)")
print("=" * 70)

result = run_agent1_full_pipeline(cv_info, results_count=results_count)

# --- 1. Agent's own Final Answer (search + evaluate + rank only) ---
print("\n" + "=" * 70)
print("AGENT'S FINAL ANSWER (search + evaluate + rank)")
print("=" * 70)
print(result.get("output", "(no output field returned)"))

# --- 2. Cover letter outcome ---
print("\n" + "=" * 70)
print("COVER LETTER")
print("=" * 70)
if result.get("cover_letter"):
    top_job = result["cover_letter_job"]
    print(f"Written for: {top_job['job_title']} @ {top_job['company']} "
          f"({top_job['score_percent']}% match)\n")
    print(result["cover_letter"])
else:
    print(f"  [FAILED] {result.get('draft_error', 'No cover letter was generated.')}")

# --- 3. Draft outcome (the part that matters most to verify) ---
print("\n" + "=" * 70)
print("GMAIL DRAFT")
print("=" * 70)
if "draft" in result:
    print(f"  [SUCCESS] Draft created (id: {result['draft']['id']}). "
          f"Check Gmail > Drafts to review.")
elif "draft_error" in result:
    print(f"  [FAILED] {result['draft_error']}")
else:
    print("  [UNKNOWN] Neither 'draft' nor 'draft_error' present in result — "
          "check run_agent1_full_pipeline() for a gap in error handling.")

# --- 4. Sanity check: how many jobs actually got evaluated ---
from agent.tools.job_evaluator import _all_evaluations
print("\n" + "=" * 70)
print(f"ALL EVALUATIONS THIS RUN ({len(_all_evaluations)} jobs)")
print("=" * 70)
for e in sorted(_all_evaluations, key=lambda r: r["score_percent"], reverse=True):
    print(f"  {e['job_title']} @ {e['company']} — {e['score_percent']}%")