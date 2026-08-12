# test/test_kimi_cv_consistency.py
"""
test_kimi_cv_consistency.py
--------------
Same reproducibility pattern as test_pipeline_on_fixture.py, but
targeted specifically at CV EXTRACTION (not job matching), and
specifically for Kimi.

Runs extract_cv_info() against the SAME raw CV text multiple times and
checks:
  1. Does the skill COUNT stay stable across runs, or does it swing?
  2. Are there any ungrounded (hallucinated) skills in ANY run?
  3. Which skills appear in some runs but not others — i.e. exactly
     which ones are unstable, not just "how many."

Deliberately does NOT use load_cv_fixture() (that loads a frozen,
ALREADY-extracted CVInfo — no LLM call happens at all). This script
needs the RAW CV TEXT so extract_cv_info() actually runs live, every
time, against identical input.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from core.cv_parser import extract_text_from_pdf, extract_cv_info
from config import _build_llm

MODEL_LABEL = "Kimi (Moonshot)"
PROVIDER = "moonshot"
MODEL = "kimi-k2.6"  # confirm exact string against Moonshot's own model list
RUNS = 3


def main():
    # Reuse the same source CV the existing fixture was captured from,
    # so this is directly comparable to whatever numbers you already
    # have on record for Qwen.
    fixture = json.loads(Path("fixtures/cv_info_fixture.json").read_text(encoding="utf-8"))
    cv_path = fixture["source_cv_path"]

    print(f"Using CV: {cv_path}")
    raw_text = extract_text_from_pdf(cv_path)

    llm = _build_llm(PROVIDER, MODEL, temperature=0.0, max_tokens=4096)

    all_runs = []
    for run_idx in range(1, RUNS + 1):
        print(f"\nRun {run_idx}/{RUNS}...")
        info = extract_cv_info(raw_text, llm=llm)
        ungrounded = [s for s in info.skills if s.lower() not in raw_text.lower()]

        all_runs.append({
            "skills": set(s.lower() for s in info.skills),
            "skill_count": len(info.skills),
            "ungrounded": ungrounded,
        })

        print(f"  -> {len(info.skills)} skills, {len(ungrounded)} ungrounded")
        if ungrounded:
            print(f"     ungrounded: {ungrounded}")

    print("\n" + "=" * 70)
    print(f"CONSISTENCY CHECK — {MODEL_LABEL}, {RUNS} runs, IDENTICAL input")
    print("=" * 70)

    counts = [r["skill_count"] for r in all_runs]
    print(f"\nSkill counts across runs: {counts}")
    if len(set(counts)) == 1:
        print("  -> STABLE: identical count every run.")
    else:
        print(f"  -> UNSTABLE: count varies by {max(counts) - min(counts)} between runs.")

    total_ungrounded = sum(len(r["ungrounded"]) for r in all_runs)
    print(f"\nTotal ungrounded (hallucinated) skills across all runs: {total_ungrounded}")
    if total_ungrounded == 0:
        print("  -> No hallucinated skills in any run.")

    # Which skills appear in SOME runs but not ALL — the actual "flip-flopping" skills
    all_skill_sets = [r["skills"] for r in all_runs]
    always_present = set.intersection(*all_skill_sets)
    ever_present = set.union(*all_skill_sets)
    unstable_skills = ever_present - always_present

    print(f"\nSkills present in EVERY run: {len(always_present)}")
    print(f"Skills present in SOME but not ALL runs (unstable): {len(unstable_skills)}")
    if unstable_skills:
        print(f"  -> {sorted(unstable_skills)}")


if __name__ == "__main__":
    main()