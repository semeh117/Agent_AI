# test/test_extraction_comparison.py
"""
test_extraction_comparison.py
--------------
Compares two LLMs specifically on the three extraction/matching stages
of the pipeline — NOT cover letter generation, NOT the agent's ReAct
loop. Scoped narrowly on purpose: "who's better at extraction" is a
different question from "who writes better prose" or "who follows a
ReAct format more reliably," and mixing them muddies the comparison.

Checks, per model:
  1. CV skill extraction: count + grounding (any hallucinated skills
     not actually present in the CV text? — the AWS-style bug).
  2. Job requirement extraction: count + grounding (same check, job side).
  3. Skill matching: does it correctly catch an EXACT overlap skill
     that's present in both CV and job text? (the Docker-style bug),
     checked across multiple runs for consistency.

Uses the frozen CV + job fixtures so both models see IDENTICAL input.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from core.cv_parser import extract_text_from_pdf, extract_cv_info
from core.job_parser import extract_job_requirements
from core.matcher import calculate_compatibility
from dev.cv_fixture import load_cv_fixture
from search.job_search_fixture import search_jobs_from_fixture
from config import _build_llm

# --- Define the two candidates for comparison here ---
# Each entry: (label, provider, model_string, api_key_env_var)
CANDIDATES = [
    ("Qwen 2.5 7B (local)", "ollama", "qwen2.5:7b-instruct"),
]


def build_candidate_llm(provider, model, max_tokens=4096):
    return _build_llm(provider, model, temperature=0.0, max_tokens=max_tokens)


def check_cv_extraction(cv_path: str, llm) -> dict:
    raw_text = extract_text_from_pdf(cv_path)
    info = extract_cv_info(raw_text, llm=llm)
    ungrounded = [s for s in info.skills if s.lower() not in raw_text.lower()]
    return {
        "skills_extracted": len(info.skills),
        "ungrounded_skills": ungrounded,
        "ungrounded_count": len(ungrounded),
    }


def check_job_extraction(job_title: str, job_description: str, llm) -> dict:
    job_req = extract_job_requirements(job_title, job_description, llm=llm)
    combined = (job_title + " " + job_description).lower()
    ungrounded = [s for s in job_req.required_skills if s.lower() not in combined]
    return {
        "skills_extracted": len(job_req.required_skills),
        "ungrounded_skills": ungrounded,
        "ungrounded_count": len(ungrounded),
        "required_experience_years": job_req.required_experience_years,
        "required_education_level": job_req.required_education_level,
    }


def check_matching_consistency(cv_info, job_req, llm, runs: int = 3) -> dict:
    """Checks whether an exact-overlap skill (present verbatim in both
    lists) gets correctly matched EVERY run, not just sometimes."""
    exact_overlap = [
        js for js in job_req.required_skills
        if any(js.lower() == cs.lower() for cs in cv_info.skills)
    ]
    if not exact_overlap:
        return {"note": "No exact-overlap skills between this CV and job fixture."}

    misses = []
    for run_idx in range(1, runs + 1):
        match = calculate_compatibility(cv_info, job_req)  # NOTE: matcher.py's
        # exact-match pre-pass (if applied) will make this trivially
        # consistent — this check is most meaningful BEFORE that fix,
        # or for skills that only match via the LLM's semantic step.
        matched_names = [m["job_skill"] for m in match["skills"]["matching"]]
        for skill in exact_overlap:
            if skill not in matched_names:
                misses.append((run_idx, skill))

    return {
        "exact_overlap_tested": exact_overlap,
        "runs": runs,
        "missed_matches": misses,
        "consistent": len(misses) == 0,
    }


def main():
    cv_info_fixture = json.loads(Path("fixtures/cv_info_fixture.json").read_text(encoding="utf-8"))
    cv_path = cv_info_fixture["source_cv_path"]
    cv_info_frozen = load_cv_fixture()

    jobs = json.loads(search_jobs_from_fixture(results_count=1))
    job = jobs[0]

    results = {}

    for label, provider, model, key_env in CANDIDATES:
        print(f"\n{'=' * 70}")
        print(f"TESTING: {label}  ({provider}/{model})")
        print("=" * 70)

        if not os.getenv(key_env):
            print(f"  [SKIP] {key_env} not set in .env")
            continue

        try:
            llm = build_candidate_llm(provider, model)
        except Exception as e:
            print(f"  [ERROR building LLM] {str(e)[:150]}")
            continue

        candidate_result = {}

        print("  Stage 1: CV extraction...")
        try:
            candidate_result["cv_extraction"] = check_cv_extraction(cv_path, llm)
        except Exception as e:
            candidate_result["cv_extraction"] = {"error": str(e)[:150]}

        print("  Stage 2: Job extraction...")
        try:
            candidate_result["job_extraction"] = check_job_extraction(job["title"], job["description"], llm)
        except Exception as e:
            candidate_result["job_extraction"] = {"error": str(e)[:150]}

        print("  Stage 3: Matching consistency...")
        try:
            job_req = extract_job_requirements(job["title"], job["description"], llm=llm)
            candidate_result["matching"] = check_matching_consistency(cv_info_frozen, job_req, llm, runs=3)
        except Exception as e:
            candidate_result["matching"] = {"error": str(e)[:150]}

        results[label] = candidate_result

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, r in results.items():
        print(f"\n--- {label} ---")
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()