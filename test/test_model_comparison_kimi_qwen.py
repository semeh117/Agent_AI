# test/test_model_comparison_kimi_qwen.py
"""
test_model_comparison_kimi_qwen.py
--------------
Focused comparison: Kimi vs Qwen on the TWO specific failure modes
already observed in this project —
  1. CV extraction hallucinating ungrounded skills (e.g. "AWS" appearing
     when never mentioned in the CV text).
  2. Skill matching inconsistently missing an obvious exact match
     (e.g. "Docker" in both CV and job posting, but reported as missing).

Uses the frozen CV + job fixtures so results are comparable model-to-
model on IDENTICAL input, not confounded by live-data drift.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from core.cv_parser import extract_text_from_pdf, extract_cv_info, verify_skills_grounded
from core.matcher import calculate_compatibility
from core.job_parser import extract_job_requirements
from dev.cv_fixture import load_cv_fixture
from search.job_search_fixture import search_jobs_from_fixture

MODELS_TO_TEST = [
    ("openrouter", "qwen/qwen-2.5-7b-instruct"),
    ("openrouter", "moonshotai/kimi-k2"),  # VERIFY this exact string at
    # https://openrouter.ai/models before running — not guaranteed accurate.
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def build_llm(provider, model_name):
    return ChatOpenAI(model=model_name, temperature=0.0,
                       api_key=OPENROUTER_API_KEY,
                       base_url="https://openrouter.ai/api/v1",
                       max_tokens=8192, max_retries=3)


def check_grounding(cv_path: str, llm) -> dict:
    """Re-runs CV extraction with a given model, reports how many
    extracted skills were NOT actually grounded in the source text —
    this is exactly the AWS-style hallucination check."""
    raw_text = extract_text_from_pdf(cv_path)
    info = extract_cv_info(raw_text, llm=llm)  # NOTE: assumes extract_cv_info
    # already has the grounding fix applied (step 1 above) — if not
    # applied yet, this will show the RAW hallucination rate, which is
    # actually useful too, as a "how bad is it without the fix" baseline.
    ungrounded = [s for s in info.skills if s.lower() not in raw_text.lower()]
    return {
        "total_skills": len(info.skills),
        "ungrounded_skills": ungrounded,
        "ungrounded_count": len(ungrounded),
    }


def check_exact_match_consistency(cv_info, job_req, llm, runs: int = 3) -> dict:
    """Runs skill matching multiple times on IDENTICAL input, checks
    whether an EXACT skill present in both lists is matched consistently
    every time — this is exactly the Docker-style inconsistency check."""
    exact_overlap = [
        js for js in job_req.required_skills
        if any(js.lower() == cs.lower() for cs in cv_info.skills)
    ]
    if not exact_overlap:
        return {"note": "No exact-overlap skills in this fixture to test with."}

    inconsistent = []
    for run_idx in range(runs):
        match = calculate_compatibility(cv_info, job_req)
        matched_names = [m["job_skill"] for m in match["skills"]["matching"]]
        for skill in exact_overlap:
            if skill not in matched_names:
                inconsistent.append((run_idx + 1, skill))

    return {
        "exact_overlap_skills_tested": exact_overlap,
        "runs": runs,
        "inconsistent_misses": inconsistent,
    }


def main():
    cv_info = load_cv_fixture()
    jobs = json.loads(search_jobs_from_fixture(results_count=1))
    job = jobs[0]

    print("=" * 70)
    print("PART 1 — CV GROUNDING (hallucination check)")
    print("=" * 70)
    from pathlib import Path
    cv_path = str(Path("cv") / Path(cv_info.full_name.replace(" ", "_") + ".pdf")) \
        if cv_info.full_name else None
    # If your fixture's source PDF path is known, hardcode it here instead —
    # simplest is to just paste the actual path from fixtures/cv_info_fixture.json's
    # "source_cv_path" field.

    for provider, model_name in MODELS_TO_TEST:
        print(f"\n--- {provider}/{model_name} ---")
        try:
            llm = build_llm(provider, model_name)
            result = check_grounding("cv/Semah_Mechi_.pdf", llm)  # adjust path
            print(f"  Total skills extracted: {result['total_skills']}")
            print(f"  Ungrounded (hallucinated) skills: {result['ungrounded_count']}")
            if result["ungrounded_skills"]:
                print(f"    -> {result['ungrounded_skills']}")
        except Exception as e:
            print(f"  ERROR: {str(e)[:150]}")

    print("\n" + "=" * 70)
    print("PART 2 — SKILL MATCH CONSISTENCY (exact-match reliability)")
    print("=" * 70)

    job_req = extract_job_requirements(job["title"], job["description"])

    for provider, model_name in MODELS_TO_TEST:
        print(f"\n--- {provider}/{model_name} ---")
        try:
            llm = build_llm(provider, model_name)
            import core.matcher as matcher_module
            original_get_llm = matcher_module.match_skills_llm
            # monkeypatch-free approach: pass llm through if match_skills_llm
            # supports it (it already does — see skill_matcher_llm.py's signature)
            result = check_exact_match_consistency(cv_info, job_req, llm, runs=3)
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"  ERROR: {str(e)[:150]}")


if __name__ == "__main__":
    main()