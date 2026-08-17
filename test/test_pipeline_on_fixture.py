"""
test_pipeline_on_fixture.py
--------------

This deliberately does NOT call extract_cv_info() or search_real_jobs()
— both of those are LLM/API calls whose output could vary run to run,
which would make it impossible to tell whether a change in output came
from your code, or from the input data shifting underneath you.

What's still live here on purpose: extract_job_requirements() and
calculate_compatibility() ARE real, un-mocked calls — that's the whole
point. This script exists to let you judge whether THOSE two are doing
a good job, with input held constant so any difference between two runs
is attributable to the model / your code, not the data.
   """

import argparse
import json

from dev.cv_fixture import load_cv_fixture
from search.job_search_fixture import search_jobs_from_fixture
from core.job_parser import extract_job_requirements
from core.matcher import calculate_compatibility


def run_once(cv_info, jobs):
    results = []
    for job in jobs:
        job_req = extract_job_requirements(job["title"], job["description"])
        print(f" [debug] {job['title']}: {len(job_req.required_skills)} skills, "
            f"exp={job_req.required_experience_years}, edu={job_req.required_education_level}")
        match = calculate_compatibility(cv_info, job_req)
        results.append({
            "job_title": job["title"],
            "company": job["company"],
            "score_percent": match["score_percent"],
            "required_skills_extracted": job_req.required_skills,
            "matching_skills": [m["job_skill"] for m in match["skills"]["matching"]],
            "missing_skills": match["skills"]["missing"],
        })
    results.sort(key=lambda r: r["score_percent"], reverse=True)
    return results


def print_results(results, label=""):
    print(f"\n{'=' * 70}")
    print(f"RESULTS{f' — {label}' if label else ''}")
    print("=" * 70)
    for r in results:
        print(f"\n{r['job_title']} @ {r['company']} — {r['score_percent']}% match")
        print(f"  Extracted requirements ({len(r['required_skills_extracted'])}): "
              f"{', '.join(r['required_skills_extracted'])}")
        print(f"  Matching ({len(r['matching_skills'])}): {', '.join(r['matching_skills']) or '-'}")
        print(f"  Missing  ({len(r['missing_skills'])}): {', '.join(r['missing_skills']) or '-'}")


def main():
    parser = argparse.ArgumentParser(
        description="Test job_parser.py + matcher.py quality against frozen CV + job fixtures."
    )
    parser.add_argument("--runs", type=int, default=1,
                         help="How many times to run (use 2+ to check run-to-run consistency)")
    parser.add_argument("--results-count", type=int, default=5,
                         help="How many jobs from the fixture to evaluate")
    args = parser.parse_args()

    cv_info = load_cv_fixture()
    jobs_json = search_jobs_from_fixture(results_count=args.results_count)
    jobs = json.loads(jobs_json)

    if isinstance(jobs, dict) and "error" in jobs:
        print(f"[ERROR] {jobs['error']}")
        return

    all_runs = []
    for i in range(args.runs):
        results = run_once(cv_info, jobs)
        print_results(results, label=f"run {i + 1}/{args.runs}")
        all_runs.append(results)

    if args.runs > 1:
        print(f"\n{'=' * 70}")
        print("CONSISTENCY CHECK ACROSS RUNS")
        print("=" * 70)
        baseline = {r["job_title"]: r["score_percent"] for r in all_runs[0]}
        stable = True
        for run_idx, run in enumerate(all_runs[1:], start=2):
            for r in run:
                base_score = baseline.get(r["job_title"])
                if base_score != r["score_percent"]:
                    stable = False
                    print(f"  [DIFF] '{r['job_title']}': run 1 = {base_score}%, "
                          f"run {run_idx} = {r['score_percent']}%")
        if stable:
            print("  All scores identical across runs — model output is stable "
                  "for this frozen input.")
        else:
            print("  Scores differ across runs on IDENTICAL input — this is "
                  "genuine model non-determinism (e.g. temperature/sampling), "
                  "not a data-reproducibility issue.")


if __name__ == "__main__":
    main()