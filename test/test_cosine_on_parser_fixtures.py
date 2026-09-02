"""Run cosine matching on the frozen CV and LinkedIn parser fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import get_embeddings
from core.cosine_matcher import (
    DEFAULT_COSINE_THRESHOLD,
    calculate_compatibility_cosine,
)
from core.cv_parser import CVInfo
from core.job_parser import JobRequirements


CV_FIXTURE = PROJECT_ROOT / "fixtures" / "cv_parser_fixture.json"
JOB_FIXTURE = PROJECT_ROOT / "fixtures" / "linkedin_job_parser_fixture.json"


def _load_fixture_data() -> tuple[
    CVInfo, list[tuple[dict, JobRequirements]], list[dict]
]:
    cv_fixture = json.loads(CV_FIXTURE.read_text(encoding="utf-8"))
    job_fixture = json.loads(JOB_FIXTURE.read_text(encoding="utf-8"))
    cv_info = CVInfo.model_validate(cv_fixture["parsed_cv"])

    valid_jobs = []
    skipped_jobs = []
    for entry in job_fixture["jobs"]:
        if entry.get("parsed_requirements") is None:
            skipped_jobs.append(entry)
            continue
        valid_jobs.append(
            (
                entry["linkedin_job"],
                JobRequirements.model_validate(entry["parsed_requirements"]),
            )
        )
    return cv_info, valid_jobs, skipped_jobs


def run_cosine_fixture_test(
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    use_esco: bool = True,
    embeddings=None,
) -> tuple[list[dict], list[dict]]:
    cv_info, valid_jobs, skipped_jobs = _load_fixture_data()
    if not cv_info.skills:
        raise AssertionError("CV parser fixture has no skills; cosine cannot be tested.")
    if not valid_jobs:
        raise AssertionError("No successfully parsed LinkedIn jobs are available.")

    embeddings = embeddings or get_embeddings()
    original_get_embeddings = config.get_embeddings
    config.get_embeddings = lambda: embeddings
    try:
        first_results = []
        second_results = []
        for raw_job, parsed_job in valid_jobs:
            first = calculate_compatibility_cosine(
                cv_info,
                parsed_job,
                threshold=threshold,
                use_esco=use_esco,
            )
            second = calculate_compatibility_cosine(
                cv_info,
                parsed_job,
                threshold=threshold,
                use_esco=use_esco,
            )
            first_results.append({"job": raw_job, "result": first})
            second_results.append({"job": raw_job, "result": second})
    finally:
        config.get_embeddings = original_get_embeddings

    assert first_results == second_results
    for item, (_, parsed_job) in zip(first_results, valid_jobs):
        result = item["result"]
        skills = result["skills"]
        assert len(skills["matching"]) + len(skills["missing"]) == len(
            parsed_job.required_skills
        )
        expected_final = round(
            (
                skills["score"] * 0.5
                + result["experience"]["score"] * 0.3
                + result["education"]["score"] * 0.2
            )
            * 100,
            1,
        )
        assert result["score_percent"] == expected_final

    first_results.sort(key=lambda item: -item["result"]["score_percent"])
    return first_results, skipped_jobs


def test_cosine_is_stable_and_accounts_for_every_parsed_skill():
    run_cosine_fixture_test()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Agent 2 cosine matching on the fresh parser fixtures."
    )
    parser.add_argument(
        "--compare-esco",
        action="store_true",
        help="Also run the same fixtures without ESCO and print an A/B table.",
    )
    args = parser.parse_args()

    shared_embeddings = get_embeddings()
    results, skipped = run_cosine_fixture_test(embeddings=shared_embeddings)
    print("COSINE RESULTS ON PARSED LINKEDIN FIXTURES")
    for rank, item in enumerate(results, start=1):
        job = item["job"]
        result = item["result"]
        skills = result["skills"]
        print(f"\n{rank}. {job['title']} @ {job['company']}")
        print(f"   URL: {job['url']}")
        print(f"   Final: {result['score_percent']}%")
        print(f"   Skills: {round(skills['score'] * 100, 1)}%")
        normalization = skills["normalization"]
        print(
            "   ESCO: "
            f"{'enabled' if normalization['enabled'] else 'disabled'} "
            f"({normalization['candidate_mapped']}/"
            f"{normalization['candidate_total']} candidate skills, "
            f"{normalization['required_mapped']}/"
            f"{normalization['required_total']} requirements mapped)"
        )
        print(f"   Experience: {round(result['experience']['score'] * 100, 1)}%")
        print(f"   Education: {round(result['education']['score'] * 100, 1)}%")
        print("   Matching:")
        for match in skills["matching"]:
            print(f"     - {match['job_skill']} <- {match['matched_via']}")
        print("   Missing:")
        for missing in skills["missing"]:
            print(f"     - {missing}")

    if skipped:
        print("\nSKIPPED BEFORE COSINE (PARSER FAILURE)")
        for entry in skipped:
            print(f"  - {entry['linkedin_job']['title']}: {entry['parser_error'][:180]}")

    print("\n[PASS] Every parsed skill was classified as matching or missing.")
    print("[PASS] Both runs returned identical cosine results.")

    if args.compare_esco:
        without_esco, _ = run_cosine_fixture_test(
            use_esco=False,
            embeddings=shared_embeddings,
        )
        baseline = {item["job"]["url"]: item for item in without_esco}
        print("\nESCO A/B ON THE SAME FRESH FIXTURES")
        print(
            f"{'job':<44} {'without':>9} {'with':>9} "
            f"{'mapped requirements':>20}"
        )
        for item in results:
            job = item["job"]
            result = item["result"]
            base_result = baseline[job["url"]]["result"]
            normalization = result["skills"]["normalization"]
            print(
                f"{job['title'][:44]:<44} "
                f"{base_result['skills']['score'] * 100:>8.1f}% "
                f"{result['skills']['score'] * 100:>8.1f}% "
                f"{normalization['required_mapped']:>9}/"
                f"{normalization['required_total']:<9}"
            )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
