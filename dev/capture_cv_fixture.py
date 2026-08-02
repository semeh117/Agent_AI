"""
capture_cv_fixture.py
--------------
Run this manually, occasionally, to (re)freeze one CV's extracted
CVInfo to disk as a test fixture. Mirrors capture_job_fixture.py's
purpose, but for the CV side of the pipeline.

Why this exists:
extract_cv_info() is itself an LLM call — if a test script re-extracts
the CV fresh on every run, any difference in test output could come
from CV-extraction drift instead of the thing actually being tested
(job_parser.py / matcher.py). Freezing CVInfo once removes that variable,
so test_pipeline_on_fixture.py can isolate model quality on the job side
with a fully controlled candidate profile on the CV side.

Usage:
    python -m dev.capture_cv_fixture cv/some_candidate.pdf
    python -m dev.capture_cv_fixture cv/some_candidate.pdf --out fixtures/cv_info_fixture.json
"""

import argparse
import json
from pathlib import Path
from core.cv_parser import extract_text_from_pdf, extract_cv_info

DEFAULT_FIXTURE_PATH = "fixtures/cv_info_fixture.json"


def main():
    parser = argparse.ArgumentParser(description="Capture one CV's extracted CVInfo as a frozen fixture.")
    parser.add_argument("cv_path", help="Path to the CV PDF, e.g. cv/candidate.pdf")
    parser.add_argument("--out", default=DEFAULT_FIXTURE_PATH, help=f"Output path (default {DEFAULT_FIXTURE_PATH})")
    args = parser.parse_args()

    print(f"Extracting text from {args.cv_path}...")
    raw_text = extract_text_from_pdf(args.cv_path)

    print("Calling LLM to extract CVInfo (this is the only LLM call this fixture will ever need)...")
    cv_info = extract_cv_info(raw_text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixture = {
        "source_cv_path": args.cv_path,
        "cv_info": cv_info.model_dump(),
    }
    out_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  -> {cv_info.full_name}, {len(cv_info.skills)} skills, "
          f"{cv_info.experience_years} yrs, {cv_info.highest_education_level}")
    print(f"  -> Captured to {out_path}")
    print("  This fixture is now frozen — reruns of load_cv_fixture() will always "
          "return this exact same CVInfo until you recapture it.")


if __name__ == "__main__":
    main()