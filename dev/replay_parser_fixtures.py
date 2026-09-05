"""Replay captured fixtures through the deterministic parser layers only.

The fixtures store the final parser output produced with a live LLM. Feeding
those skills back as the "LLM answer" lets the deterministic cleaning, region
classification, grouping, experience and education rules be evaluated and
improved offline, without any OpenRouter call.

    python -m dev.replay_parser_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

from core.agent2_cv_parser import extract_cv_info_agent2
from core.agent2_job_parser import extract_job_requirements_agent2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CV_FIXTURE = PROJECT_ROOT / "fixtures" / "cv_parser_fixture.json"
JOB_FIXTURE = PROJECT_ROOT / "fixtures" / "linkedin_job_parser_fixture.json"


class _Replay:
    def __init__(self, payload: dict):
        self.payload = payload

    def with_structured_output(self, _schema, **_kwargs):
        return self

    def invoke(self, _prompt):
        return self.payload


def replay_cv() -> dict:
    fixture = json.loads(CV_FIXTURE.read_text(encoding="utf-8"))
    parsed = fixture["parsed_cv"]
    payload = {
        "full_name": parsed.get("full_name"),
        "skills": parsed.get("skills", []),
        "contextual_skills": [],
        "job_titles": parsed.get("job_titles", []),
        "experience_years": parsed.get("experience_years"),
        "education": parsed.get("education", []),
        "highest_education_level": parsed.get("highest_education_level"),
        "phone": parsed.get("phone"),
        "linkedin": parsed.get("linkedin"),
        "mail": parsed.get("mail"),
        "github": parsed.get("github"),
    }
    result = extract_cv_info_agent2(
        fixture["raw_text"],
        llm=_Replay(payload),
        use_cache=False,
        layout_text=fixture.get("layout_text"),
    )
    return result.model_dump()


def replay_jobs() -> list[dict]:
    fixture = json.loads(JOB_FIXTURE.read_text(encoding="utf-8"))
    results = []
    for entry in fixture["jobs"]:
        raw = entry["linkedin_job"]
        parsed = entry.get("parsed_requirements") or {}
        payload = {
            "required_skills": parsed.get("required_skills", []),
            "responsibility_skills": [],
            "preferred_skills": parsed.get("preferred_skills", []),
            "job_title": parsed.get("job_title") or raw["title"],
            "seniority_level": parsed.get("seniority_level"),
            "required_experience_years": parsed.get("required_experience_years"),
            "required_education_level": parsed.get("required_education_level"),
            "preferred_education_level": parsed.get("preferred_education_level"),
        }
        try:
            result = extract_job_requirements_agent2(
                raw["title"],
                raw["description"],
                llm=_Replay(payload),
                use_cache=False,
            ).model_dump()
        except Exception as exc:  # keep reviewing the other jobs
            result = {"error": f"{type(exc).__name__}: {exc}"}
        results.append({"company": raw["company"], "title": raw["title"], **result})
    return results


def main() -> int:
    cv = replay_cv()
    print("=== CV ===")
    print("name:", cv["full_name"], "| education:", cv["highest_education_level"],
          "| experience:", cv["experience_years"])
    print("job_titles:", cv["job_titles"])
    print(f"skills ({len(cv['skills'])}):", cv["skills"])
    for job in replay_jobs():
        print("\n===", job["company"], "|", job["title"], "===")
        if "error" in job:
            print(job["error"])
            continue
        print("experience:", job["required_experience_years"],
              "| education:", job["required_education_level"],
              "| preferred education:", job["preferred_education_level"])
        print(f"required ({len(job['required_skills'])}):", job["required_skills"])
        print(f"preferred ({len(job['preferred_skills'])}):", job["preferred_skills"])
        print("groups:", job["required_skill_groups"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
